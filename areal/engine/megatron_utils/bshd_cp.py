# SPDX-License-Identifier: Apache-2.0

"""Context-parallel zigzag split/reassembly for padded BSHD ``[B, S, ...]`` tensors.

Models that must run the padded BSHD forward (VLMs and GDN/SSM architectures
such as Qwen3.5 — see ``requires_padded_seq``) cannot use the packed THD
context-parallel path in ``packed_context_parallel.py``. This module provides
the BSHD counterpart.

Layout: the sequence dimension is viewed as ``2 * cp_size`` equal chunks and
CP rank ``r`` keeps chunks ``(r, 2*cp_size - 1 - r)`` concatenated. This is
the causal load-balancing ("zigzag") layout that both Transformer Engine
attention and megatron-core's GDN context parallelism expect, and it matches
megatron-core's ``get_batch_on_this_cp_rank`` as well as the per-sequence
split used by the packed THD path here.

Everything in this module is pure torch (plus optional ``ProcessGroup``
arguments) so it is unit-testable on CPU without a megatron installation.
mpu-wired wrappers live in ``packed_context_parallel.py``.
"""

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_F


def split_padded_seqs_for_context_parallel(
    tensor: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Zigzag-split a padded ``[B, S, ...]`` tensor to this rank's CP-local view.

    Args:
        tensor: Tensor whose ``seq_dim`` has size ``S`` divisible by ``2*cp_size``.
        cp_size: Context parallel world size.
        cp_rank: This rank's index in the CP group.
        seq_dim: Sequence dimension to split along.

    Returns:
        CP-local tensor with ``seq_dim`` reduced to ``S / cp_size``, holding
        chunk ``cp_rank`` followed by chunk ``2*cp_size - 1 - cp_rank``.
    """
    if cp_size <= 1:
        return tensor
    seqlen = tensor.size(seq_dim)
    if seqlen % (2 * cp_size) != 0:
        raise ValueError(
            f"Sequence length {seqlen} must be divisible by 2*cp_size "
            f"({2 * cp_size}) for the context-parallel zigzag split."
        )
    chunks = tensor.chunk(2 * cp_size, dim=seq_dim)
    return torch.cat(
        [chunks[cp_rank], chunks[2 * cp_size - 1 - cp_rank]], dim=seq_dim
    ).contiguous()


def reorder_cp_gathered_chunks(
    gathered: list[torch.Tensor],
    seq_dim: int = 1,
) -> torch.Tensor:
    """Invert the zigzag split given every rank's CP-local tensor.

    Args:
        gathered: ``cp_size`` tensors of identical shape, ``gathered[r]`` being
            rank ``r``'s CP-local view (two chunks concatenated along ``seq_dim``).
        seq_dim: Sequence dimension.

    Returns:
        Full-sequence tensor with chunks back in original order. Differentiable
        (uses only ``narrow``/``cat``), so gradients flow to each local input.
    """
    cp_size = len(gathered)
    if cp_size == 1:
        return gathered[0]
    half = gathered[0].size(seq_dim) // 2
    pieces = []
    for chunk_idx in range(2 * cp_size):
        owner = chunk_idx if chunk_idx < cp_size else 2 * cp_size - 1 - chunk_idx
        offset = 0 if chunk_idx < cp_size else half
        pieces.append(gathered[owner].narrow(seq_dim, offset, half))
    return torch.cat(pieces, dim=seq_dim)


def gather_cp_padded_output(
    local: torch.Tensor,
    cp_group: dist.ProcessGroup,
    seq_dim: int = 1,
) -> torch.Tensor:
    """All-gather CP-local BSHD outputs and restore original sequence order.

    The all-gather is detached; this rank's own chunk is re-inserted so local
    gradients are preserved (same pattern as the packed THD postprocess). Use
    this for large tensors (logits) on forward-only paths.
    """
    cp_size = dist.get_world_size(cp_group)
    if cp_size <= 1:
        return local
    gathered = [torch.empty_like(local) for _ in range(cp_size)]
    dist.all_gather(gathered, local.detach().contiguous(), group=cp_group)
    gathered[dist.get_rank(cp_group)] = local
    return reorder_cp_gathered_chunks(gathered, seq_dim=seq_dim)


def reassemble_cp_padded_logprobs(
    local: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Differentiable inverse of the BSHD zigzag split for 1D per-token stats.

    Takes this rank's flattened CP-local values over the full ``[B, S/cp]``
    grid (padding positions included, so all ranks contribute equal shapes),
    all-gathers them differentiably (backward = reduce-scatter), restores the
    original ``[B, S]`` order, and selects valid positions so the result lines
    up with the packed ``[total_len]`` layout of the micro-batch.

    Args:
        local: 1D tensor of shape ``(B * S / cp_size,)`` — e.g. logprobs or
            entropy computed on the CP-local BSHD grid, flattened row-major.
        cu_seqlens: Cumulative (padded) sequence lengths, shape ``(B + 1,)``.
        cp_group: Context parallel process group.

    Returns:
        1D tensor of shape ``(cu_seqlens[-1],)`` in packed order.
    """
    cp_size = dist.get_world_size(cp_group)
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    batch_size = seq_lens.shape[0]
    max_seqlen = int(seq_lens.max().item())
    valid_mask = (
        torch.arange(max_seqlen, device=local.device)[None, :] < seq_lens[:, None]
    )
    if cp_size <= 1:
        return local.view(batch_size, max_seqlen)[valid_mask]

    local_2d = local.view(batch_size, max_seqlen // cp_size)
    gathered = dist_F.all_gather(local_2d, group=cp_group)
    full = reorder_cp_gathered_chunks(list(gathered), seq_dim=1)
    return full[valid_mask]


def reconstruct_padded_2d(
    packed: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild a padded ``[B, S]`` tensor from a packed 1D tensor.

    Mirrors the reconstruction in ``packed_context_parallel_forward`` so
    label building and output reassembly agree on the exact layout.

    Returns:
        ``(tensor_2d, valid_mask)`` where ``tensor_2d`` is ``[B, max_seqlen]``
        (int64 for integer inputs) and ``valid_mask`` is the boolean mask of
        non-padding positions.
    """
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    batch_size = seq_lens.shape[0]
    max_seqlen = int(seq_lens.max().item())
    valid_mask = (
        torch.arange(max_seqlen, device=packed.device)[None, :] < seq_lens[:, None]
    )
    if packed.dtype in (torch.int32, torch.int64):
        packed = packed.to(torch.long)
    tensor_2d = torch.zeros(
        batch_size, max_seqlen, dtype=packed.dtype, device=packed.device
    )
    tensor_2d[valid_mask] = packed
    return tensor_2d, valid_mask


def build_bshd_cp_local_labels(
    input_ids_2d: torch.Tensor,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Build CP-local next-token labels for the BSHD loss path.

    Rolls each row left by one (row-wise; the wrapped last position lands in
    padding or is excluded by the loss mask) and zigzag-splits to this rank.

    Returns:
        1D int64 tensor of shape ``(B * S / cp_size,)`` matching the layout of
        logprobs computed on the CP-local ``[B, S/cp, V]`` logits.
    """
    rolled = torch.roll(input_ids_2d, shifts=-1, dims=-1)
    local = split_padded_seqs_for_context_parallel(rolled, cp_size, cp_rank)
    return local.reshape(-1)
