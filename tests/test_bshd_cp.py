# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the padded BSHD context-parallel zigzag helpers.

These cover the pure-tensor layout logic (split, reorder, label building,
padded 2D reconstruction) on CPU without megatron or torch.distributed.
The collective wrappers are exercised by the GPU torchrun tests.
"""

import pytest
import torch

from areal.engine.megatron_utils.bshd_cp import (
    build_bshd_cp_local_labels,
    reconstruct_padded_2d,
    reorder_cp_gathered_chunks,
    split_padded_seqs_for_context_parallel,
)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_zigzag_split_reorder_roundtrip(cp_size, batch_size):
    seqlen = 8 * cp_size
    x = torch.randn(batch_size, seqlen, 5)

    locals_per_rank = [
        split_padded_seqs_for_context_parallel(x, cp_size, r) for r in range(cp_size)
    ]
    for local in locals_per_rank:
        assert local.shape == (batch_size, seqlen // cp_size, 5)

    restored = reorder_cp_gathered_chunks(locals_per_rank, seq_dim=1)
    assert torch.equal(restored, x)


def test_zigzag_split_chunk_placement():
    # Rank r must hold chunks r and 2*cp-1-r, in that order.
    cp_size = 2
    x = torch.arange(8).unsqueeze(0)  # chunks: [0,1] [2,3] [4,5] [6,7]
    r0 = split_padded_seqs_for_context_parallel(x, cp_size, 0)
    r1 = split_padded_seqs_for_context_parallel(x, cp_size, 1)
    assert r0.squeeze(0).tolist() == [0, 1, 6, 7]
    assert r1.squeeze(0).tolist() == [2, 3, 4, 5]


def test_zigzag_split_rejects_misaligned_seqlen():
    x = torch.randn(1, 10)
    with pytest.raises(ValueError, match="divisible by 2\\*cp_size"):
        split_padded_seqs_for_context_parallel(x, cp_size=2, cp_rank=0)


def test_reconstruct_padded_2d_roundtrip():
    cu_seqlens = torch.tensor([0, 4, 12, 16])
    packed = torch.arange(16, dtype=torch.int32) + 1  # non-zero so padding is visible
    tensor_2d, valid_mask = reconstruct_padded_2d(packed, cu_seqlens)

    assert tensor_2d.shape == (3, 8)
    assert tensor_2d.dtype == torch.long
    assert valid_mask.sum().item() == 16
    assert torch.equal(tensor_2d[valid_mask], packed.to(torch.long))
    assert (tensor_2d[~valid_mask] == 0).all()


@pytest.mark.parametrize("cp_size", [2, 4])
def test_local_labels_match_full_sequence_labels(cp_size):
    """CP-local labels reassembled across ranks must equal the row-wise
    rolled labels of the full sequence, in the same layout as reassembled
    logprobs (flattened [B, S/cp] grids, zigzag order undone)."""
    batch_size, seqlen = 2, 8 * cp_size
    cu_seqlens = torch.tensor([0, seqlen, 2 * seqlen])
    packed = torch.randint(1, 1000, (batch_size * seqlen,), dtype=torch.int32)
    input_ids_2d, valid_mask = reconstruct_padded_2d(packed, cu_seqlens)

    per_rank = [
        build_bshd_cp_local_labels(input_ids_2d, cp_size, r).view(
            batch_size, seqlen // cp_size
        )
        for r in range(cp_size)
    ]
    restored = reorder_cp_gathered_chunks(per_rank, seq_dim=1)
    expected = torch.roll(input_ids_2d, shifts=-1, dims=-1)
    assert torch.equal(restored, expected)


def test_reassemble_matches_valid_mask_selection():
    """Simulate the reassemble path without collectives: per-rank local 1D
    stats, gathered and reordered, then valid-mask selected, must equal the
    stats computed on the full [B, S] grid selected by the same mask."""
    cp_size = 2
    cu_seqlens = torch.tensor([0, 6, 14])  # padded lens 6, 8 -> max_seqlen 8
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = int(seq_lens.max().item())
    assert max_seqlen % (2 * cp_size) == 0

    full = torch.randn(2, max_seqlen)
    per_rank = [
        split_padded_seqs_for_context_parallel(full, cp_size, r) for r in range(cp_size)
    ]
    restored = reorder_cp_gathered_chunks(per_rank, seq_dim=1)

    valid_mask = torch.arange(max_seqlen)[None, :] < seq_lens[:, None]
    assert torch.equal(restored[valid_mask], full[valid_mask])
    assert restored[valid_mask].shape == (int(cu_seqlens[-1].item()),)
