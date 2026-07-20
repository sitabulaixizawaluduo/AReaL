# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from torch import Tensor

from areal.models.mcore.lightning_attention import (
    _build_zigzag_redo_indices,
    _build_zigzag_undo_indices,
)

try:
    from fla.modules import FusedRMSNormGated
except ImportError:  # pragma: no cover
    FusedRMSNormGated = None

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:  # pragma: no cover
    chunk_gated_delta_rule = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:  # pragma: no cover
    causal_conv1d_fn = None

try:
    from torch.distributed._functional_collectives import all_to_all_single_autograd
except ImportError:  # pragma: no cover
    all_to_all_single_autograd = None


def _get_tp_world_size() -> int:
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_tensor_model_parallel_world_size()
    except (RuntimeError, AttributeError):
        pass
    return 1


def _get_cp_world_size() -> int:
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_context_parallel_world_size()
    except (RuntimeError, AttributeError):
        pass
    return 1


def _get_cp_rank() -> int:
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_context_parallel_rank()
    except (RuntimeError, AttributeError):
        pass
    return 0


def _get_cp_group():
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_context_parallel_group()
    except (RuntimeError, AttributeError):
        pass
    return None


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input_: Tensor):
        ctx.group = group
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return input_
        output = torch.empty_like(input_)
        dist.all_to_all_single(output, input_.contiguous(), group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, _AllToAll.apply(ctx.group, grad_output)


def _all_to_all_equal(input_: Tensor, cp_group) -> Tensor:
    if cp_group is None:
        return input_
    cp_size = dist.get_world_size(group=cp_group)
    if cp_size == 1:
        return input_
    flat = input_.contiguous().view(-1)
    if all_to_all_single_autograd is not None:
        out = all_to_all_single_autograd(flat, None, None, group=cp_group)
    else:  # pragma: no cover
        out = _AllToAll.apply(cp_group, flat)
    return out.view_as(input_)


def _all_to_all_cp2hp(
    input_: Tensor,
    cp_group,
    split_size_or_sections: list[int] | None = None,
) -> Tensor:
    """All-to-all from CP sequence shard to head shard for [S/CP, B, H]."""
    if cp_group is None:
        return input_
    cp_size = dist.get_world_size(group=cp_group)
    if cp_size == 1:
        return input_
    if split_size_or_sections is not None:
        chunks = torch.split(input_, split_size_or_sections, dim=-1)
        return torch.cat(
            [_all_to_all_cp2hp(chunk, cp_group) for chunk in chunks],
            dim=-1,
        )

    seq_len, batch, hidden = input_.shape
    if hidden % cp_size != 0:
        raise ValueError(
            f"Qwen3.5 GDN CP all2all requires hidden dim {hidden} divisible by CP {cp_size}."
        )
    hidden_per_cp = hidden // cp_size
    flat = input_.reshape(seq_len * batch, hidden)
    flat = torch.cat(torch.split(flat, hidden_per_cp, dim=-1), dim=0)
    flat = _all_to_all_equal(flat, cp_group)
    return flat.reshape(seq_len * cp_size, batch, hidden_per_cp)


def _all_to_all_hp2cp(input_: Tensor, cp_group) -> Tensor:
    """All-to-all from head shard to CP sequence shard for [S, B, H/CP]."""
    if cp_group is None:
        return input_
    cp_size = dist.get_world_size(group=cp_group)
    if cp_size == 1:
        return input_

    seq_len, batch, hidden_per_cp = input_.shape
    if seq_len % cp_size != 0:
        raise ValueError(
            f"Qwen3.5 GDN CP all2all requires sequence dim {seq_len} divisible by CP {cp_size}."
        )
    seq_per_cp = seq_len // cp_size
    flat = input_.reshape(seq_len * batch, hidden_per_cp)
    flat = _all_to_all_equal(flat, cp_group)
    chunks = torch.split(flat, seq_per_cp * batch, dim=0)
    return torch.cat(chunks, dim=-1).reshape(seq_per_cp, batch, hidden_per_cp * cp_size)


def _get_parameter_local_cp(
    param: Tensor,
    dim: int,
    cp_rank: int,
    cp_size: int,
    split_size_or_sections: list[int] | None = None,
) -> Tensor:
    if cp_size == 1:
        return param
    if split_size_or_sections is not None:
        chunks = torch.split(param, split_size_or_sections, dim=dim)
        return torch.cat(
            [_get_parameter_local_cp(chunk, dim, cp_rank, cp_size) for chunk in chunks],
            dim=dim,
        )
    dim_size = param.size(dim)
    if dim_size % cp_size != 0:
        raise ValueError(
            f"Cannot CP-slice tensor dim {dim} of size {dim_size} by CP {cp_size}."
        )
    per_rank = dim_size // cp_size
    slices = [slice(None)] * param.dim()
    slices[dim] = slice(cp_rank * per_rank, (cp_rank + 1) * per_rank)
    return param[tuple(slices)]


class _RMSNormGatedFallback(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor, gate: Tensor | None = None) -> Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = self.weight * x.to(input_dtype)
        if gate is not None:
            x = x * F.silu(gate.to(torch.float32))
        return x.to(input_dtype)


def _split_tensor_factory(
    orig_sh_ten: ShardedTensor,
    split_sections: list[int],
    split_names: list[str],
    split_dim: int,
) -> ShardedTensorFactory:
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()

    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover dimension size: {split_sections=} vs "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )
    assert len(split_sections) == len(split_names)

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str,
        t: torch.Tensor,
        replica_id: ReplicaId,
        flattened_range: slice | None,
    ):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key,
            data=t,
            dtype=t.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )
        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
            for sh_ten in split_chunks:
                sh_ten.key = f"{sh_ten.key}.{split_name}"
            chunk_sh_tens.extend(split_chunks)
            split_start += split_size
        return chunk_sh_tens

    @torch.no_grad()
    def sh_ten_merge_fn(sub_state_dict):
        return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        orig_sh_ten.key,
        orig_sh_ten.data,
        sh_ten_build_fn,
        sh_ten_merge_fn,
        orig_sh_ten.replica_id,
    )


@dataclass
class Qwen3_5GatedDeltaNetSubmodules:
    in_proj_qkv: ModuleSpec | type
    in_proj_z: ModuleSpec | type
    in_proj_b: ModuleSpec | type
    in_proj_a: ModuleSpec | type
    out_proj: ModuleSpec | type


class Qwen3_5GatedDeltaNet(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: Qwen3_5GatedDeltaNetSubmodules,
        *,
        linear_num_key_heads: int,
        linear_num_value_heads: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_conv_kernel_dim: int,
        hidden_act: str = "silu",
        bias: bool = False,
        conv_bias: bool = False,
    ):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_k_heads = linear_num_key_heads
        self.num_v_heads = linear_num_value_heads
        self.head_k_dim = linear_key_head_dim
        self.head_v_dim = linear_value_head_dim
        self.conv_kernel_size = linear_conv_kernel_dim
        self.hidden_act = hidden_act
        self.bias = bias
        self.conv_bias = conv_bias

        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.tp_size = _get_tp_world_size()
        self.cp_size = _get_cp_world_size()
        self.cp_rank = _get_cp_rank()

        if self.num_k_heads % self.tp_size != 0 or self.num_v_heads % self.tp_size != 0:
            raise ValueError(
                f"Qwen3.5 GDN requires linear_num_key_heads ({self.num_k_heads}) and "
                f"linear_num_value_heads ({self.num_v_heads}) divisible by TP ({self.tp_size})."
            )

        self.num_k_heads_local_tp = self.num_k_heads // self.tp_size
        self.num_v_heads_local_tp = self.num_v_heads // self.tp_size
        self.key_dim_local_tp = self.num_k_heads_local_tp * self.head_k_dim
        self.value_dim_local_tp = self.num_v_heads_local_tp * self.head_v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size

        self.in_proj_qkv = build_module(
            submodules.in_proj_qkv,
            self.hidden_size,
            self.conv_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
        )
        self.in_proj_z = build_module(
            submodules.in_proj_z,
            self.hidden_size,
            self.value_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
        )
        self.in_proj_b = build_module(
            submodules.in_proj_b,
            self.hidden_size,
            self.num_v_heads,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
        )
        self.in_proj_a = build_module(
            submodules.in_proj_a,
            self.hidden_size,
            self.num_v_heads,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            bias=conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_size - 1,
            device=torch.cuda.current_device() if torch.cuda.is_available() else None,
            dtype=config.params_dtype,
        )
        setattr(self.conv1d.weight, "tensor_model_parallel", True)
        setattr(self.conv1d.weight, "partition_dim", 0)
        if conv_bias:
            setattr(self.conv1d.bias, "tensor_model_parallel", True)
            setattr(self.conv1d.bias, "partition_dim", 0)

        with get_cuda_rng_tracker().fork():
            self.dt_bias = nn.Parameter(
                torch.ones(
                    self.num_v_heads_local_tp,
                    dtype=torch.float32,
                    device=torch.cuda.current_device()
                    if torch.cuda.is_available()
                    else None,
                )
            )
            A = torch.empty(
                self.num_v_heads_local_tp,
                dtype=torch.float32,
                device=torch.cuda.current_device()
                if torch.cuda.is_available()
                else None,
            ).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
        setattr(self.dt_bias, "tensor_model_parallel", True)
        setattr(self.dt_bias, "partition_dim", 0)
        setattr(self.A_log, "tensor_model_parallel", True)
        setattr(self.A_log, "partition_dim", 0)

        if FusedRMSNormGated is not None and torch.cuda.is_available():
            self.norm = FusedRMSNormGated(
                self.head_v_dim,
                eps=self.config.layernorm_epsilon,
                activation=self.hidden_act,
                device=torch.cuda.current_device(),
                dtype=self.config.params_dtype,
            )
        else:
            self.norm = _RMSNormGatedFallback(
                self.head_v_dim,
                eps=self.config.layernorm_epsilon,
            )

        self.out_proj = build_module(
            submodules.out_proj,
            self.value_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
        )

    def _pad_packed_qkv(self, qkv: Tensor, cu_seqlens: Tensor) -> tuple[Tensor, int]:
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_size = len(seqlens)
        max_seqlen = int(seqlens.max().item())
        padded = torch.zeros(
            batch_size,
            max_seqlen,
            qkv.shape[-1],
            dtype=qkv.dtype,
            device=qkv.device,
        )
        total_tokens = int(cu_seqlens[-1].item())
        batch_indices = torch.arange(
            batch_size, device=qkv.device, dtype=torch.long
        ).repeat_interleave(seqlens)
        offsets = cu_seqlens[:-1].repeat_interleave(seqlens)
        seq_indices = (
            torch.arange(total_tokens, device=qkv.device, dtype=torch.long) - offsets
        )
        padded[batch_indices, seq_indices] = qkv
        return padded.transpose(1, 2).contiguous(), max_seqlen

    def _unpad_packed_qkv(self, padded_qkv: Tensor, cu_seqlens: Tensor) -> Tensor:
        padded_qkv = padded_qkv.transpose(1, 2)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_size = len(seqlens)
        total_tokens = int(cu_seqlens[-1].item())
        batch_indices = torch.arange(
            batch_size, device=padded_qkv.device, dtype=torch.long
        ).repeat_interleave(seqlens)
        offsets = cu_seqlens[:-1].repeat_interleave(seqlens)
        seq_indices = (
            torch.arange(total_tokens, device=padded_qkv.device, dtype=torch.long)
            - offsets
        )
        return padded_qkv[batch_indices, seq_indices]

    def _apply_conv_qkv(
        self,
        qkv: Tensor,
        conv1d_weight: Tensor,
        conv1d_bias: Tensor | None,
        seq_len_for_conv: int,
        cu_seqlens_q: Tensor | None,
    ) -> Tensor:
        # qkv: [B, S, D]
        if cu_seqlens_q is not None:
            b, s, d = qkv.shape
            if causal_conv1d_fn is None:
                qkv_flat = qkv.reshape(-1, d)
                qkv_padded, max_seqlen_q = self._pad_packed_qkv(qkv_flat, cu_seqlens_q)
                qkv_conv = F.silu(
                    F.conv1d(
                        qkv_padded,
                        conv1d_weight,
                        conv1d_bias,
                        padding=self.conv_kernel_size - 1,
                        groups=conv1d_weight.shape[0],
                    )
                )[..., :max_seqlen_q]
                return self._unpad_packed_qkv(qkv_conv, cu_seqlens_q).reshape(b, s, d)

            seqlens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            seq_idx = (
                torch.repeat_interleave(
                    torch.arange(len(seqlens), device=qkv.device, dtype=torch.int32),
                    seqlens,
                )
                .unsqueeze(0)
                .contiguous()
            )
            qkv_input = qkv.reshape(1, -1, d).contiguous().transpose(1, 2)
            qkv_conv = causal_conv1d_fn(
                x=qkv_input,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=self.hidden_act,
                seq_idx=seq_idx,
            )
            return qkv_conv.transpose(1, 2).reshape(b, s, d)

        qkv = qkv.transpose(1, 2).contiguous()
        if causal_conv1d_fn is None:
            qkv = F.silu(
                F.conv1d(
                    qkv,
                    conv1d_weight,
                    conv1d_bias,
                    padding=self.conv_kernel_size - 1,
                    groups=conv1d_weight.shape[0],
                )
            )[..., :seq_len_for_conv]
        else:
            qkv = causal_conv1d_fn(
                x=qkv,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=self.hidden_act,
            )
        return qkv.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if chunk_gated_delta_rule is None:
            raise ImportError(
                "flash-linear-attention is required for Qwen3.5 GDN "
                "(missing fla.ops.gated_delta_rule.chunk_gated_delta_rule)."
            )

        seq_len, batch, _ = hidden_states.shape
        cu_seqlens_q = None
        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q

        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        b, _ = self.in_proj_b(hidden_states)
        a, _ = self.in_proj_a(hidden_states)

        cp_group = _get_cp_group() if self.cp_size > 1 else None
        undo_idx = None
        redo_idx = None

        qkv_split_sections = [
            self.key_dim_local_tp,
            self.key_dim_local_tp,
            self.value_dim_local_tp,
        ]

        if self.cp_size > 1:
            mixed_qkv = _all_to_all_cp2hp(
                mixed_qkv,
                cp_group,
                split_size_or_sections=qkv_split_sections,
            )
            z = _all_to_all_cp2hp(z, cp_group)
            b = _all_to_all_cp2hp(b, cp_group)
            a = _all_to_all_cp2hp(a, cp_group)

            undo_idx = _build_zigzag_undo_indices(
                mixed_qkv.shape[0],
                self.cp_size,
                cu_seqlens_q,
                mixed_qkv.device,
            )
            redo_idx = _build_zigzag_redo_indices(undo_idx)

            mixed_qkv = mixed_qkv[undo_idx]
            z = z[undo_idx]
            b = b[undo_idx]
            a = a[undo_idx]

        mixed_qkv = mixed_qkv.transpose(0, 1).contiguous()
        z = z.transpose(0, 1).contiguous()
        b = b.transpose(0, 1).contiguous()
        a = a.transpose(0, 1).contiguous()

        seq_len_for_conv = mixed_qkv.shape[1]
        conv1d_weight = self.conv1d.weight
        conv1d_bias = self.conv1d.bias

        if self.cp_size > 1:
            conv1d_weight = _get_parameter_local_cp(
                conv1d_weight,
                dim=0,
                cp_rank=self.cp_rank,
                cp_size=self.cp_size,
                split_size_or_sections=qkv_split_sections,
            )
            if conv1d_bias is not None:
                conv1d_bias = _get_parameter_local_cp(
                    conv1d_bias,
                    dim=0,
                    cp_rank=self.cp_rank,
                    cp_size=self.cp_size,
                    split_size_or_sections=qkv_split_sections,
                )

        mixed_qkv = self._apply_conv_qkv(
            mixed_qkv,
            conv1d_weight,
            conv1d_bias,
            seq_len_for_conv,
            cu_seqlens_q,
        )

        cp_divisor = self.cp_size if self.cp_size > 1 else 1
        num_k_heads_local = self.num_k_heads_local_tp // cp_divisor
        num_v_heads_local = self.num_v_heads_local_tp // cp_divisor
        key_dim_local = self.key_dim_local_tp // cp_divisor
        value_dim_local = self.value_dim_local_tp // cp_divisor

        query, key, value = torch.split(
            mixed_qkv,
            [key_dim_local, key_dim_local, value_dim_local],
            dim=-1,
        )

        query = query.reshape(batch, seq_len_for_conv, num_k_heads_local, self.head_k_dim)
        key = key.reshape(batch, seq_len_for_conv, num_k_heads_local, self.head_k_dim)
        value = value.reshape(batch, seq_len_for_conv, num_v_heads_local, self.head_v_dim)
        z = z.reshape(batch, seq_len_for_conv, num_v_heads_local, self.head_v_dim)
        beta = b.sigmoid()

        A_log = self.A_log
        dt_bias = self.dt_bias
        if self.cp_size > 1:
            A_log = _get_parameter_local_cp(A_log, dim=0, cp_rank=self.cp_rank, cp_size=self.cp_size)
            dt_bias = _get_parameter_local_cp(dt_bias, dim=0, cp_rank=self.cp_rank, cp_size=self.cp_size)

        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

        if num_v_heads_local // num_k_heads_local > 1:
            repeat = num_v_heads_local // num_k_heads_local
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        if cu_seqlens_q is not None:
            query_k = query.reshape(1, batch * seq_len_for_conv, -1, self.head_k_dim)
            key_k = key.reshape(1, batch * seq_len_for_conv, -1, self.head_k_dim)
            value_k = value.reshape(1, batch * seq_len_for_conv, -1, self.head_v_dim)
            beta_k = beta.reshape(1, batch * seq_len_for_conv, -1)
            g_k = g.reshape(1, batch * seq_len_for_conv, -1)
        else:
            query_k = query.contiguous()
            key_k = key.contiguous()
            value_k = value.contiguous()
            beta_k = beta.contiguous()
            g_k = g.contiguous()

        core_attn_out, _ = chunk_gated_delta_rule(
            query_k,
            key_k,
            value_k,
            g=g_k,
            beta=beta_k,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens_q,
        )

        if cu_seqlens_q is not None:
            core_attn_out = core_attn_out.reshape(
                batch,
                seq_len_for_conv,
                num_v_heads_local,
                self.head_v_dim,
            )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_2d = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_2d)
        core_attn_out = core_attn_out.reshape(
            batch,
            seq_len_for_conv,
            num_v_heads_local,
            self.head_v_dim,
        )
        core_attn_out = core_attn_out.reshape(batch, seq_len_for_conv, value_dim_local)
        core_attn_out = core_attn_out.transpose(0, 1).contiguous()

        if self.cp_size > 1:
            core_attn_out = core_attn_out[redo_idx]
            core_attn_out = _all_to_all_hp2cp(core_attn_out, cp_group)

        out, out_bias = self.out_proj(core_attn_out)
        return out, out_bias

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        sharded_sd = {}
        self._save_to_state_dict(sharded_sd, "", keep_vars=True)
        sharded_sd = make_sharded_tensors_for_checkpoint(
            sharded_sd,
            prefix,
            tensor_parallel_layers_axis_map={"A_log": 0, "dt_bias": 0},
            sharded_offsets=sharded_offsets,
        )

        for name, module in self.named_children():
            if name == "conv1d":
                module_sd = module.state_dict(prefix="", keep_vars=True)
                tp_sharding_map = {"weight": 0}
                if self.conv_bias:
                    tp_sharding_map["bias"] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    tp_sharding_map,
                    sharded_offsets,
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module,
                    f"{prefix}{name}.",
                    sharded_offsets,
                    metadata,
                )
            sharded_sd.update(module_sharded_sd)

        qkv_key = f"{prefix}in_proj_qkv.weight"
        if qkv_key in sharded_sd and isinstance(sharded_sd[qkv_key], ShardedTensor):
            sharded_sd[qkv_key] = _split_tensor_factory(
                sharded_sd[qkv_key],
                [self.key_dim_local_tp, self.key_dim_local_tp, self.value_dim_local_tp],
                ["query", "key", "value"],
                0,
            )

        conv_key = f"{prefix}conv1d.weight"
        if conv_key in sharded_sd and isinstance(sharded_sd[conv_key], ShardedTensor):
            sharded_sd[conv_key] = _split_tensor_factory(
                sharded_sd[conv_key],
                [self.key_dim_local_tp, self.key_dim_local_tp, self.value_dim_local_tp],
                ["query", "key", "value"],
                0,
            )

        return sharded_sd


@dataclass
class Qwen3_5GatedDeltaAttentionSubmodules:
    linear_attn: ModuleSpec | type
    input_layernorm: ModuleSpec | type


class Qwen3_5GatedDeltaAttention(MegatronModule):
    """Self-attention replacement for Qwen3.5 linear-attention layers.

    Naming contract:
    - self_attention.input_layernorm.weight
    - self_attention.linear_attn.{...}
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Qwen3_5GatedDeltaAttentionSubmodules,
        layer_number: int,
        attn_mask_type=None,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.linear_attn = build_module(
            submodules.linear_attn,
            config=self.config,
            **kwargs,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params: PackedSeqParams | None = None,
        **kwargs,
    ):
        hidden_states = self.input_layernorm(hidden_states)
        return self.linear_attn(
            hidden_states,
            packed_seq_params=packed_seq_params,
        )
