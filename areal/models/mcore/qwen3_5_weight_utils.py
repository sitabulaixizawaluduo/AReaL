# SPDX-License-Identifier: Apache-2.0

import torch


def get_qwen3_5_text_config(hf_config):
    return hf_config.text_config if hasattr(hf_config, "text_config") else hf_config


def is_qwen3_5_moe_config(hf_config) -> bool:
    text_cfg = get_qwen3_5_text_config(hf_config)
    model_type = getattr(text_cfg, "model_type", None) or getattr(
        hf_config, "model_type", None
    )
    return model_type == "qwen3_5_moe"


def _qwen3_5_head_dims(hf_config) -> tuple[int, int, int, int]:
    text_cfg = get_qwen3_5_text_config(hf_config)
    num_kv_heads = text_cfg.num_key_value_heads
    num_attn_heads = text_cfg.num_attention_heads
    head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // num_attn_heads)
    n_per_group = num_attn_heads // num_kv_heads
    return num_kv_heads, num_attn_heads, head_dim, n_per_group


def qwen3_5_gdn_qkv_section_sizes(hf_config) -> tuple[int, int, int]:
    text_cfg = get_qwen3_5_text_config(hf_config)
    key_dim = text_cfg.linear_num_key_heads * text_cfg.linear_key_head_dim
    value_dim = text_cfg.linear_num_value_heads * text_cfg.linear_value_head_dim
    return key_dim, key_dim, value_dim


def _chunk_dim(t: torch.Tensor, dim: int, parts: int) -> list[torch.Tensor]:
    if t.shape[dim] % parts != 0:
        raise ValueError(
            f"Cannot split tensor dim {dim} of shape {list(t.shape)} by {parts}."
        )
    return list(torch.chunk(t, parts, dim=dim))


def relayout_fused_sections_for_tp(
    x: torch.Tensor,
    section_sizes: tuple[int, ...] | list[int],
    tp_size: int,
    dim: int = 0,
) -> torch.Tensor:
    """Relayout [sec0|sec1|...] into TP-major [sec0_tp0|sec1_tp0|...|sec0_tp1|...]."""
    if tp_size == 1:
        return x.contiguous()
    if sum(section_sizes) != x.shape[dim]:
        raise ValueError(
            f"Section sizes {section_sizes} do not cover dim-{dim} size {x.shape[dim]}"
        )
    splits = torch.split(x, list(section_sizes), dim=dim)
    section_chunks = [_chunk_dim(section, dim, tp_size) for section in splits]
    rank_locals = [
        torch.cat([chunks[rank] for chunks in section_chunks], dim=dim)
        for rank in range(tp_size)
    ]
    return torch.cat(rank_locals, dim=dim).contiguous()


def undo_relayout_fused_sections_for_tp(
    x: torch.Tensor,
    section_sizes: tuple[int, ...] | list[int],
    tp_size: int,
    dim: int = 0,
) -> torch.Tensor:
    """Inverse of relayout_fused_sections_for_tp."""
    if tp_size == 1:
        return x.contiguous()
    local_section_sizes = []
    for size in section_sizes:
        if size % tp_size != 0:
            raise ValueError(f"Section size {size} is not divisible by TP {tp_size}.")
        local_section_sizes.append(size // tp_size)

    rank_chunks = _chunk_dim(x, dim, tp_size)
    by_section: list[list[torch.Tensor]] = [[] for _ in local_section_sizes]
    for rank_chunk in rank_chunks:
        rank_sections = torch.split(rank_chunk, local_section_sizes, dim=dim)
        for i, section in enumerate(rank_sections):
            by_section[i].append(section)
    merged_sections = [
        torch.cat(section_parts, dim=dim) for section_parts in by_section
    ]
    return torch.cat(merged_sections, dim=dim).contiguous()


def qwen3_5_gated_qkv_hf_to_mcore(
    hf_config,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Convert Qwen3.5 gated full-attn QKV from HF layout to mcore layout."""
    num_kv_heads, _, head_dim, n_per_group = _qwen3_5_head_dims(hf_config)
    group_dim = n_per_group * head_dim
    real_num_kv_heads = q.shape[0] // (2 * group_dim)
    if real_num_kv_heads == 0 or q.shape[0] % (2 * group_dim) != 0:
        raise ValueError(
            f"Invalid q_proj rows {q.shape[0]} for gated Qwen3.5 conversion: "
            f"expected multiple of {2 * group_dim}."
        )
    if real_num_kv_heads > num_kv_heads:
        raise ValueError(
            f"q_proj rows imply {real_num_kv_heads} kv groups, exceeding configured "
            f"num_key_value_heads={num_kv_heads}."
        )

    if q.dim() == 1:
        q_ = (
            q.view(real_num_kv_heads, n_per_group, 2, head_dim)
            .transpose(1, 2)
            .flatten(1, 3)
        )
        k_ = k.view(real_num_kv_heads, head_dim)
        v_ = v.view(real_num_kv_heads, head_dim)
        return torch.cat([q_, k_, v_], dim=1).reshape(-1).contiguous()

    q_ = (
        q.view(real_num_kv_heads, n_per_group, 2, head_dim, -1)
        .transpose(1, 2)
        .flatten(1, 3)
    )
    k_ = k.view(real_num_kv_heads, head_dim, -1)
    v_ = v.view(real_num_kv_heads, head_dim, -1)
    return torch.cat([q_, k_, v_], dim=1).reshape(-1, q.shape[-1]).contiguous()


def qwen3_5_gated_qkv_mcore_to_hf(
    hf_config,
    mcore_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of qwen3_5_gated_qkv_hf_to_mcore."""
    _, num_attn_heads, head_dim, n_per_group = _qwen3_5_head_dims(hf_config)
    per_kv_size = (2 * n_per_group + 2) * head_dim
    real_num_kv_heads = mcore_weights.shape[0] // per_kv_size

    if mcore_weights.dim() == 1:
        w = mcore_weights.view(real_num_kv_heads, per_kv_size)
        q, k, v = torch.split(
            w,
            [2 * n_per_group * head_dim, head_dim, head_dim],
            dim=1,
        )
        q = (
            q.view(real_num_kv_heads, 2, n_per_group, head_dim)
            .transpose(1, 2)
            .reshape(-1)
            .contiguous()
        )
        return q, k.reshape(-1).contiguous(), v.reshape(-1).contiguous()

    w = mcore_weights.view(real_num_kv_heads, per_kv_size, -1)
    q, k, v = torch.split(
        w,
        [2 * n_per_group * head_dim, head_dim, head_dim],
        dim=1,
    )
    q = (
        q.view(real_num_kv_heads, 2, n_per_group, head_dim, -1)
        .transpose(1, 2)
        .reshape(num_attn_heads * 2 * head_dim, -1)
        .contiguous()
    )
    return (
        q,
        k.reshape(-1, w.shape[-1]).contiguous(),
        v.reshape(-1, w.shape[-1]).contiguous(),
    )
