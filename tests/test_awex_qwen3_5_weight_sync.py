# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Qwen3.5-MoE awex weight-sync name protocol.

Validates, without GPUs, the interlock that end-to-end transfer correctness
depends on:

1. train-side split names == inference-side unfuse names (transfer plan pairs
   by name; one mismatch means a missing weight or a failed connect);
2. for every common name, the inference rank's local tensor equals the slice
   of the train-side full tensor declared by the sharding strategy
   (dim / rank offset) -- i.e. the unfuse layouts and the sharding table are
   mutually consistent;
3. train-side PP x EP ownership tiles the full parameter set exactly;
4. the gateway inference-meta merge repairs the ``{name: meta}`` collapse for
   TP > 1 while deduplicating multi-engine duplicates;
5. awex's real ``TransferPlanBuilder`` produces full coverage for every
   inference shard under these metas.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "awex",
    reason="awex (whose import chain requires megatron-core) is not installed",
)

from areal.v2.weight_update.awex.qwen3_5 import (
    Qwen3_5MoeShardingStrategy,
    TrainOwnership,
    normalize_train_hf_name,
    split_train_hf_param,
    unfuse_sglang_param,
)

HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
GDN_K_HEADS = 2
GDN_V_HEADS = 4
GDN_HEAD_K = 8
GDN_HEAD_V = 8
KEY_DIM = GDN_K_HEADS * GDN_HEAD_K  # 16
VALUE_DIM = GDN_V_HEADS * GDN_HEAD_V  # 32
CONV_KERNEL = 4
NUM_EXPERTS = 4
MOE_INTER = 16
SHARED_INTER = 16
VOCAB = 64
VIS_HIDDEN = 16
VIS_HEADS = 2
VIS_INTER = 32
VIS_POS = 16
VIS_PATCH_IN = 48  # in_channels * temporal_patch * patch^2 flattened for conv


def make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        attn_output_gate=True,
        linear_num_key_heads=GDN_K_HEADS,
        linear_num_value_heads=GDN_V_HEADS,
        linear_key_head_dim=GDN_HEAD_K,
        linear_value_head_dim=GDN_HEAD_V,
        linear_conv_kernel_dim=CONV_KERNEL,
        num_experts=NUM_EXPERTS,
    )


def _t(*shape: int) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= s
    return torch.arange(n, dtype=torch.float32).reshape(*shape)


def make_hf_state() -> dict[str, torch.Tensor]:
    """Full (gathered) HF-name tensors: layer 0 linear-attn, layer 1 attn."""
    state: dict[str, torch.Tensor] = {}
    l0 = "model.layers.0"
    state[f"{l0}.linear_attn.in_proj_qkv.weight"] = _t(2 * KEY_DIM + VALUE_DIM, HIDDEN)
    state[f"{l0}.linear_attn.in_proj_z.weight"] = _t(VALUE_DIM, HIDDEN)
    state[f"{l0}.linear_attn.in_proj_b.weight"] = _t(GDN_V_HEADS, HIDDEN)
    state[f"{l0}.linear_attn.in_proj_a.weight"] = _t(GDN_V_HEADS, HIDDEN)
    state[f"{l0}.linear_attn.conv1d.weight"] = _t(
        2 * KEY_DIM + VALUE_DIM, 1, CONV_KERNEL
    )
    state[f"{l0}.linear_attn.A_log"] = _t(GDN_V_HEADS)
    state[f"{l0}.linear_attn.dt_bias"] = _t(GDN_V_HEADS)
    state[f"{l0}.linear_attn.norm.weight"] = _t(GDN_HEAD_V)
    state[f"{l0}.linear_attn.out_proj.weight"] = _t(HIDDEN, VALUE_DIM)

    l1 = "model.layers.1"
    state[f"{l1}.self_attn.q_proj.weight"] = _t(HEADS * 2 * HEAD_DIM, HIDDEN)
    state[f"{l1}.self_attn.k_proj.weight"] = _t(KV_HEADS * HEAD_DIM, HIDDEN)
    state[f"{l1}.self_attn.v_proj.weight"] = _t(KV_HEADS * HEAD_DIM, HIDDEN)
    state[f"{l1}.self_attn.o_proj.weight"] = _t(HIDDEN, HEADS * HEAD_DIM)
    state[f"{l1}.self_attn.q_norm.weight"] = _t(HEAD_DIM)
    state[f"{l1}.self_attn.k_norm.weight"] = _t(HEAD_DIM)

    for layer in (l0, l1):
        state[f"{layer}.input_layernorm.weight"] = _t(HIDDEN)
        state[f"{layer}.post_attention_layernorm.weight"] = _t(HIDDEN)
        state[f"{layer}.mlp.gate.weight"] = _t(NUM_EXPERTS, HIDDEN)
        state[f"{layer}.mlp.experts.gate_up_proj"] = _t(
            NUM_EXPERTS, 2 * MOE_INTER, HIDDEN
        )
        state[f"{layer}.mlp.experts.down_proj"] = _t(NUM_EXPERTS, HIDDEN, MOE_INTER)
        state[f"{layer}.mlp.shared_expert.gate_proj.weight"] = _t(SHARED_INTER, HIDDEN)
        state[f"{layer}.mlp.shared_expert.up_proj.weight"] = _t(SHARED_INTER, HIDDEN)
        state[f"{layer}.mlp.shared_expert.down_proj.weight"] = _t(HIDDEN, SHARED_INTER)
        state[f"{layer}.mlp.shared_expert_gate.weight"] = _t(1, HIDDEN)

    state["model.embed_tokens.weight"] = _t(VOCAB, HIDDEN)
    state["model.norm.weight"] = _t(HIDDEN)
    state["lm_head.weight"] = _t(VOCAB, HIDDEN)

    vb = "model.visual.blocks.0"
    state[f"{vb}.attn.qkv.weight"] = _t(3 * VIS_HIDDEN, VIS_HIDDEN)
    state[f"{vb}.attn.qkv.bias"] = _t(3 * VIS_HIDDEN)
    state[f"{vb}.attn.proj.weight"] = _t(VIS_HIDDEN, VIS_HIDDEN)
    state[f"{vb}.attn.proj.bias"] = _t(VIS_HIDDEN)
    state[f"{vb}.mlp.linear_fc1.weight"] = _t(VIS_INTER, VIS_HIDDEN)
    state[f"{vb}.mlp.linear_fc1.bias"] = _t(VIS_INTER)
    state[f"{vb}.mlp.linear_fc2.weight"] = _t(VIS_HIDDEN, VIS_INTER)
    state[f"{vb}.mlp.linear_fc2.bias"] = _t(VIS_HIDDEN)
    state[f"{vb}.norm1.weight"] = _t(VIS_HIDDEN)
    state[f"{vb}.norm1.bias"] = _t(VIS_HIDDEN)
    state[f"{vb}.norm2.weight"] = _t(VIS_HIDDEN)
    state[f"{vb}.norm2.bias"] = _t(VIS_HIDDEN)
    state["model.visual.patch_embed.proj.weight"] = _t(VIS_HIDDEN, VIS_PATCH_IN)
    state["model.visual.patch_embed.proj.bias"] = _t(VIS_HIDDEN)
    state["model.visual.pos_embed.weight"] = _t(VIS_POS, VIS_HIDDEN)
    state["model.visual.merger.norm.weight"] = _t(4 * VIS_HIDDEN)
    state["model.visual.merger.norm.bias"] = _t(4 * VIS_HIDDEN)
    state["model.visual.merger.linear_fc1.weight"] = _t(VIS_INTER, 4 * VIS_HIDDEN)
    state["model.visual.merger.linear_fc1.bias"] = _t(VIS_INTER)
    state["model.visual.merger.linear_fc2.weight"] = _t(HIDDEN, VIS_INTER)
    state["model.visual.merger.linear_fc2.bias"] = _t(HIDDEN)
    return state


def make_train_common(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cfg = make_cfg()
    common: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        for out_name, out in split_train_hf_param(name, tensor, cfg):
            common[out_name] = out
    return common


def _rows(t: torch.Tensor, start: int, size: int) -> torch.Tensor:
    return t.narrow(0, start, size)


def make_sglang_state(
    state: dict[str, torch.Tensor], tp: int, rank: int
) -> dict[str, torch.Tensor]:
    """Reproduce sglang's per-rank fused layouts from full HF tensors.

    Mirrors ``sglang/srt/models/qwen3_5.py`` (v0.5.10.post1): QKVParallelLinear
    head sharding, MergedColumnParallelLinear per-block sharding for
    in_proj_qkvz/ba, mamba_v2 block sharding for conv1d, FusedMoE w13/w2, and
    the Qwen2Moe shared-expert fusion.
    """
    l0, l1 = "model.layers.0", "model.layers.1"
    k_tp = KEY_DIM // tp
    v_tp = VALUE_DIM // tp
    nv_tp = GDN_V_HEADS // tp
    i_tp = MOE_INTER // tp
    si_tp = SHARED_INTER // tp
    out: dict[str, torch.Tensor] = {}

    qkv = state[f"{l0}.linear_attn.in_proj_qkv.weight"]
    q_blk = _rows(qkv, 0, KEY_DIM)
    k_blk = _rows(qkv, KEY_DIM, KEY_DIM)
    v_blk = _rows(qkv, 2 * KEY_DIM, VALUE_DIM)
    z = state[f"{l0}.linear_attn.in_proj_z.weight"]
    out[f"{l0}.linear_attn.in_proj_qkvz.weight"] = torch.cat(
        [
            _rows(q_blk, rank * k_tp, k_tp),
            _rows(k_blk, rank * k_tp, k_tp),
            _rows(v_blk, rank * v_tp, v_tp),
            _rows(z, rank * v_tp, v_tp),
        ]
    )
    out[f"{l0}.linear_attn.in_proj_ba.weight"] = torch.cat(
        [
            _rows(state[f"{l0}.linear_attn.in_proj_b.weight"], rank * nv_tp, nv_tp),
            _rows(state[f"{l0}.linear_attn.in_proj_a.weight"], rank * nv_tp, nv_tp),
        ]
    )
    conv = state[f"{l0}.linear_attn.conv1d.weight"]
    out[f"{l0}.linear_attn.conv1d.weight"] = torch.cat(
        [
            _rows(conv, rank * k_tp, k_tp),
            _rows(conv, KEY_DIM + rank * k_tp, k_tp),
            _rows(conv, 2 * KEY_DIM + rank * v_tp, v_tp),
        ]
    )
    out[f"{l0}.linear_attn.A_log"] = _rows(
        state[f"{l0}.linear_attn.A_log"], rank * nv_tp, nv_tp
    )
    out[f"{l0}.linear_attn.dt_bias"] = _rows(
        state[f"{l0}.linear_attn.dt_bias"], rank * nv_tp, nv_tp
    )
    out[f"{l0}.linear_attn.norm.weight"] = state[f"{l0}.linear_attn.norm.weight"]
    out[f"{l0}.linear_attn.out_proj.weight"] = state[
        f"{l0}.linear_attn.out_proj.weight"
    ].narrow(1, rank * v_tp, v_tp)

    q_sec = HEADS * 2 * HEAD_DIM // tp
    kv_sec = KV_HEADS * HEAD_DIM // tp
    out[f"{l1}.qkv_proj.weight"] = torch.cat(
        [
            _rows(state[f"{l1}.self_attn.q_proj.weight"], rank * q_sec, q_sec),
            _rows(state[f"{l1}.self_attn.k_proj.weight"], rank * kv_sec, kv_sec),
            _rows(state[f"{l1}.self_attn.v_proj.weight"], rank * kv_sec, kv_sec),
        ]
    )
    out[f"{l1}.o_proj.weight"] = state[f"{l1}.self_attn.o_proj.weight"].narrow(
        1, rank * (HEADS * HEAD_DIM // tp), HEADS * HEAD_DIM // tp
    )
    out[f"{l1}.q_norm.weight"] = state[f"{l1}.self_attn.q_norm.weight"]
    out[f"{l1}.k_norm.weight"] = state[f"{l1}.self_attn.k_norm.weight"]

    for layer in (l0, l1):
        out[f"{layer}.input_layernorm.weight"] = state[
            f"{layer}.input_layernorm.weight"
        ]
        out[f"{layer}.post_attention_layernorm.weight"] = state[
            f"{layer}.post_attention_layernorm.weight"
        ]
        out[f"{layer}.mlp.gate.weight"] = state[f"{layer}.mlp.gate.weight"]
        gup = state[f"{layer}.mlp.experts.gate_up_proj"]
        out[f"{layer}.mlp.experts.w13_weight"] = torch.stack(
            [
                torch.cat(
                    [
                        _rows(gup[e, :MOE_INTER], rank * i_tp, i_tp),
                        _rows(gup[e, MOE_INTER:], rank * i_tp, i_tp),
                    ]
                )
                for e in range(NUM_EXPERTS)
            ]
        )
        down = state[f"{layer}.mlp.experts.down_proj"]
        out[f"{layer}.mlp.experts.w2_weight"] = torch.stack(
            [down[e].narrow(1, rank * i_tp, i_tp) for e in range(NUM_EXPERTS)]
        )
        out[f"{layer}.mlp.shared_expert.gate_up_proj.weight"] = torch.cat(
            [
                _rows(
                    state[f"{layer}.mlp.shared_expert.gate_proj.weight"],
                    rank * si_tp,
                    si_tp,
                ),
                _rows(
                    state[f"{layer}.mlp.shared_expert.up_proj.weight"],
                    rank * si_tp,
                    si_tp,
                ),
            ]
        )
        out[f"{layer}.mlp.shared_expert.down_proj.weight"] = state[
            f"{layer}.mlp.shared_expert.down_proj.weight"
        ].narrow(1, rank * si_tp, si_tp)
        out[f"{layer}.mlp.shared_expert_gate.weight"] = state[
            f"{layer}.mlp.shared_expert_gate.weight"
        ]

    out["model.embed_tokens.weight"] = _rows(
        state["model.embed_tokens.weight"], rank * (VOCAB // tp), VOCAB // tp
    )
    out["model.norm.weight"] = state["model.norm.weight"]
    out["lm_head.weight"] = _rows(
        state["lm_head.weight"], rank * (VOCAB // tp), VOCAB // tp
    )

    vh_tp = VIS_HIDDEN // tp
    vi_tp = VIS_INTER // tp
    vb = "model.visual.blocks.0"
    sgl_vb = "visual.blocks.0"
    for suffix, width in (("weight", VIS_HIDDEN), ("bias", None)):
        qkv = state[f"{vb}.attn.qkv.{suffix}"]
        q_blk = _rows(qkv, 0, VIS_HIDDEN)
        k_blk = _rows(qkv, VIS_HIDDEN, VIS_HIDDEN)
        v_blk = _rows(qkv, 2 * VIS_HIDDEN, VIS_HIDDEN)
        out[f"{sgl_vb}.attn.qkv_proj.{suffix}"] = torch.cat(
            [
                _rows(q_blk, rank * vh_tp, vh_tp),
                _rows(k_blk, rank * vh_tp, vh_tp),
                _rows(v_blk, rank * vh_tp, vh_tp),
            ]
        )
    out[f"{sgl_vb}.attn.proj.weight"] = state[f"{vb}.attn.proj.weight"].narrow(
        1, rank * vh_tp, vh_tp
    )
    out[f"{sgl_vb}.attn.proj.bias"] = state[f"{vb}.attn.proj.bias"]
    out[f"{sgl_vb}.mlp.linear_fc1.weight"] = _rows(
        state[f"{vb}.mlp.linear_fc1.weight"], rank * vi_tp, vi_tp
    )
    out[f"{sgl_vb}.mlp.linear_fc1.bias"] = _rows(
        state[f"{vb}.mlp.linear_fc1.bias"], rank * vi_tp, vi_tp
    )
    out[f"{sgl_vb}.mlp.linear_fc2.weight"] = state[
        f"{vb}.mlp.linear_fc2.weight"
    ].narrow(1, rank * vi_tp, vi_tp)
    out[f"{sgl_vb}.mlp.linear_fc2.bias"] = state[f"{vb}.mlp.linear_fc2.bias"]
    for leaf in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
        out[f"{sgl_vb}.{leaf}"] = state[f"{vb}.{leaf}"]
    out["visual.patch_embed.proj.weight"] = state[
        "model.visual.patch_embed.proj.weight"
    ]
    out["visual.patch_embed.proj.bias"] = state["model.visual.patch_embed.proj.bias"]
    out["visual.pos_embed.weight"] = _rows(
        state["model.visual.pos_embed.weight"], rank * (VIS_POS // tp), VIS_POS // tp
    )
    out["visual.merger.norm.weight"] = state["model.visual.merger.norm.weight"]
    out["visual.merger.norm.bias"] = state["model.visual.merger.norm.bias"]
    out["visual.merger.linear_fc1.weight"] = _rows(
        state["model.visual.merger.linear_fc1.weight"], rank * vi_tp, vi_tp
    )
    out["visual.merger.linear_fc1.bias"] = _rows(
        state["model.visual.merger.linear_fc1.bias"], rank * vi_tp, vi_tp
    )
    out["visual.merger.linear_fc2.weight"] = state[
        "model.visual.merger.linear_fc2.weight"
    ].narrow(1, rank * vi_tp, vi_tp)
    out["visual.merger.linear_fc2.bias"] = state["model.visual.merger.linear_fc2.bias"]
    return out


def make_strategy(tp: int) -> Qwen3_5MoeShardingStrategy:
    return Qwen3_5MoeShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=None,
        tp_size=tp,
        ep_size=1,
        ep_tp_size=1,
        rank_info=SimpleNamespace(tp_size=tp),
    )


@pytest.mark.parametrize("tp", [1, 2])
def test_unfuse_matches_train_split_and_sharding_declaration(tp):
    from awex.sharding.param_sharding import ShardingType

    cfg = make_cfg()
    state = make_hf_state()
    train_common = make_train_common(state)
    strategy = make_strategy(tp)

    seen: set[str] = set()
    for rank in range(tp):
        sgl = make_sglang_state(state, tp, rank)
        for fused_name, fused in sgl.items():
            for name, local in unfuse_sglang_param(fused_name, fused, cfg, tp):
                seen.add(name)
                assert name in train_common, f"infer-only name {name}"
                full = train_common[name]
                stype, dim, num_shards = strategy.get_sharding_strategy(name)
                if stype == ShardingType.NO_SHARDING:
                    expected = full
                else:
                    assert num_shards == tp
                    expected = full.narrow(
                        dim, rank * local.shape[dim], local.shape[dim]
                    )
                torch.testing.assert_close(local, expected, rtol=0, atol=0)

    assert seen == set(train_common.keys())


def test_unfuse_maps_visual_qkv_and_passes_through_rest():
    cfg = make_cfg()
    qkv = _t(3 * VIS_HIDDEN, VIS_HIDDEN)
    parts = unfuse_sglang_param("visual.blocks.0.attn.qkv_proj.weight", qkv, cfg, 1)
    assert [n for n, _ in parts] == [
        "model.visual.blocks.0.attn.qkv_q.weight",
        "model.visual.blocks.0.attn.qkv_k.weight",
        "model.visual.blocks.0.attn.qkv_v.weight",
    ]
    torch.testing.assert_close(torch.cat([t for _, t in parts]), qkv, rtol=0, atol=0)
    pos = _t(VIS_POS, VIS_HIDDEN)
    passthrough = unfuse_sglang_param("visual.pos_embed.weight", pos, cfg, 1)
    assert [n for n, _ in passthrough] == ["model.visual.pos_embed.weight"]
    torch.testing.assert_close(passthrough[0][1], pos, rtol=0, atol=0)
    assert unfuse_sglang_param("model.mtp.layers.0.foo", _t(4, 4), cfg, 1) == []


def test_normalize_train_hf_name_strips_vl_prefix_and_keeps_visual():
    assert (
        normalize_train_hf_name("model.language_model.layers.0.mlp.gate.weight")
        == "model.layers.0.mlp.gate.weight"
    )
    assert (
        normalize_train_hf_name("model.visual.patch_embed.proj.weight")
        == "model.visual.patch_embed.proj.weight"
    )
    assert normalize_train_hf_name("mtp.layers.0.foo") is None
    assert normalize_train_hf_name("lm_head.weight") == "lm_head.weight"


def test_train_ownership_tiles_param_set_across_pp_ep():
    state = make_hf_state()
    train_common = make_train_common(state)
    names = set(train_common.keys())

    owners = {
        (pp, ep): TrainOwnership(
            owned_layers={pp},
            is_pp_first=(pp == 0),
            is_pp_last=(pp == 1),
            ep_rank=ep,
            ep_size=2,
            num_experts=NUM_EXPERTS,
        )
        for pp in (0, 1)
        for ep in (0, 1)
    }

    claimed: dict[str, list[tuple[int, int]]] = {n: [] for n in names}
    for coord, owner in owners.items():
        for name in names:
            if owner.owns(name):
                claimed[name].append(coord)

    for name, coords in claimed.items():
        assert coords, f"{name} unowned"
        if ".mlp.experts." in name:
            assert len(coords) == 1, f"expert {name} owned by {coords}"
        elif name.startswith("model.layers."):
            pp_set = {pp for pp, _ in coords}
            assert len(pp_set) == 1, f"{name} spans PP stages {pp_set}"
            assert len(coords) == 2, f"dense {name} should be on both EP ranks"
        else:
            assert len(coords) == 2, f"{name} owned by {coords}"


def test_sharding_strategy_rejects_unknown_and_dp_attention():
    strategy = make_strategy(2)
    with pytest.raises(ValueError, match="No sharding rule"):
        strategy.get_sharding_strategy("model.layers.0.mystery.weight")

    dp_strategy = make_strategy(2)
    dp_strategy.enable_dp_attention = True
    with pytest.raises(NotImplementedError):
        dp_strategy.get_sharding_strategy("model.layers.0.mlp.gate.weight")


def test_merge_infer_meta_by_name_merges_tp_and_dedupes_engines():
    from areal.v2.weight_update.gateway.app import _merge_infer_meta_by_name

    def shard(rank):
        return {"data": {"global_rank": rank}}

    def meta(rank):
        return {
            "data": {
                "name": "w",
                "shards": [shard(rank)],
                "replicas": [{"data": {"shards": [shard(rank)]}}],
            }
        }

    # tp=2 reported by two ranks, then the same instance-local metadata
    # reported again by a second DP engine replica.
    merged = _merge_infer_meta_by_name([meta(0), meta(1), meta(0), meta(1)])
    assert len(merged) == 1
    data = merged[0]["data"]
    ranks = [s["data"]["global_rank"] for s in data["shards"]]
    assert ranks == [0, 1]
    replica_ranks = [
        s["data"]["global_rank"] for s in data["replicas"][0]["data"]["shards"]
    ]
    assert replica_ranks == [0, 1]


def test_merge_training_meta_splits_identical_copies_into_replicas():
    from areal.v2.weight_update.gateway.app import _merge_training_meta_by_name

    def shard(rank, offset, shape):
        return {
            "data": {
                "global_rank": rank,
                "global_offset": list(offset),
                "shape": list(shape),
            }
        }

    def meta(name, rank, offset, shape):
        s = shard(rank, offset, shape)
        return {
            "data": {
                "name": name,
                "shards": [s],
                "replicas": [{"data": {"shards": [s]}}],
            }
        }

    # Megatron TP/CP peers: 4 identical FULL copies -> 4 separate replicas,
    # so the plan builder picks exactly one sender per parameter.
    merged = _merge_training_meta_by_name(
        [meta("w", r, (0, 0), (8, 4)) for r in range(4)]
    )
    assert len(merged) == 1
    replicas = merged[0]["data"]["replicas"]
    assert len(replicas) == 4
    assert all(len(rep["data"]["shards"]) == 1 for rep in replicas)

    # FSDP workers: complementary slices -> ONE replica holding both shards.
    merged = _merge_training_meta_by_name(
        [
            meta("w", 0, (0, 0), (4, 4)),
            meta("w", 1, (4, 0), (4, 4)),
        ]
    )
    assert len(merged) == 1
    replicas = merged[0]["data"]["replicas"]
    assert len(replicas) == 1
    assert len(replicas[0]["data"]["shards"]) == 2

    # Rotation is deterministic for a given name.
    a = _merge_training_meta_by_name([meta("w", r, (0, 0), (8, 4)) for r in range(4)])
    b = _merge_training_meta_by_name([meta("w", r, (0, 0), (8, 4)) for r in range(4)])
    ranks_a = [
        rep["data"]["shards"][0]["data"]["global_rank"]
        for rep in a[0]["data"]["replicas"]
    ]
    ranks_b = [
        rep["data"]["shards"][0]["data"]["global_rank"]
        for rep in b[0]["data"]["replicas"]
    ]
    assert ranks_a == ranks_b


def _shard_meta(name, tensor, rank, tp, stype, dim, num_shards):
    from awex.meta.weight_meta import ParameterShardMeta

    offset = [0] * tensor.dim()
    if num_shards > 1:
        offset[dim] = rank * tensor.shape[dim]
    return ParameterShardMeta(
        tp_rank=rank,
        attn_tp_rank=rank,
        pp_rank=0,
        ep_rank=0,
        ep_tp_rank=0,
        global_rank=rank,
        world_size=tp,
        engine_rank=0,
        cp_rank=0,
        cp_size=1,
        cp_mode="none",
        name=name,
        shape=tuple(tensor.shape),
        numel=int(tensor.numel()),
        dtype=tensor.dtype,
        global_offset=tuple(offset),
        sharding_type=stype,
        num_shards=num_shards,
        sharding_dim=dim,
    )


def test_transfer_plan_covers_every_infer_shard_tp2():
    from awex.meta.weight_meta import (
        ParameterMeta,
        ParameterReplicaMeta,
        ParameterShardMeta,
    )
    from awex.sharding.param_sharding import ShardingType
    from awex.transfer.transfer_plan import TransferPlanBuilder

    tp = 2
    cfg = make_cfg()
    state = make_hf_state()
    train_common = make_train_common(state)
    strategy = make_strategy(tp)

    train_meta = []
    for name, tensor in train_common.items():
        shard = ParameterShardMeta(
            tp_rank=0,
            attn_tp_rank=0,
            pp_rank=0,
            ep_rank=0,
            ep_tp_rank=0,
            global_rank=0,
            world_size=1,
            engine_rank=0,
            cp_rank=0,
            cp_size=1,
            cp_mode="none",
            name=name,
            shape=tuple(tensor.shape),
            numel=int(tensor.numel()),
            dtype=tensor.dtype,
            global_offset=tuple([0] * tensor.dim()),
            sharding_type=ShardingType.NO_SHARDING,
            num_shards=1,
            sharding_dim=0,
        )
        train_meta.append(
            ParameterMeta(
                name=name,
                global_numel=int(tensor.numel()),
                global_shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                shards=[shard],
                replicas=[ParameterReplicaMeta(shards=[shard])],
            )
        )

    infer_shards: dict[str, list] = {}
    infer_local_numel: dict[int, dict[str, int]] = {r: {} for r in range(tp)}
    for rank in range(tp):
        sgl = make_sglang_state(state, tp, rank)
        for fused_name, fused in sgl.items():
            for name, local in unfuse_sglang_param(fused_name, fused, cfg, tp):
                stype, dim, num_shards = strategy.get_sharding_strategy(name)
                infer_shards.setdefault(name, []).append(
                    _shard_meta(name, local, rank, tp, stype, dim, num_shards)
                )
                infer_local_numel[rank][name] = int(local.numel())

    infer_meta = []
    for name, shards in infer_shards.items():
        full = train_common[name]
        infer_meta.append(
            ParameterMeta(
                name=name,
                global_numel=int(full.numel()),
                global_shape=tuple(full.shape),
                dtype=full.dtype,
                shards=shards,
                replicas=[ParameterReplicaMeta(shards=shards)],
            )
        )

    builder = TransferPlanBuilder(
        infer_world_size=tp, train_world_size=1, num_infer_engines=1
    )
    for rank in range(tp):
        plan = builder.build_local_transfer_plan(
            infer_meta, train_meta, global_transfer_rank=rank
        )
        received: dict[str, int] = {}
        for ops in plan.operations.values():
            for op in ops:
                size = 1
                for dim_len in op.overlap_shape:
                    size *= dim_len
                name = op.recv_shard_meta.name
                received[name] = received.get(name, 0) + size
        for name, numel in infer_local_numel[rank].items():
            stype, _, num_shards = strategy.get_sharding_strategy(name)
            if stype == ShardingType.NO_SHARDING and rank > 0:
                # replicated params may be delivered to one replica only
                continue
            assert received.get(name, 0) >= numel, (
                f"rank {rank} param {name}: covered {received.get(name, 0)} "
                f"< local numel {numel}"
            )
