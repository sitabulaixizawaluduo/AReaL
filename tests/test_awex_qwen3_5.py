# pyright: reportMissingImports=false

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from awex.sharding.param_sharding import ShardingType

from areal.engine.awex_qwen3_5 import (
    McoreToHFWeightConverterQwen3_5Moe,
    Qwen3_5MoeShardingStrategy,
    SGlangToHFWeightConverterQwen3_5Moe,
    ensure_awex_qwen3_5_registered,
)


def _make_train_converter(
    *,
    num_attention_heads: int = 4,
    num_query_groups: int = 2,
    attn_tp_size: int = 1,
    attention_output_gate: bool = True,
    hidden_size: int = 8,
    num_experts: int = 8,
    ep_size: int = 1,
    ep_rank: int = 0,
    tp_size: int = 1,
    vision_num_heads: int = 4,
    vision_head_dim: int = 2,
) -> McoreToHFWeightConverterQwen3_5Moe:
    converter = cast(
        McoreToHFWeightConverterQwen3_5Moe,
        object.__new__(McoreToHFWeightConverterQwen3_5Moe),
    )
    setattr(
        converter,
        "tf_config",
        SimpleNamespace(
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            attention_output_gate=attention_output_gate,
            add_qkv_bias=True,
            hidden_size=hidden_size,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
        ),
    )
    setattr(
        converter,
        "hf_config",
        SimpleNamespace(
            text_config=SimpleNamespace(num_experts=num_experts),
            vision_config=SimpleNamespace(
                num_heads=vision_num_heads, head_dim=vision_head_dim
            ),
        ),
    )
    setattr(
        converter,
        "rank_info",
        SimpleNamespace(
            attn_tp_size=attn_tp_size,
            tp_size=tp_size,
            ep_size=ep_size,
            ep_rank=ep_rank,
        ),
    )
    return converter


def _make_infer_converter(
    *,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    attn_tp_size: int = 1,
    attn_output_gate: bool = True,
    hidden_size: int = 8,
    num_experts: int = 8,
    pp_size: int = 1,
    ep_size: int = 1,
    ep_rank: int = 0,
    tp_size: int = 1,
    vision_num_heads: int = 4,
    vision_head_dim: int = 2,
) -> SGlangToHFWeightConverterQwen3_5Moe:
    converter = cast(
        SGlangToHFWeightConverterQwen3_5Moe,
        object.__new__(SGlangToHFWeightConverterQwen3_5Moe),
    )
    setattr(
        converter,
        "model_config",
        SimpleNamespace(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attn_output_gate=attn_output_gate,
            hidden_size=hidden_size,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            num_experts=num_experts,
        ),
    )
    setattr(
        converter,
        "full_model_config",
        SimpleNamespace(
            vision_config=SimpleNamespace(
                num_heads=vision_num_heads, head_dim=vision_head_dim
            )
        ),
    )
    setattr(
        converter,
        "rank_info",
        SimpleNamespace(
            attn_tp_size=attn_tp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            ep_size=ep_size,
            ep_rank=ep_rank,
        ),
    )
    setattr(
        converter,
        "infer_engine_config",
        SimpleNamespace(pp_size=pp_size, ep_size=ep_size),
    )
    setattr(converter, "tp_size", tp_size)

    def _use_transposed_moe_layout(name: str, parameter: torch.Tensor) -> bool:
        del name, parameter
        return False

    setattr(converter, "_use_transposed_moe_layout", _use_transposed_moe_layout)
    return converter


def _build_mcore_q_gate_k_v_interleaved(
    *,
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    local_q_heads: int,
    local_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    heads_per_group = local_q_heads // local_kv_heads
    total_heads_per_group = 2 * heads_per_group + 2
    qkv_total_dim = total_heads_per_group * local_kv_heads
    feature_dim = q_block.shape[1]

    packed = torch.empty(qkv_total_dim, head_dim, feature_dim, dtype=q_block.dtype)

    q_heads = q_block.view(local_q_heads, 2 * head_dim, feature_dim)
    k_heads = k_block.view(local_kv_heads, head_dim, feature_dim)
    v_heads = v_block.view(local_kv_heads, head_dim, feature_dim)

    q_head_cursor = 0
    for kv_group in range(local_kv_heads):
        group_base = kv_group * total_heads_per_group
        for inner_q in range(heads_per_group):
            qkv_index = group_base + inner_q
            packed[qkv_index] = q_heads[q_head_cursor, :head_dim]
            gate_index = group_base + heads_per_group + inner_q
            packed[gate_index] = q_heads[q_head_cursor, head_dim:]
            q_head_cursor += 1

        packed[group_base + total_heads_per_group - 2] = k_heads[kv_group]
        packed[group_base + total_heads_per_group - 1] = v_heads[kv_group]

    return packed.reshape(-1, feature_dim)


def _as_dict(converted: list[tuple[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: tensor for name, tensor in converted}


def test_converter_standard_attention_equivalent_layouts_return_same_protocol_values():
    """Standard attention train/infer layouts must map to identical canonical q/k/v."""
    # Arrange
    train_converter = _make_train_converter()
    infer_converter = _make_infer_converter()
    local_q_heads = 4
    local_kv_heads = 2
    head_dim = 2
    feature_dim = 3

    q_block = torch.arange(0, 16 * feature_dim, dtype=torch.float32).view(
        16, feature_dim
    )
    k_block = torch.arange(1000, 1000 + 4 * feature_dim, dtype=torch.float32).view(
        4, feature_dim
    )
    v_block = torch.arange(2000, 2000 + 4 * feature_dim, dtype=torch.float32).view(
        4, feature_dim
    )

    mcore_local = _build_mcore_q_gate_k_v_interleaved(
        q_block=q_block,
        k_block=k_block,
        v_block=v_block,
        local_q_heads=local_q_heads,
        local_kv_heads=local_kv_heads,
        head_dim=head_dim,
    )
    sglang_local = torch.cat([q_block, k_block, v_block], dim=0)

    # Act
    train_out = _as_dict(
        train_converter._convert_language_attention_param(
            "self_attention.linear_qkv.weight", mcore_local
        )
    )
    infer_out = _as_dict(
        infer_converter._convert_language_attention_param(
            "self_attn.qkv_proj.weight", sglang_local
        )
    )

    # Assert
    assert set(train_out) == {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
    }
    assert set(infer_out) == set(train_out)
    for name in train_out:
        torch.testing.assert_close(train_out[name], infer_out[name], rtol=0.0, atol=0.0)


def test_converter_attention_gate_field_prefers_attn_output_gate_and_falls_back_compatibly():
    """Infer converter must prefer official attn_output_gate and keep compatibility fallback/default."""
    # Arrange
    converter_official = _make_infer_converter(attn_output_gate=False)
    setattr(converter_official.model_config, "attention_output_gate", True)

    converter_fallback = _make_infer_converter(attn_output_gate=True)
    delattr(converter_fallback.model_config, "attn_output_gate")
    setattr(converter_fallback.model_config, "attention_output_gate", False)

    converter_default = _make_infer_converter(attn_output_gate=True)
    delattr(converter_default.model_config, "attn_output_gate")

    # Act / Assert
    _, _, q_rows_official, _, output_gate_official = (
        converter_official._attention_qkv_layout()
    )
    _, _, q_rows_fallback, _, output_gate_fallback = (
        converter_fallback._attention_qkv_layout()
    )
    _, _, q_rows_default, _, output_gate_default = (
        converter_default._attention_qkv_layout()
    )

    assert output_gate_official is False
    assert output_gate_fallback is False
    assert output_gate_default is True
    assert q_rows_default == 16
    assert q_rows_official == 8
    assert q_rows_fallback == 8


def test_converter_gdn_and_out_norm_equivalent_layouts_return_same_protocol_values():
    """GDN split layouts and out-norm zero-centering must agree across converters."""
    # Arrange
    train_converter = _make_train_converter()
    infer_converter = _make_infer_converter()
    feature_dim = 4

    q = torch.arange(0, 8 * feature_dim, dtype=torch.float32).view(8, feature_dim)
    k = torch.arange(100, 100 + 8 * feature_dim, dtype=torch.float32).view(
        8, feature_dim
    )
    v = torch.arange(200, 200 + 8 * feature_dim, dtype=torch.float32).view(
        8, feature_dim
    )
    z = torch.arange(300, 300 + 8 * feature_dim, dtype=torch.float32).view(
        8, feature_dim
    )
    b = torch.arange(400, 400 + 4 * feature_dim, dtype=torch.float32).view(
        4, feature_dim
    )
    a = torch.arange(500, 500 + 4 * feature_dim, dtype=torch.float32).view(
        4, feature_dim
    )

    train_in_proj = torch.cat([q, k, v, z, b, a], dim=0)
    infer_qkvz = torch.cat([q, k, v, z], dim=0)
    infer_ba = torch.cat([b, a], dim=0)

    infer_norm = torch.arange(0, 7, dtype=torch.float32)
    train_norm = infer_norm - 1

    # Act
    train_out = _as_dict(
        train_converter._convert_language_attention_param(
            "self_attention.in_proj.weight", train_in_proj
        )
    )
    infer_qkvz_out = _as_dict(
        infer_converter._convert_language_attention_param(
            "linear_attn.in_proj_qkvz.weight", infer_qkvz
        )
    )
    infer_ba_out = _as_dict(
        infer_converter._convert_language_attention_param(
            "linear_attn.in_proj_ba.weight", infer_ba
        )
    )
    train_norm_out = _as_dict(
        train_converter._convert_language_attention_param(
            "self_attention.out_norm.weight", train_norm
        )
    )
    infer_norm_out = _as_dict(
        infer_converter._convert_language_attention_param(
            "linear_attn.norm.weight", infer_norm
        )
    )

    # Assert
    infer_out = {**infer_qkvz_out, **infer_ba_out}
    assert set(train_out) == set(infer_out)
    for name in train_out:
        torch.testing.assert_close(train_out[name], infer_out[name], rtol=0.0, atol=0.0)

    torch.testing.assert_close(
        train_norm_out["linear_attn.norm.weight"],
        infer_norm_out["linear_attn.norm.weight"],
        rtol=0.0,
        atol=0.0,
    )


def test_converter_vision_qkv_equivalent_layouts_return_same_protocol_values():
    """Vision QKV interleaved/consecutive layouts must produce the same canonical tensors."""
    # Arrange
    train_converter = _make_train_converter()
    infer_converter = _make_infer_converter()
    local_heads = 4
    head_dim = 2
    feature_dim = 2

    q = torch.arange(0, 8 * feature_dim, dtype=torch.float32).view(8, feature_dim)
    k = torch.arange(100, 100 + 8 * feature_dim, dtype=torch.float32).view(
        8, feature_dim
    )
    v = torch.arange(200, 200 + 8 * feature_dim, dtype=torch.float32).view(
        8, feature_dim
    )

    mcore_heads = torch.empty(
        local_heads, 3, head_dim, feature_dim, dtype=torch.float32
    )
    mcore_heads[:, 0] = q.view(local_heads, head_dim, feature_dim)
    mcore_heads[:, 1] = k.view(local_heads, head_dim, feature_dim)
    mcore_heads[:, 2] = v.view(local_heads, head_dim, feature_dim)
    mcore_local = mcore_heads.reshape(-1, feature_dim)
    sglang_local = torch.cat([q, k, v], dim=0)

    # Act
    train_out = _as_dict(
        train_converter._convert_vision_layer_param(
            "decoder.layers.0.self_attention.linear_qkv.weight", mcore_local
        )
    )
    infer_out = _as_dict(
        infer_converter._convert_vision_param(
            "blocks.0.attn.qkv_proj.weight", sglang_local
        )
    )

    # Assert
    expected_names = {
        "model.visual.blocks.0.attn.q_proj.weight",
        "model.visual.blocks.0.attn.k_proj.weight",
        "model.visual.blocks.0.attn.v_proj.weight",
    }
    assert set(train_out) == expected_names
    assert set(infer_out) == expected_names
    for name in expected_names:
        torch.testing.assert_close(train_out[name], infer_out[name], rtol=0.0, atol=0.0)


def test_converter_routed_moe_train_ep_local_and_infer_bulk_return_same_global_expert_values():
    """Train local-expert and infer bulk-expert paths must agree on global expert outputs."""
    # Arrange
    train_converter = _make_train_converter(num_experts=8, ep_size=2, ep_rank=1)
    infer_converter = _make_infer_converter(num_experts=8, ep_size=1, ep_rank=0)
    fc1_local = torch.arange(0, 32, dtype=torch.float32).view(8, 4)
    global_expert_id = 5

    bulk_w13 = torch.zeros(8, 8, 4, dtype=torch.float32)
    bulk_w13[global_expert_id] = fc1_local

    # Act
    train_out = _as_dict(
        train_converter._convert_language_mlp_param(
            "mlp.experts.local_experts.1.linear_fc1.weight", fc1_local
        )
    )
    infer_out = _as_dict(
        infer_converter._convert_language_mlp_param("mlp.experts.w13_weight", bulk_w13)
    )

    # Assert
    train_gate_name = f"mlp.experts.{global_expert_id}.gate_proj.weight"
    train_up_name = f"mlp.experts.{global_expert_id}.up_proj.weight"
    assert train_gate_name in train_out
    assert train_up_name in train_out
    assert train_gate_name in infer_out
    assert train_up_name in infer_out
    torch.testing.assert_close(
        train_out[train_gate_name], infer_out[train_gate_name], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        train_out[train_up_name], infer_out[train_up_name], rtol=0.0, atol=0.0
    )


def test_qwen3_5_sharding_strategy_classifies_train_and_infer_rules_exactly():
    """Sharding strategy must classify replicated, TP, EP, and EP_TP cases exactly."""
    # Arrange
    train = cast(Qwen3_5MoeShardingStrategy, object.__new__(Qwen3_5MoeShardingStrategy))
    setattr(train, "engine_name", "mcore")
    setattr(
        train,
        "rank_info",
        SimpleNamespace(tp_size=2, attn_tp_size=2, ep_size=2, ep_tp_size=1, pp_size=2),
    )
    setattr(train, "enable_dp_attention", False)
    setattr(train, "enable_dp_lm_head", False)

    train_ep_tp = cast(
        Qwen3_5MoeShardingStrategy, object.__new__(Qwen3_5MoeShardingStrategy)
    )
    setattr(train_ep_tp, "engine_name", "mcore")
    setattr(
        train_ep_tp,
        "rank_info",
        SimpleNamespace(tp_size=2, attn_tp_size=2, ep_size=2, ep_tp_size=4, pp_size=2),
    )
    setattr(train_ep_tp, "enable_dp_attention", False)
    setattr(train_ep_tp, "enable_dp_lm_head", False)

    infer = cast(Qwen3_5MoeShardingStrategy, object.__new__(Qwen3_5MoeShardingStrategy))
    setattr(infer, "engine_name", "sglang")
    setattr(
        infer,
        "rank_info",
        SimpleNamespace(tp_size=2, attn_tp_size=2, ep_size=1, ep_tp_size=1, pp_size=1),
    )
    setattr(infer, "enable_dp_attention", False)
    setattr(infer, "enable_dp_lm_head", False)

    # Act / Assert
    assert train.get_sharding_strategy(
        "model.language_model.layers.0.input_layernorm.weight"
    ) == (
        ShardingType.NO_SHARDING,
        0,
        1,
    )
    assert train.get_sharding_strategy(
        "model.language_model.layers.0.self_attn.q_proj.weight"
    ) == (
        ShardingType.TP_SHARDING,
        0,
        2,
    )
    assert train.get_sharding_strategy(
        "model.language_model.layers.0.self_attn.o_proj.weight"
    ) == (
        ShardingType.TP_SHARDING,
        1,
        2,
    )
    assert train.get_sharding_strategy(
        "model.language_model.layers.0.mlp.experts.3.gate_proj.weight"
    ) == (
        ShardingType.EP_SHARDING,
        0,
        2,
    )
    assert train_ep_tp.get_sharding_strategy(
        "model.language_model.layers.0.mlp.experts.3.down_proj.weight"
    ) == (ShardingType.EP_TP_SHARDING, 1, 4)
    assert infer.get_sharding_strategy(
        "model.language_model.layers.0.mlp.experts.3.gate_proj.weight"
    ) == (
        ShardingType.TP_SHARDING,
        0,
        2,
    )


def test_qwen3_5_sharding_strategy_dp_attention_uses_attn_tp_for_language_only():
    """Language attention uses DP_TP(attn_tp); vision stays TP(tp_size) under DP attention."""
    # Arrange
    infer = cast(Qwen3_5MoeShardingStrategy, object.__new__(Qwen3_5MoeShardingStrategy))
    setattr(infer, "engine_name", "sglang")
    setattr(
        infer,
        "rank_info",
        SimpleNamespace(tp_size=2, attn_tp_size=4, ep_size=1, ep_tp_size=1, pp_size=1),
    )
    setattr(infer, "enable_dp_attention", True)
    setattr(infer, "enable_dp_lm_head", True)

    # Act / Assert
    assert infer.get_sharding_strategy(
        "model.language_model.layers.0.self_attn.q_proj.weight"
    ) == (ShardingType.DP_TP_SHARDING, 0, 4)
    assert infer.get_sharding_strategy(
        "model.language_model.layers.0.linear_attn.out_proj.weight"
    ) == (ShardingType.DP_TP_SHARDING, 1, 4)
    assert infer.get_sharding_strategy("lm_head.weight") == (
        ShardingType.DP_TP_SHARDING,
        0,
        4,
    )

    # Vision must stay on model TP, not attention TP.
    assert infer.get_sharding_strategy("model.visual.blocks.0.attn.q_proj.weight") == (
        ShardingType.TP_SHARDING,
        0,
        2,
    )
    assert infer.get_sharding_strategy("model.visual.blocks.0.attn.proj.weight") == (
        ShardingType.TP_SHARDING,
        1,
        2,
    )


def test_qwen3_5_sharding_strategy_raises_when_infer_pp_or_ep_not_one():
    """Inference sharding must reject PP!=1 or EP!=1 topology inputs."""
    # Arrange
    infer_bad_pp = cast(
        Qwen3_5MoeShardingStrategy, object.__new__(Qwen3_5MoeShardingStrategy)
    )
    setattr(infer_bad_pp, "engine_name", "sglang")
    setattr(
        infer_bad_pp,
        "rank_info",
        SimpleNamespace(tp_size=2, attn_tp_size=2, ep_size=1, ep_tp_size=1, pp_size=2),
    )
    setattr(infer_bad_pp, "enable_dp_attention", False)
    setattr(infer_bad_pp, "enable_dp_lm_head", False)

    infer_bad_ep = cast(
        Qwen3_5MoeShardingStrategy, object.__new__(Qwen3_5MoeShardingStrategy)
    )
    setattr(infer_bad_ep, "engine_name", "sglang")
    setattr(
        infer_bad_ep,
        "rank_info",
        SimpleNamespace(tp_size=2, attn_tp_size=2, ep_size=2, ep_tp_size=1, pp_size=1),
    )
    setattr(infer_bad_ep, "enable_dp_attention", False)
    setattr(infer_bad_ep, "enable_dp_lm_head", False)

    # Act / Assert
    with pytest.raises(ValueError, match="requires PP=1"):
        infer_bad_pp.get_sharding_strategy(
            "model.language_model.layers.0.self_attn.q_proj.weight"
        )

    with pytest.raises(ValueError, match="requires EP=1"):
        infer_bad_ep.get_sharding_strategy(
            "model.language_model.layers.0.self_attn.q_proj.weight"
        )


def test_ensure_awex_qwen3_5_registered_is_idempotent_and_keeps_exact_classes(
    monkeypatch: pytest.MonkeyPatch,
):
    """Registry install must be idempotent and keep the exact converter/strategy classes."""
    # Arrange
    from awex.models import registry as awex_registry
    from awex.models.registry import ModelConfig

    monkeypatch.setattr(awex_registry.ModelRegistry, "models", {}, raising=False)

    # Act
    ensure_awex_qwen3_5_registered()
    first = awex_registry.ModelRegistry.models["Qwen3_5MoeForConditionalGeneration"]
    ensure_awex_qwen3_5_registered()
    second = awex_registry.ModelRegistry.models["Qwen3_5MoeForConditionalGeneration"]

    # Assert
    assert isinstance(first, ModelConfig)
    assert first is second
    assert first.sharding_strategy is Qwen3_5MoeShardingStrategy
    assert first.mcore_converter is McoreToHFWeightConverterQwen3_5Moe
    assert first.sglang_converter is SGlangToHFWeightConverterQwen3_5Moe


def test_converter_mtp_parameters_are_skipped_with_empty_outputs():
    """Both train and infer converters must skip MTP parameters with empty outputs."""
    # Arrange
    train_converter = _make_train_converter()
    infer_converter = _make_infer_converter()
    tensor = torch.arange(0, 6, dtype=torch.float32)

    # Act / Assert
    assert (
        train_converter.convert_param(
            "language_model.decoder.layers.0.mtp.proj.weight", tensor
        )
        == []
    )
    assert (
        infer_converter.convert_param(
            "model.language_model.layers.0.mtp.proj.weight", tensor
        )
        == []
    )


def test_converter_attention_rejects_tp_greater_than_kv_groups_with_v1_limitation_message():
    """Train attention conversion must reject TP>KV-group split with v1 limitation message."""
    # Arrange
    converter = _make_train_converter(
        num_attention_heads=8,
        num_query_groups=2,
        attn_tp_size=4,
        attention_output_gate=True,
        hidden_size=16,
    )
    malformed = torch.zeros(32, 2, dtype=torch.float32)

    # Act / Assert
    with pytest.raises(
        ValueError,
        match=("AWEX v1 cannot express the required split-group permutation/transform"),
    ):
        converter._convert_language_attention_param(
            "self_attention.linear_qkv.weight", malformed
        )


def test_converter_infer_attention_rejects_tp_greater_than_kv_heads_with_v1_limitation_message():
    """Infer attention conversion must reject attn_tp_size > num_key_value_heads."""
    # Arrange
    converter = _make_infer_converter(
        num_attention_heads=8,
        num_key_value_heads=2,
        attn_tp_size=4,
        attn_output_gate=True,
        hidden_size=16,
        tp_size=4,
    )
    malformed = torch.zeros(48, 2, dtype=torch.float32)

    # Act / Assert
    with pytest.raises(
        ValueError,
        match=(
            "requires KV-head replication that AWEX v1 sharding metadata cannot represent"
        ),
    ):
        converter._convert_language_attention_param(
            "self_attn.qkv_proj.weight", malformed
        )


def test_qwen3_5_sharding_strategy_rejects_attn_tp_model_tp_mismatch_when_dp_attention_disabled():
    """Language attention sharding must reject attn_tp_size!=tp_size when DP attention is disabled."""
    # Arrange
    infer = cast(Qwen3_5MoeShardingStrategy, object.__new__(Qwen3_5MoeShardingStrategy))
    setattr(infer, "engine_name", "sglang")
    setattr(
        infer,
        "rank_info",
        SimpleNamespace(tp_size=2, attn_tp_size=1, ep_size=1, ep_tp_size=1, pp_size=1),
    )
    setattr(infer, "enable_dp_attention", False)
    setattr(infer, "enable_dp_lm_head", False)

    # Act / Assert
    with pytest.raises(ValueError, match="language attention sharding mismatch"):
        infer.get_sharding_strategy(
            "model.language_model.layers.0.self_attn.q_proj.weight"
        )


def test_converter_runtime_visual_prefix_is_accepted_for_patch_embed_weight():
    """Runtime visual.* prefix must dispatch to vision conversion."""
    # Arrange
    converter = _make_infer_converter()
    parameter = torch.arange(0, 6, dtype=torch.float32).view(2, 3)

    # Act
    converted = converter.convert_param("visual.patch_embed.proj.weight", parameter)

    # Assert
    assert len(converted) == 1
    assert converted[0][0] == "model.visual.patch_embed.proj.weight"
    torch.testing.assert_close(converted[0][1], parameter, rtol=0.0, atol=0.0)


def test_converter_runtime_model_prefix_accepts_embed_norm_and_layers_parameters():
    """Runtime model.* language names must support embed/norm/layers dispatch."""
    # Arrange
    converter = _make_infer_converter()
    embed = torch.arange(0, 12, dtype=torch.float32).view(3, 4)
    norm = torch.arange(0, 4, dtype=torch.float32)
    layer_norm = torch.arange(0, 4, dtype=torch.float32)

    # Act
    embed_converted = converter.convert_param("model.embed_tokens.weight", embed)
    norm_converted = converter.convert_param("model.norm.weight", norm)
    layer_converted = converter.convert_param(
        "model.layers.0.input_layernorm.weight", layer_norm
    )

    # Assert
    assert embed_converted[0][0] == "model.language_model.embed_tokens.weight"
    assert norm_converted[0][0] == "model.language_model.norm.weight"
    assert (
        layer_converted[0][0] == "model.language_model.layers.0.input_layernorm.weight"
    )
    torch.testing.assert_close(embed_converted[0][1], embed, rtol=0.0, atol=0.0)
    torch.testing.assert_close(norm_converted[0][1], norm, rtol=0.0, atol=0.0)
    torch.testing.assert_close(layer_converted[0][1], layer_norm, rtol=0.0, atol=0.0)


def test_converter_checkpoint_alias_prefixes_remain_compatible():
    """Checkpoint-style aliases must remain accepted after runtime prefix expansion."""
    # Arrange
    converter = _make_infer_converter()
    embed = torch.arange(0, 12, dtype=torch.float32).view(3, 4)
    vision = torch.arange(0, 6, dtype=torch.float32).view(2, 3)
    layer_norm = torch.arange(0, 4, dtype=torch.float32)
    lm_head = torch.arange(0, 6, dtype=torch.float32).view(2, 3)

    # Act
    language_alias = converter.convert_param(
        "model.language_model.embed_tokens.weight", embed
    )
    vision_alias = converter.convert_param(
        "model.visual.patch_embed.proj.weight", vision
    )
    layer_alias = converter.convert_param(
        "model.layers.0.input_layernorm.weight", layer_norm
    )
    lm_head_alias = converter.convert_param("model.lm_head.weight", lm_head)

    # Assert
    assert language_alias[0][0] == "model.language_model.embed_tokens.weight"
    assert vision_alias[0][0] == "model.visual.patch_embed.proj.weight"
    assert layer_alias[0][0] == "model.language_model.layers.0.input_layernorm.weight"
    assert lm_head_alias[0][0] == "lm_head.weight"
