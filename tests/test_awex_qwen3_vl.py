# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for Qwen3-VL support in the v1 AWEX colocate path."""

from types import SimpleNamespace

import pytest
import torch

param_sharding = pytest.importorskip("awex.sharding.param_sharding")

ShardingType = param_sharding.ShardingType

from areal.engine.awex.qwen3_vl import (  # noqa: E402
    QWEN3_VL_ARCHITECTURES,
    Qwen3VLMcoreToHFWeightConverter,
    Qwen3VLMoeMcoreToHFWeightConverter,
    Qwen3VLSGlangToHFWeightConverter,
    Qwen3VLShardingStrategy,
    register_qwen3_vl_awex_models,
)


@pytest.fixture
def text_config():
    return SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        tie_word_embeddings=False,
    )


@pytest.fixture
def vl_config(text_config):
    return SimpleNamespace(
        architectures=["Qwen3VLForConditionalGeneration"],
        text_config=text_config,
        vision_config=SimpleNamespace(num_heads=2),
    )


@pytest.fixture
def tf_config():
    return SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        num_layers=2,
    )


@pytest.fixture
def rank_info():
    return SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
    )


@pytest.fixture
def infer_engine_config():
    return SimpleNamespace(
        tp_size=1,
        ep_size=1,
        device_backend="cpu",
    )


def _train_converter(cls, vl_config, rank_info, tf_config):
    return cls(
        vl_config,
        rank_info,
        {"infer_atten_tp_size": 1},
        tf_config,
    )


def _infer_converter(vl_config, rank_info, infer_engine_config):
    return Qwen3VLSGlangToHFWeightConverter(
        vl_config, infer_engine_config, rank_info
    )


def test_register_qwen3_vl_awex_models_is_idempotent():
    from awex.models.registry import ModelRegistry

    register_qwen3_vl_awex_models()
    first_entries = {
        architecture: ModelRegistry.models[architecture]
        for architecture in QWEN3_VL_ARCHITECTURES
    }

    register_qwen3_vl_awex_models()

    assert {
        architecture: ModelRegistry.models[architecture]
        for architecture in QWEN3_VL_ARCHITECTURES
    } == first_entries
    assert all(
        entry["sglang_converter"] is Qwen3VLSGlangToHFWeightConverter
        for entry in first_entries.values()
    )
    assert first_entries["Qwen3VLForConditionalGeneration"][
        "mcore_converter"
    ]() is Qwen3VLMcoreToHFWeightConverter
    assert first_entries["Qwen3VLMoeForConditionalGeneration"][
        "mcore_converter"
    ]() is Qwen3VLMoeMcoreToHFWeightConverter


def test_colocate_reader_uses_nested_text_config_for_awex_metadata():
    from areal.engine.awex.colocate_reader import _get_text_config

    text_config = SimpleNamespace(num_hidden_layers=28, router_dtype="fp32")
    vl_config = SimpleNamespace(text_config=text_config)

    assert _get_text_config(vl_config) is text_config
    assert _get_text_config(text_config) is text_config


def test_colocate_writer_exposes_text_depth_without_losing_vl_config():
    from areal.engine.awex.colocate_writer import _get_awex_train_hf_config

    text_config = SimpleNamespace(num_hidden_layers=28)
    vision_config = SimpleNamespace(num_heads=16)
    vl_config = SimpleNamespace(
        architectures=["Qwen3VLForConditionalGeneration"],
        text_config=text_config,
        vision_config=vision_config,
    )

    awex_config = _get_awex_train_hf_config(vl_config)

    assert awex_config is not vl_config
    assert not hasattr(vl_config, "num_hidden_layers")
    assert awex_config.num_hidden_layers == 28
    assert awex_config.architectures == ["Qwen3VLForConditionalGeneration"]
    assert awex_config.text_config is text_config
    assert awex_config.vision_config is vision_config


def test_dense_language_qkv_has_train_infer_parity(
    vl_config, tf_config, rank_info, infer_engine_config
):
    train_converter = _train_converter(
        Qwen3VLMcoreToHFWeightConverter, vl_config, rank_info, tf_config
    )
    infer_converter = _infer_converter(vl_config, rank_info, infer_engine_config)
    mcore_qkv = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)

    train_params = train_converter.convert_param(
        "module.module.language_model.decoder.layers.0."
        "self_attention.linear_qkv.weight",
        mcore_qkv,
    )
    sglang_qkv = torch.cat([param for _, param in train_params], dim=0)
    infer_params = infer_converter.convert_param(
        "model.layers.0.self_attn.qkv_proj.weight", sglang_qkv
    )

    assert [name for name, _ in train_params] == [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
    ]
    assert [name for name, _ in infer_params] == [
        name for name, _ in train_params
    ]
    for (_, train_param), (_, infer_param) in zip(train_params, infer_params):
        torch.testing.assert_close(train_param, infer_param, rtol=0, atol=0)


def test_vision_qkv_has_train_infer_parity(
    vl_config, tf_config, rank_info, infer_engine_config
):
    train_converter = Qwen3VLMcoreToHFWeightConverter(
        vl_config,
        rank_info,
        {
            "infer_atten_tp_size": 1,
            "train_pp_stage_layer_id_map": {(0, 0): {0: 0}},
        },
        tf_config,
    )
    infer_converter = _infer_converter(vl_config, rank_info, infer_engine_config)
    mcore_qkv = torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)

    train_params = train_converter.convert_param(
        "module.module.vision_model.decoder.layers.1."
        "self_attention.linear_qkv.weight",
        mcore_qkv,
    )
    infer_params = infer_converter.convert_param(
        "visual.blocks.1.attn.qkv_proj.weight", train_params[0][1]
    )

    assert train_params[0][0] == "model.visual.blocks.1.attn.qkv.weight"
    assert infer_params[0][0] == train_params[0][0]
    torch.testing.assert_close(
        train_params[0][1], infer_params[0][1], rtol=0, atol=0
    )
    assert not torch.equal(train_params[0][1], mcore_qkv)


def test_dense_mlp_has_train_infer_parity(
    vl_config, tf_config, rank_info, infer_engine_config
):
    train_converter = _train_converter(
        Qwen3VLMcoreToHFWeightConverter, vl_config, rank_info, tf_config
    )
    infer_converter = _infer_converter(vl_config, rank_info, infer_engine_config)
    mcore_fc1 = torch.arange(12 * 8, dtype=torch.float32).reshape(12, 8)

    train_params = train_converter.convert_param(
        "module.module.language_model.decoder.layers.0.mlp.linear_fc1.weight",
        mcore_fc1,
    )
    infer_params = infer_converter.convert_param(
        "model.layers.0.mlp.gate_up_proj.weight", mcore_fc1
    )

    assert [name for name, _ in train_params] == [
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
    ]
    assert [name for name, _ in infer_params] == [
        name for name, _ in train_params
    ]
    for (_, train_param), (_, infer_param) in zip(train_params, infer_params):
        torch.testing.assert_close(train_param, infer_param, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mcore_name", "sglang_name", "canonical_name"),
    [
        (
            "module.module.vision_model.merger.linear_fc1.weight",
            "visual.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc1.weight",
        ),
        (
            "module.module.vision_model.decoder.deepstack_merger_list.2."
            "patch_norm.weight",
            "visual.deepstack_merger_list.2.norm.weight",
            "model.visual.deepstack_merger_list.2.norm.weight",
        ),
    ],
)
def test_vision_merger_names_have_train_infer_parity(
    mcore_name,
    sglang_name,
    canonical_name,
    vl_config,
    tf_config,
    rank_info,
    infer_engine_config,
):
    train_converter = _train_converter(
        Qwen3VLMcoreToHFWeightConverter, vl_config, rank_info, tf_config
    )
    infer_converter = _infer_converter(vl_config, rank_info, infer_engine_config)
    parameter = torch.arange(8, dtype=torch.float32)

    train_params = train_converter.convert_param(mcore_name, parameter)
    infer_params = infer_converter.convert_param(sglang_name, parameter)

    assert [name for name, _ in train_params] == [canonical_name]
    assert [name for name, _ in infer_params] == [canonical_name]
    torch.testing.assert_close(train_params[0][1], parameter, rtol=0, atol=0)
    torch.testing.assert_close(infer_params[0][1], parameter, rtol=0, atol=0)


def test_moe_expert_has_train_infer_parity(
    vl_config, tf_config, rank_info, infer_engine_config
):
    train_converter = _train_converter(
        Qwen3VLMoeMcoreToHFWeightConverter, vl_config, rank_info, tf_config
    )
    infer_converter = _infer_converter(vl_config, rank_info, infer_engine_config)
    expert_fc1 = torch.arange(12 * 8, dtype=torch.float32).reshape(12, 8)

    train_params = train_converter.convert_param(
        "module.module.language_model.decoder.layers.1."
        "mlp.experts.linear_fc1.weight0",
        expert_fc1,
    )
    sglang_w13 = expert_fc1.unsqueeze(0)
    infer_params = infer_converter.convert_param(
        "model.layers.1.mlp.experts.w13_weight", sglang_w13
    )

    assert [name for name, _ in train_params] == [
        "model.language_model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.1.mlp.experts.0.up_proj.weight",
    ]
    assert [name for name, _ in infer_params] == [
        name for name, _ in train_params
    ]
    for (_, train_param), (_, infer_param) in zip(train_params, infer_params):
        torch.testing.assert_close(train_param, infer_param, rtol=0, atol=0)


def test_moe_training_converter_offsets_local_expert_id(
    vl_config, tf_config, rank_info
):
    rank_info.ep_size = 2
    rank_info.ep_rank = 1
    converter = _train_converter(
        Qwen3VLMoeMcoreToHFWeightConverter, vl_config, rank_info, tf_config
    )

    converted = converter.convert_param(
        "module.module.language_model.decoder.layers.0."
        "mlp.experts.linear_fc2.weight0",
        torch.ones(8, 6),
    )

    assert converted[0][0] == (
        "model.language_model.layers.0.mlp.experts.2.down_proj.weight"
    )


def test_moe_router_uses_sglang_router_dtype(vl_config, tf_config, rank_info):
    converter = Qwen3VLMoeMcoreToHFWeightConverter(
        vl_config,
        rank_info,
        {"infer_atten_tp_size": 1, "router_dtype": "fp32"},
        tf_config,
    )

    converted = converter.convert_param(
        "module.module.language_model.decoder.layers.0.mlp.router.weight",
        torch.ones(4, 8, dtype=torch.bfloat16),
    )

    assert converted[0][0] == "model.language_model.layers.0.mlp.gate.weight"
    assert converted[0][1].dtype is torch.float32


def test_tied_embedding_alias_has_train_infer_parity(
    vl_config, tf_config, rank_info, infer_engine_config
):
    vl_config.text_config.tie_word_embeddings = True
    train_converter = _train_converter(
        Qwen3VLMcoreToHFWeightConverter, vl_config, rank_info, tf_config
    )
    infer_converter = _infer_converter(vl_config, rank_info, infer_engine_config)
    embedding = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    train_params = train_converter.convert_param(
        "module.module.language_model.embedding.word_embeddings.weight", embedding
    )
    infer_params = infer_converter.convert_param(
        "model.embed_tokens.weight", embedding
    )

    expected_names = ["model.language_model.embed_tokens.weight", "lm_head.weight"]
    assert [name for name, _ in train_params] == expected_names
    assert [name for name, _ in infer_params] == expected_names


@pytest.mark.parametrize(
    ("name", "expected_type", "expected_dim"),
    [
        ("model.visual.patch_embed.proj.weight", ShardingType.NO_SHARDING, 0),
        ("model.visual.blocks.0.attn.qkv.weight", ShardingType.TP_SHARDING, 0),
        ("model.visual.blocks.0.attn.proj.weight", ShardingType.TP_SHARDING, 1),
        (
            "model.visual.deepstack_merger_list.0.linear_fc1.weight",
            ShardingType.TP_SHARDING,
            0,
        ),
        (
            "model.visual.deepstack_merger_list.0.linear_fc2.weight",
            ShardingType.TP_SHARDING,
            1,
        ),
        (
            "model.visual.deepstack_merger_list.0.linear_fc2.bias",
            ShardingType.NO_SHARDING,
            0,
        ),
    ],
)
def test_vision_sharding_strategy_matches_sglang_layout(
    name, expected_type, expected_dim, rank_info
):
    rank_info.tp_size = 2
    rank_info.attn_tp_size = 2
    strategy = Qwen3VLShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=2,
        tp_size=2,
        ep_size=1,
        ep_tp_size=1,
        rank_info=rank_info,
        device_backend="cpu",
    )

    sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy(name)

    assert sharding_type is expected_type
    assert sharding_dim == expected_dim
    assert num_shards == (2 if expected_type is ShardingType.TP_SHARDING else 1)


def test_unknown_megatron_vision_parameter_fails_fast(
    vl_config, tf_config, rank_info
):
    converter = _train_converter(
        Qwen3VLMcoreToHFWeightConverter, vl_config, rank_info, tf_config
    )

    with pytest.raises(ValueError, match="Unknown Qwen3-VL parameter"):
        converter.convert_param(
            "module.module.vision_model.unknown.weight", torch.ones(2)
        )
