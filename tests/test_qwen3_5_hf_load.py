# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _stub_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_module(monkeypatch, name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def qwen3_5_hf_modules(monkeypatch):
    class _Logger:
        def debug(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

    class _Bridge:
        pass

    class _LLMBridge:
        def _weight_to_mcore_format(self, mcore_weights_name, hf_weights):
            if len(hf_weights) != 1:
                raise AssertionError(f"unexpected fallback for {mcore_weights_name}")
            return hf_weights[0]

        def _weight_to_hf_format(self, mcore_weights_name, mcore_weights):
            raise AssertionError(f"unexpected fallback for {mcore_weights_name}")

    class _AttnBackend:
        fused = "fused"

    class _FP8BlockwiseTensorHelper:
        pass

    tp_state = {"initialized": True, "tp_size": 1, "tp_rank": 0}

    parallel_state = _stub_module(
        monkeypatch,
        "megatron.core.parallel_state",
        model_parallel_is_initialized=lambda: tp_state["initialized"],
        get_tensor_model_parallel_world_size=lambda: tp_state["tp_size"],
        get_tensor_model_parallel_rank=lambda: tp_state["tp_rank"],
        get_expert_model_parallel_world_size=lambda: 1,
        get_expert_model_parallel_rank=lambda: 0,
        get_expert_tensor_parallel_world_size=lambda: 1,
        get_expert_tensor_parallel_rank=lambda: 0,
        is_pipeline_last_stage=lambda: True,
    )

    _stub_module(monkeypatch, "megatron")
    _stub_module(
        monkeypatch,
        "megatron.core",
        parallel_state=parallel_state,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.fp8_utils",
        is_float8tensor=lambda _: False,
    )
    _stub_module(monkeypatch, "megatron.core.transformer", TransformerConfig=object)
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.enums",
        AttnBackend=_AttnBackend,
    )

    _stub_module(monkeypatch, "mbridge")
    _stub_module(
        monkeypatch,
        "mbridge.core",
        LLMBridge=_LLMBridge,
        register_model=lambda names: lambda cls: cls,
    )
    _stub_module(monkeypatch, "mbridge.core.bridge", Bridge=_Bridge)

    _stub_module(monkeypatch, "safetensors", safe_open=lambda *args, **kwargs: None)

    _stub_module(monkeypatch, "areal")
    _stub_module(monkeypatch, "areal.models")
    _stub_module(monkeypatch, "areal.models.mcore")
    _stub_module(monkeypatch, "areal.engine")
    _stub_module(monkeypatch, "areal.engine.core")
    _stub_module(
        monkeypatch,
        "areal.engine.core.model",
        lang_config=lambda config: config.text_config
        if hasattr(config, "text_config")
        else config,
    )
    _stub_module(monkeypatch, "areal.engine.megatron_utils")
    _stub_module(
        monkeypatch,
        "areal.engine.megatron_utils.fp8",
        FP8BlockwiseTensorHelper=_FP8BlockwiseTensorHelper,
        dequantize_params=lambda *args, **kwargs: None,
        get_block_size_from_config=lambda *args, **kwargs: None,
    )
    _stub_module(monkeypatch, "areal.infra")
    _stub_module(
        monkeypatch,
        "areal.infra.platforms",
        current_platform=SimpleNamespace(device_type="cpu"),
    )
    _stub_module(
        monkeypatch,
        "areal.models.mcore.registry",
        unwrap_to_gpt_model=lambda model: model,
    )
    _stub_module(monkeypatch, "areal.utils")
    logging_module = _stub_module(
        monkeypatch,
        "areal.utils.logging",
        getLogger=lambda name: _Logger(),
    )
    sys.modules["areal.utils"].logging = logging_module
    _stub_module(
        monkeypatch,
        "areal.models.mcore.qwen3_5",
        make_mcore_layer_specs_qwen3_5_moe=lambda *args, **kwargs: None,
    )

    utils = _load_module(
        monkeypatch,
        "areal.models.mcore.qwen3_5_weight_utils",
        "areal/models/mcore/qwen3_5_weight_utils.py",
    )
    bridge = _load_module(
        monkeypatch,
        "areal.models.mcore.qwen3_5_bridge",
        "areal/models/mcore/qwen3_5_bridge.py",
    )
    hf_load = _load_module(
        monkeypatch,
        "areal.models.mcore.hf_load",
        "areal/models/mcore/hf_load.py",
    )
    return SimpleNamespace(
        utils=utils, bridge=bridge, hf_load=hf_load, tp_state=tp_state
    )


def _make_hf_config(geometry: str):
    if geometry == "tiny":
        text = SimpleNamespace(
            model_type="qwen3_5_moe",
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
        )
    elif geometry == "large":
        text = SimpleNamespace(
            model_type="qwen3_5_moe",
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        )
    else:
        raise ValueError(f"unknown geometry {geometry}")
    return SimpleNamespace(model_type="qwen3_5_moe", text_config=text)


@pytest.mark.parametrize("geometry", ["tiny", "large"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("is_bias", [False, True])
def test_hf_load_qwen3_5_gated_qkv_tp_shards_reconstruct_bridge_full(
    qwen3_5_hf_modules,
    geometry,
    tp_size,
    is_bias,
):
    modules = qwen3_5_hf_modules
    hf_config = _make_hf_config(geometry)
    text = hf_config.text_config
    if text.num_key_value_heads % tp_size != 0:
        pytest.skip("kv groups not divisible by TP for this geometry")

    q_rows = 2 * text.num_attention_heads * text.head_dim
    kv_rows = text.num_key_value_heads * text.head_dim
    if is_bias:
        q = torch.arange(q_rows)
        k = torch.arange(kv_rows) + 100_000
        v = torch.arange(kv_rows) + 200_000
        name = "decoder.layers.0.self_attention.linear_qkv.bias"
    else:
        q = torch.arange(q_rows * text.hidden_size).reshape(q_rows, text.hidden_size)
        k = (
            torch.arange(kv_rows * text.hidden_size).reshape(kv_rows, text.hidden_size)
            + 100_000
        )
        v = (
            torch.arange(kv_rows * text.hidden_size).reshape(kv_rows, text.hidden_size)
            + 200_000
        )
        name = "decoder.layers.0.self_attention.linear_qkv.weight"

    modules.tp_state["tp_size"] = tp_size
    bridge = modules.bridge.Qwen3_5MoeBridge.__new__(modules.bridge.Qwen3_5MoeBridge)
    bridge.hf_config = hf_config
    full = bridge._weight_to_mcore_format(name, [q, k, v])
    local_dim0 = full.shape[0] // tp_size
    local_shape = [local_dim0] if is_bias else [local_dim0, text.hidden_size]

    locals_ = []
    for tp_rank in range(tp_size):
        shard = modules.hf_load._weight_to_mcore_tp(
            hf_config=hf_config,
            mcore_weights_name=name,
            mcore_param_shape=local_shape,
            hf_weights_safe_slice=[q, k, v],
            tp_rank=tp_rank,
            tp_size=tp_size,
            dtype=None,
        )
        assert list(shard.shape) == local_shape
        locals_.append(shard)

    reconstructed = torch.cat(locals_, dim=0)
    torch.testing.assert_close(reconstructed, full, rtol=0, atol=0)


@pytest.mark.parametrize("geometry", ["tiny", "large"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize(
    ("name", "kernel"),
    [
        ("decoder.layers.0.self_attention.linear_attn.in_proj_qkv.weight", 1),
        ("decoder.layers.0.self_attention.linear_attn.conv1d.weight", 3),
    ],
)
def test_hf_load_qwen3_5_gdn_fused_qkv_tp_shards_preserve_sections_and_reconstruct_bridge_full(
    qwen3_5_hf_modules,
    geometry,
    tp_size,
    name,
    kernel,
):
    modules = qwen3_5_hf_modules
    hf_config = _make_hf_config(geometry)
    text = hf_config.text_config
    sections = modules.utils.qwen3_5_gdn_qkv_section_sizes(hf_config)
    assert all(section % tp_size == 0 for section in sections)
    total = sum(sections)

    if kernel == 1:
        source = torch.arange(total * text.hidden_size).reshape(total, text.hidden_size)
        local_shape = [total // tp_size, text.hidden_size]
        dim = 0
    else:
        source = torch.arange(total * kernel).reshape(total, 1, kernel)
        local_shape = [total // tp_size, 1, kernel]
        dim = 0

    modules.tp_state["tp_size"] = tp_size
    bridge = modules.bridge.Qwen3_5MoeBridge.__new__(modules.bridge.Qwen3_5MoeBridge)
    bridge.hf_config = hf_config
    full = bridge._weight_to_mcore_format(name, [source])

    locals_ = []
    local_sections = [section // tp_size for section in sections]
    source_sections = torch.split(source, sections, dim=dim)
    for tp_rank in range(tp_size):
        shard = modules.hf_load._weight_to_mcore_tp(
            hf_config=hf_config,
            mcore_weights_name=name,
            mcore_param_shape=local_shape,
            hf_weights_safe_slice=[source],
            tp_rank=tp_rank,
            tp_size=tp_size,
            dtype=None,
        )
        assert list(shard.shape) == local_shape
        locals_.append(shard)

        shard_sections = torch.split(shard, local_sections, dim=dim)
        for section_id, expected_full_section in enumerate(source_sections):
            expected = torch.chunk(expected_full_section, tp_size, dim=dim)[tp_rank]
            torch.testing.assert_close(
                shard_sections[section_id],
                expected,
                rtol=0,
                atol=0,
            )

    reconstructed = torch.cat(locals_, dim=dim)
    torch.testing.assert_close(reconstructed, full, rtol=0, atol=0)
