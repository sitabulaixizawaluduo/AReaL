# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

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


def _load_qwen3_5_modules(monkeypatch):
    class _CudaRngTracker:
        def fork(self):
            return self

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _MegatronModule(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config

    class _ModuleSpec:
        def __init__(self, module=None, submodules=None, params=None):
            self.module = module
            self.submodules = submodules
            self.params = params

    class _FakeParallelLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=False):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter("bias", None)

        def forward(self, input_):
            return nn.functional.linear(input_, self.weight, self.bias), None

    class _Logger:
        def info(self, *args, **kwargs):
            return None

    class _LLMBridge:
        def _weight_to_mcore_format(self, mcore_weights_name, hf_weights):
            raise AssertionError(f"unexpected fallback for {mcore_weights_name}")

        def _weight_to_hf_format(self, mcore_weights_name, mcore_weights):
            raise AssertionError(f"unexpected fallback for {mcore_weights_name}")

    class _AttnBackend:
        fused = "fused"

    def _build_module(spec, *args, **kwargs):
        if len(args) >= 2:
            return _FakeParallelLinear(args[0], args[1], bias=kwargs.get("bias", False))
        return nn.Identity()

    parallel_state = _stub_module(
        monkeypatch,
        "megatron.core.parallel_state",
        model_parallel_is_initialized=lambda: False,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: 0,
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_context_parallel_group=lambda: None,
    )
    mpu = _stub_module(
        monkeypatch,
        "megatron.core.mpu",
        get_expert_model_parallel_world_size=lambda: 1,
        get_expert_model_parallel_rank=lambda: 0,
    )
    _stub_module(monkeypatch, "megatron")
    megatron_core = _stub_module(
        monkeypatch,
        "megatron.core",
        parallel_state=parallel_state,
        mpu=mpu,
    )
    _stub_module(monkeypatch, "megatron.core.models")
    _stub_module(monkeypatch, "megatron.core.models.common")
    _stub_module(monkeypatch, "megatron.core.models.common.embeddings")
    _stub_module(
        monkeypatch,
        "megatron.core.models.common.embeddings.rotary_pos_embedding",
        apply_rotary_pos_emb=lambda x, *args, **kwargs: x,
    )
    _stub_module(monkeypatch, "megatron.core.models.gpt")
    _stub_module(
        monkeypatch,
        "megatron.core.models.gpt.gpt_layer_specs",
        get_gpt_decoder_block_spec=lambda *args, **kwargs: None,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.dist_checkpointing",
        ShardedTensor=type("ShardedTensor", (), {}),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.dist_checkpointing.mapping",
        ReplicaId=type("ReplicaId", (), {}),
        ShardedTensorFactory=type("ShardedTensorFactory", (), {}),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.packed_seq_params",
        PackedSeqParams=type("PackedSeqParams", (), {}),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.tensor_parallel",
        get_cuda_rng_tracker=lambda: _CudaRngTracker(),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer",
        TransformerConfig=type("TransformerConfig", (), {}),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.enums",
        AttnBackend=_AttnBackend,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.identity_op",
        IdentityOp=type("IdentityOp", (nn.Module,), {}),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.module",
        MegatronModule=_MegatronModule,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.spec_utils",
        ModuleSpec=_ModuleSpec,
        build_module=_build_module,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.transformer_block",
        get_num_layers_to_build=lambda *args, **kwargs: 0,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.transformer_layer",
        get_transformer_layer_offset=lambda *args, **kwargs: 0,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.utils",
        make_sharded_tensors_for_checkpoint=lambda *args, **kwargs: {},
        sharded_state_dict_default=lambda *args, **kwargs: {},
    )

    _stub_module(monkeypatch, "fla")
    _stub_module(
        monkeypatch,
        "fla.modules",
        FusedRMSNormGated=type("FusedRMSNormGated", (nn.Module,), {}),
    )
    _stub_module(monkeypatch, "fla.ops")
    _stub_module(
        monkeypatch,
        "fla.ops.gated_delta_rule",
        chunk_gated_delta_rule=lambda *args, **kwargs: None,
    )

    _stub_module(monkeypatch, "mbridge")
    _stub_module(
        monkeypatch,
        "mbridge.core",
        LLMBridge=_LLMBridge,
        register_model=lambda names: lambda cls: cls,
    )
    _stub_module(
        monkeypatch, "transformers", PretrainedConfig=type("PretrainedConfig", (), {})
    )

    _stub_module(monkeypatch, "areal")
    _stub_module(monkeypatch, "areal.models")
    _stub_module(monkeypatch, "areal.models.mcore")
    _stub_module(monkeypatch, "areal.utils")
    logging_module = _stub_module(
        monkeypatch,
        "areal.utils.logging",
        getLogger=lambda name: _Logger(),
    )
    sys.modules["areal.utils"].logging = logging_module
    _stub_module(
        monkeypatch,
        "areal.models.mcore.common",
        check_and_construct_configs=lambda args, cls: cls(**args),
        hf_to_mcore_base_args=lambda **kwargs: {},
    )

    lightning = _load_module(
        monkeypatch,
        "areal.models.mcore.lightning_attention",
        "areal/models/mcore/lightning_attention.py",
    )
    gdn = _load_module(
        monkeypatch,
        "areal.models.mcore.qwen3_5_gdn",
        "areal/models/mcore/qwen3_5_gdn.py",
    )
    qwen = _load_module(
        monkeypatch,
        "areal.models.mcore.qwen3_5",
        "areal/models/mcore/qwen3_5.py",
    )
    bridge = _load_module(
        monkeypatch,
        "areal.models.mcore.qwen3_5_bridge",
        "areal/models/mcore/qwen3_5_bridge.py",
    )
    megatron_core.mpu = mpu
    return SimpleNamespace(lightning=lightning, gdn=gdn, qwen=qwen, bridge=bridge)


@pytest.fixture
def qwen3_5_modules(monkeypatch):
    return _load_qwen3_5_modules(monkeypatch)


def _collective_received(prepared_inputs, target_rank, cp_size):
    return torch.cat(
        [
            torch.chunk(input_, cp_size, dim=0)[target_rank]
            for input_ in prepared_inputs
        ],
        dim=0,
    )


def _simulate_cp2hp(monkeypatch, gdn, rank_inputs, split_sections):
    cp_size = len(rank_inputs)
    sections_by_rank = [
        torch.split(input_, split_sections, dim=-1) for input_ in rank_inputs
    ]
    outputs = []
    monkeypatch.setattr(gdn.dist, "get_world_size", lambda group=None: cp_size)

    for target_rank in range(cp_size):
        call_index = 0

        def fake_all_to_all(input_, cp_group):
            nonlocal call_index
            section_index = call_index
            call_index += 1
            prepared = []
            for rank_sections in sections_by_rank:
                section = rank_sections[section_index]
                seq_len, batch, hidden = section.shape
                hidden_per_cp = hidden // cp_size
                flat = section.reshape(seq_len * batch, hidden)
                prepared.append(
                    torch.cat(torch.split(flat, hidden_per_cp, dim=-1), dim=0)
                )
            return _collective_received(prepared, target_rank, cp_size)

        monkeypatch.setattr(gdn, "_all_to_all_equal", fake_all_to_all)
        outputs.append(
            gdn._all_to_all_cp2hp(
                rank_inputs[target_rank],
                cp_group=object(),
                split_size_or_sections=split_sections,
            )
        )
        assert call_index == len(split_sections)
    return outputs


def _simulate_hp2cp(monkeypatch, gdn, rank_inputs, split_sections):
    cp_size = len(rank_inputs)
    local_sections = [section // cp_size for section in split_sections]
    sections_by_rank = [
        torch.split(input_, local_sections, dim=-1) for input_ in rank_inputs
    ]
    outputs = []
    monkeypatch.setattr(gdn.dist, "get_world_size", lambda group=None: cp_size)

    for target_rank in range(cp_size):
        call_index = 0

        def fake_all_to_all(input_, cp_group):
            nonlocal call_index
            section_index = call_index
            call_index += 1
            flat_inputs = [
                rank_sections[section_index].reshape(
                    -1, rank_sections[section_index].shape[-1]
                )
                for rank_sections in sections_by_rank
            ]
            return _collective_received(flat_inputs, target_rank, cp_size)

        monkeypatch.setattr(gdn, "_all_to_all_equal", fake_all_to_all)
        outputs.append(
            gdn._all_to_all_hp2cp(
                rank_inputs[target_rank],
                cp_group=object(),
                split_size_or_sections=split_sections,
            )
        )
        assert call_index == len(split_sections)
    return outputs


def test_get_parameter_local_cp_each_qkv_section_returns_rank_local_heads(
    qwen3_5_modules,
):
    gdn = qwen3_5_modules.gdn
    cp_size = 4
    head_counts = [4, 4, 8]
    head_dim = 2
    sections = []
    for section_id, head_count in enumerate(head_counts):
        head_ids = torch.arange(head_count).repeat_interleave(head_dim)
        sections.append((section_id + 1) * 100 + head_ids)
    conv_weight = torch.cat(sections).reshape(-1, 1, 1)
    split_sections = [head_count * head_dim for head_count in head_counts]

    for cp_rank in range(cp_size):
        actual = gdn._get_parameter_local_cp(
            conv_weight,
            dim=0,
            cp_rank=cp_rank,
            cp_size=cp_size,
            split_size_or_sections=split_sections,
        )
        expected_sections = []
        for section_id, head_count in enumerate(head_counts):
            heads_per_rank = head_count // cp_size
            local_heads = torch.arange(
                cp_rank * heads_per_rank,
                (cp_rank + 1) * heads_per_rank,
            ).repeat_interleave(head_dim)
            expected_sections.append((section_id + 1) * 100 + local_heads)
        expected = torch.cat(expected_sections).reshape(-1, 1, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_all_to_all_cp2hp_with_qkv_sections_preserves_section_boundaries(
    qwen3_5_modules, monkeypatch
):
    gdn = qwen3_5_modules.gdn
    cp_size = 2
    split_sections = [4, 4, 8]
    rank_inputs = []
    for source_rank in range(cp_size):
        sections = []
        for section_id, width in enumerate(split_sections):
            values = torch.empty(2, 1, width, dtype=torch.int64)
            for token in range(2):
                values[token, 0] = (
                    (section_id + 1) * 1000
                    + source_rank * 100
                    + token * 10
                    + torch.arange(width)
                )
            sections.append(values)
        rank_inputs.append(torch.cat(sections, dim=-1))

    outputs = _simulate_cp2hp(monkeypatch, gdn, rank_inputs, split_sections)

    for target_rank, actual in enumerate(outputs):
        expected_sections = []
        for section_id, width in enumerate(split_sections):
            width_per_rank = width // cp_size
            expected_sections.append(
                torch.cat(
                    [
                        torch.split(rank_inputs[source], split_sections, dim=-1)[
                            section_id
                        ][
                            ...,
                            target_rank * width_per_rank : (target_rank + 1)
                            * width_per_rank,
                        ]
                        for source in range(cp_size)
                    ],
                    dim=0,
                )
            )
        expected = torch.cat(expected_sections, dim=-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_all_to_all_hp2cp_after_cp2hp_returns_original(
    qwen3_5_modules, monkeypatch, cp_size
):
    gdn = qwen3_5_modules.gdn
    split_sections = [2 * cp_size, 2 * cp_size, 3 * cp_size]
    hidden = sum(split_sections)
    rank_inputs = [
        torch.arange(rank * 10000, rank * 10000 + 2 * hidden).reshape(2, 1, hidden)
        for rank in range(cp_size)
    ]

    head_parallel = _simulate_cp2hp(monkeypatch, gdn, rank_inputs, split_sections)
    roundtrip = _simulate_hp2cp(monkeypatch, gdn, head_parallel, split_sections)

    for actual, expected in zip(roundtrip, rank_inputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("cp_size", "sequence_lengths"),
    [(2, [4, 8, 12]), (4, [8, 16])],
)
def test_zigzag_undo_redo_with_unequal_packed_sequences_returns_original(
    qwen3_5_modules, cp_size, sequence_lengths
):
    lightning = qwen3_5_modules.lightning
    cu_seqlens = torch.tensor([0, *torch.tensor(sequence_lengths).cumsum(0).tolist()])
    total_len = int(cu_seqlens[-1])
    sequential = torch.arange(total_len)
    undo = lightning._build_zigzag_undo_indices(
        total_len,
        cp_size,
        cu_seqlens,
        sequential.device,
    )
    redo = lightning._build_zigzag_redo_indices(undo)
    zigzag = sequential[redo]

    restored_sequential = zigzag[undo]
    restored_zigzag = restored_sequential[redo]

    torch.testing.assert_close(restored_sequential, sequential, rtol=0, atol=0)
    torch.testing.assert_close(restored_zigzag, zigzag, rtol=0, atol=0)
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        expected = torch.arange(int(start), int(end))
        torch.testing.assert_close(
            restored_sequential[int(start) : int(end)],
            expected,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("key_heads", "value_heads", "tp_size", "cp_size", "message"),
    [
        (5, 8, 2, 2, "linear_num_key_heads must be divisible by TP"),
        (4, 9, 2, 2, "linear_num_value_heads must be divisible by TP"),
        (6, 8, 2, 2, "linear_num_key_heads / TP must be divisible by CP"),
        (8, 6, 2, 2, "linear_num_value_heads / TP must be divisible by CP"),
    ],
)
def test_validate_linear_attn_cp_divisibility_with_invalid_geometry_raises_clear_error(
    qwen3_5_modules,
    key_heads,
    value_heads,
    tp_size,
    cp_size,
    message,
):
    tf_config = SimpleNamespace(
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
    )
    text_config = SimpleNamespace(
        linear_num_key_heads=key_heads,
        linear_num_value_heads=value_heads,
    )

    with pytest.raises(ValueError, match=message):
        qwen3_5_modules.qwen._validate_linear_attn_cp_divisibility(
            tf_config,
            text_config,
            ["linear_attention"],
        )


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("cp_size", [1, 2])
def test_validate_linear_attn_cp_divisibility_with_tiny_geometry_passes(
    qwen3_5_modules, tp_size, cp_size
):
    tf_config = SimpleNamespace(
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
    )
    text_config = SimpleNamespace(
        linear_num_key_heads=4,
        linear_num_value_heads=8,
    )

    qwen3_5_modules.qwen._validate_linear_attn_cp_divisibility(
        tf_config,
        text_config,
        ["linear_attention"],
    )


def test_get_qwen3_5_layer_types_with_explicit_types_returns_copy(qwen3_5_modules):
    layer_types = ["full_attention", "linear_attention"]
    text_config = SimpleNamespace(layer_types=layer_types, num_hidden_layers=2)

    actual = qwen3_5_modules.qwen._get_qwen3_5_layer_types(text_config)

    assert actual == layer_types
    assert actual is not layer_types


def test_get_qwen3_5_layer_types_with_interval_four_returns_three_to_one_pattern(
    qwen3_5_modules,
):
    text_config = SimpleNamespace(num_hidden_layers=8, full_attention_interval=4)

    actual = qwen3_5_modules.qwen._get_qwen3_5_layer_types(text_config)

    assert (
        actual
        == [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        * 2
    )


def test_bridge_q_gate_conversion_roundtrip_returns_original_hf_weights(
    qwen3_5_modules,
):
    bridge_module = qwen3_5_modules.bridge
    bridge = bridge_module.Qwen3_5MoeBridge.__new__(bridge_module.Qwen3_5MoeBridge)
    bridge.hf_config = SimpleNamespace(
        num_key_value_heads=2,
        num_attention_heads=8,
        head_dim=64,
        hidden_size=512,
    )
    input_dim = 3
    q = torch.arange(2 * 8 * 64 * input_dim).reshape(2 * 8 * 64, input_dim)
    k = torch.arange(2 * 64 * input_dim).reshape(2 * 64, input_dim) + 100000
    v = torch.arange(2 * 64 * input_dim).reshape(2 * 64, input_dim) + 200000
    mcore_name = "decoder.layers.0.self_attention.linear_qkv.weight"

    mcore_weight = bridge._weight_to_mcore_format(mcore_name, [q, k, v])
    hf_names, roundtrip = bridge._weight_to_hf_format(mcore_name, mcore_weight)

    assert hf_names == [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
    ]
    for actual, expected in zip(roundtrip, [q, k, v]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_bridge_fused_expert_conversion_with_3d_tensor_extracts_requested_expert(
    qwen3_5_modules,
):
    bridge_module = qwen3_5_modules.bridge
    bridge = bridge_module.Qwen3_5MoeBridge.__new__(bridge_module.Qwen3_5MoeBridge)
    bridge.hf_config = SimpleNamespace()
    fused_experts = torch.arange(4 * 6 * 3).reshape(4, 6, 3)

    actual = bridge._weight_to_mcore_format(
        "decoder.layers.0.mlp.experts.linear_fc1.weight2",
        [fused_experts],
    )

    torch.testing.assert_close(actual, fused_experts[2], rtol=0, atol=0)


def test_gated_delta_net_named_parameters_match_bridge_linear_attn_mapping(
    qwen3_5_modules,
):
    gdn = qwen3_5_modules.gdn
    bridge = qwen3_5_modules.bridge
    config = SimpleNamespace(
        hidden_size=16,
        init_method=lambda tensor: tensor,
        output_layer_init_method=lambda tensor: tensor,
        params_dtype=torch.float32,
        layernorm_epsilon=1e-6,
    )
    submodules = gdn.Qwen3_5GatedDeltaNetSubmodules(
        in_proj_qkv=object,
        in_proj_z=object,
        in_proj_b=object,
        in_proj_a=object,
        out_proj=object,
    )

    module = gdn.Qwen3_5GatedDeltaNet(
        config,
        submodules,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
    )
    actual = {name for name, _ in module.named_parameters()}
    prefix = "self_attention.linear_attn."
    expected = {
        name.removeprefix(prefix)
        for name in bridge.Qwen3_5MoeBridge._ATTENTION_MAPPING
        if name.startswith(prefix)
    }

    assert actual == expected


def test_linear_attn_spec_places_gdn_params_on_inner_spec(qwen3_5_modules, monkeypatch):
    monkeypatch.setattr(
        qwen3_5_modules.qwen,
        "_te_linear_and_norm",
        lambda: (object, object, object),
    )
    text_config = SimpleNamespace(
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
        hidden_act="silu",
    )

    spec = qwen3_5_modules.qwen._build_qwen3_5_linear_attn_spec(text_config)

    inner_params = spec.submodules.linear_attn.params
    required = {
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    }
    assert required <= set(inner_params), (
        "GDN constructor hyperparams must live on the inner linear_attn "
        f"ModuleSpec; missing {required - set(inner_params)}"
    )
    assert not spec.params, (
        "outer Qwen3_5GatedDeltaAttention spec must stay params-free: mcore's "
        "TransformerLayer injects its own kwargs (pg_collection, ...) there, "
        "and the wrapper does not forward arbitrary kwargs to the GDN module"
    )
