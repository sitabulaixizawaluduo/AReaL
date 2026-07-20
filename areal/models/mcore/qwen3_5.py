# SPDX-License-Identifier: Apache-2.0

import copy

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import PretrainedConfig

from areal.models.mcore.common import check_and_construct_configs, hf_to_mcore_base_args
from areal.models.mcore.qwen3_5_gdn import (
    Qwen3_5GatedDeltaAttention,
    Qwen3_5GatedDeltaAttentionSubmodules,
    Qwen3_5GatedDeltaNet,
    Qwen3_5GatedDeltaNetSubmodules,
)


def _get_text_config(hf_config: PretrainedConfig) -> PretrainedConfig:
    return hf_config.text_config if hasattr(hf_config, "text_config") else hf_config


def _resolve_dtype(
    text_config: PretrainedConfig,
    fallback_dtype: torch.dtype,
) -> torch.dtype:
    for key in ("dtype", "torch_dtype"):
        value = getattr(text_config, key, None)
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            normalized = value.removeprefix("torch.")
            if normalized == "bfloat16":
                return torch.bfloat16
            if normalized == "float16":
                return torch.float16
            if normalized == "float32":
                return torch.float32
    return fallback_dtype


def _get_qwen3_5_layer_types(text_config: PretrainedConfig) -> list[str]:
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)
    interval = getattr(text_config, "full_attention_interval", 4)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(text_config.num_hidden_layers)
    ]


def hf_to_mcore_config_qwen3_5_moe(
    hf_config: PretrainedConfig,
    dtype: torch.dtype,
) -> TransformerConfig:
    text_config = _get_text_config(hf_config)
    resolved_dtype = _resolve_dtype(text_config, dtype)

    base_args = hf_to_mcore_base_args(
        hf_config=text_config,
        dtype=resolved_dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        add_qkv_bias=getattr(text_config, "attention_bias", False),
        qk_layernorm=True,
    )

    rope_parameters = getattr(text_config, "rope_parameters", None) or {}
    rotary_base = rope_parameters.get(
        "rope_theta", getattr(text_config, "rope_theta", 10000.0)
    )
    rotary_percent = getattr(text_config, "partial_rotary_factor", 1.0)

    ffn_hidden_size = getattr(text_config, "intermediate_size", None)
    if ffn_hidden_size is None:
        ffn_hidden_size = getattr(text_config, "shared_expert_intermediate_size")

    moe_args = {
        "ffn_hidden_size": ffn_hidden_size,
        "num_moe_experts": getattr(text_config, "num_experts", None),
        "moe_router_topk": getattr(text_config, "num_experts_per_tok", 1),
        "moe_ffn_hidden_size": getattr(text_config, "moe_intermediate_size", None),
        "moe_shared_expert_intermediate_size": getattr(
            text_config,
            "shared_expert_intermediate_size",
            None,
        ),
        "moe_shared_expert_gate": True,
        "moe_router_score_function": "softmax",
        "moe_aux_loss_coeff": getattr(text_config, "router_aux_loss_coef", 0.0),
        "moe_router_load_balancing_type": "aux_loss",
        "moe_grouped_gemm": True,
        "moe_router_dtype": "fp32",
        "moe_token_dispatcher_type": "alltoall",
        "moe_layer_freq": 1,
    }

    qwen3_5_args = {
        "attention_output_gate": True,
        "qk_layernorm": True,
        "layernorm_epsilon": text_config.rms_norm_eps,
        "num_layers": text_config.num_hidden_layers,
        "hidden_size": text_config.hidden_size,
        "num_attention_heads": text_config.num_attention_heads,
        "num_query_groups": text_config.num_key_value_heads,
        "kv_channels": getattr(text_config, "head_dim", None),
        "rotary_base": rotary_base,
        "rotary_percent": rotary_percent,
        "bf16": resolved_dtype == torch.bfloat16,
        "params_dtype": resolved_dtype,
        "pipeline_dtype": resolved_dtype,
        "untie_embeddings_and_output_weights": not getattr(
            text_config,
            "tie_word_embeddings",
            False,
        ),
    }

    return check_and_construct_configs(
        {**base_args, **qwen3_5_args, **moe_args},
        TransformerConfig,
    )


def _te_linear_and_norm():
    try:
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TENorm,
            TERowParallelLinear,
        )

        return TEColumnParallelLinear, TERowParallelLinear, TENorm
    except ImportError:
        from megatron.core.tensor_parallel import (
            ColumnParallelLinear as TEColumnParallelLinear,
        )
        from megatron.core.tensor_parallel import (
            RowParallelLinear as TERowParallelLinear,
        )
        from megatron.core.transformer.torch_norm import WrappedTorchNorm as TENorm

        return TEColumnParallelLinear, TERowParallelLinear, TENorm


def _build_qwen3_5_linear_attn_spec(
    text_config: PretrainedConfig,
) -> ModuleSpec:
    col, row, norm = _te_linear_and_norm()
    return ModuleSpec(
        module=Qwen3_5GatedDeltaAttention,
        submodules=Qwen3_5GatedDeltaAttentionSubmodules(
            linear_attn=ModuleSpec(
                module=Qwen3_5GatedDeltaNet,
                submodules=Qwen3_5GatedDeltaNetSubmodules(
                    in_proj_qkv=col,
                    in_proj_z=col,
                    in_proj_b=col,
                    in_proj_a=col,
                    out_proj=row,
                ),
            ),
            input_layernorm=norm,
        ),
        params={
            "linear_num_key_heads": text_config.linear_num_key_heads,
            "linear_num_value_heads": text_config.linear_num_value_heads,
            "linear_key_head_dim": text_config.linear_key_head_dim,
            "linear_value_head_dim": text_config.linear_value_head_dim,
            "linear_conv_kernel_dim": text_config.linear_conv_kernel_dim,
            "hidden_act": text_config.hidden_act,
            "bias": False,
            "conv_bias": False,
        },
    )


def _validate_linear_attn_cp_divisibility(
    tf_config: TransformerConfig,
    text_config: PretrainedConfig,
    layer_types: list[str],
) -> None:
    if tf_config.context_parallel_size <= 1:
        return
    if "linear_attention" not in layer_types:
        return

    tp_size = tf_config.tensor_model_parallel_size
    cp_size = tf_config.context_parallel_size
    k_heads_per_tp = text_config.linear_num_key_heads // tp_size
    v_heads_per_tp = text_config.linear_num_value_heads // tp_size

    if k_heads_per_tp * tp_size != text_config.linear_num_key_heads:
        raise ValueError(
            "For Qwen3.5 GDN with CP, linear_num_key_heads must be divisible by TP. "
            f"Got linear_num_key_heads={text_config.linear_num_key_heads}, TP={tp_size}."
        )
    if v_heads_per_tp * tp_size != text_config.linear_num_value_heads:
        raise ValueError(
            "For Qwen3.5 GDN with CP, linear_num_value_heads must be divisible by TP. "
            f"Got linear_num_value_heads={text_config.linear_num_value_heads}, TP={tp_size}."
        )
    if k_heads_per_tp % cp_size != 0:
        raise ValueError(
            "For Qwen3.5 GDN with CP, linear_num_key_heads / TP must be divisible by CP. "
            f"Got linear_num_key_heads={text_config.linear_num_key_heads}, TP={tp_size}, CP={cp_size}."
        )
    if v_heads_per_tp % cp_size != 0:
        raise ValueError(
            "For Qwen3.5 GDN with CP, linear_num_value_heads / TP must be divisible by CP. "
            f"Got linear_num_value_heads={text_config.linear_num_value_heads}, TP={tp_size}, CP={cp_size}."
        )


def make_mcore_layer_specs_qwen3_5_moe(
    tf_config: TransformerConfig,
    hf_config: PretrainedConfig,
    use_te: bool = True,
    vp_stage: int | None = None,
):
    text_config = _get_text_config(hf_config)
    layer_types = _get_qwen3_5_layer_types(text_config)
    _validate_linear_attn_cp_divisibility(tf_config, text_config, layer_types)

    decoder_spec = get_gpt_decoder_block_spec(
        tf_config,
        use_transformer_engine=use_te,
    )
    linear_attn_spec = _build_qwen3_5_linear_attn_spec(text_config)

    num_layers_to_build = get_num_layers_to_build(tf_config, vp_stage=vp_stage)
    layer_offset = get_transformer_layer_offset(tf_config, vp_stage=vp_stage)

    for local_layer_id in range(num_layers_to_build):
        global_layer_id = local_layer_id + layer_offset
        if layer_types[global_layer_id] != "linear_attention":
            continue
        layer_spec = copy.deepcopy(decoder_spec.layer_specs[local_layer_id])
        layer_spec.submodules.self_attention = linear_attn_spec
        layer_spec.submodules.input_layernorm = IdentityOp
        decoder_spec.layer_specs[local_layer_id] = layer_spec

    return decoder_spec
