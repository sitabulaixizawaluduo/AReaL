# SPDX-License-Identifier: Apache-2.0

"""AWEX compatibility registrations for Qwen3-VL colocated weight updates.

AWEX 0.8 ships converters for Qwen3-MoE, but not for the dense or MoE
Qwen3-VL architectures.  The colocate path needs both sides to describe every
parameter with the same canonical Hugging Face name, including the vision
tower and deepstack mergers.  This module supplies those missing converters
without changing AWEX globally unless the v1 colocate path is initialized.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import torch
from awex.models.qwen3_moe import SGlangToHFWeightConverterQwen3Moe
from awex.sharding.param_sharding import (
    ShardingStrategy,
    ShardingType,
    get_default_sharding_dim,
)

QWEN3_VL_ARCHITECTURES = (
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
)


def get_qwen3_vl_text_config(config: Any) -> Any:
    """Return the nested text config used by Qwen3-VL when it is present."""
    return getattr(config, "text_config", config)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _uses_tied_embeddings(config: Any) -> bool:
    return bool(
        _config_value(get_qwen3_vl_text_config(config), "tie_word_embeddings", False)
    )


class Qwen3VLShardingStrategy(ShardingStrategy):
    """Add the vision-tower TP layout missing from AWEX's base strategy."""

    def _vision_tp_strategy(self, sharding_dim: int):
        if self.enable_dp_attention:
            tp_size = self.rank_info.attn_tp_size
            sharding_type = ShardingType.DP_TP_SHARDING
        else:
            tp_size = self.rank_info.tp_size
            sharding_type = ShardingType.TP_SHARDING
        if tp_size > 1:
            return sharding_type, sharding_dim, tp_size
        return ShardingType.NO_SHARDING, sharding_dim, 1

    def get_sharding_strategy(self, parameter_name: str, **kwargs):
        if not parameter_name.startswith("model.visual."):
            return super().get_sharding_strategy(parameter_name, **kwargs)

        sharding_dim = get_default_sharding_dim(parameter_name)
        replicated_suffixes = (
            ".bias",
            ".norm.weight",
            ".norm.bias",
            ".norm1.weight",
            ".norm1.bias",
            ".norm2.weight",
            ".norm2.bias",
        )
        if parameter_name.startswith("model.visual.patch_embed.") or (
            parameter_name.endswith(replicated_suffixes)
            and not parameter_name.endswith(("qkv.bias", "linear_fc1.bias"))
        ):
            return ShardingType.NO_SHARDING, sharding_dim, 1

        if parameter_name.endswith(
            (
                "pos_embed.weight",
                "attn.qkv.weight",
                "attn.qkv.bias",
                "mlp.linear_fc1.weight",
                "mlp.linear_fc1.bias",
                "merger.linear_fc1.weight",
                "merger.linear_fc1.bias",
            )
        ) or (
            ".deepstack_merger_list." in parameter_name
            and parameter_name.endswith(("linear_fc1.weight", "linear_fc1.bias"))
        ):
            return self._vision_tp_strategy(0)

        if parameter_name.endswith(
            (
                "attn.proj.weight",
                "mlp.linear_fc2.weight",
                "merger.linear_fc2.weight",
            )
        ) or (
            ".deepstack_merger_list." in parameter_name
            and parameter_name.endswith("linear_fc2.weight")
        ):
            return self._vision_tp_strategy(1)

        return ShardingType.NO_SHARDING, sharding_dim, 1


class Qwen3VLSGlangToHFWeightConverter(SGlangToHFWeightConverterQwen3Moe):
    """Reuse AWEX Qwen3 text conversion and add Qwen3-VL namespaces."""

    def __init__(self, model_config, infer_engine_config, rank_info):
        self.vl_model_config = model_config
        super().__init__(
            get_qwen3_vl_text_config(model_config), infer_engine_config, rank_info
        )

    @torch.no_grad()
    def convert_param(self, name: str, parameter: torch.Tensor):
        if name.startswith("visual."):
            hf_name = f"model.{name}"
            hf_name = hf_name.replace(".attn.qkv_proj.", ".attn.qkv.")
            return [(hf_name, parameter)]

        converted = super().convert_param(name, parameter)
        prefixed = []
        for hf_name, hf_param in converted:
            if hf_name.startswith("model."):
                hf_name = hf_name.replace("model.", "model.language_model.", 1)
            prefixed.append((hf_name, hf_param))

        if name == "model.embed_tokens.weight" and _uses_tied_embeddings(
            self.vl_model_config
        ):
            prefixed.append(("lm_head.weight", parameter))
        return prefixed


class _Qwen3VLMcoreToHFWeightConverter:
    """Delegate Megatron conversion to AReaL's mbridge-aligned VL mapping."""

    model_name = "qwen3_vl"
    _vision_qkv_pattern = re.compile(
        r"vision_model\.decoder\.layers\.(\d+)\."
        r"self_attention\.linear_qkv\.(weight|bias)"
    )

    def __init__(self, hf_config, rank_info, infer_conf, tf_config):
        self.hf_config = hf_config
        self.rank_info = rank_info
        self.infer_conf = infer_conf
        self.tf_config = tf_config
        self._pp_stage_layer_id_map = infer_conf.get("train_pp_stage_layer_id_map")
        router_dtype = infer_conf.get("router_dtype", "bf16")
        try:
            self.router_dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }[router_dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported router dtype: {router_dtype}") from exc

    def _normalize_expert_id(self, name: str) -> str:
        if self.rank_info.ep_size <= 1:
            return name
        match = re.search(r"(\.mlp\.experts\.[^.]+\.weight)(\d+)$", name)
        if match is None:
            return name

        text_config = get_qwen3_vl_text_config(self.hf_config)
        num_experts = _config_value(text_config, "num_experts")
        if num_experts is None:
            return name
        experts_per_rank = int(num_experts) // int(self.rank_info.ep_size)
        global_expert_id = int(match.group(2)) + (
            int(self.rank_info.ep_rank) * experts_per_rank
        )
        return f"{name[: match.start(2)]}{global_expert_id}"

    def _convert_vision_qkv(
        self, name: str, parameter: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]] | None:
        match = self._vision_qkv_pattern.fullmatch(name)
        if match is None:
            return None

        from awex.converter.mcore_converter import (
            convert_qkv_weight_along_tp_attention,
        )

        vision_config = _config_value(self.hf_config, "vision_config")
        num_heads = _config_value(vision_config, "num_heads")
        hidden_size = _config_value(vision_config, "hidden_size")
        if num_heads is None or hidden_size is None:
            raise ValueError(
                "hf_config.vision_config.num_heads and hidden_size are required "
                "for Qwen3-VL vision QKV conversion"
            )

        # SGLang stores fused QKV per TP rank as [Q_rank, K_rank, V_rank].
        # Repack Megatron's head-interleaved tensor into rank-major blocks before
        # AWEX applies its ordinary dim-0 transfer-plan sharding.
        vision_tf_config = SimpleNamespace(
            hidden_size=int(hidden_size),
            num_attention_heads=int(num_heads),
            num_query_groups=int(num_heads),
            kv_channels=int(hidden_size) // int(num_heads),
        )
        packed = convert_qkv_weight_along_tp_attention(
            parameter,
            max(1, int(self.infer_conf.get("infer_atten_tp_size", 1))),
            vision_tf_config,
            train_tp_rank=int(self.rank_info.attn_tp_rank),
            train_tp_size=max(1, int(self.rank_info.attn_tp_size)),
        )
        layer_number, kind = match.groups()
        return [(f"model.visual.blocks.{layer_number}.attn.qkv.{kind}", packed)]

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor, vp_stage: int | None = None
    ):
        from awex.converter.mcore_converter import _process_mcore_pp_name

        from areal.engine.megatron_utils.megatron import convert_to_hf

        while name.startswith("module."):
            name = name[len("module.") :]

        vision_qkv = self._convert_vision_qkv(name, parameter)
        if vision_qkv is not None:
            return vision_qkv

        # Only the language decoder is pipeline-partitioned. Vision block ids
        # are global and must never be remapped through the language PP table.
        if name.startswith("language_model.decoder.layers."):
            name = _process_mcore_pp_name(
                name,
                self.rank_info,
                get_qwen3_vl_text_config(self.hf_config),
                self.tf_config,
                vp_stage=vp_stage,
                pp_stage_layer_id_map=self._pp_stage_layer_id_map,
            )

        name = self._normalize_expert_id(name)
        converted = convert_to_hf(
            self.tf_config,
            self.model_name,
            f"module.module.{name}",
            parameter,
            hf_config=self.hf_config,
        )
        converted = [
            (
                hf_name,
                hf_param.to(self.router_dtype)
                if hf_name.endswith(".mlp.gate.weight")
                else hf_param,
            )
            for hf_name, hf_param in converted
        ]
        if (
            name == "language_model.embedding.word_embeddings.weight"
            and _uses_tied_embeddings(self.hf_config)
        ):
            converted.append(("lm_head.weight", parameter))
        return converted


class Qwen3VLMcoreToHFWeightConverter(_Qwen3VLMcoreToHFWeightConverter):
    model_name = "qwen3_vl"


class Qwen3VLMoeMcoreToHFWeightConverter(_Qwen3VLMcoreToHFWeightConverter):
    model_name = "qwen3_vl_moe"


def _build_qwen3_vl_mcore_converter():
    return Qwen3VLMcoreToHFWeightConverter


def _build_qwen3_vl_moe_mcore_converter():
    return Qwen3VLMoeMcoreToHFWeightConverter


_AWEX_CONFIGS = {
    "Qwen3VLForConditionalGeneration": {
        "model_name": "Qwen3VLForConditionalGeneration",
        "sharding_strategy": Qwen3VLShardingStrategy,
        "mcore_converter": _build_qwen3_vl_mcore_converter,
        "sglang_converter": Qwen3VLSGlangToHFWeightConverter,
    },
    "Qwen3VLMoeForConditionalGeneration": {
        "model_name": "Qwen3VLMoeForConditionalGeneration",
        "sharding_strategy": Qwen3VLShardingStrategy,
        "mcore_converter": _build_qwen3_vl_moe_mcore_converter,
        "sglang_converter": Qwen3VLSGlangToHFWeightConverter,
    },
}


def register_qwen3_vl_awex_models() -> None:
    """Idempotently install Qwen3-VL entries into AWEX's process registry."""
    from awex.models.registry import ModelRegistry

    ModelRegistry.models.update(_AWEX_CONFIGS)


__all__ = [
    "QWEN3_VL_ARCHITECTURES",
    "Qwen3VLMcoreToHFWeightConverter",
    "Qwen3VLMoeMcoreToHFWeightConverter",
    "Qwen3VLSGlangToHFWeightConverter",
    "Qwen3VLShardingStrategy",
    "get_qwen3_vl_text_config",
    "register_qwen3_vl_awex_models",
]
