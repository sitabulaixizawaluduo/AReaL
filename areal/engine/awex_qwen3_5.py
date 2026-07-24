# SPDX-License-Identifier: Apache-2.0

"""AWEX training-side weight conversion for Qwen3.5 MoE vision-language models."""

from __future__ import annotations

# pyright: reportMissingImports=false
import re
from typing import Any, TypeAlias, cast

import torch
from awex.converter.mcore_converter import (
    McoreToHFWeightConverter,
    _process_mcore_pp_name,
)
from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.sharding.param_sharding import ShardingStrategy, ShardingType

ConvertedParam: TypeAlias = tuple[str, torch.Tensor]
ConvertedParams: TypeAlias = list[ConvertedParam]


class McoreToHFWeightConverterQwen3_5Moe(McoreToHFWeightConverter):
    """Convert local MCore shards to Qwen3.5-MoE HF names without global gathers."""

    _LANGUAGE_PREFIX = "language_model."
    _HF_LANGUAGE_PREFIX = "model.language_model."
    _VISION_PREFIX = "vision_model."
    _HF_VISION_PREFIX = "model.visual."
    _LOCAL_EXPERT_PATTERN = re.compile(
        r"^mlp\.experts\.local_experts\.(\d+)\.linear_fc([12])\.weight$"
    )
    _PACKED_EXPERT_PATTERN = re.compile(r"^mlp\.experts\.linear_fc([12])\.weight(\d+)$")

    @torch.no_grad()
    def convert_param(
        self,
        name: str,
        parameter: torch.Tensor,
        vp_stage: int | None = None,
    ) -> ConvertedParams:
        name = name.replace("module.", "")
        if self._is_mtp_param(name):
            return []

        if name.startswith(self._LANGUAGE_PREFIX):
            # train_pp_stage_layer_id_map is derived from converted `.layers.` names
            # in the language decoder path; vision blocks must bypass this mapping.
            name = _process_mcore_pp_name(
                name,
                self.rank_info,
                self.hf_config,
                self.tf_config,
                vp_stage=0 if vp_stage is None else vp_stage,
                pp_stage_layer_id_map=self._pp_stage_layer_id_map,
            )
            if self._is_mtp_param(name):
                return []
            return self._convert_language_param(
                name.removeprefix(self._LANGUAGE_PREFIX), parameter
            )
        if name.startswith(self._VISION_PREFIX):
            return self._convert_vision_param(
                name.removeprefix(self._VISION_PREFIX), parameter
            )

        raise NotImplementedError(
            "Unsupported Qwen3.5 parameter outside language_model./vision_model.: "
            f"{name}"
        )

    @staticmethod
    def _is_mtp_param(name: str) -> bool:
        return "mtp" in name.split(".")

    @staticmethod
    def _cfg_get(config: object, key: str, default: object = None) -> object:
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _cfg_int(self, config: object, key: str) -> int:
        value = self._cfg_get(config, key, None)
        if value is None:
            raise ValueError(f"Missing required config field: {key}")
        if not isinstance(value, int):
            raise ValueError(
                f"Config field must be int: {key}={value} (type={type(value).__name__})"
            )
        return value

    def _hf_text_config(self) -> object:
        text_cfg = getattr(self.hf_config, "text_config", None)
        return text_cfg if text_cfg is not None else self.hf_config

    def _num_experts(self) -> int:
        text_cfg = self._hf_text_config()
        num_experts = self._cfg_get(text_cfg, "num_experts", None)
        if num_experts is None:
            num_experts = self._cfg_get(self.hf_config, "num_experts", None)
        if num_experts is None or not isinstance(num_experts, int):
            raise ValueError(
                "Qwen3.5 MoE conversion requires hf_config(.text_config).num_experts"
            )
        return num_experts

    def _attention_tp_size(self) -> int:
        tp_size = int(getattr(self.rank_info, "attn_tp_size", 1) or 1)
        if tp_size <= 0:
            raise ValueError(f"Invalid attn_tp_size={tp_size}")
        return tp_size

    def _attention_head_dim(self) -> int:
        kv_channels = self._cfg_get(self.tf_config, "kv_channels", None)
        if kv_channels is not None:
            if not isinstance(kv_channels, int):
                raise ValueError(
                    "Transformer config kv_channels must be int when provided, "
                    f"got {kv_channels} (type={type(kv_channels).__name__})"
                )
            return kv_channels
        hidden_size = self._cfg_int(self.tf_config, "hidden_size")
        num_heads = self._cfg_int(self.tf_config, "num_attention_heads")
        if hidden_size % num_heads != 0:
            raise ValueError(
                "Invalid hidden_size/num_attention_heads for attention head dim: "
                f"hidden_size={hidden_size}, num_attention_heads={num_heads}"
            )
        return hidden_size // num_heads

    def _attention_output_gate_enabled(self) -> bool:
        enabled = self._cfg_get(self.tf_config, "attention_output_gate", None)
        if not isinstance(enabled, bool):
            raise ValueError(
                "Transformer config must provide boolean attention_output_gate for "
                f"Qwen3.5 attention conversion, got {enabled} "
                f"(type={type(enabled).__name__})"
            )
        return enabled

    def _attention_qkv_bias_enabled(self) -> bool:
        enabled = self._cfg_get(self.tf_config, "add_qkv_bias", False)
        if not isinstance(enabled, bool):
            raise ValueError(
                f"Transformer config add_qkv_bias must be bool, got {enabled} "
                f"(type={type(enabled).__name__})"
            )
        return enabled

    def _attention_tp_kv_layout_mode(self, name: str) -> tuple[int, int, int]:
        """Resolve local TP↔KV-group layout for linear_qkv conversion.

        Returns:
            (local_q_heads, local_kv_heads, head_dim)

        AWEX v1 sharding can only describe contiguous shard slices and cannot
        transform packed partial-query-group layouts into canonical q/k/v views.
        Therefore TP > KV-groups is explicitly rejected here.
        """
        num_heads = self._cfg_int(self.tf_config, "num_attention_heads")
        num_query_groups = self._cfg_int(self.tf_config, "num_query_groups")
        tp_size = self._attention_tp_size()
        head_dim = self._attention_head_dim()

        if num_query_groups <= 0:
            raise ValueError(
                f"num_query_groups must be positive, got {num_query_groups}"
            )
        if num_heads % tp_size != 0:
            raise ValueError(
                "num_attention_heads must be divisible by attn_tp_size: "
                f"num_attention_heads={num_heads}, attn_tp_size={tp_size}"
            )

        local_q_heads = num_heads // tp_size
        if num_query_groups % tp_size == 0:
            local_kv_heads = num_query_groups // tp_size
            if local_kv_heads <= 0:
                raise ValueError(
                    "Invalid local_kv_heads resolved from num_query_groups % attn_tp_size == 0: "
                    f"num_query_groups={num_query_groups}, attn_tp_size={tp_size}"
                )
            return local_q_heads, local_kv_heads, head_dim

        raise ValueError(
            "Unsupported TP/KV-group relationship for Qwen3.5 AWEX v1 attention conversion: "
            f"num_query_groups={num_query_groups}, attn_tp_size={tp_size}. "
            "This converter requires num_query_groups % TP == 0. "
            "When TP > num_query_groups, local linear_qkv shards cut through query groups and "
            "AWEX v1 cannot express the required split-group permutation/transform without extra collectives."
        )

    @staticmethod
    def _split_gate_up(
        weight: torch.Tensor, name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weight.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor for gate/up split in {name}, got shape={tuple(weight.shape)}"
            )
        rows = weight.shape[0]
        if rows % 2 != 0:
            raise ValueError(
                f"Expected even dim0 for gate/up split in {name}, got dim0={rows}"
            )
        half = rows // 2
        return weight.narrow(0, 0, half), weight.narrow(0, half, half)

    def _split_attention_qkv(
        self, name: str, parameter: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_q_heads, local_kv_heads, head_dim = self._attention_tp_kv_layout_mode(
            name
        )
        output_gate = self._attention_output_gate_enabled()
        if local_q_heads % local_kv_heads != 0:
            raise ValueError(
                "Invalid local grouped-query layout: local_q_heads must be divisible "
                f"by local_kv_heads, got local_q_heads={local_q_heads}, "
                f"local_kv_heads={local_kv_heads}"
            )
        heads_per_group = local_q_heads // local_kv_heads
        total_heads_per_group = (
            (2 * heads_per_group + 2) if output_gate else (heads_per_group + 2)
        )
        qkv_total_dim = total_heads_per_group * local_kv_heads
        expected_dim0 = qkv_total_dim * head_dim

        if parameter.shape[0] != expected_dim0:
            raise ValueError(
                "Malformed local linear_qkv tensor for configured attention_output_gate "
                f"layout: name={name}, shape={tuple(parameter.shape)}, "
                f"expected_dim0={expected_dim0}, head_dim={head_dim}, "
                f"local_q_heads={local_q_heads}, local_kv_heads={local_kv_heads}, "
                f"attention_output_gate={output_gate}"
            )

        if parameter.ndim == 1:
            packed = parameter.view(qkv_total_dim, head_dim)
            feature_dim = 1
        elif parameter.ndim == 2:
            feature_dim = parameter.shape[1]
            packed = parameter.view(qkv_total_dim, head_dim, feature_dim)
        else:
            raise ValueError(
                f"Expected attention linear_qkv tensor rank 1 or 2, got shape={tuple(parameter.shape)}"
            )

        q_slice = torch.cat(
            [
                torch.arange(
                    total_heads_per_group * i,
                    total_heads_per_group * i + heads_per_group,
                )
                for i in range(local_kv_heads)
            ]
        )
        k_slice = torch.arange(
            total_heads_per_group - 2, qkv_total_dim, total_heads_per_group
        )
        v_slice = torch.arange(
            total_heads_per_group - 1, qkv_total_dim, total_heads_per_group
        )

        if output_gate:
            gate_slice = torch.cat(
                [
                    torch.arange(
                        total_heads_per_group * i + heads_per_group,
                        total_heads_per_group * i + 2 * heads_per_group,
                    )
                    for i in range(local_kv_heads)
                ]
            )
            q = torch.cat([packed[q_slice], packed[gate_slice]], dim=1)
        else:
            q = packed[q_slice]
        k = packed[k_slice]
        v = packed[v_slice]

        expected_q_rows = (
            (2 * local_q_heads * head_dim)
            if output_gate
            else (local_q_heads * head_dim)
        )
        expected_kv_rows = local_kv_heads * head_dim
        if parameter.ndim == 1:
            q = q.reshape(-1)
            k = k.reshape(-1)
            v = v.reshape(-1)
            if (
                q.shape[0] != expected_q_rows
                or k.shape[0] != expected_kv_rows
                or v.shape[0] != expected_kv_rows
            ):
                raise ValueError(
                    "Unexpected local q/k/v bias rows after split: "
                    f"q={q.shape}, k={k.shape}, v={v.shape}, "
                    f"expected_q_rows={expected_q_rows}, expected_kv_rows={expected_kv_rows}, "
                    f"attention_output_gate={output_gate}"
                )
        else:
            q = q.reshape(-1, feature_dim)
            k = k.reshape(-1, feature_dim)
            v = v.reshape(-1, feature_dim)
            if (
                q.shape[0] != expected_q_rows
                or k.shape[0] != expected_kv_rows
                or v.shape[0] != expected_kv_rows
            ):
                raise ValueError(
                    "Unexpected local q/k/v weight rows after split: "
                    f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}, "
                    f"expected_q_rows={expected_q_rows}, expected_kv_rows={expected_kv_rows}, "
                    f"attention_output_gate={output_gate}"
                )

        return q, k, v

    def _hf_vision_config(self) -> object:
        vision_cfg = getattr(self.hf_config, "vision_config", None)
        if vision_cfg is None:
            raise ValueError("Qwen3.5 MoE conversion requires hf_config.vision_config")
        return vision_cfg

    def _vision_tp_size(self) -> int:
        tp_size = int(getattr(self.rank_info, "tp_size", 1) or 1)
        if tp_size <= 0:
            raise ValueError(f"Invalid tp_size={tp_size} for vision conversion")
        return tp_size

    def _vision_head_layout(self) -> tuple[int, int, int]:
        vision_cfg = self._hf_vision_config()
        num_heads = self._cfg_get(vision_cfg, "num_heads", None)
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                "Qwen3.5 vision conversion requires integer vision_config.num_heads"
            )
        head_dim = self._cfg_get(vision_cfg, "head_dim", None)
        if not isinstance(head_dim, int) or head_dim <= 0:
            hidden_size = self._cfg_get(vision_cfg, "hidden_size", None)
            if not isinstance(hidden_size, int) or hidden_size <= 0:
                raise ValueError(
                    "Qwen3.5 vision conversion requires vision_config.hidden_size"
                )
            if hidden_size % num_heads != 0:
                raise ValueError(
                    "vision_config.hidden_size must be divisible by num_heads: "
                    f"hidden_size={hidden_size}, num_heads={num_heads}"
                )
            head_dim = hidden_size // num_heads

        tp_size = self._vision_tp_size()
        if num_heads % tp_size != 0:
            raise ValueError(
                "Qwen3.5 vision conversion requires num_heads % tp_size == 0: "
                f"num_heads={num_heads}, tp_size={tp_size}"
            )
        local_heads = num_heads // tp_size
        return local_heads, head_dim, tp_size

    def _split_vision_attention_qkv(
        self, name: str, parameter: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_heads, head_dim, _ = self._vision_head_layout()
        expected_rows = 3 * local_heads * head_dim
        if parameter.shape[0] != expected_rows:
            raise ValueError(
                "Malformed local MCore vision linear_qkv tensor: "
                f"name={name}, shape={tuple(parameter.shape)}, expected_dim0={expected_rows}, "
                f"local_heads={local_heads}, head_dim={head_dim}"
            )
        if parameter.ndim == 1:
            packed = parameter.view(local_heads, 3, head_dim)
            q = packed[:, 0, :].reshape(-1)
            k = packed[:, 1, :].reshape(-1)
            v = packed[:, 2, :].reshape(-1)
            return q, k, v
        if parameter.ndim == 2:
            feature_dim = parameter.shape[1]
            packed = parameter.view(local_heads, 3, head_dim, feature_dim)
            q = packed[:, 0, :, :].reshape(-1, feature_dim)
            k = packed[:, 1, :, :].reshape(-1, feature_dim)
            v = packed[:, 2, :, :].reshape(-1, feature_dim)
            return q, k, v
        raise ValueError(
            f"Expected rank-1/2 local MCore vision linear_qkv tensor, got shape={tuple(parameter.shape)}"
        )

    def _convert_language_param(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        direct_name_mapping = {
            "embedding.word_embeddings.weight": f"{self._HF_LANGUAGE_PREFIX}embed_tokens.weight",
            "decoder.final_layernorm.weight": f"{self._HF_LANGUAGE_PREFIX}norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        hf_name = direct_name_mapping.get(name)
        if hf_name is not None:
            return [(hf_name, parameter)]
        return self._convert_language_layer_param(name, parameter)

    def _convert_language_layer_param(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        if not name.startswith("decoder.layers."):
            raise NotImplementedError(
                f"Unsupported language parameter (expected decoder.layers.*): {name}"
            )
        layer_number, remaining_name = name.removeprefix("decoder.layers.").split(
            ".", 1
        )
        layer_prefix = f"{self._HF_LANGUAGE_PREFIX}layers.{layer_number}."

        if remaining_name.startswith("self_attention."):
            converted = self._convert_language_attention_param(
                remaining_name, parameter
            )
            return [
                (f"{layer_prefix}{hf_suffix}", tensor)
                for hf_suffix, tensor in converted
            ]

        if (
            remaining_name.startswith("mlp.")
            or remaining_name == "pre_mlp_layernorm.weight"
        ):
            converted = self._convert_language_mlp_param(remaining_name, parameter)
            return [
                (f"{layer_prefix}{hf_suffix}", tensor)
                for hf_suffix, tensor in converted
            ]

        raise NotImplementedError(f"Unsupported language layer parameter: {name}")

    def _convert_language_attention_param(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        if name == "self_attention.linear_qkv.layer_norm_weight":
            return [("input_layernorm.weight", parameter)]
        if name == "self_attention.q_layernorm.weight":
            return [("self_attn.q_norm.weight", parameter)]
        if name == "self_attention.k_layernorm.weight":
            return [("self_attn.k_norm.weight", parameter)]
        if name == "self_attention.linear_proj.weight":
            return [("self_attn.o_proj.weight", parameter)]
        if name == "self_attention.linear_proj.bias":
            return [("self_attn.o_proj.bias", parameter)]
        if name in {
            "self_attention.linear_qkv.weight",
            "self_attention.linear_qkv.bias",
        }:
            if (
                name == "self_attention.linear_qkv.bias"
                and not self._attention_qkv_bias_enabled()
            ):
                raise ValueError(
                    "Received self_attention.linear_qkv.bias while add_qkv_bias=False"
                )

            q, k, v = self._split_attention_qkv(name, parameter)
            return [
                (
                    "self_attn.q_proj.weight"
                    if parameter.ndim == 2
                    else "self_attn.q_proj.bias",
                    q,
                ),
                (
                    "self_attn.k_proj.weight"
                    if parameter.ndim == 2
                    else "self_attn.k_proj.bias",
                    k,
                ),
                (
                    "self_attn.v_proj.weight"
                    if parameter.ndim == 2
                    else "self_attn.v_proj.bias",
                    v,
                ),
            ]

        # Gated DeltaNet path.
        if name == "self_attention.in_proj.layer_norm_weight":
            return [("input_layernorm.weight", parameter)]
        if name == "self_attention.out_proj.weight":
            return [("linear_attn.out_proj.weight", parameter)]
        if name == "self_attention.conv1d.weight":
            return [("linear_attn.conv1d.weight", parameter)]
        if name == "self_attention.A_log":
            return [("linear_attn.A_log", parameter)]
        if name == "self_attention.dt_bias":
            return [("linear_attn.dt_bias", parameter)]
        if name == "self_attention.out_norm.weight":
            return [("linear_attn.norm.weight", parameter + 1)]
        if name == "self_attention.in_proj.weight":
            return self._convert_gdn_in_proj(name, parameter)

        raise NotImplementedError(f"Unsupported language attention parameter: {name}")

    def _convert_gdn_in_proj(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        if parameter.ndim != 2:
            raise ValueError(
                f"Expected 2D GDN in_proj.weight for {name}, got shape={tuple(parameter.shape)}"
            )

        qk_head_dim = self._cfg_int(self.tf_config, "linear_key_head_dim")
        v_head_dim = self._cfg_int(self.tf_config, "linear_value_head_dim")
        num_qk_heads = self._cfg_int(self.tf_config, "linear_num_key_heads")
        num_v_heads = self._cfg_int(self.tf_config, "linear_num_value_heads")
        tp_size = self._attention_tp_size()

        if num_qk_heads % tp_size != 0 or num_v_heads % tp_size != 0:
            raise ValueError(
                "GDN local split requires head counts divisible by attn_tp_size: "
                f"linear_num_key_heads={num_qk_heads}, linear_num_value_heads={num_v_heads}, attn_tp_size={tp_size}"
            )

        num_qk_heads_local = num_qk_heads // tp_size
        num_v_heads_local = num_v_heads // tp_size
        qk_dim_local = qk_head_dim * num_qk_heads_local
        v_dim_local = v_head_dim * num_v_heads_local
        expected_rows = 2 * qk_dim_local + 2 * v_dim_local + 2 * num_v_heads_local

        if parameter.shape[0] != expected_rows:
            raise ValueError(
                "Malformed GDN in_proj local shard shape: "
                f"got={tuple(parameter.shape)}, expected_dim0={expected_rows}, "
                f"qk_dim_local={qk_dim_local}, v_dim_local={v_dim_local}, "
                f"num_v_heads_local={num_v_heads_local}"
            )

        cursor = 0
        q = parameter.narrow(0, cursor, qk_dim_local)
        cursor += qk_dim_local
        k = parameter.narrow(0, cursor, qk_dim_local)
        cursor += qk_dim_local
        v = parameter.narrow(0, cursor, v_dim_local)
        cursor += v_dim_local
        z = parameter.narrow(0, cursor, v_dim_local)
        cursor += v_dim_local
        b = parameter.narrow(0, cursor, num_v_heads_local)
        cursor += num_v_heads_local
        a = parameter.narrow(0, cursor, num_v_heads_local)

        in_proj_qkv = torch.cat([q, k, v], dim=0)
        return [
            ("linear_attn.in_proj_qkv.weight", in_proj_qkv),
            ("linear_attn.in_proj_z.weight", z),
            ("linear_attn.in_proj_b.weight", b),
            ("linear_attn.in_proj_a.weight", a),
        ]

    def _global_expert_id(self, local_expert_id: int) -> int:
        num_experts = self._num_experts()
        ep_size = int(getattr(self.rank_info, "ep_size", 1) or 1)
        ep_rank = int(getattr(self.rank_info, "ep_rank", 0) or 0)
        if ep_size <= 0:
            raise ValueError(f"Invalid ep_size={ep_size}")
        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts must be divisible by ep_size: num_experts={num_experts}, ep_size={ep_size}"
            )
        experts_per_partition = num_experts // ep_size
        return local_expert_id + ep_rank * experts_per_partition

    def _convert_language_mlp_param(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        if name == "pre_mlp_layernorm.weight":
            return [("post_attention_layernorm.weight", parameter)]
        if name == "mlp.router.weight":
            return [("mlp.gate.weight", parameter)]
        if name == "mlp.shared_experts.gate_weight":
            return [("mlp.shared_expert_gate.weight", parameter)]
        if name == "mlp.shared_experts.linear_fc1.weight":
            gate, up = self._split_gate_up(parameter, name)
            return [
                ("mlp.shared_expert.gate_proj.weight", gate),
                ("mlp.shared_expert.up_proj.weight", up),
            ]
        if name == "mlp.shared_experts.linear_fc2.weight":
            return [("mlp.shared_expert.down_proj.weight", parameter)]

        local_match = self._LOCAL_EXPERT_PATTERN.match(name)
        if local_match is not None:
            local_expert_id = int(local_match.group(1))
            fc_index = local_match.group(2)
            global_id = self._global_expert_id(local_expert_id)
            if fc_index == "1":
                gate, up = self._split_gate_up(parameter, name)
                return [
                    (f"mlp.experts.{global_id}.gate_proj.weight", gate),
                    (f"mlp.experts.{global_id}.up_proj.weight", up),
                ]
            return [(f"mlp.experts.{global_id}.down_proj.weight", parameter)]

        packed_match = self._PACKED_EXPERT_PATTERN.match(name)
        if packed_match is not None:
            fc_index = packed_match.group(1)
            local_expert_id = int(packed_match.group(2))
            global_id = self._global_expert_id(local_expert_id)
            if fc_index == "1":
                gate, up = self._split_gate_up(parameter, name)
                return [
                    (f"mlp.experts.{global_id}.gate_proj.weight", gate),
                    (f"mlp.experts.{global_id}.up_proj.weight", up),
                ]
            return [(f"mlp.experts.{global_id}.down_proj.weight", parameter)]

        raise NotImplementedError(f"Unsupported language MLP parameter: {name}")

    def _convert_vision_param(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        if name.startswith("patch_embed.proj."):
            return [(f"{self._HF_VISION_PREFIX}{name}", parameter)]
        if name == "pos_embed.weight":
            return [(f"{self._HF_VISION_PREFIX}{name}", parameter)]
        if name.startswith("merger.patch_norm."):
            return [
                (
                    f"{self._HF_VISION_PREFIX}{name.replace('merger.patch_norm.', 'merger.norm.', 1)}",
                    parameter,
                )
            ]
        if name in {
            "merger.linear_fc1.weight",
            "merger.linear_fc1.bias",
            "merger.linear_fc2.weight",
            "merger.linear_fc2.bias",
        }:
            return [(f"{self._HF_VISION_PREFIX}{name}", parameter)]
        return self._convert_vision_layer_param(name, parameter)

    def _convert_vision_layer_param(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        if not name.startswith("decoder.layers."):
            raise NotImplementedError(
                f"Unsupported vision parameter (expected decoder.layers.*): {name}"
            )

        layer_number, remaining_name = name.removeprefix("decoder.layers.").split(
            ".", 1
        )
        layer_prefix = f"{self._HF_VISION_PREFIX}blocks.{layer_number}."

        if remaining_name in {
            "self_attention.linear_qkv.weight",
            "self_attention.linear_qkv.bias",
        }:
            q, k, v = self._split_vision_attention_qkv(name, parameter)
            leaf = "weight" if remaining_name.endswith(".weight") else "bias"
            return [
                (f"{layer_prefix}attn.q_proj.{leaf}", q),
                (f"{layer_prefix}attn.k_proj.{leaf}", k),
                (f"{layer_prefix}attn.v_proj.{leaf}", v),
            ]

        direct_mapping = {
            "self_attention.linear_proj.weight": "attn.proj.weight",
            "self_attention.linear_proj.bias": "attn.proj.bias",
            "mlp.linear_fc1.weight": "mlp.linear_fc1.weight",
            "mlp.linear_fc1.bias": "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight": "mlp.linear_fc2.weight",
            "mlp.linear_fc2.bias": "mlp.linear_fc2.bias",
            "self_attention.linear_qkv.layer_norm_weight": "norm1.weight",
            "self_attention.linear_qkv.layer_norm_bias": "norm1.bias",
            "mlp.linear_fc1.layer_norm_weight": "norm2.weight",
            "mlp.linear_fc1.layer_norm_bias": "norm2.bias",
        }

        hf_suffix = direct_mapping.get(remaining_name)
        if hf_suffix is None:
            raise NotImplementedError(f"Unsupported vision layer parameter: {name}")
        return [(f"{layer_prefix}{hf_suffix}", parameter)]


class SGlangToHFWeightConverterQwen3_5Moe(SGlangToHFWeightConverter):
    """Convert SGLang Qwen3.5-MoE VLM runtime names to canonical HF names."""

    _HF_LANGUAGE_PREFIX = "model.language_model."
    _HF_VISION_PREFIX = "model.visual."

    _EXPERT_W13_BULK_PATTERN = re.compile(r"^mlp\.experts\.w13_weight$")
    _EXPERT_W2_BULK_PATTERN = re.compile(r"^mlp\.experts\.w2_weight$")
    _EXPERT_PER_EXPERT_PATTERN = re.compile(
        r"^mlp\.experts\.(\d+)\.(w13_weight|w2_weight|gate_proj\.weight|up_proj\.weight|down_proj\.weight|gate_up_proj\.weight)$"
    )

    def __init__(
        self,
        model_config: object,
        infer_engine_config: object,
        rank_info: object,
    ) -> None:
        text_config = getattr(model_config, "text_config", None)
        if text_config is None:
            text_config = model_config

        for required_attr in ("num_attention_heads", "num_key_value_heads"):
            value = getattr(text_config, required_attr, None)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "Qwen3.5 SGLang converter requires text config with integer "
                    f"`{required_attr}` before AWEX init; got {value} "
                    f"(type={type(value).__name__})"
                )

        self.full_model_config = model_config
        self._resolved_text_config = text_config
        super().__init__(cast(Any, text_config), infer_engine_config, rank_info)

    @staticmethod
    def _cfg_get(config: object, key: str, default: object = None) -> object:
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _cfg_int(self, config: object, key: str) -> int:
        value = self._cfg_get(config, key, None)
        if value is None:
            raise ValueError(f"Missing required config field: {key}")
        if not isinstance(value, int):
            raise ValueError(
                f"Config field must be int: {key}={value} (type={type(value).__name__})"
            )
        return value

    def _hf_text_config(self) -> object:
        return self.model_config

    @staticmethod
    def _is_mtp_param(name: str) -> bool:
        return "mtp" in name.split(".")

    def _validate_infer_parallel_constraints(self) -> None:
        def _require_int(value: object, field: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{field} must be int when provided, got {value} "
                    f"(type={type(value).__name__})"
                )
            return value

        rank_pp_size = getattr(self.rank_info, "pp_size", None)
        if (
            rank_pp_size is not None
            and _require_int(rank_pp_size, "rank_info.pp_size") != 1
        ):
            raise ValueError(
                "Qwen3.5 SGLang inference converter requires PP=1, got "
                f"rank_info.pp_size={rank_pp_size}"
            )
        cfg_pp_size = self._cfg_get(self.infer_engine_config, "pp_size", None)
        if (
            cfg_pp_size is not None
            and _require_int(cfg_pp_size, "infer_engine_config.pp_size") != 1
        ):
            raise ValueError(
                "Qwen3.5 SGLang inference converter requires PP=1, got "
                f"infer_engine_config.pp_size={cfg_pp_size}"
            )

        rank_ep_size = getattr(self.rank_info, "ep_size", None)
        if (
            rank_ep_size is not None
            and _require_int(rank_ep_size, "rank_info.ep_size") != 1
        ):
            raise ValueError(
                "Qwen3.5 SGLang inference converter requires EP=1, got "
                f"rank_info.ep_size={rank_ep_size}"
            )
        cfg_ep_size = self._cfg_get(self.infer_engine_config, "ep_size", None)
        if (
            cfg_ep_size is not None
            and _require_int(cfg_ep_size, "infer_engine_config.ep_size") != 1
        ):
            raise ValueError(
                "Qwen3.5 SGLang inference converter requires EP=1, got "
                f"infer_engine_config.ep_size={cfg_ep_size}"
            )

        rank_ep_rank = getattr(self.rank_info, "ep_rank", None)
        if (
            rank_ep_rank is not None
            and _require_int(rank_ep_rank, "rank_info.ep_rank") != 0
        ):
            raise ValueError(
                "Qwen3.5 SGLang inference converter requires EP rank 0 under EP=1, got "
                f"rank_info.ep_rank={rank_ep_rank}"
            )

    def _attention_tp_size(self) -> int:
        tp_size = int(getattr(self.rank_info, "attn_tp_size", 0) or self.tp_size or 1)
        if tp_size <= 0:
            raise ValueError(f"Invalid attention TP size: {tp_size}")
        return tp_size

    def _attention_head_dim(self) -> int:
        cfg = self._hf_text_config()
        head_dim = self._cfg_get(cfg, "head_dim", None)
        if isinstance(head_dim, int) and head_dim > 0:
            return head_dim
        hidden_size = self._cfg_int(cfg, "hidden_size")
        num_heads = self._cfg_int(cfg, "num_attention_heads")
        if hidden_size % num_heads != 0:
            raise ValueError(
                "Invalid hidden_size/num_attention_heads for attention head dim: "
                f"hidden_size={hidden_size}, num_attention_heads={num_heads}"
            )
        return hidden_size // num_heads

    def _attention_qkv_layout(self) -> tuple[int, int, int, int, bool]:
        cfg = self._hf_text_config()
        num_heads = self._cfg_int(cfg, "num_attention_heads")
        num_kv_heads = self._cfg_get(cfg, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = self._cfg_get(cfg, "num_query_groups", None)
        if not isinstance(num_kv_heads, int):
            raise ValueError(
                "Qwen3.5 conversion requires num_key_value_heads or num_query_groups"
            )
        tp_size = self._attention_tp_size()
        if num_heads % tp_size != 0:
            raise ValueError(
                "num_attention_heads must be divisible by TP for local split: "
                f"num_attention_heads={num_heads}, tp={tp_size}"
            )

        # Official SGLang field is `attn_output_gate` (default True).
        # Keep compatibility for older configs that used `attention_output_gate`.
        output_gate = self._cfg_get(cfg, "attn_output_gate", None)
        if output_gate is None:
            output_gate = self._cfg_get(cfg, "attention_output_gate", None)
        if output_gate is None:
            output_gate = True
        if not isinstance(output_gate, bool):
            raise ValueError(
                "Qwen3.5 conversion requires boolean attn_output_gate (or compatibility field attention_output_gate) in HF text config"
            )

        head_dim = self._attention_head_dim()
        local_q_heads = num_heads // tp_size
        local_q_rows = local_q_heads * head_dim * (2 if output_gate else 1)

        if tp_size < num_kv_heads:
            if num_kv_heads % tp_size != 0:
                raise ValueError(
                    "num_key_value_heads must be divisible by TP when TP < KV heads: "
                    f"num_key_value_heads={num_kv_heads}, tp={tp_size}"
                )
            local_kv_heads = num_kv_heads // tp_size
        elif tp_size == num_kv_heads:
            local_kv_heads = 1
        else:
            raise ValueError(
                "Unsupported SGLang attention TP topology for AWEX v1: "
                f"attn_tp_size={tp_size} > num_key_value_heads={num_kv_heads}. "
                "This requires KV-head replication that AWEX v1 sharding metadata cannot represent."
            )

        local_kv_rows = local_kv_heads * head_dim
        return (
            local_q_heads,
            local_kv_heads,
            local_q_rows,
            local_kv_rows,
            output_gate,
        )

    @staticmethod
    def _split_gate_up_tensor(
        weight: torch.Tensor, name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weight.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor for gate/up split in {name}, got shape={tuple(weight.shape)}"
            )
        rows = weight.shape[0]
        if rows % 2 != 0:
            raise ValueError(
                f"Expected even dim0 for gate/up split in {name}, got dim0={rows}"
            )
        half = rows // 2
        return weight.narrow(0, 0, half), weight.narrow(0, half, half)

    def _split_attention_qkv(
        self, name: str, parameter: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            local_q_heads,
            local_kv_heads,
            local_q_rows,
            local_kv_rows,
            output_gate,
        ) = self._attention_qkv_layout()

        expected_rows = local_q_rows + 2 * local_kv_rows
        if parameter.shape[0] != expected_rows:
            raise ValueError(
                "Malformed local self_attn.qkv_proj tensor for SGLang consecutive [Q;K;V] "
                "layout: "
                f"name={name}, shape={tuple(parameter.shape)}, "
                f"expected_dim0={expected_rows}, "
                f"local_q_rows={local_q_rows}, local_kv_rows={local_kv_rows}, "
                f"local_q_heads={local_q_heads}, local_kv_heads={local_kv_heads}, "
                f"attention_output_gate={output_gate}"
            )

        if parameter.ndim not in {1, 2}:
            raise ValueError(
                f"Expected self_attn.qkv_proj rank 1 or 2, got shape={tuple(parameter.shape)}"
            )

        # Official SGLang Qwen3.5 runtime layout is consecutive [Q_block; K; V].
        q = parameter.narrow(0, 0, local_q_rows)
        k = parameter.narrow(0, local_q_rows, local_kv_rows)
        v = parameter.narrow(0, local_q_rows + local_kv_rows, local_kv_rows)

        if (
            q.shape[0] != local_q_rows
            or k.shape[0] != local_kv_rows
            or v.shape[0] != local_kv_rows
        ):
            raise ValueError(
                "Unexpected q/k/v rows after unfusing self_attn.qkv_proj: "
                f"q_shape={tuple(q.shape)}, k_shape={tuple(k.shape)}, "
                f"v_shape={tuple(v.shape)}, expected_q_rows={local_q_rows}, "
                f"expected_kv_rows={local_kv_rows}, "
                f"attention_output_gate={output_gate}"
            )
        return q, k, v

    def _split_gdn_in_proj_qkvz(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        if parameter.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor for {name}, got shape={tuple(parameter.shape)}"
            )

        cfg = self._hf_text_config()
        qk_head_dim = self._cfg_int(cfg, "linear_key_head_dim")
        v_head_dim = self._cfg_int(cfg, "linear_value_head_dim")
        num_qk_heads = self._cfg_int(cfg, "linear_num_key_heads")
        num_v_heads = self._cfg_int(cfg, "linear_num_value_heads")
        tp_size = self._attention_tp_size()

        if num_qk_heads % tp_size != 0 or num_v_heads % tp_size != 0:
            raise ValueError(
                "GDN in_proj_qkvz local split requires head counts divisible by TP: "
                f"linear_num_key_heads={num_qk_heads}, "
                f"linear_num_value_heads={num_v_heads}, tp={tp_size}"
            )

        num_qk_heads_local = num_qk_heads // tp_size
        num_v_heads_local = num_v_heads // tp_size
        qk_dim_local = qk_head_dim * num_qk_heads_local
        v_dim_local = v_head_dim * num_v_heads_local
        expected_rows = 2 * qk_dim_local + 2 * v_dim_local
        if parameter.shape[0] != expected_rows:
            raise ValueError(
                "Malformed GDN in_proj_qkvz local shape: "
                f"name={name}, shape={tuple(parameter.shape)}, expected_dim0={expected_rows}, "
                f"qk_dim_local={qk_dim_local}, v_dim_local={v_dim_local}"
            )

        cursor = 0
        q = parameter.narrow(0, cursor, qk_dim_local)
        cursor += qk_dim_local
        k = parameter.narrow(0, cursor, qk_dim_local)
        cursor += qk_dim_local
        v = parameter.narrow(0, cursor, v_dim_local)
        cursor += v_dim_local
        z = parameter.narrow(0, cursor, v_dim_local)

        return [
            ("linear_attn.in_proj_qkv.weight", torch.cat([q, k, v], dim=0)),
            ("linear_attn.in_proj_z.weight", z),
        ]

    def _split_gdn_in_proj_ba(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        if parameter.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor for {name}, got shape={tuple(parameter.shape)}"
            )

        cfg = self._hf_text_config()
        num_v_heads = self._cfg_int(cfg, "linear_num_value_heads")
        tp_size = self._attention_tp_size()
        if num_v_heads % tp_size != 0:
            raise ValueError(
                "GDN in_proj_ba local split requires linear_num_value_heads divisible by TP: "
                f"linear_num_value_heads={num_v_heads}, tp={tp_size}"
            )
        num_v_heads_local = num_v_heads // tp_size
        expected_rows = 2 * num_v_heads_local
        if parameter.shape[0] != expected_rows:
            raise ValueError(
                "Malformed GDN in_proj_ba local shape: "
                f"name={name}, shape={tuple(parameter.shape)}, expected_dim0={expected_rows}, "
                f"num_v_heads_local={num_v_heads_local}"
            )
        b = parameter.narrow(0, 0, num_v_heads_local)
        a = parameter.narrow(0, num_v_heads_local, num_v_heads_local)
        return [
            ("linear_attn.in_proj_b.weight", b),
            ("linear_attn.in_proj_a.weight", a),
        ]

    def _num_experts(self) -> int:
        cfg = self._hf_text_config()
        num_experts = self._cfg_get(cfg, "num_experts", None)
        if num_experts is None:
            num_experts = self._cfg_get(cfg, "n_routed_experts", None)
        if not isinstance(num_experts, int):
            raise ValueError(
                "Qwen3.5 MoE conversion requires num_experts (or n_routed_experts) in text config"
            )
        return num_experts

    def _global_expert_id(self, local_expert_id: int) -> int:
        self._validate_infer_parallel_constraints()
        return local_expert_id

    @torch.no_grad()
    def convert_param(
        self,
        name: str,
        parameter: torch.Tensor,
    ) -> ConvertedParams:
        name = name.replace("module.", "")
        self._validate_infer_parallel_constraints()
        if self._is_mtp_param(name):
            return []

        if name in {"lm_head.weight", "model.lm_head.weight"}:
            return [("lm_head.weight", parameter)]

        if name.startswith(self._HF_LANGUAGE_PREFIX):
            return self._convert_language_param(
                name.removeprefix(self._HF_LANGUAGE_PREFIX), parameter
            )
        if name.startswith(self._HF_VISION_PREFIX):
            return self._convert_vision_param(
                name.removeprefix(self._HF_VISION_PREFIX), parameter
            )

        if name.startswith("visual."):
            return self._convert_vision_param(name.removeprefix("visual."), parameter)

        if name.startswith("model.layers."):
            return self._convert_language_param(name.removeprefix("model."), parameter)

        if name.startswith("model."):
            return self._convert_language_param(name.removeprefix("model."), parameter)

        raise NotImplementedError(
            "Unsupported Qwen3.5 SGLang parameter. Accepted prefixes: "
            "runtime {lm_head., model., visual.}; "
            "checkpoint aliases {model.language_model., model.visual., model.layers., model.lm_head.}. "
            f"Got: {name}"
        )

    def _convert_language_param(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        direct_mapping = {
            "embed_tokens.weight": f"{self._HF_LANGUAGE_PREFIX}embed_tokens.weight",
            "norm.weight": f"{self._HF_LANGUAGE_PREFIX}norm.weight",
            "language_model.embed_tokens.weight": f"{self._HF_LANGUAGE_PREFIX}embed_tokens.weight",
            "language_model.norm.weight": f"{self._HF_LANGUAGE_PREFIX}norm.weight",
        }
        mapped = direct_mapping.get(name)
        if mapped is not None:
            return [(mapped, parameter)]

        if name.startswith("language_model.layers."):
            name = name.removeprefix("language_model.")

        if not name.startswith("layers."):
            raise NotImplementedError(
                f"Unsupported language parameter (expected layers.*): {name}"
            )

        layer_number, remaining = name.removeprefix("layers.").split(".", 1)
        layer_prefix = f"{self._HF_LANGUAGE_PREFIX}layers.{layer_number}."

        converted = self._convert_language_layer_param(remaining, parameter)
        return [(f"{layer_prefix}{suffix}", tensor) for suffix, tensor in converted]

    def _convert_language_layer_param(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        if name == "input_layernorm.weight":
            return [("input_layernorm.weight", parameter)]

        if name.startswith("self_attn.") or name.startswith("linear_attn."):
            return self._convert_language_attention_param(name, parameter)

        if (
            name == "post_attention_layernorm.weight"
            or name == "pre_mlp_layernorm.weight"
            or name.startswith("mlp.")
        ):
            return self._convert_language_mlp_param(name, parameter)

        raise NotImplementedError(f"Unsupported language layer parameter: {name}")

    def _convert_language_attention_param(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        direct = {
            "self_attn.q_norm.weight": "self_attn.q_norm.weight",
            "self_attn.k_norm.weight": "self_attn.k_norm.weight",
            "self_attn.o_proj.weight": "self_attn.o_proj.weight",
            "self_attn.o_proj.bias": "self_attn.o_proj.bias",
            "linear_attn.conv1d.weight": "linear_attn.conv1d.weight",
            "linear_attn.A_log": "linear_attn.A_log",
            "linear_attn.dt_bias": "linear_attn.dt_bias",
            "linear_attn.out_proj.weight": "linear_attn.out_proj.weight",
            "linear_attn.out_proj.bias": "linear_attn.out_proj.bias",
            "linear_attn.norm.weight": "linear_attn.norm.weight",
            "linear_attn.in_proj_qkv.weight": "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight": "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_b.weight": "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_a.weight": "linear_attn.in_proj_a.weight",
        }
        mapped = direct.get(name)
        if mapped is not None:
            return [(mapped, parameter)]

        if name in {
            "self_attn.qkv_proj.weight",
            "self_attn.qkv_proj.bias",
        }:
            q, k, v = self._split_attention_qkv(name, parameter)
            if parameter.ndim == 2:
                return [
                    ("self_attn.q_proj.weight", q),
                    ("self_attn.k_proj.weight", k),
                    ("self_attn.v_proj.weight", v),
                ]
            return [
                ("self_attn.q_proj.bias", q),
                ("self_attn.k_proj.bias", k),
                ("self_attn.v_proj.bias", v),
            ]

        if name in {"self_attn.in_proj_qkvz.weight", "linear_attn.in_proj_qkvz.weight"}:
            return self._split_gdn_in_proj_qkvz(name, parameter)

        if name in {"self_attn.in_proj_ba.weight", "linear_attn.in_proj_ba.weight"}:
            return self._split_gdn_in_proj_ba(name, parameter)

        raise NotImplementedError(f"Unsupported Qwen3.5 attention parameter: {name}")

    def _convert_bulk_routed_experts(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        if parameter.ndim != 3:
            raise ValueError(
                f"Expected 3D routed expert tensor for {name}, got shape={tuple(parameter.shape)}"
            )

        expected_experts = self._num_experts()
        if parameter.shape[0] != expected_experts:
            raise ValueError(
                "Under inference EP=1, routed bulk expert tensor must contain all experts: "
                f"name={name}, got_num_experts={parameter.shape[0]}, "
                f"expected_num_experts={expected_experts}"
            )

        converted: ConvertedParams = []
        if self._EXPERT_W13_BULK_PATTERN.match(name):
            for local_expert_id in range(parameter.shape[0]):
                global_expert_id = self._global_expert_id(local_expert_id)
                expert_parameter = parameter[local_expert_id]
                if self._use_transposed_moe_layout(name, expert_parameter):
                    expert_parameter = expert_parameter.transpose(0, 1).contiguous()
                gate, up = self._split_gate_up_tensor(
                    expert_parameter, f"{name}[{local_expert_id}]"
                )
                converted.append(
                    (f"mlp.experts.{global_expert_id}.gate_proj.weight", gate)
                )
                converted.append((f"mlp.experts.{global_expert_id}.up_proj.weight", up))
            return converted

        if self._EXPERT_W2_BULK_PATTERN.match(name):
            for local_expert_id in range(parameter.shape[0]):
                global_expert_id = self._global_expert_id(local_expert_id)
                expert_parameter = parameter[local_expert_id]
                if self._use_transposed_moe_layout(name, expert_parameter):
                    expert_parameter = expert_parameter.transpose(0, 1).contiguous()
                converted.append(
                    (
                        f"mlp.experts.{global_expert_id}.down_proj.weight",
                        expert_parameter,
                    )
                )
            return converted

        raise NotImplementedError(f"Unsupported bulk routed expert parameter: {name}")

    def _convert_language_mlp_param(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        if self._use_transposed_moe_layout(name, parameter):
            parameter = parameter.transpose(0, 1).contiguous()

        if name in {"post_attention_layernorm.weight", "pre_mlp_layernorm.weight"}:
            return [("post_attention_layernorm.weight", parameter)]

        if name in {"mlp.router.weight", "mlp.gate.weight"}:
            return [("mlp.gate.weight", parameter)]

        if name in {"mlp.shared_expert_gate.weight", "mlp.shared_experts.gate_weight"}:
            return [("mlp.shared_expert_gate.weight", parameter)]

        if name in {
            "mlp.shared_experts.linear_fc1.weight",
            "mlp.shared_experts.gate_up_proj.weight",
            "mlp.shared_experts.gate_up_weight",
            "mlp.shared_experts.w13_weight",
        }:
            gate, up = self._split_gate_up_tensor(parameter, name)
            return [
                ("mlp.shared_expert.gate_proj.weight", gate),
                ("mlp.shared_expert.up_proj.weight", up),
            ]

        if name in {
            "mlp.shared_experts.linear_fc2.weight",
            "mlp.shared_experts.down_proj.weight",
            "mlp.shared_experts.down_weight",
            "mlp.shared_experts.w2_weight",
        }:
            return [("mlp.shared_expert.down_proj.weight", parameter)]

        per_expert_match = self._EXPERT_PER_EXPERT_PATTERN.match(name)
        if per_expert_match is not None:
            expert_id = int(per_expert_match.group(1))
            param_name = per_expert_match.group(2)
            if param_name in {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}:
                return [(f"mlp.experts.{expert_id}.{param_name}", parameter)]
            if param_name in {"w13_weight", "gate_up_proj.weight"}:
                gate, up = self._split_gate_up_tensor(parameter, name)
                return [
                    (f"mlp.experts.{expert_id}.gate_proj.weight", gate),
                    (f"mlp.experts.{expert_id}.up_proj.weight", up),
                ]
            return [(f"mlp.experts.{expert_id}.down_proj.weight", parameter)]

        if self._EXPERT_W13_BULK_PATTERN.match(
            name
        ) or self._EXPERT_W2_BULK_PATTERN.match(name):
            return self._convert_bulk_routed_experts(name, parameter)

        raise NotImplementedError(f"Unsupported Qwen3.5 MLP parameter: {name}")

    def _vision_head_layout(self) -> tuple[int, int]:
        full_cfg = self.full_model_config
        vision_cfg = getattr(full_cfg, "vision_config", None)
        if vision_cfg is None:
            raise ValueError(
                "Qwen3.5 SGLang conversion requires full_model_config.vision_config"
            )

        num_heads = self._cfg_get(vision_cfg, "num_heads", None)
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                "Qwen3.5 vision conversion requires integer vision_config.num_heads"
            )

        head_dim = self._cfg_get(vision_cfg, "head_dim", None)
        if not isinstance(head_dim, int) or head_dim <= 0:
            hidden_size = self._cfg_get(vision_cfg, "hidden_size", None)
            if not isinstance(hidden_size, int) or hidden_size <= 0:
                raise ValueError(
                    "Qwen3.5 vision conversion requires vision_config.hidden_size"
                )
            if hidden_size % num_heads != 0:
                raise ValueError(
                    "vision_config.hidden_size must be divisible by num_heads: "
                    f"hidden_size={hidden_size}, num_heads={num_heads}"
                )
            head_dim = hidden_size // num_heads

        tp_size = int(getattr(self.rank_info, "tp_size", 0) or self.tp_size or 1)
        if tp_size <= 0:
            raise ValueError(f"Invalid TP size for vision conversion: {tp_size}")
        if num_heads % tp_size != 0:
            raise ValueError(
                "Qwen3.5 vision conversion requires num_heads % TP == 0: "
                f"num_heads={num_heads}, tp={tp_size}"
            )
        local_heads = num_heads // tp_size
        return local_heads, head_dim

    def _split_vision_qkv_consecutive(
        self, name: str, parameter: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if parameter.ndim not in {1, 2}:
            raise ValueError(
                f"Expected rank-1/2 tensor for vision attn.qkv in {name}, got shape={tuple(parameter.shape)}"
            )

        local_heads, head_dim = self._vision_head_layout()
        local_rows = local_heads * head_dim
        expected_rows = 3 * local_rows
        if parameter.shape[0] != expected_rows:
            raise ValueError(
                "Malformed SGLang vision attn.qkv_proj tensor for local consecutive [Q;K;V] layout: "
                f"name={name}, shape={tuple(parameter.shape)}, expected_dim0={expected_rows}, "
                f"local_heads={local_heads}, head_dim={head_dim}"
            )

        q = parameter.narrow(0, 0, local_rows)
        k = parameter.narrow(0, local_rows, local_rows)
        v = parameter.narrow(0, 2 * local_rows, local_rows)
        return q, k, v

    def _convert_vision_param(
        self, name: str, parameter: torch.Tensor
    ) -> ConvertedParams:
        direct_mapping = {
            "patch_embed.proj.weight": f"{self._HF_VISION_PREFIX}patch_embed.proj.weight",
            "patch_embed.proj.bias": f"{self._HF_VISION_PREFIX}patch_embed.proj.bias",
            "pos_embed.weight": f"{self._HF_VISION_PREFIX}pos_embed.weight",
            "merger.linear_fc1.weight": f"{self._HF_VISION_PREFIX}merger.linear_fc1.weight",
            "merger.linear_fc1.bias": f"{self._HF_VISION_PREFIX}merger.linear_fc1.bias",
            "merger.linear_fc2.weight": f"{self._HF_VISION_PREFIX}merger.linear_fc2.weight",
            "merger.linear_fc2.bias": f"{self._HF_VISION_PREFIX}merger.linear_fc2.bias",
        }
        mapped = direct_mapping.get(name)
        if mapped is not None:
            return [(mapped, parameter)]

        if name.startswith("merger.patch_norm."):
            return [
                (
                    f"{self._HF_VISION_PREFIX}{name.replace('merger.patch_norm.', 'merger.norm.', 1)}",
                    parameter,
                )
            ]

        if name.startswith("decoder.layers."):
            name = name.replace("decoder.layers.", "blocks.", 1)

        if not name.startswith("blocks."):
            raise NotImplementedError(
                f"Unsupported vision parameter (expected blocks.*): {name}"
            )

        layer_number, remaining = name.removeprefix("blocks.").split(".", 1)
        layer_prefix = f"{self._HF_VISION_PREFIX}blocks.{layer_number}."

        if remaining in {
            "attn.qkv_proj.weight",
            "attn.qkv_proj.bias",
            "attn.qkv.weight",
            "attn.qkv.bias",
        }:
            q, k, v = self._split_vision_qkv_consecutive(name, parameter)
            leaf = "weight" if remaining.endswith(".weight") else "bias"
            return [
                (f"{layer_prefix}attn.q_proj.{leaf}", q),
                (f"{layer_prefix}attn.k_proj.{leaf}", k),
                (f"{layer_prefix}attn.v_proj.{leaf}", v),
            ]

        direct_layer_mapping = {
            "attn.proj.weight": "attn.proj.weight",
            "attn.proj.bias": "attn.proj.bias",
            "mlp.linear_fc1.weight": "mlp.linear_fc1.weight",
            "mlp.linear_fc1.bias": "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight": "mlp.linear_fc2.weight",
            "mlp.linear_fc2.bias": "mlp.linear_fc2.bias",
            "norm1.weight": "norm1.weight",
            "norm1.bias": "norm1.bias",
            "norm2.weight": "norm2.weight",
            "norm2.bias": "norm2.bias",
        }
        mapped = direct_layer_mapping.get(remaining)
        if mapped is None:
            raise NotImplementedError(
                f"Unsupported Qwen3.5 vision block parameter: {name}"
            )
        return [(f"{layer_prefix}{mapped}", parameter)]


_QWEN35_ROUTED_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")

_QWEN35_REPLICATED_SUFFIXES = (
    "norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "linear_attn.norm.weight",
    "mlp.gate.weight",
    "mlp.shared_expert_gate.weight",
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "pos_embed.weight",
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "merger.norm.weight",
    "merger.norm.bias",
    "attn.proj.bias",
    "mlp.linear_fc2.bias",
    "merger.linear_fc2.bias",
)

_QWEN35_TP_DIM0_SUFFIXES = (
    "embed_tokens.weight",
    "lm_head.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.bias",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
    "attn.q_proj.bias",
    "attn.k_proj.bias",
    "attn.v_proj.bias",
    "mlp.linear_fc1.weight",
    "mlp.linear_fc1.bias",
    "merger.linear_fc1.weight",
    "merger.linear_fc1.bias",
)

_QWEN35_TP_DIM1_SUFFIXES = (
    "self_attn.o_proj.weight",
    "linear_attn.out_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "attn.proj.weight",
    "mlp.linear_fc2.weight",
    "merger.linear_fc2.weight",
)


class Qwen3_5MoeShardingStrategy(ShardingStrategy):
    """Qwen3.5-MoE explicit sharding strategy for AWEX v1 transfer metadata."""

    def _tp_or_replicated(self, dim: int):
        tp_size = int(getattr(self.rank_info, "tp_size", 1) or 1)
        if tp_size > 1:
            return ShardingType.TP_SHARDING, dim, tp_size
        return ShardingType.NO_SHARDING, dim, 1

    def _attn_tp_size(self) -> int:
        attn_tp_size = int(getattr(self.rank_info, "attn_tp_size", 0) or 0)
        if attn_tp_size <= 0:
            attn_tp_size = int(getattr(self.rank_info, "tp_size", 1) or 1)
        if attn_tp_size <= 0:
            raise ValueError(f"Invalid attn_tp_size={attn_tp_size}")
        return attn_tp_size

    def _language_attention_sharding(self, parameter_name: str, dim: int):
        enable_dp_attention = bool(getattr(self, "enable_dp_attention", False))
        model_tp_size = int(getattr(self.rank_info, "tp_size", 1) or 1)
        attn_tp_size = self._attn_tp_size()
        if enable_dp_attention:
            if attn_tp_size > 1:
                return ShardingType.DP_TP_SHARDING, dim, attn_tp_size
            return ShardingType.NO_SHARDING, dim, 1

        if attn_tp_size != model_tp_size:
            raise ValueError(
                "Qwen3.5 language attention sharding mismatch when enable_dp_attention=False: "
                f"attn_tp_size={attn_tp_size}, tp_size={model_tp_size}, parameter={parameter_name}"
            )
        return self._tp_or_replicated(dim)

    @staticmethod
    def _is_language_attention_parameter(parameter_name: str) -> bool:
        if ".language_model." not in parameter_name:
            return False
        return ".self_attn." in parameter_name or ".linear_attn." in parameter_name

    def _expert_sharding(self, dim: int):
        if self.engine_name == "mcore":
            ep_size = int(getattr(self.rank_info, "ep_size", 1) or 1)
            ep_tp_size = int(getattr(self.rank_info, "ep_tp_size", 1) or 1)
            if ep_size > 1 and ep_tp_size > 1:
                return ShardingType.EP_TP_SHARDING, dim, ep_tp_size
            if ep_size > 1:
                return ShardingType.EP_SHARDING, dim, ep_size
            return self._tp_or_replicated(dim)

        # Inference side must keep EP=1 (validated by converter). Under EP=1,
        # routed experts are plain TP shards like other dense projections.
        return self._tp_or_replicated(dim)

    def get_sharding_strategy(self, parameter_name, **kwargs):
        del kwargs

        if self.engine_name != "mcore":
            pp_size = int(getattr(self.rank_info, "pp_size", 1) or 1)
            if pp_size != 1:
                raise ValueError(
                    "Qwen3.5 AWEX v1 inference sharding requires PP=1, got "
                    f"rank_info.pp_size={pp_size}"
                )
            ep_size = int(getattr(self.rank_info, "ep_size", 1) or 1)
            if ep_size != 1:
                raise ValueError(
                    "Qwen3.5 AWEX v1 inference sharding requires EP=1, got "
                    f"rank_info.ep_size={ep_size}"
                )

        if any(
            parameter_name.endswith(suffix) for suffix in _QWEN35_REPLICATED_SUFFIXES
        ):
            return ShardingType.NO_SHARDING, 0, 1

        if parameter_name.endswith("lm_head.weight"):
            if bool(getattr(self, "enable_dp_lm_head", False)):
                attn_tp_size = self._attn_tp_size()
                if attn_tp_size > 1:
                    return ShardingType.DP_TP_SHARDING, 0, attn_tp_size
                return ShardingType.NO_SHARDING, 0, 1
            return self._tp_or_replicated(0)

        if _QWEN35_ROUTED_EXPERT_RE.search(parameter_name):
            if parameter_name.endswith("gate_proj.weight") or parameter_name.endswith(
                "up_proj.weight"
            ):
                return self._expert_sharding(0)
            if parameter_name.endswith("down_proj.weight"):
                return self._expert_sharding(1)

        if self._is_language_attention_parameter(parameter_name):
            if any(
                parameter_name.endswith(suffix)
                for suffix in (
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.q_proj.bias",
                    "self_attn.k_proj.bias",
                    "self_attn.v_proj.bias",
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.in_proj_z.weight",
                    "linear_attn.in_proj_b.weight",
                    "linear_attn.in_proj_a.weight",
                    "linear_attn.conv1d.weight",
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                )
            ):
                return self._language_attention_sharding(parameter_name, 0)
            if any(
                parameter_name.endswith(suffix)
                for suffix in (
                    "self_attn.o_proj.weight",
                    "linear_attn.out_proj.weight",
                )
            ):
                return self._language_attention_sharding(parameter_name, 1)

        if any(parameter_name.endswith(suffix) for suffix in _QWEN35_TP_DIM0_SUFFIXES):
            return self._tp_or_replicated(0)

        if any(parameter_name.endswith(suffix) for suffix in _QWEN35_TP_DIM1_SUFFIXES):
            return self._tp_or_replicated(1)

        raise ValueError(
            f"No Qwen3.5 AWEX sharding rule for parameter: {parameter_name}"
        )


def ensure_awex_qwen3_5_registered() -> None:
    """Register Qwen3.5-MoE AWEX v1 model config exactly once per process."""
    from awex.models import registry as _reg
    from awex.models.registry import ModelConfig

    model_key = "Qwen3_5MoeForConditionalGeneration"
    desired_entry = ModelConfig(
        sharding_strategy=Qwen3_5MoeShardingStrategy,
        mcore_converter=McoreToHFWeightConverterQwen3_5Moe,
        sglang_converter=SGlangToHFWeightConverterQwen3_5Moe,
    )

    current_entry = _reg.ModelRegistry.models.get(model_key)
    if isinstance(current_entry, ModelConfig):
        if (
            current_entry.sharding_strategy == desired_entry.sharding_strategy
            and current_entry.mcore_converter == desired_entry.mcore_converter
            and current_entry.sglang_converter == desired_entry.sglang_converter
        ):
            return

    if isinstance(current_entry, dict):
        if (
            current_entry.get("sharding_strategy") == desired_entry.sharding_strategy
            and current_entry.get("mcore_converter") == desired_entry.mcore_converter
            and current_entry.get("sglang_converter") == desired_entry.sglang_converter
        ):
            return

    if current_entry == desired_entry:
        return

    _reg.ModelRegistry.models[model_key] = desired_entry
