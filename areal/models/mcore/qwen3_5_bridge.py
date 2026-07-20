# SPDX-License-Identifier: Apache-2.0

import inspect

import torch
from mbridge.core import LLMBridge, register_model
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend

from areal.models.mcore.qwen3_5 import make_mcore_layer_specs_qwen3_5_moe


def _get_text_config(hf_config):
    return hf_config.text_config if hasattr(hf_config, "text_config") else hf_config


@register_model(["qwen3_5_moe"])
class Qwen3_5MoeBridge(LLMBridge):
    """mbridge bridge for Qwen3.5-MoE text model path on top of a VL checkpoint."""

    TransformerConfigClass = TransformerConfig

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": [
            "model.language_model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.q_layernorm.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_norm.weight"
        ],
        "self_attention.k_layernorm.weight": [
            "model.language_model.layers.{layer_number}.self_attn.k_norm.weight"
        ],
        "self_attention.linear_qkv.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    } | {
        f"self_attention.{weight_name}": [
            "model.language_model.layers.{layer_number}." + weight_name
        ]
        for weight_name in [
            "input_layernorm.weight",
            # linear attention
            "linear_attn.A_log",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            # full attention
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ]
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.down_proj.weight"
        ],
        "shared_experts.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
        ],
        "pre_mlp_layernorm": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "shared_experts.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"
        ],
        "mlp.router.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate.weight"
        ],
        "shared_experts.gate_weight": [
            "model.language_model.layers.{layer_number}.mlp.shared_expert_gate.weight"
        ],
        # Fused expert format: one 3D tensor for all experts in HF.
        "mlp.experts.linear_fc1": [
            "model.language_model.layers.{layer_number}.mlp.experts.gate_up_proj",
        ],
        "mlp.experts.linear_fc2": [
            "model.language_model.layers.{layer_number}.mlp.experts.down_proj"
        ],
    }

    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "ffn_hidden_size": ("intermediate_size", None),
        "attention_dropout": "attention_dropout",
        "layernorm_epsilon": "rms_norm_eps",
        "hidden_dropout": ("hidden_dropout", 0.0),
        "kv_channels": ("head_dim", None),
    }

    def _supports_transformer_config_kwarg(self, kwarg_name: str) -> bool:
        transformer_config_class = getattr(self, "TransformerConfigClass", None)
        if transformer_config_class is None:
            return True

        dataclass_fields = getattr(
            transformer_config_class, "__dataclass_fields__", None
        )
        if dataclass_fields is not None:
            return kwarg_name in dataclass_fields

        try:
            signature = inspect.signature(transformer_config_class)
        except (TypeError, ValueError):
            return True
        return kwarg_name in signature.parameters

    def _adjust_mapping_for_shared_weights(self):
        text_config = _get_text_config(self.hf_config)
        tie_word_embeddings = getattr(
            text_config, "tie_word_embeddings", False
        ) or getattr(self.hf_config, "tie_word_embeddings", False)
        if tie_word_embeddings:
            self._DIRECT_MAPPING = dict(self._DIRECT_MAPPING)
            self._DIRECT_MAPPING["output_layer.weight"] = (
                "model.language_model.embed_tokens.weight"
            )

    def _build_config(self):
        text_config = _get_text_config(self.hf_config)

        ffn_hidden_size = getattr(text_config, "intermediate_size", None)
        if ffn_hidden_size is None:
            ffn_hidden_size = getattr(text_config, "shared_expert_intermediate_size")

        rope_parameters = getattr(text_config, "rope_parameters", None) or {}
        rotary_base = rope_parameters.get(
            "rope_theta",
            getattr(text_config, "rope_theta", 10000.0),
        )

        kwargs = dict(
            text_config_key="text_config"
            if hasattr(self.hf_config, "text_config")
            else None,
            attention_backend=AttnBackend.fused,
            layernorm_epsilon=text_config.rms_norm_eps,
            ffn_hidden_size=ffn_hidden_size,
            qk_layernorm=True,
            attention_output_gate=True,
            rotary_base=rotary_base,
            rotary_percent=getattr(text_config, "partial_rotary_factor", 1.0),
            # MoE
            moe_ffn_hidden_size=getattr(text_config, "moe_intermediate_size", None),
            moe_shared_expert_intermediate_size=getattr(
                text_config, "shared_expert_intermediate_size", None
            ),
            moe_shared_expert_gate=True,
            moe_router_topk=getattr(text_config, "num_experts_per_tok", 1),
            num_moe_experts=getattr(text_config, "num_experts", None),
            moe_aux_loss_coeff=getattr(text_config, "router_aux_loss_coef", 0.0),
            moe_router_score_function="softmax",
            moe_router_load_balancing_type="aux_loss",
            moe_grouped_gemm=True,
            moe_router_dtype="fp32",
            moe_token_dispatcher_type="alltoall",
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
        )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if self._supports_transformer_config_kwarg("use_gated_attention"):
            kwargs["use_gated_attention"] = True

        return self._build_base_config(**kwargs)

    def _get_transformer_layer_spec(self, vp_stage=None):
        self.has_vp_stage = False
        return make_mcore_layer_specs_qwen3_5_moe(
            self.config,
            self.hf_config,
            use_te=True,
            vp_stage=vp_stage,
        )

    def _get_gptmodel_args(self) -> dict:
        text_config = _get_text_config(self.hf_config)
        return {
            "vocab_size": text_config.vocab_size,
            "max_sequence_length": text_config.max_position_embeddings,
            "position_embedding_type": "rope",
            "rotary_base": (getattr(text_config, "rope_parameters", None) or {}).get(
                "rope_theta", getattr(text_config, "rope_theta", 10000.0)
            ),
        }

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if (
            "self_attention" in mcore_weights_name
            or "input_layernorm.weight" in mcore_weights_name
        ):
            return self._weight_name_mapping_attention(mcore_weights_name)
        if "mlp" in mcore_weights_name or "pre_mlp_layernorm" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._ATTENTION_MAPPING.items():
            if keyword in name:
                convert_names.extend(
                    [x.format(layer_number=layer_number) for x in mapping_names]
                )
                break
        if not convert_names:
            raise NotImplementedError(f"Unsupported attention parameter: {name}")
        return convert_names

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    expert_id = name.split("weight")[-1]
                    convert_names.extend(
                        [
                            x.format(layer_number=layer_number, expert_id=expert_id)
                            for x in mapping_names
                        ]
                    )
                else:
                    convert_names.extend(
                        [x.format(layer_number=layer_number) for x in mapping_names]
                    )
                break
        if not convert_names:
            raise NotImplementedError(f"Unsupported MLP parameter: {name}")
        return convert_names

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ):
        # Full-attention q_proj in Qwen3.5 stores [query, gate] interleaved per query head.
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
            and len(hf_weights) == 3
        ):
            text_config = _get_text_config(self.hf_config)
            num_kv_heads = text_config.num_key_value_heads
            num_attn_heads = text_config.num_attention_heads
            head_dim = getattr(
                text_config, "head_dim", text_config.hidden_size // num_attn_heads
            )
            n_per_group = num_attn_heads // num_kv_heads
            group_dim = head_dim * n_per_group

            q, k, v = hf_weights
            real_num_kv_heads = q.shape[0] // (2 * group_dim)
            q = (
                q.view(real_num_kv_heads, n_per_group, 2, head_dim, -1)
                .transpose(1, 2)
                .flatten(1, 3)
            )
            k = k.view(real_num_kv_heads, head_dim, -1)
            v = v.view(real_num_kv_heads, head_dim, -1)
            if ".bias" in mcore_weights_name:
                return torch.cat([q, k, v], dim=1).reshape(-1).contiguous()
            return torch.cat([q, k, v], dim=1).reshape(-1, q.shape[-1]).contiguous()

        if "mlp.experts.linear_fc" in mcore_weights_name and len(hf_weights) == 1:
            w = hf_weights[0]
            if w.dim() == 3:
                local_expert_id = int(mcore_weights_name.split("weight")[-1])
                from megatron.core import mpu

                ep_size = mpu.get_expert_model_parallel_world_size()
                if ep_size > 1:
                    ep_rank = mpu.get_expert_model_parallel_rank()
                    num_local_experts = w.shape[0] // ep_size
                    global_expert_id = ep_rank * num_local_experts + local_expert_id
                else:
                    global_expert_id = local_expert_id
                return w[global_expert_id].contiguous()

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ):
        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
            and len(hf_names) == 3
        ):
            text_config = _get_text_config(self.hf_config)
            num_kv_heads = text_config.num_key_value_heads
            num_attn_heads = text_config.num_attention_heads
            head_dim = getattr(
                text_config, "head_dim", text_config.hidden_size // num_attn_heads
            )
            n_per_group = num_attn_heads // num_kv_heads
            per_kv_size = (2 * n_per_group + 2) * head_dim
            real_num_kv_heads = mcore_weights.shape[0] // per_kv_size

            if ".bias" in mcore_weights_name:
                w = mcore_weights.view(real_num_kv_heads, per_kv_size)
            else:
                w = mcore_weights.view(real_num_kv_heads, per_kv_size, -1)

            q, k, v = torch.split(
                w,
                [2 * n_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )

            if ".bias" in mcore_weights_name:
                q = (
                    q.view(real_num_kv_heads, 2, n_per_group, head_dim)
                    .transpose(1, 2)
                    .reshape(-1)
                    .contiguous()
                )
                k = k.reshape(-1).contiguous()
                v = v.reshape(-1).contiguous()
            else:
                q = (
                    q.view(real_num_kv_heads, 2, n_per_group, head_dim, -1)
                    .transpose(1, 2)
                    .reshape(-1, w.shape[-1])
                    .contiguous()
                )
                k = k.reshape(-1, w.shape[-1]).contiguous()
                v = v.reshape(-1, w.shape[-1]).contiguous()

            return hf_names, [q, k, v]

        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)
