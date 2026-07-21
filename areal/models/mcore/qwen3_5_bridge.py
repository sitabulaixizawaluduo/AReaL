# SPDX-License-Identifier: Apache-2.0

import inspect
import os
from collections.abc import Callable

import torch
from mbridge.core import LLMBridge, register_model
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend

from areal.models.mcore.qwen3_5 import make_mcore_layer_specs_qwen3_5_moe
from areal.models.mcore.qwen3_5_weight_utils import (
    qwen3_5_gated_qkv_hf_to_mcore,
    qwen3_5_gated_qkv_mcore_to_hf,
    qwen3_5_gdn_qkv_section_sizes,
    relayout_fused_sections_for_tp,
    undo_relayout_fused_sections_for_tp,
)

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeVisionModel,
    )
except ImportError:  # pragma: no cover
    Qwen3_5MoeVisionModel = None


def _get_text_config(hf_config):
    return hf_config.text_config if hasattr(hf_config, "text_config") else hf_config


def _parse_expert_weight_idx(mcore_weights_name: str) -> int:
    return int(mcore_weights_name.split(".weight")[-1])


@register_model(["qwen3_5_moe"])
class Qwen3_5MoeBridge(LLMBridge):
    """mbridge bridge for Qwen3.5-MoE text model path on top of a VL checkpoint."""

    TransformerConfigClass = TransformerConfig
    HfVisionClass: type | None = Qwen3_5MoeVisionModel

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "output_layer.weight": "lm_head.weight",
        "language_model.output_layer.weight": "lm_head.weight",
        "vision_model.patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
        "vision_model.patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
        "vision_model.pos_embed.weight": "model.visual.pos_embed.weight",
        "vision_model.merger.norm.weight": "model.visual.merger.norm.weight",
        "vision_model.merger.norm.bias": "model.visual.merger.norm.bias",
        "vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
        "vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
        "vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
        "vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
    }

    _VISUAL_MAPPING = {
        "vision_model.blocks.{layer_number}.attn.proj.weight": [
            "model.visual.blocks.{layer_number}.attn.proj.weight"
        ],
        "vision_model.blocks.{layer_number}.attn.proj.bias": [
            "model.visual.blocks.{layer_number}.attn.proj.bias"
        ],
        "vision_model.blocks.{layer_number}.attn.qkv.weight": [
            "model.visual.blocks.{layer_number}.attn.qkv.weight"
        ],
        "vision_model.blocks.{layer_number}.attn.qkv.bias": [
            "model.visual.blocks.{layer_number}.attn.qkv.bias"
        ],
        "vision_model.blocks.{layer_number}.mlp.linear_fc1.weight": [
            "model.visual.blocks.{layer_number}.mlp.linear_fc1.weight"
        ],
        "vision_model.blocks.{layer_number}.mlp.linear_fc1.bias": [
            "model.visual.blocks.{layer_number}.mlp.linear_fc1.bias"
        ],
        "vision_model.blocks.{layer_number}.mlp.linear_fc2.weight": [
            "model.visual.blocks.{layer_number}.mlp.linear_fc2.weight"
        ],
        "vision_model.blocks.{layer_number}.mlp.linear_fc2.bias": [
            "model.visual.blocks.{layer_number}.mlp.linear_fc2.bias"
        ],
        "vision_model.blocks.{layer_number}.norm1.weight": [
            "model.visual.blocks.{layer_number}.norm1.weight"
        ],
        "vision_model.blocks.{layer_number}.norm1.bias": [
            "model.visual.blocks.{layer_number}.norm1.bias"
        ],
        "vision_model.blocks.{layer_number}.norm2.weight": [
            "model.visual.blocks.{layer_number}.norm2.weight"
        ],
        "vision_model.blocks.{layer_number}.norm2.bias": [
            "model.visual.blocks.{layer_number}.norm2.bias"
        ],
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
            self._DIRECT_MAPPING["language_model.output_layer.weight"] = (
                "model.language_model.embed_tokens.weight"
            )

    def _build_config(self):
        text_config = _get_text_config(self.hf_config)

        ffn_hidden_size = getattr(text_config, "intermediate_size", None)
        if ffn_hidden_size is None:
            ffn_hidden_size = getattr(text_config, "shared_expert_intermediate_size")

        kwargs = dict(
            text_config_key="text_config"
            if hasattr(self.hf_config, "text_config")
            else None,
            attention_backend=AttnBackend.fused,
            layernorm_epsilon=text_config.rms_norm_eps,
            ffn_hidden_size=ffn_hidden_size,
            qk_layernorm=True,
            attention_output_gate=True,
            # TE CP attention algorithm ("p2p"/"all_gather"/"a2a"/"a2a+p2p").
            # Env override to bisect/work around TE AttnFuncWithCPAndKVP2P
            # backward-NaN bugs without a config schema change.
            cp_comm_type=os.environ.get("AREAL_CP_COMM_TYPE") or None,
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

        rope_parameters = getattr(text_config, "rope_parameters", None) or {}
        if self._supports_transformer_config_kwarg("mrope_section"):
            kwargs["mrope_section"] = rope_parameters.get("mrope_section", [11, 11, 10])

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
            "rotary_percent": getattr(text_config, "partial_rotary_factor", 1.0),
        }

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if mcore_weights_name.startswith("vision_model."):
            return self._weight_name_mapping_visual(mcore_weights_name)

        if mcore_weights_name.startswith("language_model."):
            mcore_weights_name = mcore_weights_name.removeprefix("language_model.")

        if (
            "self_attention" in mcore_weights_name
            or "input_layernorm.weight" in mcore_weights_name
        ):
            return self._weight_name_mapping_attention(mcore_weights_name)
        if "mlp" in mcore_weights_name or "pre_mlp_layernorm" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_name_mapping_visual(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[2]
        split_name[2] = "{layer_number}"
        key = ".".join(split_name)
        mapping_names = self._VISUAL_MAPPING[key]
        convert_names = [x.format(layer_number=layer_number) for x in mapping_names]
        if not convert_names:
            raise NotImplementedError(f"Unsupported visual parameter: {name}")
        return convert_names

    def _convert_vision_qkv_hf_to_mcore(
        self,
        mcore_weights_name: str,
        hf_weights: list[torch.Tensor],
    ) -> torch.Tensor:
        x = hf_weights[0]
        vision_num_heads = self.hf_config.vision_config.num_heads
        head_dim = self.hf_config.vision_config.hidden_size // vision_num_heads
        is_bias = ".bias" in mcore_weights_name

        if is_bias:
            q, k, v = x.view(3, vision_num_heads, head_dim)
            return torch.cat(
                [
                    q.reshape(vision_num_heads, head_dim),
                    k.reshape(vision_num_heads, head_dim),
                    v.reshape(vision_num_heads, head_dim),
                ],
                dim=1,
            ).reshape(-1)

        hidden_size = x.shape[-1]
        q, k, v = x.view(3, vision_num_heads, head_dim, hidden_size)
        return torch.cat(
            [
                q.reshape(vision_num_heads, head_dim, hidden_size),
                k.reshape(vision_num_heads, head_dim, hidden_size),
                v.reshape(vision_num_heads, head_dim, hidden_size),
            ],
            dim=1,
        ).reshape(-1, hidden_size)

    def _convert_vision_qkv_mcore_to_hf(
        self,
        mcore_weights_name: str,
        mcore_weights: torch.Tensor,
    ) -> torch.Tensor:
        vision_num_heads = self.hf_config.vision_config.num_heads
        hidden_vision = mcore_weights.shape[0] // 3
        head_dim = hidden_vision // vision_num_heads
        is_bias = ".bias" in mcore_weights_name

        if is_bias:
            x = mcore_weights.view(vision_num_heads, 3, head_dim)
            return x.permute(1, 0, 2).contiguous().view(-1)

        in_features = mcore_weights.shape[-1]
        x = mcore_weights.view(vision_num_heads, 3, head_dim, in_features)
        return x.permute(1, 0, 2, 3).contiguous().view(-1, in_features)

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
        if (
            mcore_weights_name.startswith("vision_model.blocks.")
            and ".attn.qkv." in mcore_weights_name
            and len(hf_weights) == 1
        ):
            return self._convert_vision_qkv_hf_to_mcore(mcore_weights_name, hf_weights)

        # Full-attention q_proj in Qwen3.5 stores [query, gate] interleaved per query head.
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
            and len(hf_weights) == 3
        ):
            q, k, v = hf_weights
            return qwen3_5_gated_qkv_hf_to_mcore(self.hf_config, q, k, v)

        if len(hf_weights) == 1 and (
            "self_attention.linear_attn.in_proj_qkv.weight" in mcore_weights_name
            or "self_attention.linear_attn.conv1d.weight" in mcore_weights_name
        ):
            tp_size = 1
            try:
                from megatron.core import parallel_state as mpu

                if mpu.model_parallel_is_initialized():
                    tp_size = mpu.get_tensor_model_parallel_world_size()
            except (ImportError, RuntimeError, AttributeError):  # pragma: no cover
                pass
            return relayout_fused_sections_for_tp(
                hf_weights[0],
                section_sizes=qwen3_5_gdn_qkv_section_sizes(self.hf_config),
                tp_size=tp_size,
                dim=0,
            )

        if "mlp.experts.linear_fc" in mcore_weights_name and len(hf_weights) == 1:
            w = hf_weights[0]
            if w.dim() == 3:
                local_expert_id = _parse_expert_weight_idx(mcore_weights_name)
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

    def _stack_qwen3_5_grouped_expert_to_hf(
        self,
        mcore_weights_name: str,
        hf_name: str,
        mcore_weights: torch.Tensor,
    ) -> tuple[list[str], list[torch.Tensor]]:
        if not hasattr(self, "_qwen3_5_grouped_expert_buffers"):
            self._qwen3_5_grouped_expert_buffers = {}

        text_cfg = _get_text_config(self.hf_config)
        num_experts = getattr(text_cfg, "num_experts", None)
        if num_experts is None:
            raise ValueError(
                "Qwen3.5-MoE grouped expert export requires text_config.num_experts"
            )

        expert_idx = _parse_expert_weight_idx(mcore_weights_name)
        key = (hf_name, str(mcore_weights.dtype), str(mcore_weights.device))
        slots = self._qwen3_5_grouped_expert_buffers.setdefault(
            key, [None] * num_experts
        )
        slots[expert_idx] = mcore_weights.contiguous()

        if any(v is None for v in slots):
            return [], []

        stacked = torch.stack(slots, dim=0)
        del self._qwen3_5_grouped_expert_buffers[key]
        return [hf_name], [stacked]

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ):
        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)

        if (
            len(hf_names) == 1
            and "mlp.experts.linear_fc" in mcore_weights_name
            and ".weight" in mcore_weights_name
        ):
            return self._stack_qwen3_5_grouped_expert_to_hf(
                mcore_weights_name,
                hf_names[0],
                mcore_weights,
            )

        if (
            mcore_weights_name.startswith("vision_model.blocks.")
            and ".attn.qkv." in mcore_weights_name
            and len(hf_names) == 1
        ):
            return hf_names, [
                self._convert_vision_qkv_mcore_to_hf(mcore_weights_name, mcore_weights)
            ]

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
            and len(hf_names) == 3
        ):
            q, k, v = qwen3_5_gated_qkv_mcore_to_hf(self.hf_config, mcore_weights)
            return hf_names, [q, k, v]

        if len(hf_names) == 1 and (
            "self_attention.linear_attn.in_proj_qkv.weight" in mcore_weights_name
            or "self_attention.linear_attn.conv1d.weight" in mcore_weights_name
        ):
            tp_size = 1
            try:
                from megatron.core import parallel_state as mpu

                if mpu.model_parallel_is_initialized():
                    tp_size = mpu.get_tensor_model_parallel_world_size()
            except (ImportError, RuntimeError, AttributeError):  # pragma: no cover
                pass
            return hf_names, [
                undo_relayout_fused_sections_for_tp(
                    mcore_weights,
                    section_sizes=qwen3_5_gdn_qkv_section_sizes(self.hf_config),
                    tp_size=tp_size,
                    dim=0,
                )
            ]

        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _model_provider(
        self, post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]]
    ):
        share_embeddings_and_output_weights = getattr(
            _get_text_config(self.hf_config), "tie_word_embeddings", False
        )

        def provider(pre_process, post_process, vp_stage=None):
            if not hasattr(self.hf_config, "vision_config"):
                return super(Qwen3_5MoeBridge, self)._model_provider(
                    post_model_creation_callbacks
                )(pre_process, post_process, vp_stage=vp_stage)

            from areal.models.mcore.qwen3_5_vl_model import Qwen3_5MoeVLModel

            if self.HfVisionClass is None:
                raise ImportError(
                    "Qwen3_5MoeVisionModel is unavailable. Please install a transformers "
                    "build that includes qwen3_5_moe vision classes."
                )

            transformer_layer_spec = self._get_transformer_layer_spec(vp_stage)
            text_config = _get_text_config(self.hf_config)
            rope_parameters = getattr(text_config, "rope_parameters", None) or {}
            model = Qwen3_5MoeVLModel(
                language_transformer_config=self.config,
                language_transformer_layer_spec=transformer_layer_spec,
                language_vocab_size=text_config.vocab_size,
                language_max_sequence_length=text_config.max_position_embeddings,
                hf_config=self.hf_config,
                hf_vision_cls=self.HfVisionClass,
                parallel_output=True,
                language_rotary_percent=getattr(
                    text_config, "partial_rotary_factor", 1.0
                ),
                language_rotary_base=rope_parameters.get(
                    "rope_theta",
                    getattr(text_config, "rope_theta", 10000.0),
                ),
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=False,
                language_share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                image_token_id=self.hf_config.image_token_id,
                video_token_id=self.hf_config.video_token_id,
                vision_start_token_id=self.hf_config.vision_start_token_id,
                freeze_vision_model=getattr(self, "freeze_vision_model", False),
            )

            for callback in post_model_creation_callbacks:
                callback(
                    model,
                    pre_process=pre_process,
                    post_process=post_process,
                    config=self.config,
                    hf_config=self.hf_config,
                )
            return model

        return provider
