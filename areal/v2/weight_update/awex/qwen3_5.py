# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false
"""Qwen3.5(-MoE) name protocol for awex disaggregated weight update.

Qwen3.5 is a hybrid model (gated-deltanet linear attention + gated full
attention + MoE).  Both the Megatron (train) and SGLang (inference) sides
must report parameter metadata under one *common name space* so awex's
``TransferPlanBuilder`` can pair send/recv shards by name.

Common name space
-----------------
HF text-model checkpoint names (``model.layers.N...``, i.e. the VL composite
``model.language_model.`` prefix stripped), with three deviations forced by
tensor-parallel contiguity -- a ``ParameterShardMeta`` can only describe ONE
contiguous slice per rank, so any fused tensor whose per-rank shard maps to
multiple disjoint slices of the HF tensor must be split into per-block
synthetic names on BOTH sides:

1. ``linear_attn.in_proj_qkv.weight`` (HF ``[2*key_dim+value_dim, H]``) ->
   ``linear_attn.in_proj_q/k/v.weight``.  SGLang's ``in_proj_qkvz`` is a
   ``MergedColumnParallelLinear`` over ``[key, key, value, value]`` blocks:
   each block is TP-sharded independently, so a rank's fused rows map to 3
   disjoint slices of HF ``in_proj_qkv``.
2. ``linear_attn.conv1d.weight`` (HF ``[2*key_dim+value_dim, 1, kernel]``) ->
   ``linear_attn.conv1d_q/k/v.weight`` (same block structure, see SGLang's
   ``mamba_v2_sharded_weight_loader([qk, qk, v])``).
3. ``mlp.experts.gate_up_proj`` / ``down_proj`` (HF fused-3D
   ``[E, 2I, H]`` / ``[E, H, I]``) -> per-expert
   ``mlp.experts.{e}.gate_proj/up_proj/down_proj.weight``.  Per-expert
   splitting keeps TP shards contiguous (a fused ``gate_up`` TP shard would
   be two disjoint slices) and gives EP a natural per-expert ownership unit.

Kept as-is (contiguous under TP by construction):

- ``self_attn.q_proj.weight``: HF fuses the attention output gate per head
  (``[q_h; gate_h]`` blocks of ``2*head_dim``).  SGLang's ``qkv_proj`` treats
  them as ``2*num_heads`` q-heads, so a TP shard is a contiguous slice of the
  HF tensor.  Same for ``k/v_proj`` (plain GQA heads).
- ``linear_attn.in_proj_z/b/a``, ``A_log``, ``dt_bias``: single-block
  column-parallel tensors.

Skipped on BOTH sides (documented limitation):

- ``model.visual.*``: RL fine-tuning on this branch keeps the vision tower
  frozen; SGLang loads it from the base checkpoint at server start.
- MTP / nextn speculative weights.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import torch
from awex.sharding.param_sharding import ShardingStrategy, ShardingType

from areal.utils import logging

logger = logging.getLogger("AwexQwen3_5")

# SGLang architecture strings served by ``EntryClass`` in
# ``sglang/srt/models/qwen3_5.py`` plus the plain text-model class used when
# serving a text-only checkpoint.
QWEN3_5_MOE_ARCHITECTURES = (
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
)

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.mlp\.experts\.(\d+)\.")


def text_config(hf_config: Any) -> Any:
    """Return the text side of a (possibly VL-composite) Qwen3.5 config."""
    return getattr(hf_config, "text_config", hf_config)


def is_qwen3_5_moe_hf_config(hf_config: Any) -> bool:
    cfg = text_config(hf_config)
    return getattr(cfg, "model_type", "") in ("qwen3_5_moe", "qwen3_5_moe_text")


class _GdnDims:
    """Gated-deltanet block dimensions derived from the HF text config."""

    def __init__(self, cfg: Any):
        self.num_k_heads = int(cfg.linear_num_key_heads)
        self.num_v_heads = int(cfg.linear_num_value_heads)
        self.head_k_dim = int(cfg.linear_key_head_dim)
        self.head_v_dim = int(cfg.linear_value_head_dim)
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads


# ---------------------------------------------------------------------------
# Name normalization (train side, bridge export names)
# ---------------------------------------------------------------------------


def normalize_train_hf_name(name: str) -> str | None:
    """Map a megatron-bridge exported HF name into the common name space.

    Returns ``None`` for parameters excluded from weight sync (vision tower,
    MTP, rotary buffers).
    """
    if "visual" in name or "mtp" in name or "rotary_emb.inv_freq" in name:
        return None
    if "language_model" in name:
        name = name.replace("model.language_model.", "model.")
    return name


# ---------------------------------------------------------------------------
# Train-side splitting (full HF tensors -> common names)
# ---------------------------------------------------------------------------


def split_train_hf_param(
    name: str, tensor: torch.Tensor, hf_config: Any
) -> Iterator[tuple[str, torch.Tensor]]:
    """Split a full (TP/EP/PP-gathered) HF tensor into common-name tensors.

    ``name`` must already be normalized via :func:`normalize_train_hf_name`.
    Tensors are yielded as views where possible; callers own contiguity.
    """
    cfg = text_config(hf_config)

    if name.endswith("linear_attn.in_proj_qkv.weight"):
        gdn = _GdnDims(cfg)
        q, k, v = torch.split(tensor, [gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=0)
        base = name[: -len("in_proj_qkv.weight")]
        yield f"{base}in_proj_q.weight", q
        yield f"{base}in_proj_k.weight", k
        yield f"{base}in_proj_v.weight", v
        return

    if name.endswith("linear_attn.conv1d.weight"):
        gdn = _GdnDims(cfg)
        q, k, v = torch.split(tensor, [gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=0)
        base = name[: -len("conv1d.weight")]
        yield f"{base}conv1d_q.weight", q
        yield f"{base}conv1d_k.weight", k
        yield f"{base}conv1d_v.weight", v
        return

    if name.endswith("mlp.experts.gate_up_proj"):
        # HF fused-3D [E, 2I, H]; gate rows first, up rows second
        # (``linear(x, w[e]).chunk(2, dim=-1)`` on the output side).
        inter = tensor.shape[1] // 2
        base = name[: -len("gate_up_proj")]
        for e in range(tensor.shape[0]):
            yield f"{base}{e}.gate_proj.weight", tensor[e, :inter]
            yield f"{base}{e}.up_proj.weight", tensor[e, inter:]
        return

    if name.endswith("mlp.experts.down_proj"):
        # HF fused-3D [E, H, I]
        base = name[: -len("down_proj")]
        for e in range(tensor.shape[0]):
            yield f"{base}{e}.down_proj.weight", tensor[e]
        return

    yield name, tensor


# ---------------------------------------------------------------------------
# Train-side ownership (PP stage x EP expert)
# ---------------------------------------------------------------------------


class TrainOwnership:
    """Decides which common-name params this train rank reports and sends.

    ``bridge.export_hf_weights`` iterates the GLOBAL parameter set on every
    rank (all ranks participate in the gather collectives).  awex, however,
    expects each rank to claim only the params it will actually send:

    - transformer layers  -> ranks whose PP stage holds the layer,
    - expert tensors      -> additionally, ranks whose EP range owns the
      expert (matching the native EP placement, which bounds memory),
    - embeddings          -> first PP stage,
    - lm_head / final norm-> last PP stage.

    TP/CP/DP peers within the owning (pp, ep) coordinate all report the same
    full tensors; ``dp_replicated=True`` lets awex pick a single sender.
    """

    def __init__(
        self,
        owned_layers: set[int],
        is_pp_first: bool,
        is_pp_last: bool,
        ep_rank: int,
        ep_size: int,
        num_experts: int | None,
    ):
        if ep_size > 1:
            if not num_experts or num_experts % ep_size != 0:
                raise ValueError(
                    f"num_experts={num_experts} must be divisible by "
                    f"ep_size={ep_size} for awex expert ownership"
                )
        self.owned_layers = owned_layers
        self.is_pp_first = is_pp_first
        self.is_pp_last = is_pp_last
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.num_experts = num_experts

    def owns(self, common_name: str) -> bool:
        m = _LAYER_RE.match(common_name)
        if m is not None:
            if int(m.group(1)) not in self.owned_layers:
                return False
            if self.ep_size > 1:
                em = _EXPERT_RE.search(common_name)
                if em is not None:
                    assert self.num_experts is not None
                    per_rank = self.num_experts // self.ep_size
                    return int(em.group(1)) // per_rank == self.ep_rank
            return True
        if common_name == "model.embed_tokens.weight":
            return self.is_pp_first
        if common_name in ("model.norm.weight", "lm_head.weight"):
            return self.is_pp_last
        # Unrecognized non-layer param: claim on the last stage only so it is
        # reported exactly once per DP replica group.
        logger.warning(
            "Unrecognized non-layer param %s; claiming on last PP stage",
            common_name,
        )
        return self.is_pp_last


# ---------------------------------------------------------------------------
# Inference-side unfuse (SGLang runtime params -> common names)
# ---------------------------------------------------------------------------


def unfuse_sglang_param(
    name: str,
    tensor: torch.Tensor,
    hf_config: Any,
    tp_size: int,
) -> list[tuple[str, torch.Tensor]]:
    """Map one SGLang runtime parameter to (common_name, local_shard) pairs.

    SGLang's ``Qwen3_5(Moe)ForCausalLM`` attaches attention modules directly
    on the decoder layer (``model.layers.N.qkv_proj``), so the ``self_attn.``
    segment must be re-inserted to match HF names.  All splits below mirror
    the per-rank layouts established in ``sglang/srt/models/qwen3_5.py``.
    """
    cfg = text_config(hf_config)

    if name.startswith("visual."):
        return []
    if ".mtp" in name or "rotary_emb" in name:
        return []

    # --- full attention (module attrs live directly on the layer) --------
    if ".qkv_proj.weight" in name:
        num_heads = int(cfg.num_attention_heads)
        num_kv = int(getattr(cfg, "num_key_value_heads", num_heads))
        head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // num_heads))
        gate = bool(getattr(cfg, "attn_output_gate", True))
        if num_heads % tp_size != 0 or num_kv % tp_size != 0:
            raise ValueError(
                f"attention heads ({num_heads}/{num_kv}) must be divisible "
                f"by tp_size={tp_size} for awex qwen3.5 weight sync"
            )
        q_local = (num_heads // tp_size) * (2 if gate else 1) * head_dim
        kv_local = (num_kv // tp_size) * head_dim
        q, k, v = torch.split(tensor, [q_local, kv_local, kv_local], dim=0)
        base = name.replace(".qkv_proj.weight", ".self_attn.")
        return [
            (f"{base}q_proj.weight", q),
            (f"{base}k_proj.weight", k),
            (f"{base}v_proj.weight", v),
        ]
    for attn_leaf in ("o_proj.weight", "q_norm.weight", "k_norm.weight"):
        marker = f".{attn_leaf}"
        if name.endswith(marker) and ".self_attn." not in name:
            return [(name[: -len(marker)] + f".self_attn.{attn_leaf}", tensor)]

    # --- gated deltanet (linear attention) --------------------------------
    if ".linear_attn.in_proj_qkvz.weight" in name:
        gdn = _GdnDims(cfg)
        if gdn.key_dim % tp_size != 0 or gdn.value_dim % tp_size != 0:
            raise ValueError(
                f"linear-attention dims (key_dim={gdn.key_dim}, "
                f"value_dim={gdn.value_dim}) must be divisible by "
                f"tp_size={tp_size}"
            )
        k_tp = gdn.key_dim // tp_size
        v_tp = gdn.value_dim // tp_size
        q, k, v, z = torch.split(tensor, [k_tp, k_tp, v_tp, v_tp], dim=0)
        base = name.replace("in_proj_qkvz.weight", "")
        return [
            (f"{base}in_proj_q.weight", q),
            (f"{base}in_proj_k.weight", k),
            (f"{base}in_proj_v.weight", v),
            (f"{base}in_proj_z.weight", z),
        ]
    if ".linear_attn.in_proj_ba.weight" in name:
        gdn = _GdnDims(cfg)
        nv_tp = gdn.num_v_heads // tp_size
        b, a = torch.split(tensor, [nv_tp, nv_tp], dim=0)
        base = name.replace("in_proj_ba.weight", "")
        return [
            (f"{base}in_proj_b.weight", b),
            (f"{base}in_proj_a.weight", a),
        ]
    if ".linear_attn.conv1d.weight" in name:
        gdn = _GdnDims(cfg)
        k_tp = gdn.key_dim // tp_size
        v_tp = gdn.value_dim // tp_size
        q, k, v = torch.split(tensor, [k_tp, k_tp, v_tp], dim=0)
        base = name.replace("conv1d.weight", "")
        return [
            (f"{base}conv1d_q.weight", q),
            (f"{base}conv1d_k.weight", k),
            (f"{base}conv1d_v.weight", v),
        ]

    # --- MoE ---------------------------------------------------------------
    if ".mlp.experts.w13_weight" in name:
        # FusedMoE per-rank [num_local_experts, 2*I/tp, H]; gate rows first.
        inter = tensor.shape[1] // 2
        base = name.replace(".w13_weight", "")
        out: list[tuple[str, torch.Tensor]] = []
        for e in range(tensor.shape[0]):
            out.append((f"{base}.{e}.gate_proj.weight", tensor[e, :inter]))
            out.append((f"{base}.{e}.up_proj.weight", tensor[e, inter:]))
        return out
    if ".mlp.experts.w2_weight" in name:
        base = name.replace(".w2_weight", "")
        return [
            (f"{base}.{e}.down_proj.weight", tensor[e]) for e in range(tensor.shape[0])
        ]
    if ".mlp.shared_expert.gate_up_proj.weight" in name:
        half = tensor.shape[0] // 2
        return [
            (
                name.replace("gate_up_proj.weight", "gate_proj.weight"),
                tensor[:half],
            ),
            (
                name.replace("gate_up_proj.weight", "up_proj.weight"),
                tensor[half:],
            ),
        ]

    # --- passthrough (names already match HF) ------------------------------
    return [(name, tensor)]


# ---------------------------------------------------------------------------
# Inference-side sharding declaration
# ---------------------------------------------------------------------------

# Suffix -> (sharded, dim). Matched against the COMMON (post-unfuse) name.
_REPLICATED_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".linear_attn.norm.weight",
    ".mlp.gate.weight",
    ".mlp.shared_expert_gate.weight",
    "model.norm.weight",
)

_DIM0_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".linear_attn.in_proj_q.weight",
    ".linear_attn.in_proj_k.weight",
    ".linear_attn.in_proj_v.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.in_proj_b.weight",
    ".linear_attn.in_proj_a.weight",
    ".linear_attn.conv1d_q.weight",
    ".linear_attn.conv1d_k.weight",
    ".linear_attn.conv1d_v.weight",
    ".linear_attn.A_log",
    ".linear_attn.dt_bias",
    ".gate_proj.weight",  # experts.{e}. and shared_expert.
    ".up_proj.weight",
    "model.embed_tokens.weight",
    "lm_head.weight",
)

_DIM1_SUFFIXES = (
    ".self_attn.o_proj.weight",
    ".linear_attn.out_proj.weight",
    ".down_proj.weight",  # experts.{e}. and shared_expert.
)


class Qwen3_5MoeShardingStrategy(ShardingStrategy):
    """Explicit TP sharding table for Qwen3.5(-MoE) common names.

    awex's default :class:`ShardingStrategy` keys off substrings such as
    ``attention``/``mlp`` that do not appear in Qwen3.5's HF names
    (``self_attn``/``linear_attn``) and knows nothing about the gated
    deltanet tensors, so every declaration here is explicit.  Only plain TP
    inference deployments are supported (``dp_attention`` off, ``ep_size=1``
    on the inference side).
    """

    def get_sharding_strategy(self, parameter_name, **kwargs):
        del kwargs
        tp_size = self.rank_info.tp_size
        if self.enable_dp_attention:
            raise NotImplementedError(
                "awex qwen3.5 weight sync does not support "
                "enable_dp_attention on the inference side"
            )
        if tp_size == 1:
            return ShardingType.NO_SHARDING, 0, 1
        for suffix in _REPLICATED_SUFFIXES:
            if parameter_name.endswith(suffix):
                return ShardingType.NO_SHARDING, 0, 1
        for suffix in _DIM0_SUFFIXES:
            if parameter_name.endswith(suffix):
                return ShardingType.TP_SHARDING, 0, tp_size
        for suffix in _DIM1_SUFFIXES:
            if parameter_name.endswith(suffix):
                return ShardingType.TP_SHARDING, 1, tp_size
        raise ValueError(
            f"No sharding rule for qwen3.5 parameter {parameter_name!r}; "
            "extend areal/v2/weight_update/awex/qwen3_5.py"
        )
