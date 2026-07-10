# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import os
import re
import threading
import time
from typing import TYPE_CHECKING

import httpx
import torch
import torch.distributed as dist
from awex.meta.weight_meta import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_send_ops
from awex.transfer.transfer_plan import TransferPlan, TransferPlanBuilder
from awex.util.tensor_util import (
    cuda_ipc_serialize,
    group_tensors_by_shape_and_dtype,
)

from areal.engine.core.model import is_qwen3_5_vl_model, is_qwen_vl_model
from areal.utils import logging
from areal.v2.weight_update.awex import (
    awex_wu_use_group,
    fetch_kv_metadata,
)
from areal.v2.weight_update.nccl_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)
from areal.v2.weight_update.training_adapter import (
    AwexTrainingAdapter,
)

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronEngine

logger = logging.getLogger("AwexMegatronAdapter")


_QWEN_VL_VISUAL_QKV_RE = re.compile(
    r"^(visual\.blocks\.\d+\.attn\.)qkv(\.(?:weight|bias))$"
)


def _split_qwen_vl_visual_qkv(
    hf_name: str, tensor: torch.Tensor
) -> list[tuple[str, torch.Tensor]] | None:
    """Split mbridge's fused visual-tower QKV into HF's q/k/v_proj triplet.

    Why: HF's Qwen VL modeling refactored the visual attention from a fused
    ``qkv`` linear into separate ``q_proj``/``k_proj``/``v_proj``. mbridge
    still exports the fused HF name, while SGLang/vLLM already expect the
    split names — so awex's transfer plan rejects the mismatch. The vision
    tower has no GQA, and mbridge lays the fused tensor out as HF grouped
    ``[Q_all | K_all | V_all]``, so a plain ``chunk(3, dim=0)`` recovers the
    three projections.
    """
    m = _QWEN_VL_VISUAL_QKV_RE.match(hf_name)
    if m is None:
        return None
    prefix, suffix = m.group(1), m.group(2)
    q, k, v = tensor.chunk(3, dim=0)
    return [
        (f"{prefix}q_proj{suffix}", q),
        (f"{prefix}k_proj{suffix}", k),
        (f"{prefix}v_proj{suffix}", v),
    ]


_QWEN3_5_VL_VISUAL_QKV_RE = re.compile(
    r"^(visual\.blocks\.\d+\.attn\.)qkv(\.(?:weight|bias))$"
)


def _apply_qwen3_5_vl_fixups(
    hf_name: str, tensor: torch.Tensor
) -> tuple[str, torch.Tensor]:
    """Align bridge output to SGLang memory names/dtypes for Qwen3.5-VL.

    Five independent normalizations, applied in this order:

    1. Strip ``model.language_model.`` → ``model.`` (SGLang omits this level;
       ``qwen3_5.py`` mounts the language backbone directly under ``model``).
    2. Strip ``model.visual.`` → ``visual.`` (SGLang has no top-level
       ``model.`` prefix on the visual tower).
    3. Strip ``.self_attn.`` mid-level from ``model.layers.N.self_attn.<p>``
       → ``model.layers.N.<p>``. Unlike most HF-family SGLang models,
       ``Qwen3_5AttentionDecoderLayer`` (``qwen3_5.py:721``) hangs
       ``qkv_proj`` / ``o_proj`` / ``q_norm`` / ``k_norm`` directly on the
       decoder layer with no ``self_attn`` submodule wrapper. Linear-attn
       layers still use a ``linear_attn`` submodule, so leave those alone.
    4. Rename visual ``.attn.qkv.`` → ``.attn.qkv_proj.``. Both sides keep
       QKV fused for the visual tower (no GQA, Q=K=V), so this is a pure
       rename — no ``chunk(3)`` split needed, unlike Qwen2/2.5-VL.
    5. Cast ``.linear_attn.A_log`` to fp32. SGLang stores this SSM state
       param as fp32 for numerical stability of the ``exp(A * dt)``
       recurrence; mcore currently stores it as bf16 (should be fixed
       upstream — see megatron/mcore Qwen3-Next impl), so the transport
       dtype must be upcast to match the receiver buffer. This does NOT
       recover precision lost from bf16 training; it only aligns bytes.

    ``dt_bias`` is bf16 on both sides — no cast needed.
    """
    if hf_name.startswith("model.language_model."):
        hf_name = "model." + hf_name[len("model.language_model.") :]
    elif hf_name.startswith("model.visual."):
        hf_name = hf_name[len("model.") :]  # → "visual...."

    if hf_name.startswith("model.layers.") and ".self_attn." in hf_name:
        hf_name = hf_name.replace(".self_attn.", ".", 1)

    m = _QWEN3_5_VL_VISUAL_QKV_RE.match(hf_name)
    if m is not None:
        hf_name = f"{m.group(1)}qkv_proj{m.group(2)}"

    if hf_name.endswith(".linear_attn.A_log"):
        tensor = tensor.float()

    return hf_name, tensor


class AwexMegatronAdapter(AwexTrainingAdapter):
    """Awex training adapter for MegatronEngine supporting DP, TP, and PP.

    PP: get_named_parameters already yields only the current stage's layers
    (with globally-correct HF layer indices via get_transformer_layer_offset),
    so each rank naturally reports and sends only its own subset of parameters.
    The gateway's _merge_training_meta_by_name unions disjoint PP stage params
    by name, so the full model is covered across all PP ranks.

    TP: all_gather_param gathers the full tensor on every TP rank before
    convert_to_hf. dp_replicated=True tells awex that TP ranks within a DP
    group hold identical full tensors and only one needs to send.
    """

    def __init__(self, engine: MegatronEngine):
        self._engine = engine
        self._transfer_plan: TransferPlan | None = None
        self._weights_update_group = None
        self._transfer_rank: int | None = None
        self._offloaded_optimizer_states: dict = {}
        self._offloaded_weights: dict[str, torch.Tensor] = {}
        self._released_tags: set[str] = set()
        self._colocate_lock = threading.Lock()
        self._colocate_admin_api_key: str = "areal-admin-key"
        self._colocate_http_client: httpx.Client | None = None
        self._colocate_timeout_s: float = 120.0

    @property
    def parallelism_strategy(self) -> dict:
        from megatron.core import parallel_state as mpu

        tp_size = mpu.get_tensor_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        return {
            "world_size": self._engine.world_size,
            "tp_size": tp_size,
            "pp_size": mpu.get_pipeline_model_parallel_world_size(),
            "dp_size": self._engine.data_parallel_world_size,
            "ep_size": mpu.get_expert_model_parallel_world_size(),
            "dp_replicated": tp_size > 1 or cp_size > 1,
        }

    def get_weight_metadata(self) -> list[ParameterMeta]:
        rank_info = self._build_rank_info()
        metadata: list[ParameterMeta] = []

        for hf_name, tensor in self._iter_hf_params():
            shape = tuple(tensor.shape)
            numel = int(tensor.numel())
            shard_meta = ParameterShardMeta(
                tp_rank=rank_info.tp_rank,
                attn_tp_rank=rank_info.attn_tp_rank,
                pp_rank=rank_info.pp_rank,
                ep_rank=rank_info.ep_rank,
                ep_tp_rank=rank_info.ep_tp_rank,
                global_rank=rank_info.global_rank,
                world_size=rank_info.world_size,
                engine_rank=rank_info.engine_rank,
                cp_rank=rank_info.cp_rank,
                cp_size=rank_info.cp_size,
                cp_mode=rank_info.cp_mode,
                name=hf_name,
                shape=shape,
                numel=numel,
                dtype=tensor.dtype,
                global_offset=tuple([0] * len(shape)),
                sharding_type=ShardingType.NO_SHARDING,
                num_shards=1,
                sharding_dim=0,
            )
            replica = ParameterReplicaMeta(shards=[shard_meta])
            metadata.append(
                ParameterMeta(
                    name=hf_name,
                    global_numel=numel,
                    global_shape=shape,
                    dtype=tensor.dtype,
                    shards=[shard_meta],
                    replicas=[replica],
                )
            )

        return metadata

    def get_local_shard_parameters(
        self, required_names: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        required = set(required_names) if required_names else None
        result: dict[str, torch.Tensor] = {}
        for hf_name, tensor in self._iter_hf_params():
            if required is not None and hf_name not in required:
                continue
            result[hf_name] = tensor
        return result

    def save_parameters(self, save_path: str, names: list[str] | None = None) -> None:
        weights_offloaded = "weights" in self._released_tags
        if weights_offloaded:
            self.resume_memory(tags=["weights"])
        try:
            params = self.get_local_shard_parameters(names)
            cpu_params = {k: v.detach().cpu().clone() for k, v in params.items()}
            torch.save(cpu_params, save_path)
        finally:
            if weights_offloaded:
                self.release_memory(tags=["weights"])

    def init_weight_update_group(
        self,
        pair_name: str,
        master_addr: str,
        master_port: int,
        transfer_rank: int,
        world_size: int,
        kv_store_url: str,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
    ) -> None:
        self._transfer_rank = transfer_rank

        infer_meta, train_meta = fetch_kv_metadata(kv_store_url, pair_name)

        builder = TransferPlanBuilder(
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            num_infer_engines=num_engines,
        )
        self._transfer_plan = builder.build_local_transfer_plan(
            infer_meta, train_meta, global_transfer_rank=transfer_rank
        )

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        self._weights_update_group = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=transfer_rank,
            world_size=world_size,
            group_name=f"awex_{pair_name}",
            role="training",
        )

    def execute_weight_update(self, version: int) -> None:
        del version
        if self._transfer_plan is None:
            raise RuntimeError("Transfer plan is not initialized")
        if self._weights_update_group is None:
            raise RuntimeError("Weight update group is not initialized")
        if self._transfer_rank is None:
            raise RuntimeError("Transfer rank is not initialized")

        params = self.get_local_shard_parameters()
        send_ops, _, _ = nccl_build_send_ops(
            params,
            self._transfer_plan,
            self._weights_update_group,
            copy_rank=self._transfer_rank,
        )
        batch_send_recv(
            send_ops=send_ops,
            recv_ops=[],
            blocking=True,
            use_group=awex_wu_use_group(),
        )
        dist.barrier(group=self._weights_update_group)

    def batch_isend_irecv(self, **kwargs) -> None:
        setup_kwargs = {k: v for k, v in kwargs.items() if k != "world_size"}
        setup_batch_isend_irecv(
            self._weights_update_group,
            self._transfer_rank,
            kwargs.get("world_size", 0),
            **setup_kwargs,
        )

    def teardown_weight_update_group(self) -> None:
        if self._weights_update_group is not None and dist.is_initialized():
            dist.destroy_process_group(self._weights_update_group)
        self._weights_update_group = None
        self._transfer_plan = None
        self._transfer_rank = None
        if self._colocate_http_client is not None:
            self._colocate_http_client.close()
            self._colocate_http_client = None

    def _build_rank_info(self) -> RankInfo:
        from megatron.core import parallel_state as mpu

        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        ep_size = mpu.get_expert_model_parallel_world_size()
        ep_rank = mpu.get_expert_model_parallel_rank()
        etp_size = mpu.get_expert_tensor_parallel_world_size()
        etp_rank = mpu.get_expert_tensor_parallel_rank()
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", self._engine.rank))

        return RankInfo(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            dp_size=self._engine.data_parallel_world_size,
            dp_rank=self._engine.data_parallel_rank,
            ep_rank=ep_rank,
            ep_size=ep_size,
            ep_tp_rank=etp_rank,
            ep_tp_size=etp_size,
            attn_tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_dp_rank=self._engine.data_parallel_rank,
            world_size=self._engine.world_size,
            global_rank=self._engine.rank,
            local_rank=local_rank,
            engine_rank=0,
            is_infer=False,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_mode="ring" if cp_size > 1 else "none",
        )

    def _iter_hf_params(self):
        """Yield (hf_name, tensor) for every parameter on this rank.

        Two backends:

        * ``_iter_hf_params_via_registry`` (default): mcore → HF via AReaL's
          ``convert_to_hf`` registry. Fast and self-contained, but the
          registry has no ``qwen3_5`` entry — Qwen3.5-VL falls through
          substring matching to ``convert_qwen2_to_hf`` and crashes on
          visual tower / linear-attn params.
        * ``_iter_hf_params_via_bridge``: delegate to
          ``megatron-bridge.export_hf_weights``. Bridge already knows
          Qwen3.5-VL, so this is the only viable path there. Gated on the
          same conditions as ``MegatronEngine._update_weights_via_bridge``:
          bridge_cls == 'megatron-bridge', use_bridge_for_update_weights
          opt-in, no FP8, no LoRA, bridge instance available. For Qwen3.5-VL
          this becomes effectively required — a clear error is raised when
          the required conditions aren't met.

        Both backends pass through ``_apply_awex_name_fixups`` so the HF
        names/dtypes match what SGLang's non-disk weight loaders expect
        (visual QKV split for Qwen2/2.5-VL, prefix/qkv/A_log fixups for
        Qwen3.5-VL).
        """
        model_name = self._engine.hf_config.model_type
        needs_bridge = self._engine.is_vision_model and is_qwen3_5_vl_model(
            model_name
        )
        if self._should_use_bridge():
            source = self._iter_hf_params_via_bridge()
        elif needs_bridge:
            raise RuntimeError(
                f"Model type {model_name!r} requires megatron-bridge for awex "
                "weight update (no entry in _CONVERSION_FN_REGISTRY, and the "
                "substring fallback lands on convert_qwen2_to_hf which does "
                "not know the visual tower or linear-attn params). Set "
                "megatron.bridge_type='megatron-bridge' and "
                "megatron.use_bridge_for_update_weights=True in your training "
                "config."
            )
        else:
            source = self._iter_hf_params_via_registry()
        yield from self._apply_awex_name_fixups(source)

    def _should_use_bridge(self) -> bool:
        """Mirror ``MegatronEngine._update_weights_from_distributed``'s
        bridge dispatch. Bridge is only viable when all of: bridge_cls is
        megatron-bridge, opt-in flag is on, no FP8/LoRA (bridge doesn't
        cover those yet), and a bridge instance exists.
        """
        engine = self._engine
        if getattr(engine, "bridge_cls", None) != "megatron-bridge":
            return False
        mcore_config = getattr(engine, "mcore_config", None)
        if mcore_config is None:
            return False
        if not getattr(mcore_config, "use_bridge_for_update_weights", False):
            return False
        if getattr(engine, "quantization_config", None):
            return False
        if getattr(engine.config, "use_lora", False):
            return False
        if getattr(engine, "bridge", None) is None:
            return False
        return True

    def _iter_hf_params_via_bridge(self):
        """Stream (hf_name, hf_tensor) from megatron-bridge.export_hf_weights.

        The bridge handles TP/EP/PP gather and HF layout transformation
        internally, so no ``all_gather_param`` / ``convert_to_hf`` is
        needed here. MoE expert weights come out inline (single pass, no
        second expert loop).

        Note: with PP>1 each rank sees only its own stage's params, and
        the awex training-side metadata contract has each rank report its
        own shards — matches how ``_iter_hf_params_via_registry`` behaves
        under PP.
        """
        tie_word_embeddings = getattr(
            self._engine.hf_config, "tie_word_embeddings", False
        )
        for hf_name, hf_tensor in self._engine.bridge.export_hf_weights(
            self._engine.model,
            cpu=False,
            show_progress=False,
        ):
            if tie_word_embeddings and hf_name == "lm_head.weight":
                continue
            yield hf_name, hf_tensor.detach().contiguous()

    def _iter_hf_params_via_registry(self):
        """mcore → HF via ``convert_to_hf`` registry. Fast path for models
        that have a registry entry; Qwen3.5-VL doesn't and must go through
        bridge.
        """
        from areal.engine.megatron_utils.megatron import (
            all_gather_param,
            convert_to_hf,
            get_named_parameters,
        )

        num_moe_experts = getattr(self._engine.tf_config, "num_moe_experts", None)
        model_name = self._engine.hf_config.model_type
        tie_word_embeddings = getattr(
            self._engine.hf_config, "tie_word_embeddings", False
        )

        for mcore_name, param in get_named_parameters(
            self._engine.model, num_moe_experts
        ):
            gathered = all_gather_param(
                mcore_name,
                param,
                fp8_direct_convert=False,
                quantization_config=None,
                duplicated_param_names=self._engine._duplicated_param_names,
            )
            if not isinstance(gathered, torch.Tensor):
                gathered = gathered.data

            for hf_name, tensor in convert_to_hf(
                self._engine.tf_config,
                model_name,
                mcore_name,
                gathered,
            ):
                if tie_word_embeddings and hf_name == "lm_head.weight":
                    continue
                yield hf_name, tensor.detach()

    def _apply_awex_name_fixups(self, source):
        """Adapt bridge/registry output to what SGLang stores in memory.

        - Qwen3.5-VL: prefix strip, visual qkv rename, A_log fp32 cast.
        - Qwen2/2.5-VL: chunk-3 split of fused visual qkv into q/k/v_proj.
        - Other models: pass-through.
        """
        model_name = self._engine.hf_config.model_type
        split_visual_qkv = self._engine.is_vision_model and is_qwen_vl_model(
            model_name
        )
        apply_qwen3_5_vl_fixups = (
            self._engine.is_vision_model and is_qwen3_5_vl_model(model_name)
        )
        for hf_name, tensor in source:
            if apply_qwen3_5_vl_fixups:
                hf_name, tensor = _apply_qwen3_5_vl_fixups(hf_name, tensor)
                yield hf_name, tensor
                continue
            split = (
                _split_qwen_vl_visual_qkv(hf_name, tensor)
                if split_visual_qkv
                else None
            )
            if split is None:
                yield hf_name, tensor
            else:
                yield from split

    # ── Colocated weight transfer methods ─────────────────────────────────

    def init_colocate_weight_update(
        self,
        pair_name: str,
        kv_store_url: str,
        transfer_rank: int,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
        master_port: int,
        admin_api_key: str = "areal-admin-key",
        timeout_s: float = 120.0,
    ) -> None:
        self._colocate_pair_name = pair_name
        self._colocate_kv_store_url = kv_store_url
        self._colocate_transfer_rank = transfer_rank
        self._colocate_infer_world_size = infer_world_size
        self._colocate_admin_api_key = admin_api_key
        self._colocate_timeout_s = timeout_s
        if self._colocate_http_client is None:
            self._colocate_http_client = httpx.Client()
        logger.info(
            "Initialized colocate weight update for pair '%s', transfer_rank=%d",
            pair_name,
            transfer_rank,
        )

    def execute_colocate_weight_update(self, version: int) -> None:
        with self._colocate_lock:
            self._execute_colocate_weight_update_locked(version)

    def _execute_colocate_weight_update_locked(self, version: int) -> None:
        kv_store_url = self._colocate_kv_store_url
        pair_name = self._colocate_pair_name
        transfer_rank = self._colocate_transfer_rank
        assert self._colocate_http_client is not None, (
            "init_colocate_weight_update must be called first"
        )
        client = self._colocate_http_client
        auth_headers = {"Authorization": f"Bearer {self._colocate_admin_api_key}"}
        timeout_s = self._colocate_timeout_s

        weights_offloaded = "weights" in self._released_tags
        if weights_offloaded:
            self.resume_memory(tags=["weights"])

        params = self.get_local_shard_parameters()
        tensors = list(params.values())
        names = list(params.keys())

        group_tensors, metadata = group_tensors_by_shape_and_dtype(tensors)
        torch.cuda.synchronize()

        del tensors

        group_shared = [t.share_memory_() for t in group_tensors]
        serialized_weights = cuda_ipc_serialize((group_shared, metadata, names))
        torch.cuda.synchronize()

        kv_key = f"colocate_weights_rank{transfer_rank}_{version}"

        client.put(
            f"{kv_store_url}/weight_meta/{pair_name}/{kv_key}",
            json={"value": serialized_weights.hex()},
            headers=auth_headers,
            timeout=timeout_s,
        )

        logger.info(
            "Serialized %d params (%d groups) for colocate transfer v%d, rank %d",
            len(names),
            len(group_shared),
            version,
            transfer_rank,
        )

        done_key = f"colocate_done_rank{transfer_rank}_{version}"
        deadline = time.monotonic() + timeout_s
        poll_count = 0
        last_status = -1
        while time.monotonic() < deadline:
            resp = client.get(
                f"{kv_store_url}/weight_meta/{pair_name}/{done_key}",
                timeout=5.0,
            )
            last_status = resp.status_code
            if resp.status_code == 200:
                break
            poll_count += 1
            time.sleep(0.1)
        else:
            raise TimeoutError(
                f"Inference did not signal completion within {timeout_s}s "
                f"(waiting_key={done_key}, put_key={kv_key}, "
                f"polls={poll_count}, last_status={last_status})"
            )

        del group_shared, group_tensors, serialized_weights
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        if weights_offloaded:
            self.release_memory(tags=["weights"])

    def release_memory(self, tags: list[str] | None = None) -> None:
        """Release GPU memory for specified tags by offloading to CPU.

        Supported tags:
            - "optimizer": Offload optimizer state tensors (exp_avg, exp_avg_sq, etc.)
            - "weights": Offload model parameters
        """
        tags = tags or ["optimizer", "weights"]
        tags_to_release = [t for t in tags if t not in self._released_tags]
        if not tags_to_release:
            logger.info("release_memory: tags=%s already released, skipping", tags)
            return

        logger.info("release_memory: offloading tags=%s", tags_to_release)

        if "optimizer" in tags_to_release:
            self._offload_optimizer_states()
            self._released_tags.add("optimizer")

        if "weights" in tags_to_release:
            self._offload_model_weights()
            self._released_tags.add("weights")

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("release_memory: done for tags=%s", tags_to_release)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        """Resume GPU memory for specified tags by reloading from CPU.

        Supported tags:
            - "optimizer": Reload optimizer state tensors to GPU
            - "weights": Reload model parameters to GPU
        """
        tags = tags or ["optimizer", "weights"]
        tags_to_resume = [t for t in tags if t in self._released_tags]
        if not tags_to_resume:
            logger.info("resume_memory: tags=%s not released, skipping", tags)
            return

        logger.info("resume_memory: reloading tags=%s", tags_to_resume)

        if "weights" in tags_to_resume:
            self._reload_model_weights()
            self._released_tags.discard("weights")

        if "optimizer" in tags_to_resume:
            self._reload_optimizer_states()
            self._released_tags.discard("optimizer")

        torch.cuda.synchronize()
        logger.info("resume_memory: done for tags=%s", tags_to_resume)

    def _offload_optimizer_states(self) -> None:
        """Move optimizer state tensors to CPU, keeping references for reload."""
        optimizer = self._engine.optimizer
        if optimizer is None:
            logger.warning("No optimizer found, skipping optimizer offload")
            return

        # Megatron's ChainedOptimizer wraps per-model-chunk optimizers;
        # each in turn wraps a base torch optimizer holding the state dict.
        if hasattr(optimizer, "optimizers"):
            inner_optimizers = optimizer.optimizers
        else:
            inner_optimizers = [optimizer]
            logger.warning(
                "Optimizer does not have 'optimizers' attribute. "
                "Treating it as a single optimizer; offload may be incomplete "
                "for non-standard Megatron optimizer structures."
            )
        for opt in inner_optimizers:
            base_opt = getattr(opt, "optimizer", opt)
            for param, state in base_opt.state.items():
                cpu_state: dict[str, torch.Tensor] = {}
                for key, val in state.items():
                    if isinstance(val, torch.Tensor) and val.is_cuda:
                        cpu_state[key] = val.detach().to("cpu", non_blocking=True)
                        state[key] = torch.empty(0, device="cpu")
                if cpu_state:
                    self._offloaded_optimizer_states[param] = cpu_state

        logger.info(
            "Offloaded optimizer states for %d params",
            len(self._offloaded_optimizer_states),
        )

    def _reload_optimizer_states(self) -> None:
        """Restore optimizer state tensors from CPU back to GPU."""
        if not self._offloaded_optimizer_states:
            return

        optimizer = self._engine.optimizer
        if optimizer is None:
            return

        inner_optimizers = getattr(optimizer, "optimizers", [optimizer])
        for opt in inner_optimizers:
            base_opt = getattr(opt, "optimizer", opt)
            for param, state in base_opt.state.items():
                if param in self._offloaded_optimizer_states:
                    cpu_state = self._offloaded_optimizer_states[param]
                    for key, val in cpu_state.items():
                        state[key] = val.to(param.device, non_blocking=True)

        self._offloaded_optimizer_states.clear()
        logger.info("Reloaded optimizer states to GPU")

    def _offload_model_weights(self) -> None:
        """Move model parameters to CPU, keeping references for reload."""
        if self._engine.model is None:
            return

        for name, param in self._engine.model.named_parameters():
            if param.is_cuda:
                self._offloaded_weights[name] = param.data.detach().to(
                    "cpu", non_blocking=True
                )
                param.data = torch.empty(0, device="cpu")

        logger.info(
            "Offloaded %d model weight tensors to CPU",
            len(self._offloaded_weights),
        )

    def _reload_model_weights(self) -> None:
        """Restore model parameters from CPU back to GPU."""
        if not self._offloaded_weights:
            return
        if self._engine.model is None:
            return

        device = self._engine.device
        for name, param in self._engine.model.named_parameters():
            if name in self._offloaded_weights:
                param.data = self._offloaded_weights[name].to(device, non_blocking=True)

        self._offloaded_weights.clear()
        logger.info("Reloaded model weights to GPU")
