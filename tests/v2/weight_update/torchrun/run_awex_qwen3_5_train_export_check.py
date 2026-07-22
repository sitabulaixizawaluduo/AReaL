# SPDX-License-Identifier: Apache-2.0
"""Single-side GPU check for the Qwen3.5-MoE awex Megatron bridge export.

Runs the awex train adapter WITHOUT any inference side and validates:

1. the union of ``get_weight_metadata`` names across all ranks equals the
   name set derived from the checkpoint (ownership tiles the model);
2. expert params are claimed by exactly one (pp, ep) coordinate;
3. every rank's ``get_local_shard_parameters`` tensors match the checkpoint
   bitwise (bridge export produced correct HF values).

Usage (GPU node; adjust nproc/parallelism as needed):
    python tests/make_tiny_qwen3_5_moe.py --output /tmp/qwen3_5_moe_tiny
    torchrun --nproc_per_node=4 \
        tests/v2/weight_update/torchrun/run_awex_qwen3_5_train_export_check.py \
        --model-path /tmp/qwen3_5_moe_tiny --backend megatron:d1t2p2

Expected: rank 0 prints "ALL CHECKS PASSED".
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch
import torch.distributed as dist

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine.megatron_engine import MegatronEngine
from areal.v2.weight_update.awex.megatron_adapter import AwexMegatronAdapter
from areal.v2.weight_update.awex.qwen3_5 import (
    normalize_train_hf_name,
    split_train_hf_param,
)


def load_checkpoint_common(model_path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    common: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        for raw_name, tensor in load_file(shard).items():
            name = normalize_train_hf_name(raw_name)
            if name is None:
                continue
            for out_name, out in split_train_hf_param(name, tensor, hf_config):
                common[out_name] = out
    if getattr(hf_config, "tie_word_embeddings", False):
        common.pop("lm_head.weight", None)
    return common


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backend", default="megatron:d1")
    args = parser.parse_args()

    config = TrainEngineConfig(
        backend=args.backend,
        experiment_name="awex-qwen35-export-check",
        trial_name="t0",
        path=args.model_path,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=256),
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(
            bridge_type="megatron-bridge",
            use_bridge_for_update_weights=True,
        ),
    )
    alloc_mode = ModelAllocation.from_str(args.backend)
    engine = MegatronEngine(config)
    engine.create_process_group(parallel_strategy=alloc_mode.parallel)
    engine.initialize(
        addr=None,
        ft_spec=FinetuneSpec(
            total_train_epochs=1, dataset_size=128, train_batch_size=8
        ),
    )
    rank = dist.get_rank()

    adapter = AwexMegatronAdapter(engine)
    assert adapter._use_bridge_export(), (
        "bridge export path not active; check bridge_type / "
        "use_bridge_for_update_weights / model_type"
    )

    meta = adapter.get_weight_metadata()
    local_names = sorted(m.name for m in meta)
    print(f"[rank {rank}] reports {len(local_names)} params")

    gathered: list[list[str]] = [None] * dist.get_world_size()  # type: ignore[list-item]
    dist.all_gather_object(gathered, local_names)

    failures = 0
    if rank == 0:
        expected = load_checkpoint_common(args.model_path)
        union: set[str] = set()
        for names in gathered:
            union.update(names)
        missing = sorted(set(expected) - union)
        extra = sorted(union - set(expected))
        if missing or extra:
            failures += 1
            print(f"[rank 0] FAIL names: missing={missing[:10]} extra={extra[:10]}")
        else:
            print(f"[rank 0] ownership union OK ({len(union)} params)")

        expert_claims: dict[str, int] = {}
        for names in gathered:
            for name in names:
                if ".mlp.experts." in name:
                    expert_claims[name] = expert_claims.get(name, 0) + 1
        from megatron.core import parallel_state as mpu

        replicas_per_expert = dist.get_world_size() // (
            mpu.get_expert_model_parallel_world_size()
            * mpu.get_pipeline_model_parallel_world_size()
        )
        bad = {n: c for n, c in expert_claims.items() if c != replicas_per_expert}
        if bad:
            failures += 1
            print(f"[rank 0] FAIL expert ownership counts: {list(bad.items())[:5]}")
        else:
            print(
                f"[rank 0] expert ownership OK "
                f"({replicas_per_expert} replica claims per expert)"
            )

    expected_local = load_checkpoint_common(args.model_path)
    params = adapter.get_local_shard_parameters()
    for name, tensor in params.items():
        ref = expected_local[name].to(tensor.dtype).to(tensor.device)
        if not torch.equal(tensor, ref):
            failures += 1
            print(f"[rank {rank}] FAIL value mismatch: {name}")
    print(f"[rank {rank}] value check done ({len(params)} params)")

    fail_tensor = torch.tensor(
        [failures], dtype=torch.int64, device=torch.cuda.current_device()
    )
    dist.all_reduce(fail_tensor)
    if rank == 0:
        if int(fail_tensor.item()) == 0:
            print("ALL CHECKS PASSED")
        else:
            print(f"FAILED with {int(fail_tensor.item())} errors")
    dist.barrier()
    return 0 if int(fail_tensor.item()) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
