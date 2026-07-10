#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dump the HF (name, shape, dtype) triples that mbridge / megatron-bridge
export for a given HF checkpoint, without spinning up any AReaL training loop.

Answers "which HF names does bridge actually emit for THIS model on THIS
rank?" ahead of a real weight-update round-trip -- useful when preparing
awex support for a new arch (Qwen3-Next linear attention, Qwen2.5-VL visual
tower, MoE variants, ...).

Usage:

    # dump every emitted name for a checkpoint
    uv run python -m areal.tools.dump_bridge_hf_names /path/to/hf/model

    # filter to linear-attn params only (answer "does bridge fuse in_proj_qkvz
    # or split into q/k/v/z_proj?")
    uv run python -m areal.tools.dump_bridge_hf_names /path/to/qwen3-next \\
        --pattern 'linear_attn|in_proj|conv1d'

    # switch to mbridge (default: megatron-bridge, matches
    # AwexMegatronAdapter._iter_hf_params_via_bridge)
    uv run python -m areal.tools.dump_bridge_hf_names /path/to/model --bridge mbridge

    # dump a JSON report (name/shape/dtype + optional HF-safetensors diff)
    uv run python -m areal.tools.dump_bridge_hf_names /path/to/model \\
        --json /tmp/bridge_names.json --hf-diff

Runs single-process gloo, TP=PP=1, CPU-materialized. No GPU required. Bridge
is built with the same knobs ``areal.models.mcore.registry.make_mcore_model``
uses in production, so emitted names match what awex sees on rank 0 of a
PP=1 job.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
import torch.distributed as dist


def _resolve_model_path(user_path: str) -> str:
    """Prefer /storage/openpsi/models cache if the user handed us a HF id."""
    if os.path.isdir(user_path):
        return user_path
    cached = os.path.join("/storage/openpsi/models", user_path.replace("/", "__"))
    if os.path.isdir(cached):
        return cached
    return user_path


def _init_single_process() -> None:
    from areal.utils.network import find_free_ports
    from megatron.core import parallel_state as mpu
    from megatron.core import tensor_parallel

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(find_free_ports(1)[0]))
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    mpu.initialize_model_parallel()
    tensor_parallel.model_parallel_cuda_manual_seed(0)


def _teardown() -> None:
    from megatron.core import parallel_state as mpu

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


def _build_mbridge_model(model_path: str):
    import mbridge

    bridge = mbridge.AutoBridge.from_pretrained(model_path, trust_remote_code=True)
    bridge.dtype = torch.bfloat16
    models = bridge.get_model(wrap_with_ddp=False)
    return bridge, models


def _build_megatron_bridge_model(model_path: str):
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state as mpu

    bridge = AutoBridge.from_hf_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16
    )
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
    provider.pipeline_model_parallel_size = mpu.get_pipeline_model_parallel_world_size()
    provider.virtual_pipeline_model_parallel_size = None
    provider.context_parallel_size = mpu.get_context_parallel_world_size()
    provider.expert_model_parallel_size = mpu.get_expert_model_parallel_world_size()
    provider.expert_tensor_parallel_size = mpu.get_expert_tensor_parallel_world_size()
    provider.sequence_parallel = False
    provider.pipeline_dtype = torch.bfloat16
    # Mirror registry.py defaults so emitted names match production awex.
    provider.variable_seq_lengths = True
    provider.moe_token_dispatcher_type = "alltoall"
    provider.batch_p2p_comm = False
    provider.overlap_p2p_comm = False
    provider.account_for_embedding_in_pipeline_split = False
    provider.account_for_loss_in_pipeline_split = False
    # MTP head is not used at RL update time -- drop it so provide_ succeeds.
    if getattr(provider, "mtp_num_layers", None):
        provider.mtp_num_layers = None
    provider.finalize()

    models = provider.provide_distributed_model(
        ddp_config=None,
        fp16=False,
        bf16=True,
        use_megatron_fsdp=False,
        use_torch_fsdp2=False,
        wrap_with_ddp=False,
        overlap_param_gather_with_optimizer_step=False,
    )
    return bridge, list(models)


def _load_hf_safetensors_keys(model_path: str) -> set[str] | None:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return set(json.load(f).get("weight_map", {}).keys())
    if os.path.exists(single_path):
        from safetensors import safe_open

        with safe_open(single_path, framework="pt") as f:
            return set(f.keys())
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_path", help="HF model directory or HF hub id.")
    parser.add_argument(
        "--bridge",
        choices=("megatron-bridge", "mbridge"),
        default="megatron-bridge",
        help=(
            "Which bridge to invoke. Default matches "
            "AwexMegatronAdapter._iter_hf_params_via_bridge."
        ),
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Regex; only names for which re.search matches are printed.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Optional path to dump a machine-readable report to.",
    )
    parser.add_argument(
        "--hf-diff",
        action="store_true",
        help=(
            "Also compute HF safetensors coverage diff (missing / extra) "
            "against the checkpoint at model_path. Ignored if model_path is a "
            "HF hub id without a local safetensors index."
        ),
    )
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model_path)
    regex = re.compile(args.pattern) if args.pattern else None

    print(
        f"[dump_bridge_hf_names] bridge={args.bridge} model={model_path}",
        file=sys.stderr,
    )

    _init_single_process()
    try:
        if args.bridge == "mbridge":
            bridge, models = _build_mbridge_model(model_path)
        else:
            bridge, models = _build_megatron_bridge_model(model_path)

        rows: list[dict] = []
        for hf_name, hf_tensor in bridge.export_hf_weights(
            models, cpu=True, show_progress=False
        ):
            if regex is not None and not regex.search(hf_name):
                continue
            rows.append(
                {
                    "name": hf_name,
                    "shape": list(hf_tensor.shape),
                    "dtype": str(hf_tensor.dtype),
                }
            )
    finally:
        _teardown()

    for r in rows:
        print(f"  {r['name']:70s} shape={tuple(r['shape'])} dtype={r['dtype']}")
    print(f"[dump_bridge_hf_names] {len(rows)} names emitted", file=sys.stderr)

    if args.json_path or args.hf_diff:
        report: dict = {
            "bridge": args.bridge,
            "model_path": model_path,
            "pattern": args.pattern,
            "entries": rows,
        }
        if args.hf_diff:
            hf_keys = _load_hf_safetensors_keys(model_path)
            if hf_keys is None:
                print(
                    "[dump_bridge_hf_names] --hf-diff skipped: no safetensors "
                    f"index under {model_path}",
                    file=sys.stderr,
                )
            else:
                emitted = {r["name"] for r in rows}
                report["missing_in_bridge"] = sorted(hf_keys - emitted)
                report["extra_in_bridge"] = sorted(emitted - hf_keys)
                print(
                    f"[dump_bridge_hf_names] hf_diff: "
                    f"{len(report['missing_in_bridge'])} missing, "
                    f"{len(report['extra_in_bridge'])} extra",
                    file=sys.stderr,
                )
        if args.json_path:
            with open(args.json_path, "w") as f:
                json.dump(report, f, indent=2)
            print(
                f"[dump_bridge_hf_names] json written to {args.json_path}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
