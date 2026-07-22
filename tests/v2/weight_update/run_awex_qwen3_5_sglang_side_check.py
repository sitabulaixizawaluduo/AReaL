# SPDX-License-Identifier: Apache-2.0
"""Single-side GPU check for the Qwen3.5-MoE awex SGLang integration.

Launches AReaL's v2 SGLang server on a Qwen3.5-MoE checkpoint, then verifies
WITHOUT any training side:

1. /awex/report_weight_meta name set == names derived from the checkpoint via
   the train-side split rules (name-protocol agreement);
2. the rank-0 parameter dump matches the checkpoint slice declared by
   ``Qwen3_5MoeShardingStrategy`` (unfuse layout vs sharding declaration);
3. reports every meta dtype that differs from the checkpoint dtype (these are
   the entries the train adapter will cast before sending, e.g. fp32 A_log).

Usage (GPU node):
    python tests/make_tiny_qwen3_5_moe.py --output /tmp/qwen3_5_moe_tiny
    python tests/v2/weight_update/run_awex_qwen3_5_sglang_side_check.py \
        --model-path /tmp/qwen3_5_moe_tiny --tp 2

Expected: prints "ALL CHECKS PASSED" and exits 0.
"""

from __future__ import annotations

import argparse
import glob
import os
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import httpx
import torch

from areal.utils.network import find_free_ports
from areal.v2.weight_update.awex.qwen3_5 import (
    Qwen3_5MoeShardingStrategy,
    normalize_train_hf_name,
    split_train_hf_param,
)

SERVER_STARTUP_TIMEOUT = 600


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


def launch_server(model_path: str, tp: int) -> tuple[str, subprocess.Popen]:
    port = find_free_ports(1)[0]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "areal.v2.inference_service.sglang.launch_server",
            "--model-path",
            model_path,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--tp-size",
            str(tp),
            "--mem-fraction-static",
            "0.7",
            "--log-level",
            "warning",
        ],
        stdout=sys.stdout,
        stderr=sys.stdout,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=5.0).status_code == 200:
                return base_url, proc
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (code {proc.returncode})")
        time.sleep(2.0)
    proc.kill()
    raise RuntimeError("server did not become healthy in time")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    print(f"[check] loading checkpoint reference from {args.model_path} ...")
    expected = load_checkpoint_common(args.model_path)
    print(f"[check] checkpoint yields {len(expected)} common-name params")

    base_url, proc = launch_server(args.model_path, args.tp)
    try:
        print("[check] fetching /awex/report_weight_meta ...")
        resp = httpx.post(f"{base_url}/awex/report_weight_meta", timeout=600.0)
        resp.raise_for_status()
        meta_entries = resp.json()["meta"]

        reported: dict[str, str] = {}
        for entry in meta_entries:
            data = entry.get("data", entry)
            reported[data["name"]] = str(data.get("dtype", ""))

        missing = sorted(set(expected) - set(reported))
        extra = sorted(set(reported) - set(expected))
        if missing or extra:
            print(f"[check] FAIL name mismatch:\n  missing={missing}\n  extra={extra}")
            return 1
        print(f"[check] name protocol OK: {len(reported)} params on both sides")

        dtype_diffs = []
        for name, dtype in sorted(reported.items()):
            ckpt_dtype = str(expected[name].dtype)
            if dtype and dtype.replace("torch.", "") not in ckpt_dtype:
                dtype_diffs.append((name, ckpt_dtype, dtype))
        if dtype_diffs:
            print(
                f"[check] {len(dtype_diffs)} params hold a different runtime "
                "dtype (train side will cast before send):"
            )
            for name, ck, rt in dtype_diffs[:20]:
                print(f"    {name}: checkpoint={ck} runtime={rt}")

        dump_path = os.path.join(tempfile.mkdtemp(), "infer_params.pt")
        print("[check] dumping rank-0 parameters ...")
        resp = httpx.post(
            f"{base_url}/awex/debug/get_parameters",
            json={"save_path": dump_path},
            timeout=600.0,
        )
        resp.raise_for_status()
        dumped = torch.load(dump_path, map_location="cpu", weights_only=True)

        strategy = Qwen3_5MoeShardingStrategy(
            engine_name="sglang",
            enable_dp_attention=False,
            enable_dp_lm_head=False,
            moe_dense_tp_size=None,
            tp_size=args.tp,
            ep_size=1,
            ep_tp_size=1,
            rank_info=SimpleNamespace(tp_size=args.tp),
        )
        from awex.sharding.param_sharding import ShardingType

        mismatches = 0
        for name, local in sorted(dumped.items()):
            full = expected[name]
            stype, dim, _ = strategy.get_sharding_strategy(name)
            ref = (
                full
                if stype == ShardingType.NO_SHARDING
                else full.narrow(dim, 0, local.shape[dim])
            )
            if local.shape != ref.shape or not torch.equal(
                local.float(), ref.to(local.dtype).float()
            ):
                mismatches += 1
                print(
                    f"[check] MISMATCH {name}: local shape {list(local.shape)} "
                    f"vs ref {list(ref.shape)}"
                )
        if mismatches:
            print(f"[check] FAIL: {mismatches} rank-0 params differ from checkpoint")
            return 1
        print(f"[check] rank-0 value check OK for {len(dumped)} params")
        print("ALL CHECKS PASSED")
        return 0
    finally:
        os.kill(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
