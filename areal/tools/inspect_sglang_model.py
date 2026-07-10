#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Boot a real single-process SGLang ``ModelRunner`` (weights actually loaded
onto GPU) and drop into an interactive REPL with the model instance available.

Contrast with ``dump_sglang_param_names``: that tool instantiates on meta
device just to inspect the param tree; this one gives you a **live**, fully-
loaded model you can walk with ``named_modules()``, run forward passes on,
or attach a debugger to. Use when the meta-device dump isn't enough —
e.g. verifying which submodules exist for a given arch, checking runtime
attention backend selection, or eyeballing dtypes after quant.

Usage:

    uv run python -m areal.tools.inspect_sglang_model /path/to/hf/model

    # smaller footprint (less KV cache) for faster boot on small GPUs
    uv run python -m areal.tools.inspect_sglang_model /path/to/model \\
        --mem-fraction-static 0.1

    # non-interactive: just print module tree and exit
    uv run python -m areal.tools.inspect_sglang_model /path/to/model --no-repl

Inside the REPL you have: ``model_runner``, ``model``, ``server_args``,
``model_config``. Try::

    for name, mod in model.named_modules():
        print(name, type(mod).__name__)

    for name, p in model.named_parameters():
        print(name, tuple(p.shape), p.dtype)
"""

from __future__ import annotations

import argparse
import code
import os
import sys

import torch


def _resolve_model_path(user_path: str) -> str:
    if os.path.isdir(user_path):
        return user_path
    cached = os.path.join("/storage/openpsi/models", user_path.replace("/", "__"))
    if os.path.isdir(cached):
        return cached
    return user_path


def _print_module_tree(model, max_depth: int = 6) -> None:
    print(f"\n=== Model: {type(model).__module__}.{type(model).__name__} ===\n")
    for name, module in model.named_modules():
        if not name:
            continue
        depth = name.count(".")
        if depth > max_depth:
            continue
        indent = "  " * depth
        leaf = name.split(".")[-1]
        cls = type(module).__name__
        # Show shape suffix for parameter-carrying leaves so the tree is
        # useful without also dumping named_parameters separately.
        param_summary = ""
        direct_params = list(module._parameters.items())
        if direct_params:
            param_summary = " [" + ", ".join(
                f"{n}: {tuple(p.shape)}"
                for n, p in direct_params
                if p is not None
            ) + "]"
        print(f"{indent}{leaf}: {cls}{param_summary}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_path", help="HF model directory or HF hub id.")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.5,
        help=(
            "Fraction of free GPU memory the KV cache pool takes. Lower "
            "(e.g. 0.1) speeds up init on small GPUs; higher matches "
            "production."
        ),
    )
    parser.add_argument(
        "--no-repl",
        action="store_true",
        help="Just print the module tree and exit; skip the interactive prompt.",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=6,
        help="Prune the printed module tree at this depth. Default 6.",
    )
    args = parser.parse_args()

    from areal.utils.network import find_free_ports
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    model_path = _resolve_model_path(args.model_path)
    # ModelRunner does the full distributed init itself (init_distributed_environment
    # -> initialize_model_parallel -> initialize_dp_attention, see
    # model_runner.py:1234-1258). We only need to hand it a free port for the
    # TCPStore.
    nccl_port = find_free_ports(1)[0]

    server_args = ServerArgs(
        model_path=model_path,
        trust_remote_code=True,
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
    )
    set_global_server_args_for_scheduler(server_args)

    model_config = ModelConfig.from_server_args(server_args, model_path=model_path)

    print(
        f"[inspect_sglang_model] booting ModelRunner (tp={args.tp_size}, "
        f"loads real weights on GPU) model={model_path}",
        file=sys.stderr,
    )
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=args.mem_fraction_static,
        gpu_id=args.gpu_id,
        tp_rank=0,
        tp_size=args.tp_size,
        moe_ep_rank=0,
        moe_ep_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=nccl_port,
        server_args=server_args,
    )
    print("[inspect_sglang_model] ModelRunner ready.", file=sys.stderr)

    model = model_runner.model
    _print_module_tree(model, max_depth=args.max_tree_depth)

    if args.no_repl:
        return 0

    banner = (
        "\n=== SGLang model loaded. Locals: "
        "model_runner, model, server_args, model_config, torch ===\n"
        "Try: [n for n, _ in model.named_parameters()][:10]\n"
        "     dict(model.named_modules())['visual.blocks.0.attn']\n"
        "Ctrl-D to exit."
    )
    # Prefer IPython if available, fall back to stdlib `code`.
    try:
        from IPython import embed  # type: ignore

        embed(
            header=banner,
            user_ns={
                "model_runner": model_runner,
                "model": model,
                "server_args": server_args,
                "model_config": model_config,
                "torch": torch,
            },
        )
    except ImportError:
        code.interact(
            banner=banner,
            local={
                "model_runner": model_runner,
                "model": model,
                "server_args": server_args,
                "model_config": model_config,
                "torch": torch,
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
