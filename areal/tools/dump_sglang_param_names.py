#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dump the (name, shape, dtype) triples that SGLang instantiates locally for
a given HF checkpoint's architecture -- WITHOUT booting an SGLang scheduler,
launching any HTTP server, or loading real weights.

Symmetric to ``areal.tools.dump_bridge_hf_names``: that tool answers "what
names does the training side (megatron-bridge) emit?", this tool answers
"what names does the inference side (SGLang) actually store?" for the same
checkpoint. Diff the two dumps to build the exact fixup map awex needs
(fuse/split/rename) BEFORE running a real weight-update round-trip.

Usage:

    # every param SGLang would keep on rank 0 of a TP=PP=1 setup
    uv run python -m areal.tools.dump_sglang_param_names /path/to/hf/model

    # filter to just the linear-attention params
    uv run python -m areal.tools.dump_sglang_param_names /path/to/qwen3-next \\
        --pattern 'linear_attn|in_proj|conv1d'

    # dump JSON, e.g. for feeding into a diff script
    uv run python -m areal.tools.dump_sglang_param_names /path/to/model \\
        --json /tmp/sglang_names.json

Runs single-process gloo, TP=PP=1. Model is instantiated on ``torch.device
("meta")`` so no weights are ever allocated -- just walks the parameter
tree that SGLang's model class registers. No GPU required.
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
    if os.path.isdir(user_path):
        return user_path
    cached = os.path.join("/storage/openpsi/models", user_path.replace("/", "__"))
    if os.path.isdir(cached):
        return cached
    return user_path


def _init_single_process() -> None:
    """Bring up gloo world_size=1 + SGLang's parallel_state (TP=PP=1).

    Enough for SGLang's custom Linear layers (ColumnParallelLinear, etc.) to
    query TP rank/size at ``__init__`` time without deadlocking.
    """
    from areal.utils.network import find_free_ports
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(find_free_ports(1)[0]))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        local_rank=0,
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)


def _teardown() -> None:
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    try:
        destroy_model_parallel()
    finally:
        destroy_distributed_environment()
        if dist.is_initialized():
            dist.destroy_process_group()


def _force_sglang_subconfigs(config):
    """Rewire ``config.text_config`` / ``config.vision_config`` (etc.) so
    they are instances of the SGLang-registered subclasses, not the stock
    ``transformers`` variants.

    Why: ``sglang.srt.utils.hf_transformers.config.get_config`` reloads the
    top-level config through SGLang's ``_CONFIG_REGISTRY``, but nested
    sub-configs are still populated by HF's ``PretrainedConfig.from_dict``,
    which resolves them via ``AutoConfig`` and often picks HF's own class
    (e.g. transformers' Qwen3_5TextConfig). SGLang model code then accesses
    properties like ``layers_block_type`` that only exist on SGLang's
    subclass, and blows up with AttributeError. Round-tripping through
    ``to_dict()`` + SGLang class re-init fixes that.
    """
    try:
        from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY
    except ImportError:
        # Older layout; skip the rewire — likely nothing to fix on that
        # version either.
        return config
    for attr_name in ("text_config", "vision_config", "audio_config"):
        sub = getattr(config, attr_name, None)
        if sub is None:
            continue
        sub_model_type = getattr(sub, "model_type", None)
        if sub_model_type is None:
            continue
        sglang_cls = _CONFIG_REGISTRY.get(sub_model_type)
        if sglang_cls is None or isinstance(sub, sglang_cls):
            continue
        sub_dict = sub.to_dict()
        # Drop transformers-only keys that would trip strict __init__.
        sub_dict.pop("_name_or_path", None)
        setattr(config, attr_name, sglang_cls(**sub_dict))
        print(
            f"[dump_sglang_param_names] rewired {attr_name}: "
            f"{type(sub).__module__}.{type(sub).__name__} -> "
            f"{sglang_cls.__module__}.{sglang_cls.__name__}",
            file=sys.stderr,
        )
    return config


def _seed_global_server_args(model_path: str):
    """Seed the SGLang ``_global_server_args`` singleton and return it.

    Some model classes (e.g. Qwen3-VL family) read ``get_global_server_args().
    mm_enable_dp_encoder`` at ``__init__`` time. In production this is set by
    the scheduler; here we set a defaults-only stub so the read succeeds.
    """
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    server_args = ServerArgs(model_path=model_path, trust_remote_code=True)
    set_global_server_args_for_scheduler(server_args)
    return server_args


def _seed_dp_attention(server_args, model_path: str) -> None:
    """Seed SGLang's dp_attention globals (``_ATTN_DP_SIZE`` etc.).

    ``LayerCommunicator.__init__`` (invoked by any decoder layer in the
    Qwen3-VL / Qwen3.5 family) asserts these are set. In production the
    scheduler calls ``initialize_dp_attention`` right after parallel state
    setup; we do the same here with defaults-only ServerArgs +
    ModelConfig, which lands in the ``enable_dp_attention=False`` branch
    and sets ``_ATTN_DP_SIZE=1``.
    """
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.dp_attention import initialize_dp_attention

    model_config = ModelConfig.from_server_args(server_args, model_path=model_path)
    initialize_dp_attention(server_args, model_config)


def _instantiate_model_on_meta(model_path: str, device: str = "meta"):
    """Resolve the SGLang model class for ``model_path`` and materialize its
    parameter tree on ``torch.device(device)`` (default 'meta').
    """
    from sglang.srt.models.registry import ModelRegistry

    # SGLang's get_config re-dispatches model_type through its own
    # _CONFIG_REGISTRY so classes like Qwen3_5TextConfig become SGLang's
    # variant (inherits Qwen3NextConfig, carries the ``layers_block_type``
    # property etc.) rather than transformers' stock class, which has only
    # ``layer_types`` and would trip AttributeError on load.
    # Import location differs across SGLang versions.
    try:
        from sglang.srt.utils.hf_transformers.config import get_config
    except ImportError:
        from sglang.srt.utils.hf_transformers_utils import get_config

    server_args = _seed_global_server_args(model_path)
    _seed_dp_attention(server_args, model_path)
    hf_config = get_config(model_path, trust_remote_code=True)
    hf_config = _force_sglang_subconfigs(hf_config)
    architectures = getattr(hf_config, "architectures", None) or []
    if not architectures:
        raise ValueError(
            f"HF config at {model_path} has no 'architectures' field; cannot "
            "resolve an SGLang model class."
        )
    model_cls, matched_arch = ModelRegistry.resolve_model_cls(architectures)
    print(
        f"[dump_sglang_param_names] arch={matched_arch} -> "
        f"{model_cls.__module__}.{model_cls.__name__}",
        file=sys.stderr,
    )

    with torch.device(device):
        model = model_cls(config=hf_config, quant_config=None, prefix="")
    return model, matched_arch


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_path", help="HF model directory or HF hub id.")
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
        "--device",
        choices=("meta", "cpu"),
        default="meta",
        help=(
            "Where to allocate the parameter tree. Default 'meta' allocates "
            "no memory; switch to 'cpu' if the model class does something "
            "meta-incompatible at __init__ (e.g. Mamba conv init)."
        ),
    )
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model_path)
    regex = re.compile(args.pattern) if args.pattern else None

    print(
        f"[dump_sglang_param_names] model={model_path}",
        file=sys.stderr,
    )

    _init_single_process()
    try:
        model, arch = _instantiate_model_on_meta(model_path, device=args.device)
        rows: list[dict] = []
        for name, param in model.named_parameters():
            if regex is not None and not regex.search(name):
                continue
            rows.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                }
            )
    finally:
        _teardown()

    for r in rows:
        print(f"  {r['name']:70s} shape={tuple(r['shape'])} dtype={r['dtype']}")
    print(
        f"[dump_sglang_param_names] {len(rows)} names emitted (arch={arch})",
        file=sys.stderr,
    )

    if args.json_path:
        report = {
            "model_path": model_path,
            "arch": arch,
            "pattern": args.pattern,
            "entries": rows,
        }
        with open(args.json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(
            f"[dump_sglang_param_names] json written to {args.json_path}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
