# SPDX-License-Identifier: Apache-2.0
"""Env-gated GPU memory instrumentation for leak hunting.

Enable with AREAL_MEM_DEBUG=1. Optionally set AREAL_MEM_SNAPSHOT to a
directory to record allocation history and dump a pickle readable by
https://pytorch.org/memory_viz. Zero overhead when disabled.
"""

import os

import torch

from areal.utils import logging

logger = logging.getLogger("MemDebug")

_ENABLED = os.environ.get("AREAL_MEM_DEBUG", "0") == "1"
_SNAPSHOT_DIR = os.environ.get("AREAL_MEM_SNAPSHOT", "")
_history_started = False


def mem_debug(tag: str) -> None:
    if not _ENABLED or not torch.cuda.is_available():
        return
    global _history_started
    if _SNAPSHOT_DIR and not _history_started:
        torch.cuda.memory._record_memory_history(max_entries=200000)
        _history_started = True
    logger.info(
        f"[mem-debug] {tag}: alloc={torch.cuda.memory_allocated() / 2**30:.2f}GB "
        f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GB"
    )


def mem_snapshot(tag: str) -> None:
    if not _ENABLED or not _SNAPSHOT_DIR or not torch.cuda.is_available():
        return
    rank = os.environ.get("RANK", "0")
    path = os.path.join(_SNAPSHOT_DIR, f"mem_snapshot_{tag}_rank{rank}.pickle")
    torch.cuda.memory._dump_snapshot(path)
    logger.info(f"[mem-debug] snapshot dumped: {path}")
