# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from areal.experimental.cli.state import areal_home, atomic_write_json, pid_alive


def inf_root() -> Path:
    d = areal_home() / "inf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path() -> Path:
    return inf_root() / "state.json"


def logs_dir() -> Path:
    d = inf_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class DaemonState:
    gateway_pid: int
    gateway_url: str
    router_pid: int
    router_url: str
    admin_api_key: str
    started_at: float

    def save(self) -> None:
        atomic_write_json(state_path(), asdict(self))

    @classmethod
    def load(cls) -> DaemonState:
        p = state_path()
        if not p.exists():
            raise FileNotFoundError(f"No daemon state at {p}")
        with open(p) as f:
            return cls(**json.load(f))

    @classmethod
    def remove(cls) -> None:
        p = state_path()
        if p.exists():
            p.unlink()


def gateway_alive(state: DaemonState) -> bool:
    return pid_alive(state.gateway_pid)
