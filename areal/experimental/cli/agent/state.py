# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from areal.experimental.cli.process import pid_alive
from areal.experimental.cli.state import (
    ServiceStateBase,
    SupportsComponentProbe,
    atomic_write_json,
    clear_current_service,
    service_state_path,
    set_current_service,
)

AGENT_NAMESPACE = "agent"


@dataclass
class ProcessState:
    component: str
    pid: int
    url: str
    log_file: str

    @property
    def addr(self) -> str:
        return self.url


@dataclass
class PairState:
    index: int
    worker: ProcessState
    data_proxy: ProcessState


@dataclass
class ServiceState(ServiceStateBase):
    service: str
    launch_mode: str
    agent: str
    admin_api_key: str
    gateway: ProcessState
    router: ProcessState
    pairs: list[PairState]
    inf_addr: str = ""
    inf_api_key: str = ""
    inf_model: str = ""
    session_timeout: float = 1800.0
    health_poll_interval: float = 5.0
    drain_timeout: float = 30.0
    started_at: float = field(default_factory=time.time)

    def save(self) -> None:
        atomic_write_json(
            service_state_path(AGENT_NAMESPACE, self.service), asdict(self)
        )
        set_current_service(AGENT_NAMESPACE, self.service)

    @classmethod
    def load(cls, service: str) -> ServiceState:
        with open(service_state_path(AGENT_NAMESPACE, service)) as f:
            raw = json.load(f)
        raw["gateway"] = ProcessState(**raw["gateway"])
        raw["router"] = ProcessState(**raw["router"])
        raw["pairs"] = [
            PairState(
                index=pair["index"],
                worker=ProcessState(**pair["worker"]),
                data_proxy=ProcessState(**pair["data_proxy"]),
            )
            for pair in raw.get("pairs", [])
        ]
        # ``created_at`` is the legacy field name; accept it for forward
        # compatibility with any state files written before the rename.
        if "created_at" in raw and "started_at" not in raw:
            raw["started_at"] = raw.pop("created_at")
        raw.pop("created_at", None)
        return cls(**raw)

    @classmethod
    def remove(cls, service: str) -> None:
        path = service_state_path(AGENT_NAMESPACE, service)
        if path.exists():
            path.unlink()
        clear_current_service(AGENT_NAMESPACE, service)

    def gateway_alive(self) -> bool:
        # The CLI treats "running" as "gateway process still alive". The
        # gateway is the entry point — without it, the rest of the stack
        # is unreachable even if some children survived.
        return pid_alive(self.gateway.pid)

    def components(self) -> Iterable[tuple[str, SupportsComponentProbe]]:
        yield "gateway", self.gateway
        yield "router", self.router
        for pair in self.pairs:
            yield pair.worker.component, pair.worker
            yield pair.data_proxy.component, pair.data_proxy
