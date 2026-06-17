# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from areal.experimental.cli.agent.process import kill_pids
from areal.experimental.cli.agent.state import (
    ServiceState,
    SessionsState,
    resolve_service_name,
    service_state_path,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("stop", help="Stop an agent service")
    parser.add_argument("--service", default=None)
    parser.add_argument("--grace-period", type=float, default=10.0)
    parser.add_argument("--keep-state", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(handler=handle)

    destroy = subparsers.add_parser("destroy", help="Alias for stop")
    destroy.add_argument("--service", default=None)
    destroy.add_argument("--grace-period", type=float, default=10.0)
    destroy.add_argument("--keep-state", action="store_true")
    destroy.add_argument("--force", action="store_true")
    destroy.set_defaults(handler=handle)


def handle(args: argparse.Namespace) -> int:
    service = resolve_service_name(args.service)
    return do_stop(
        service=service,
        grace_period=args.grace_period,
        keep_state=args.keep_state,
        force=args.force,
    )


def do_stop(
    *,
    service: str,
    grace_period: float,
    keep_state: bool,
    force: bool,
) -> int:
    if not service_state_path(service).exists():
        print(f"service {service!r} is not running")
        return 0
    try:
        state = ServiceState.load(service)
    except Exception:
        if not keep_state:
            ServiceState.remove(service)
            SessionsState.remove(service)
        print(f"removed stale state for {service!r}")
        return 0

    kill_pids(state.all_pids(), grace_s=0.0 if force else grace_period)
    if not keep_state:
        ServiceState.remove(service)
        SessionsState.remove(service)
    print(f"service {service!r} stopped")
    return 0
