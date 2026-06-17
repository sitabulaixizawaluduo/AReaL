# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json

from areal.experimental.cli.agent.process import pid_alive
from areal.experimental.cli.agent.state import ServiceState, list_service_names


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("ps", help="List locally known agent services")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--all", action="store_true", dest="include_all")
    parser.set_defaults(handler=handle)


def handle(args: argparse.Namespace) -> int:
    rows = []
    for service in list_service_names():
        try:
            state = ServiceState.load(service)
        except Exception:
            if args.include_all:
                rows.append({"service": service, "status": "stale"})
            continue
        running = any(pid_alive(pid) for pid in state.all_pids())
        if running or args.include_all:
            rows.append(
                {
                    "service": service,
                    "status": "running" if running else "stale",
                    "gateway_url": state.gateway.url,
                    "agent": state.agent,
                }
            )

    if args.as_json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("no agent services")
        return 0
    cols = ("SERVICE", "STATUS", "GATEWAY", "AGENT")
    table = [
        (
            row["service"],
            row["status"],
            row.get("gateway_url", ""),
            row.get("agent", ""),
        )
        for row in rows
    ]
    widths = [max(len(str(row[i])) for row in (cols, *table)) for i in range(4)]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*cols))
    for row in table:
        print(fmt.format(*row))
    return 0
