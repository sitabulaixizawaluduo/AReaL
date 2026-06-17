# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os

from areal.experimental.cli.agent.state import resolve_service_name, service_logs_dir


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("logs", help="Show agent service logs")
    parser.add_argument("--service", default=None)
    parser.add_argument("--component", default="gateway")
    parser.add_argument("-f", "--follow", action="store_true")
    parser.add_argument("-n", "--lines", type=int, default=200)
    parser.set_defaults(handler=handle)


def handle(args: argparse.Namespace) -> int:
    service = resolve_service_name(args.service)
    log_dir = service_logs_dir(service)
    target = log_dir / f"{args.component}.log"
    if not target.exists():
        available = sorted(path.stem for path in log_dir.glob("*.log"))
        if not available:
            print(f"no logs found under {log_dir}")
            return 1
        print(f"no log named {args.component!r}; available: {', '.join(available)}")
        return 1

    cmd = ["tail", f"-n{args.lines}"]
    if args.follow:
        cmd.append("-F")
    cmd.append(str(target))
    os.execvp(cmd[0], cmd)
    return 0
