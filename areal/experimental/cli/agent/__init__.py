# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from areal.experimental.cli.agent.commands import (
    logs,
    new_session,
    ps,
    run,
    status,
    stop,
    switch_session,
)


def register_agent_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "agent",
        help="Manage agent services and sessions",
        description="Manage agent services and sessions.",
    )
    agent_subparsers = parser.add_subparsers(dest="agent_command", metavar="<command>")
    run.register(agent_subparsers)
    stop.register(agent_subparsers)
    status.register(agent_subparsers)
    ps.register(agent_subparsers)
    new_session.register(agent_subparsers)
    switch_session.register(agent_subparsers)
    logs.register(agent_subparsers)

    def _missing(_: argparse.Namespace) -> int:
        parser.print_help()
        return 2

    parser.set_defaults(handler=_missing)
