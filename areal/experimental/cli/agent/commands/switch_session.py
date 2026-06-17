# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys

from areal.experimental.cli.agent.state import SessionsState, resolve_service_name


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "switch_session",
        help="Switch the current default agent session",
    )
    parser.add_argument("session_key")
    parser.add_argument("--service", default=None)
    parser.set_defaults(handler=handle)


def handle(args: argparse.Namespace) -> int:
    service = resolve_service_name(args.service)
    return do_switch_session(service=service, session_key=args.session_key)


def do_switch_session(*, service: str, session_key: str) -> int:
    sessions_state = SessionsState.load(service)
    try:
        sessions_state.require_active(session_key)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sessions_state.current_session = session_key
    sessions_state.save()
    print(f"current session: {session_key}")
    return 0
