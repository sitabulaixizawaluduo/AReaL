# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys

from areal.experimental.cli.agent.http import AgentCLIHTTPError, AgentCLIUnreachable
from areal.experimental.cli.agent.session_ops import create_session
from areal.experimental.cli.agent.state import (
    ServiceState,
    SessionsState,
    resolve_service_name,
    service_state_path,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("new_session", help="Start a new agent session")
    parser.add_argument("session_key", nargs="?", default=None)
    parser.add_argument("--service", default=None)
    parser.add_argument("--no-switch", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.set_defaults(handler=handle)


def handle(args: argparse.Namespace) -> int:
    service = resolve_service_name(args.service)
    return do_new_session(
        service=service,
        session_key=args.session_key,
        no_switch=args.no_switch,
        as_json=args.as_json,
    )


def do_new_session(
    *,
    service: str,
    session_key: str | None,
    no_switch: bool,
    as_json: bool,
) -> int:
    if not service_state_path(service).exists():
        print(f"error: service {service!r} is not running", file=sys.stderr)
        return 1
    try:
        service_state = ServiceState.load(service)
        sessions_state = SessionsState.load(service)
        session = create_session(
            service_state,
            sessions_state,
            session_key=session_key,
            switch=not no_switch,
        )
    except (AgentCLIHTTPError, AgentCLIUnreachable, ValueError) as exc:
        print(f"error: failed to create session: {exc}", file=sys.stderr)
        return 1

    payload = {
        "service": service,
        "session_key": session.key,
        "current": session.key == SessionsState.load(service).current_session,
        "rl_negotiated": session.rl_negotiated,
    }
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"session={session.key} "
            f"current={'yes' if payload['current'] else 'no'} "
            f"rl={'yes' if session.rl_negotiated else 'no'}"
        )
    return 0
