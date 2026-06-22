# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import click

from areal.experimental.cli.agent.http import AgentCLIHTTPError, AgentCLIUnreachable
from areal.experimental.cli.agent.session_ops import create_session
from areal.experimental.cli.agent.state import (
    ServiceState,
    SessionsState,
    resolve_service_name,
    service_state_path,
)
from areal.utils import logging

logger = logging.getLogger("AgentCLI")


@click.command(name="new_session", help="Start a new agent session.")
@click.argument("session_key", required=False)
@click.option("--service", default=None)
@click.option("--no-switch", is_flag=True)
@click.option("--json", "as_json", is_flag=True)
def new_session_cmd(
    session_key: str | None,
    service: str | None,
    no_switch: bool,
    as_json: bool,
) -> None:
    raise SystemExit(
        handle(
            session_key=session_key,
            service=service,
            no_switch=no_switch,
            as_json=as_json,
        )
        or 0
    )


def handle(
    *,
    session_key: str | None,
    service: str | None,
    no_switch: bool,
    as_json: bool,
) -> int:
    service = resolve_service_name(service)
    return do_new_session(
        service=service,
        session_key=session_key,
        no_switch=no_switch,
        as_json=as_json,
    )


def do_new_session(
    *,
    service: str,
    session_key: str | None,
    no_switch: bool,
    as_json: bool,
) -> int:
    if not service_state_path(service).exists():
        logger.error("service %r is not running", service)
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
        logger.error("failed to create session: %s", exc)
        return 1

    payload = {
        "service": service,
        "session_key": session.key,
        "current": session.key == SessionsState.load(service).current_session,
    }
    if as_json:
        logger.info("%s", json.dumps(payload, indent=2))
    else:
        logger.info(
            "session=%s current=%s",
            session.key,
            "yes" if payload["current"] else "no",
        )
    return 0
