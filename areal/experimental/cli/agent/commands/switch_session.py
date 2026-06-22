# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.experimental.cli.agent.state import SessionsState, resolve_service_name
from areal.utils import logging

logger = logging.getLogger("AgentCLI")


@click.command(name="switch_session", help="Switch the current default agent session.")
@click.argument("session_key")
@click.option("--service", default=None)
def switch_session_cmd(session_key: str, service: str | None) -> None:
    raise SystemExit(handle(session_key=session_key, service=service) or 0)


def handle(*, session_key: str, service: str | None) -> int:
    service = resolve_service_name(service)
    return do_switch_session(service=service, session_key=session_key)


def do_switch_session(*, service: str, session_key: str) -> int:
    sessions_state = SessionsState.load(service)
    try:
        sessions_state.require_active(session_key)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    sessions_state.current_session = session_key
    sessions_state.save()
    logger.info("current session: %s", session_key)
    return 0
