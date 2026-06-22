# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
from pathlib import Path

from areal.experimental.cli.agent.commands.new_session import do_new_session
from areal.experimental.cli.agent.commands.status import do_status
from areal.experimental.cli.agent.commands.stop import do_stop
from areal.experimental.cli.agent.commands.switch_session import do_switch_session
from areal.experimental.cli.agent.state import ServiceState, SessionsState, agent_root
from areal.utils import logging

logger = logging.getLogger("AgentCLI")


def run_shell(
    service_state: ServiceState,
    *,
    stop_on_exit: bool,
    history_file: Path | None = None,
) -> int:
    service = service_state.service
    _setup_history(service, history_file)
    logger.info("agent service %r ready at %s", service, service_state.gateway.url)
    logger.info(
        "type /help for commands; chat and reward are not implemented in this CLI yet"
    )
    while True:
        try:
            line = input("agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("")
            break
        if not line:
            continue
        if not line.startswith("/"):
            logger.info("chat is not implemented yet")
            continue
        parts = line.split()
        command = parts[0]
        if command in {"/exit", "/quit"}:
            break
        if command == "/help":
            logger.info(
                "/session, /sessions, /new_session [key], /switch_session <key>"
            )
            logger.info("/status, /stop, /exit")
        elif command == "/session":
            sessions = SessionsState.load(service)
            current = sessions.current_session
            session = sessions.sessions.get(current) if current else None
            if session is None:
                logger.info("no current session")
            else:
                logger.info("%s status=%s", session.key, session.status)
        elif command == "/sessions":
            sessions = SessionsState.load(service)
            for session in sessions.sessions.values():
                marker = "*" if session.key == sessions.current_session else " "
                logger.info("%s %s %s", marker, session.key, session.status)
        elif command == "/new_session":
            key = parts[1] if len(parts) > 1 else None
            do_new_session(
                service=service, session_key=key, no_switch=False, as_json=False
            )
        elif command == "/switch_session" and len(parts) == 2:
            do_switch_session(service=service, session_key=parts[1])
        elif command == "/status":
            do_status(service=service, as_json=False, watch=False, interval=2.0)
        elif command == "/stop":
            do_stop(service=service, grace_period=10.0, keep_state=False, force=True)
            return 0
        elif command in {"/reward", "/chat"}:
            logger.info("%s is not implemented yet", command[1:])
        else:
            logger.info("unknown command: %s", command)
    if stop_on_exit:
        return do_stop(service=service, grace_period=10.0, keep_state=False, force=True)
    return 0


def _setup_history(service: str, history_file: Path | None) -> None:
    try:
        import readline
    except ImportError:
        return

    path = history_file or agent_root() / "history" / f"{service}.history"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(str(path))
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, str(path))
