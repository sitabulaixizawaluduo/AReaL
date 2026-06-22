# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import click

from areal.experimental.cli.agent.config import (
    cfg_get,
    load_config,
    resolve_admin_api_key,
    resolve_default_service,
    resolve_inf_addr,
    resolve_inf_api_key,
    resolve_inf_model,
)
from areal.experimental.cli.agent.http import AgentCLIHTTPError, AgentCLIUnreachable
from areal.experimental.cli.agent.interactive import run_shell
from areal.experimental.cli.agent.launcher import launch_agent_stack
from areal.experimental.cli.agent.process import kill_pids, pid_alive
from areal.experimental.cli.agent.session_ops import create_session
from areal.experimental.cli.agent.state import (
    ServiceState,
    SessionsState,
    service_state_path,
)
from areal.utils import logging

logger = logging.getLogger("AgentCLI")


@click.command(name="run", help="Launch an agent service.")
@click.option("--agent", default=None, help="Agent import path.")
@click.option("--service", default=None, help="Service instance name.")
@click.option("--num-pairs", type=int, default=None)
@click.option("--admin-api-key", default=None)
@click.option("--setup-timeout", type=float, default=None)
@click.option("--health-poll-interval", type=float, default=None)
@click.option("--drain-timeout", type=float, default=None)
@click.option("--session-timeout", type=float, default=None)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default=None,
)
@click.option("--config", type=click.Path(path_type=Path), default=None)
@click.option("--force", is_flag=True)
@click.option("--inf-addr", default=None)
@click.option("--inf-api-key", default=None)
@click.option("--inf-model", default=None)
@click.option("--interactive", "-i", is_flag=True)
@click.option("--stop-on-exit", is_flag=True)
@click.option("--history-file", type=click.Path(path_type=Path), default=None)
@click.option("--session-key", default=None)
def run_cmd(**opts) -> None:
    raise SystemExit(do_run(**opts) or 0)


def do_run(
    *,
    agent: str | None,
    service: str | None,
    num_pairs: int | None,
    admin_api_key: str | None,
    setup_timeout: float | None,
    health_poll_interval: float | None,
    drain_timeout: float | None,
    session_timeout: float | None,
    log_level: str | None,
    config: Path | None,
    force: bool,
    inf_addr: str | None,
    inf_api_key: str | None,
    inf_model: str | None,
    interactive: bool,
    stop_on_exit: bool,
    history_file: Path | None,
    session_key: str | None,
) -> int:
    loaded_config = load_config(config)
    service = resolve_default_service(loaded_config, service)
    agent = agent or cfg_get(loaded_config, "run", "agent", "")
    if not agent:
        logger.error("--agent is required")
        return 2

    admin_api_key = resolve_admin_api_key(loaded_config, admin_api_key)
    num_pairs = int(num_pairs or cfg_get(loaded_config, "run", "num_pairs", 1))
    setup_timeout = float(
        setup_timeout or cfg_get(loaded_config, "run", "setup_timeout", 120.0)
    )
    health_poll_interval = float(
        health_poll_interval
        or cfg_get(loaded_config, "run", "health_poll_interval", 5.0)
    )
    drain_timeout = float(
        drain_timeout or cfg_get(loaded_config, "run", "drain_timeout", 30.0)
    )
    session_timeout = float(
        session_timeout or cfg_get(loaded_config, "run", "session_timeout", 1800.0)
    )
    log_level = log_level or cfg_get(loaded_config, "default", "log_level", "info")
    inf_addr = resolve_inf_addr(loaded_config, inf_addr)
    inf_api_key = resolve_inf_api_key(loaded_config, inf_api_key)
    inf_model = resolve_inf_model(loaded_config, inf_model)

    existing = _load_existing(service)
    if existing is not None and any(pid_alive(pid) for pid in existing.all_pids()):
        if not force:
            logger.error(
                "service %r already has live processes; "
                "run `areal agent stop` first or use --force",
                service,
            )
            return 1
        kill_pids(existing.all_pids(), grace_s=5.0)
    elif service_state_path(service).exists() and not force:
        logger.error(
            "stale state exists for %r; use `areal agent run --force`",
            service,
        )
        return 1

    launched: ServiceState | None = None
    try:
        launched = launch_agent_stack(
            service=service,
            agent=agent,
            admin_api_key=admin_api_key,
            num_pairs=num_pairs,
            setup_timeout=setup_timeout,
            session_timeout=session_timeout,
            health_poll_interval=health_poll_interval,
            drain_timeout=drain_timeout,
            log_level=log_level,
            inf_addr=inf_addr,
            inf_api_key=inf_api_key,
            inf_model=inf_model,
            interactive=interactive,
        )
        launched.save()
        sessions = SessionsState(service=service)
        session = create_session(
            launched,
            sessions,
            session_key=session_key,
            switch=True,
        )
    except (AgentCLIHTTPError, AgentCLIUnreachable, RuntimeError, ValueError) as exc:
        if launched is not None:
            kill_pids(launched.all_pids(), grace_s=5.0)
        logger.error("failed to launch agent service: %s", exc)
        return 1

    logger.info("service=%s gateway=%s", service, launched.gateway.url)
    logger.info("session=%s", session.key)
    if interactive:
        return run_shell(
            launched,
            stop_on_exit=stop_on_exit,
            history_file=history_file,
        )
    return 0


def _load_existing(service: str) -> ServiceState | None:
    if not service_state_path(service).exists():
        return None
    try:
        return ServiceState.load(service)
    except Exception:
        return None
