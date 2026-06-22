# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.experimental.cli.agent.process import kill_pids
from areal.experimental.cli.agent.state import (
    ServiceState,
    SessionsState,
    resolve_service_name,
    service_state_path,
)
from areal.utils import logging

logger = logging.getLogger("AgentCLI")


@click.command(name="stop", help="Stop an agent service.")
@click.option("--service", default=None)
@click.option("--grace-period", type=float, default=10.0, show_default=True)
@click.option("--keep-state", is_flag=True)
@click.option("--force", is_flag=True)
def stop_cmd(
    service: str | None,
    grace_period: float,
    keep_state: bool,
    force: bool,
) -> None:
    raise SystemExit(
        handle(
            service=service,
            grace_period=grace_period,
            keep_state=keep_state,
            force=force,
        )
        or 0
    )


@click.command(name="destroy", help="Alias for stop.")
@click.option("--service", default=None)
@click.option("--grace-period", type=float, default=10.0, show_default=True)
@click.option("--keep-state", is_flag=True)
@click.option("--force", is_flag=True)
def destroy_cmd(
    service: str | None,
    grace_period: float,
    keep_state: bool,
    force: bool,
) -> None:
    raise SystemExit(
        handle(
            service=service,
            grace_period=grace_period,
            keep_state=keep_state,
            force=force,
        )
        or 0
    )


def handle(
    *,
    service: str | None,
    grace_period: float,
    keep_state: bool,
    force: bool,
) -> int:
    service = resolve_service_name(service)
    return do_stop(
        service=service,
        grace_period=grace_period,
        keep_state=keep_state,
        force=force,
    )


def do_stop(
    *,
    service: str,
    grace_period: float,
    keep_state: bool,
    force: bool,
) -> int:
    if not service_state_path(service).exists():
        logger.info("service %r is not running", service)
        return 0
    try:
        state = ServiceState.load(service)
    except Exception:
        if not keep_state:
            ServiceState.remove(service)
            SessionsState.remove(service)
        logger.info("removed stale state for %r", service)
        return 0

    kill_pids(state.all_pids(), grace_s=0.0 if force else grace_period)
    if not keep_state:
        ServiceState.remove(service)
        SessionsState.remove(service)
    logger.info("service %r stopped", service)
    return 0
