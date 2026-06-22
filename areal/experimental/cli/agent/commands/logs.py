# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import click

from areal.experimental.cli.agent.state import resolve_service_name, service_logs_dir
from areal.utils import logging

logger = logging.getLogger("AgentCLI")


@click.command(name="logs", help="Show agent service logs.")
@click.option("--service", default=None)
@click.option("--component", default="gateway", show_default=True)
@click.option("-f", "--follow", is_flag=True)
@click.option("-n", "--lines", type=int, default=200, show_default=True)
def logs_cmd(
    service: str | None,
    component: str,
    follow: bool,
    lines: int,
) -> None:
    raise SystemExit(
        handle(service=service, component=component, follow=follow, lines=lines) or 0
    )


def handle(
    *,
    service: str | None,
    component: str,
    follow: bool,
    lines: int,
) -> int:
    service = resolve_service_name(service)
    log_dir = service_logs_dir(service)
    target = log_dir / f"{component}.log"
    if not target.exists():
        available = sorted(path.stem for path in log_dir.glob("*.log"))
        if not available:
            logger.error("no logs found under %s", log_dir)
            return 1
        logger.error("no log named %r; available: %s", component, ", ".join(available))
        return 1

    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-F")
    cmd.append(str(target))
    os.execvp(cmd[0], cmd)
    return 0
