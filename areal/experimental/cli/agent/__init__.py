# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.experimental.cli.agent.commands.logs import logs_cmd
from areal.experimental.cli.agent.commands.new_session import new_session_cmd
from areal.experimental.cli.agent.commands.ps import ps_cmd
from areal.experimental.cli.agent.commands.run import run_cmd
from areal.experimental.cli.agent.commands.status import health_cmd, status_cmd
from areal.experimental.cli.agent.commands.stop import destroy_cmd, stop_cmd
from areal.experimental.cli.agent.commands.switch_session import switch_session_cmd


@click.group(help="Manage agent services and sessions.")
def agent() -> None:
    pass


agent.add_command(run_cmd)
agent.add_command(stop_cmd)
agent.add_command(destroy_cmd)
agent.add_command(status_cmd)
agent.add_command(health_cmd)
agent.add_command(ps_cmd)
agent.add_command(new_session_cmd)
agent.add_command(switch_session_cmd)
agent.add_command(logs_cmd)
