# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from click.testing import CliRunner

from areal.experimental.cli.cli import cli


def test_agent_help_contains_expected_commands():
    result = CliRunner().invoke(cli, ["agent", "--help"])

    assert result.exit_code == 0
    assert "run" in result.output
    assert "switch_session" in result.output
    assert "new_session" in result.output


def test_agent_switch_session_accepts_session_argument():
    result = CliRunner().invoke(cli, ["agent", "switch_session", "--help"])

    assert result.exit_code == 0
    assert "SESSION_KEY" in result.output


def test_agent_parser_does_not_register_chat_or_reward():
    runner = CliRunner()

    chat = runner.invoke(cli, ["agent", "chat"])
    reward = runner.invoke(cli, ["agent", "reward", "1.0"])

    assert chat.exit_code != 0
    assert reward.exit_code != 0
    assert "No such command" in chat.output
    assert "No such command" in reward.output
