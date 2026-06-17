# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from areal.experimental.cli.cli import build_parser


def test_agent_help_parser_contains_expected_commands():
    parser = build_parser()

    args = parser.parse_args(["agent", "switch_session", "abc"])

    assert args.agent_command == "switch_session"
    assert args.session_key == "abc"


def test_agent_parser_does_not_register_chat_or_reward():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["agent", "chat"])
    with pytest.raises(SystemExit):
        parser.parse_args(["agent", "reward", "1.0"])
