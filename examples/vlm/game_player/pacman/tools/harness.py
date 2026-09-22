# SPDX-License-Identifier: Apache-2.0

"""Parse executable moves separately from strict response serialization."""

import re
from dataclasses import dataclass

from examples.vlm.game_player.protocols import Decision, EpisodeStop


@dataclass(frozen=True)
class PacmanCommand:
    kind: str
    value: str


class PacmanHarness:
    def __init__(self) -> None:
        self._all_strict = True
        self.format_decisions = 0

    @property
    def all_strict(self) -> bool:
        """Whether every observed model decision used the exact visible format."""
        return self.format_decisions > 0 and self._all_strict

    def record_format(self, decision: Decision) -> None:
        """Account for one authoritative execution attempt exactly once."""
        self.format_decisions += 1
        self._all_strict = self._all_strict and bool(
            decision.evidence.get("strict_format_valid", False)
        )

    @staticmethod
    def parse(decision: Decision) -> PacmanCommand:
        answers = re.findall(r"<answer>(.*?)</answer>", decision.text, re.DOTALL)
        match = (
            re.fullmatch(
                r"\s*(?:(MOVE) ([UDLR])|(OPTION) (A0)|(A0))\s*",
                answers[0],
            )
            if len(answers) == 1
            else None
        )
        unsupported = decision.choice.get("tool_calls") or decision.choice.get(
            "refusal"
        )
        parseable = match is not None and not unsupported
        strict = bool(
            parseable
            and re.fullmatch(
                r"\s*<answer>(?:MOVE [UDLR]|OPTION A0)</answer>\s*",
                decision.text,
            )
        )
        decision.evidence.update(
            action_parseable=parseable,
            parse_valid=parseable,
            strict_format_valid=strict,
            # Compatibility field now means the strict visible serialization.
            format_valid=strict,
            command_form=(
                "option_alias"
                if parseable and match is not None and match.group(5) is not None
                else "canonical"
                if parseable
                else None
            ),
        )
        if not parseable:
            raise EpisodeStop("invalid_format")
        assert match is not None
        move_kind, move_value, option_kind, option_value, option_alias = match.groups()
        kind, value = (
            (move_kind, move_value)
            if move_kind
            else (option_kind or "OPTION", option_value or option_alias)
        )
        assert kind is not None and value is not None
        return PacmanCommand(kind, value)
