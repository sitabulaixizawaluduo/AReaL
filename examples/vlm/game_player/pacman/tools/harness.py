# SPDX-License-Identifier: Apache-2.0

"""Parse executable moves separately from strict response serialization."""

import re

from examples.vlm.game_player.protocols import Decision, EpisodeStop


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
    def parse(decision: Decision) -> str:
        answers = re.findall(r"<answer>(.*?)</answer>", decision.text, re.DOTALL)
        match = (
            re.fullmatch(r"\s*MOVE ([UDLR])\s*", answers[0])
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
                r"\s*<answer>MOVE [UDLR]</answer>\s*",
                decision.text,
            )
        )
        decision.evidence.update(
            action_parseable=parseable,
            parse_valid=parseable,
            strict_format_valid=strict,
            # Compatibility field now means the strict visible serialization.
            format_valid=strict,
        )
        if not parseable:
            raise EpisodeStop("invalid_format")
        assert match is not None
        return match.group(1)
