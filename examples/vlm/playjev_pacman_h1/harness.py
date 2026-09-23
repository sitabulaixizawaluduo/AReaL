# SPDX-License-Identifier: Apache-2.0

"""One-decision Pacman harness with an isolated source-state verifier."""

from __future__ import annotations

import json
from typing import Any

from .policy import policy_decision, verify_decision


class OneStepPacmanHarness:
    def __init__(self, sample: dict[str, Any], max_env_steps: int = 1):
        if max_env_steps != 1:
            raise ValueError("The H1 harness supports exactly one game decision")
        self._image = sample["image"]
        oracle_info = sample["oracle_info"]
        self._oracle_info = (
            json.loads(oracle_info) if isinstance(oracle_info, str) else oracle_info
        )
        self._steps = 0

    @property
    def image(self) -> bytes:
        return self._image

    @staticmethod
    def call_policy_tool(inferred_state: dict[str, Any]) -> dict[str, Any]:
        """The public tool has no handle to the oracle or browser."""
        return policy_decision(inferred_state)

    def step(self, action: str, inferred_state: dict[str, Any]) -> dict[str, Any]:
        """Finish this one-decision task and score the chosen action."""
        if self._steps >= 1:
            raise RuntimeError("Pacman H1 decision already ended")
        self._steps += 1
        metrics = verify_decision(self._oracle_info, inferred_state, action)
        return {"done": True, "reward": metrics["reward"], "metrics": metrics}
