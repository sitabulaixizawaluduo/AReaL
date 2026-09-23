# SPDX-License-Identifier: Apache-2.0

"""Tests for the single-state PlayJev teacher wrapper and verifier."""

from __future__ import annotations

import pytest

from examples.vlm.playjev_pacman_h1.policy import (
    ACTIONS,
    policy_decision,
    state_accuracy,
    verify_decision,
)


def _state() -> dict:
    rows = ["#" * 19] + ["#" + "." * 17 + "#" for _ in range(20)] + ["#" * 19]
    return {
        "map": "|".join(rows),
        "pac": {"x": 9.0, "y": 12.0, "dir": "left"},
        "ghosts": [
            {"x": 5.0, "y": 5.0, "dir": "up"},
            {"x": 13.0, "y": 5.0, "dir": "down"},
            {"x": 5.0, "y": 15.0, "dir": "left"},
            {"x": 13.0, "y": 15.0, "dir": "right"},
        ],
    }


def _oracle(state: dict) -> dict:
    return {
        **state,
        "ghosts": [
            {**ghost, "vulnerable": False, "eaten": False} for ghost in state["ghosts"]
        ],
        "tick": 200,
        "level": 1,
        "pellets": state["map"].count("."),
        "pills": 0,
    }


def test_policy_and_verifier_perfect_visual_state_get_full_reward():
    state = _state()
    teacher = policy_decision(state)
    score = verify_decision(_oracle(state), state, teacher["action"])

    assert set(teacher["probabilities"]) == set(ACTIONS)
    assert sum(teacher["probabilities"].values()) == pytest.approx(1.0)
    assert score["reward"] == pytest.approx(1.0)
    assert score["state"] == pytest.approx(1.0)


def test_state_accuracy_penalizes_wrong_player_coordinate():
    state = _state()
    incorrect = {**state, "pac": {**state["pac"], "x": 5.0}}
    assert state_accuracy(incorrect, _oracle(state)) < 1.0
