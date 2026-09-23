# SPDX-License-Identifier: Apache-2.0

"""Pure, single-state wrapper around the PlayJev Pacman teacher."""

from __future__ import annotations

import itertools
import math
from typing import Any

from .pacman_teacher import PacmanTeacher

ACTIONS = ("up", "down", "left", "right")
ACTION_SPEC = [{"name": name, "index": i} for i, name in enumerate(ACTIONS)]
MAP_CHARS = frozenset("#=.o ")


def normalize_visual_state(state: dict[str, Any]) -> dict[str, Any]:
    """Validate a model-supplied state without looking up the true game state.

    H1 uses only ordinary ghosts. This lets each image be scored independently:
    the original teacher's hidden pill/ghost timers are irrelevant in this slice.
    """
    if not isinstance(state, dict):
        raise ValueError("state must be an object")
    map_string = state.get("map")
    if not isinstance(map_string, str):
        raise ValueError("map must be a string")
    rows = map_string.split("|")
    if len(rows) != 22 or len(rows[0]) != 19:
        raise ValueError("Pacman map must have 22 rows and 19 columns")
    if any(len(row) != len(rows[0]) or set(row) - MAP_CHARS for row in rows):
        raise ValueError("map rows must have equal width and valid tiles")

    pac = _entity(state.get("pac"), "pac")
    ghosts_input = state.get("ghosts")
    if not isinstance(ghosts_input, list) or len(ghosts_input) != 4:
        raise ValueError("ghosts must contain four entities")
    ghosts = []
    for i, ghost in enumerate(ghosts_input):
        normalized = _entity(ghost, f"ghosts[{i}]")
        normalized.update(vulnerable=False, eaten=False)
        ghosts.append(normalized)

    return {
        "map": map_string,
        "pac": pac,
        "ghosts": ghosts,
        "tick": 0,
        "level": 1,
        "pellets": map_string.count("."),
        "pills": map_string.count("o"),
    }


def _entity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    x, y, direction = value.get("x"), value.get("y"), value.get("dir")
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, (int, float))
        or not isinstance(y, (int, float))
        or not math.isfinite(x)
        or not math.isfinite(y)
        or not -1 <= x <= 19
        or not 0 <= y <= 21
        or direction not in (*ACTIONS, "none")
    ):
        raise ValueError(f"{label} has invalid coordinates or direction")
    return {"x": float(x), "y": float(y), "dir": direction}


def project_oracle_info(info: dict[str, Any]) -> dict[str, Any]:
    """Select only fields that the image-side tool schema can represent."""
    if any(g.get("vulnerable") or g.get("eaten") for g in info["ghosts"]):
        raise ValueError("H1 excludes active power-pill and eaten-ghost states")
    return normalize_visual_state(
        {
            "map": info["map"],
            "pac": info["pac"],
            "ghosts": info["ghosts"],
        }
    )


def policy_decision(state: dict[str, Any]) -> dict[str, Any]:
    """Return the original teacher's action distribution for one state."""
    info = normalize_visual_state(state)
    teacher = PacmanTeacher(ACTION_SPEC)
    teacher.reset()
    probs = teacher.act({"info": info})
    scores = {name: round(float(probs[i]), 6) for i, name in enumerate(ACTIONS)}
    return {
        "action": ACTIONS[max(range(len(probs)), key=probs.__getitem__)],
        "probabilities": scores,
        "mode": teacher.last_mode,
    }


def state_accuracy(predicted: dict[str, Any], oracle: dict[str, Any]) -> float:
    """Give dense visual grounding credit without rewarding unchanged wall pixels."""
    pred = normalize_visual_state(predicted)
    gold = project_oracle_info(oracle)

    def entity_score(a: dict[str, Any], b: dict[str, Any]) -> float:
        distance = abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])
        return 0.8 * max(0.0, 1.0 - distance / 4.0) + 0.2 * float(a["dir"] == b["dir"])

    pac_score = entity_score(pred["pac"], gold["pac"])
    ghost_score = max(
        sum(entity_score(a, b) for a, b in zip(pred["ghosts"], ordered)) / 4
        for ordered in itertools.permutations(gold["ghosts"])
    )
    pred_targets = {
        (x, y, char)
        for y, row in enumerate(pred["map"].split("|"))
        for x, char in enumerate(row)
        if char in ".o"
    }
    gold_targets = {
        (x, y, char)
        for y, row in enumerate(gold["map"].split("|"))
        for x, char in enumerate(row)
        if char in ".o"
    }
    target_score = len(pred_targets & gold_targets) / max(
        1, len(pred_targets | gold_targets)
    )
    # Walls are a gate on map fidelity, not the main score: most pixels are walls.
    pred_walls = {
        (x, y, char)
        for y, row in enumerate(pred["map"].split("|"))
        for x, char in enumerate(row)
        if char in "#="
    }
    gold_walls = {
        (x, y, char)
        for y, row in enumerate(gold["map"].split("|"))
        for x, char in enumerate(row)
        if char in "#="
    }
    wall_score = len(pred_walls & gold_walls) / max(1, len(pred_walls | gold_walls))
    return (
        0.30 * pac_score + 0.30 * ghost_score + 0.30 * target_score + 0.10 * wall_score
    )


def verify_decision(
    oracle_info: dict[str, Any],
    predicted_state: dict[str, Any],
    final_action: str,
) -> dict[str, float]:
    """Score one agent decision using the isolated source-state oracle."""
    if final_action not in ACTIONS:
        return {"reward": 0.0, "state": 0.0, "action": 0.0}
    oracle = policy_decision(project_oracle_info(oracle_info))
    grounding = state_accuracy(predicted_state, oracle_info)
    best = max(oracle["probabilities"].values())
    action_quality = oracle["probabilities"][final_action] / best
    return {
        "reward": 0.40 * grounding + 0.60 * action_quality,
        "state": grounding,
        "action": action_quality,
    }
