# SPDX-License-Identifier: Apache-2.0

"""Bounded episode reward, computed once from authoritative game events."""

import math
from typing import Any

DEFAULT_STEP_EFFICIENCY_PENALTY_WEIGHT = 0.05


class EpisodeReward:
    def __init__(
        self,
        initial_info: dict[str, Any],
        ghost_reward_target: float = 4,
        *,
        max_steps: int = 512,
        step_efficiency_penalty_weight: float = (
            DEFAULT_STEP_EFFICIENCY_PENALTY_WEIGHT
        ),
    ):
        if not math.isfinite(ghost_reward_target) or ghost_reward_target <= 0:
            raise ValueError("ghost_reward_target must be finite and positive")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if (
            not math.isfinite(step_efficiency_penalty_weight)
            or step_efficiency_penalty_weight < 0
            or step_efficiency_penalty_weight > 0.1
        ):
            raise ValueError(
                "step_efficiency_penalty_weight must be finite and in [0, 0.1]"
            )
        self.ghost_reward_target = ghost_reward_target
        self.max_steps = max_steps
        self.step_efficiency_penalty_weight = step_efficiency_penalty_weight
        self.initial_normal = int(initial_info["normal_pellets_remaining"])
        self.initial_power = int(initial_info["power_pellets_remaining"])
        if self.initial_normal <= 0 or self.initial_power < 0:
            raise ValueError("Initial pellet counts are invalid")
        self.initial_step = int(initial_info["step"])
        self.last_step = self.initial_step
        self.ghosts_eaten = 0
        self.finished = False
        self.total = 0.0
        self.components: dict[str, float] = {}
        self.bounded_game_reward = 0.0
        self.weighted_game_reward = 0.0
        self.strict_format_bonus = 0.0
        self.all_strict = False
        self.env_steps = 0
        self.completion_step_ratio: float | None = None
        self.completion_step_penalty = 0.0

    def step(
        self, previous: dict[str, Any], info: dict[str, Any], base_reward: float
    ) -> tuple[float, dict[str, float]]:
        if self.finished or int(info["step"]) != self.last_step + 1:
            raise ValueError("Each live environment transition must be observed once")
        events = info.get("logic_frame_events")
        if not isinstance(events, list):
            raise ValueError("Reward accounting requires the environment event ledger")
        for key, event_type in (
            ("normal_pellets_remaining", "normal_pellet_eaten"),
            ("power_pellets_remaining", "power_pellet_eaten"),
        ):
            eaten = sum(event["event_type"] == event_type for event in events)
            if int(previous[key]) - int(info[key]) != eaten or int(info[key]) < 0:
                raise ValueError(f"{key} does not reconcile with event ledger")
        deaths = sum(event["event_type"] == "death" for event in events)
        if int(info["death_count"]) - int(previous["death_count"]) != deaths:
            raise ValueError("Death count does not reconcile with event ledger")
        if sum(event["score_delta"] for event in events) != base_reward:
            raise ValueError("Game score does not reconcile with event ledger")
        self.ghosts_eaten += sum(
            event["event_type"] == "ghost_eaten" for event in events
        )
        self.last_step = int(info["step"])
        return 0.0, {}

    def finish(
        self,
        reason: str,
        info: dict[str, Any],
        *,
        all_strict: bool,
    ) -> tuple[float, dict[str, float]]:
        if self.finished:
            raise ValueError("Episode reward must be settled exactly once")
        normal = int(info["normal_pellets_remaining"])
        power = int(info["power_pellets_remaining"])
        deaths = int(info["death_count"])
        if (
            not 0 <= normal <= self.initial_normal
            or not 0 <= power <= self.initial_power
        ):
            raise ValueError("Final pellet counts exceed the initial game")
        if not 0 <= deaths <= 3:
            raise ValueError("The game must end by its third death")
        win = reason == "all_normal_pellets"
        if win and normal != 0:
            raise ValueError("Winning requires clearing every normal pellet")
        self.env_steps = int(info["step"]) - self.initial_step
        if self.env_steps < 0:
            raise ValueError("Final environment step precedes the initial step")
        completion_step_ratio = (
            min(1.0, max(0.0, self.env_steps / self.max_steps)) if win else None
        )
        completion_step_penalty = (
            -self.step_efficiency_penalty_weight * completion_step_ratio
            if completion_step_ratio is not None
            else 0.0
        )
        p = 1 - normal / self.initial_normal
        s = 1 - power / self.initial_power if self.initial_power else 0.0
        game_components = {
            "completion_or_progress": 0.9 if win else 0.5 * p,
            "special_pellets": 0.05 * s,
            "ghosts": 0.05 * min(self.ghosts_eaten / self.ghost_reward_target, 1.0),
            "deaths": -0.02 * deaths,
            "completion_step_penalty": completion_step_penalty,
        }
        raw_game_reward = sum(game_components.values())
        bounded_game_reward = min(1.0, max(0.0, raw_game_reward))
        game_components["clipping"] = bounded_game_reward - raw_game_reward
        weighted_game_reward = 0.9 * bounded_game_reward
        strict_format_bonus = 0.1 * float(all_strict)
        components = {key: 0.9 * value for key, value in game_components.items()}
        components["strict_serialization"] = strict_format_bonus
        total = min(1.0, max(0.0, weighted_game_reward + strict_format_bonus))
        if reason in {"invalid_format", "invalid_action"}:
            components["strict_serialization"] = 0.0
            components["invalid_output"] = -weighted_game_reward
            total = 0.0
            strict_format_bonus = 0.0
        self.bounded_game_reward = bounded_game_reward
        self.weighted_game_reward = weighted_game_reward
        self.strict_format_bonus = strict_format_bonus
        self.all_strict = bool(all_strict)
        self.completion_step_ratio = completion_step_ratio
        self.completion_step_penalty = completion_step_penalty
        self.total, self.components, self.finished = total, components, True
        return total, dict(components)
