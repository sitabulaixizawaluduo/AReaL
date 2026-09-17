# SPDX-License-Identifier: Apache-2.0

"""Bounded episode reward, computed once from authoritative game events."""

import math
from typing import Any


class EpisodeReward:
    def __init__(self, initial_info: dict[str, Any], ghost_reward_target: float = 4):
        if not math.isfinite(ghost_reward_target) or ghost_reward_target <= 0:
            raise ValueError("ghost_reward_target must be finite and positive")
        self.ghost_reward_target = ghost_reward_target
        self.initial_normal = int(initial_info["normal_pellets_remaining"])
        self.initial_power = int(initial_info["power_pellets_remaining"])
        if self.initial_normal <= 0 or self.initial_power < 0:
            raise ValueError("Initial pellet counts are invalid")
        self.last_step = int(initial_info["step"])
        self.ghosts_eaten = 0
        self.finished = False
        self.total = 0.0
        self.components: dict[str, float] = {}

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
        self, reason: str, info: dict[str, Any]
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
        p = 1 - normal / self.initial_normal
        s = 1 - power / self.initial_power if self.initial_power else 0.0
        components = {
            "completion_or_progress": 0.9 if win else 0.5 * p,
            "special_pellets": 0.05 * s,
            "ghosts": 0.05 * min(self.ghosts_eaten / self.ghost_reward_target, 1.0),
            "deaths": -0.02 * deaths,
        }
        raw = sum(components.values())
        total = min(1.0, max(0.0, raw))
        components["clipping"] = total - raw
        if reason in {"invalid_format", "invalid_action"}:
            components["invalid_output"] = -total
            total = 0.0
        self.total, self.components, self.finished = total, components, True
        return total, dict(components)
