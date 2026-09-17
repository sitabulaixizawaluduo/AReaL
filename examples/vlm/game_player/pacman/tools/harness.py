# SPDX-License-Identifier: Apache-2.0

"""Advisory safe options; the model may execute any physically legal direction."""

from typing import Any


class PacmanHarness:
    def __init__(self, planner: Any = None):
        from maapacman.planner import EdwardPlanner

        self.planner = planner if planner is not None else EdwardPlanner()
        self.refusal_triggers = 0

    def candidates(self, state: dict[str, Any]) -> tuple[Any, ...]:
        from maapacman.planner import EdwardSafetyRefusal

        try:
            return self.planner.advertised_candidates(state)
        except EdwardSafetyRefusal:
            self.refusal_triggers += 1
            return ()

    def observe_transition(self, info: dict[str, Any], state: dict[str, Any]) -> None:
        for event in info["logic_frame_events"]:
            if event["event_type"] in {"normal_pellet_eaten", "power_pellet_eaten"}:
                self.planner.observe({"pacman_position": event["pacman_position"]})
        self.planner.observe(state)

    def continue_option(
        self,
        option: Any,
        previous: dict[str, Any],
        info: dict[str, Any],
        state: dict[str, Any],
        remaining_moves: int,
    ) -> tuple[str | None, str]:
        from maapacman.planner import EdwardSafetyRefusal

        events = {event["event_type"] for event in info["logic_frame_events"]}
        if (
            info.get("respawned")
            or (int(previous["edible_ticks"]) > 0) != (int(info["edible_ticks"]) > 0)
            or events.intersection({"power_pellet_eaten", "ghost_eaten", "death"})
        ):
            return None, "state_event_interrupt"
        if remaining_moves <= 0:
            return None, "max_commit"
        try:
            action, status = self.planner.continue_option(option, state)
        except EdwardSafetyRefusal:
            return None, "safe_hint_interrupt"
        if status == "active" and action not in info["legal_actions"]:
            return None, "option_no_longer_legal"
        return action, status
