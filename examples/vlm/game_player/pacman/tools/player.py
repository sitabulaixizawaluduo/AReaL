# SPDX-License-Identifier: Apache-2.0

"""Independent Pacman SDK player; all model processing stays on the server."""

import math
from dataclasses import dataclass
from typing import Any

from examples.vlm.game_player.pacman.tools.game import PacmanSessionFactory
from examples.vlm.game_player.player import GamePlayer


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 512
    # Standalone SDK players opt into provider reasoning unless explicitly disabled.
    reasoning: bool = True
    # Only the training proxy consumes this total-context limit.
    max_tokens: int | None = None
    seed: int | None = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.max_tokens is not None and self.max_tokens <= self.max_new_tokens:
            raise ValueError("Proxy context limit must exceed max_new_tokens")


class PacmanPlayer(GamePlayer):
    def __init__(
        self,
        model: str,
        generation: dict[str, Any],
        options: dict[str, Any],
    ):
        self.model, self.options = model, dict(options)
        self.options.setdefault("planner_assisted", True)
        if not isinstance(self.options["planner_assisted"], bool):
            raise TypeError("planner_assisted must be a boolean")
        self.gconfig = GenerationSettings(**generation)
        super().__init__(PacmanSessionFactory(self))
