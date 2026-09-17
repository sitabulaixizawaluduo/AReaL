# SPDX-License-Identifier: Apache-2.0

"""Independent Pacman SDK player; training adapters may inject HF components."""

import math
import threading
from dataclasses import dataclass
from typing import Any

from examples.vlm.game_player.pacman.tools.game import PacmanSessionFactory
from examples.vlm.game_player.player import GamePlayer


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 32
    max_tokens: int = 32768
    seed: int | None = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_new_tokens < 1 or self.max_tokens <= self.max_new_tokens:
            raise ValueError("Generation and context token budgets are invalid")


class PacmanPlayer(GamePlayer):
    def __init__(
        self,
        model: str,
        generation: dict[str, Any],
        options: dict[str, Any],
        *,
        processor: Any = None,
        tokenizer: Any = None,
    ):
        self.model, self.options = model, dict(options)
        self.gconfig = GenerationSettings(**generation)
        if (processor is None) != (tokenizer is None):
            raise ValueError("Inject both processor and tokenizer, or neither")
        if processor is None:
            from transformers import AutoProcessor, AutoTokenizer

            # The independent player only counts prompt lengths; no training
            # tensor or inference engine is created locally.
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(
                model,
                trust_remote_code=True,
                use_fast=False,
            )
        self.processor, self.tokenizer = processor, tokenizer
        self.processor_lock = threading.Lock()
        if int(self.options.get("context_safety_margin", 256)) < 0:
            raise ValueError("context_safety_margin must be nonnegative")
        super().__init__(PacmanSessionFactory(self))
