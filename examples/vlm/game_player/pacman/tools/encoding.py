# SPDX-License-Identifier: Apache-2.0

"""Free SDK generation with a complete linear multimodal conversation."""

import base64
import hashlib
import json
from collections.abc import Callable
from io import BytesIO
from typing import Any

from examples.vlm.game_player.pacman.tools.harness import PacmanHarness
from examples.vlm.game_player.protocols import Decision, EpisodeStop, Observation

SYSTEM_PROMPT = """Play Pacman. Clear all normal pellets to win. Power pellets let you
 eat ghosts temporarily. The game ends immediately on your third death.
Safe hints are suggestions, not restrictions or guarantees. You may choose any
listed physically legal direction, including one that risks a ghost collision.
An OPTION executes its advertised route until completion or interruption. MOVE
executes one game step in U, D, L, or R. Reply with exactly one of:
<answer>OPTION A0</answer> (replace A0 by a current safe hint ID), or
<answer>MOVE U</answer> (replace U by a current legal direction).
Do not include reasoning, explanations, or other text. Invalid format or an
unavailable option/direction ends the game with zero training reward."""


class PacmanPolicyCodec:
    def __init__(
        self,
        owner: Any,
        harness: PacmanHarness,
        *,
        generation_seed: int,
        proxy_session: bool = True,
    ):
        self.owner, self.harness = owner, harness
        self.generation_seed, self.proxy_session = generation_seed, proxy_session
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.generated_tokens = 0

    def _prompt_length(self, messages: list[dict[str, Any]]) -> int:
        """Count the full expanded prompt; discard temporary image tensors.

        Training token IDs still come exclusively from the native proxy. Add
        all prior generated tokens separately below as a conservative allowance
        for concat's raw-token prefix versus re-tokenized assistant text.
        """
        from PIL import Image

        rendered, images = [], []
        try:
            for message in messages:
                copy = dict(message)
                if isinstance(message["content"], list):
                    parts = []
                    for part in message["content"]:
                        if part["type"] == "image_url":
                            encoded = part["image_url"]["url"].split(",", 1)[1]
                            with Image.open(
                                BytesIO(base64.b64decode(encoded, validate=True))
                            ) as image:
                                images.append(image.convert("RGB"))
                            parts.append({"type": "image"})
                        else:
                            parts.append(part)
                    copy["content"] = parts
                rendered.append(copy)
            with self.owner.processor_lock:
                prompt = self.owner.tokenizer.apply_chat_template(
                    rendered,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
                processed = self.owner.processor(
                    text=[prompt],
                    images=images,
                    padding=False,
                    return_tensors="np",
                )
                length = int(processed["input_ids"].shape[-1])
                del processed
            return length
        finally:
            for image in images:
                image.close()

    def decide(
        self,
        observation: Observation,
        context: Any,
        generate: Callable[[dict[str, Any]], Any],
        decision_index: int,
    ) -> Decision:
        from PIL import Image

        candidates, legal = context
        image, state, info = observation.value, observation.state, observation.info
        user = json.dumps(
            {
                "position": info["pacman_position"],
                "facing": state["facing"],
                "normal_pellets_remaining": info["normal_pellets_remaining"],
                "power_pellets_remaining": info["power_pellets_remaining"],
                "deaths": info["death_count"],
                "ghosts": state["ghosts"],
                "edible_ticks": state["edible_ticks"],
                "legal_moves": legal,
                "safe_hints": [candidate.as_dict() for candidate in candidates],
            },
            sort_keys=True,
        )
        with Image.fromarray(image) as pil_image:
            with BytesIO() as buffer:
                pil_image.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        messages = [
            *self.messages,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            },
        ]
        prompt_length = self._prompt_length(messages)
        margin = int(self.owner.options.get("context_safety_margin", 256))
        upper_bound = prompt_length + self.generated_tokens + decision_index + margin
        if (
            upper_bound + self.owner.gconfig.max_new_tokens
            > self.owner.gconfig.max_tokens
        ):
            raise EpisodeStop("context_budget_exhausted", "budget")
        generation_seed = (self.generation_seed + decision_index) % 0x80000000
        template = {"chat_template_kwargs": {"enable_thinking": False}}
        extra = (
            {
                "max_total_tokens": self.owner.gconfig.max_tokens,
                "top_k": -1,
                "extra_body": template,
            }
            if self.proxy_session
            else template
        )
        completion = generate(
            {
                "model": self.owner.model,
                "messages": messages,
                "max_completion_tokens": self.owner.gconfig.max_new_tokens,
                "n": 1,
                "stream": False,
                "temperature": self.owner.gconfig.temperature,
                "top_p": self.owner.gconfig.top_p,
                "seed": generation_seed,
                "extra_body": extra,
            }
        )
        if len(completion.choices) != 1:
            raise ValueError("Pacman requires exactly one SDK completion choice")
        choice = completion.choices[0]
        # Keep the actual SDK assistant message, including invalid model output.
        assistant = choice.message.model_dump(exclude_none=True)
        self.messages = [*messages, assistant]
        usage = completion.usage
        self.generated_tokens += (
            int(usage.completion_tokens)
            if usage is not None
            else self.owner.gconfig.max_new_tokens
        )
        text = choice.message.content
        # Missing textual output is a model format error, not a synthetic token.
        text = text if isinstance(text, str) else ""
        evidence = {
            "completion_id": completion.id,
            "completion": text,
            "finish_reason": choice.finish_reason,
            "model_version": None,
            "inference_evidence_status": "proxy_owned" if self.proxy_session else "sdk",
            "prompt_sha256": hashlib.sha256(
                json.dumps(messages, sort_keys=True).encode()
            ).hexdigest(),
            "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "prompt_tokens_counted": prompt_length,
            "context_budget_upper_bound": upper_bound,
            "safe_hints": [candidate.as_dict() for candidate in candidates],
            "legal_moves": legal,
            "generation_seed": generation_seed,
            "format_valid": False,
            "action_legal": False,
            "safe_advice_match": False,
        }
        return Decision(
            completion_id=completion.id,
            choice=assistant,
            text=text,
            finish_reason=choice.finish_reason,
            evidence=evidence,
        )
