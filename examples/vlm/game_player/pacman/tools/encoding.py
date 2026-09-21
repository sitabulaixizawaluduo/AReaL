# SPDX-License-Identifier: Apache-2.0

"""Free SDK generation with proxy concat and current-frame standalone prompts."""

import base64
import hashlib
import json
from collections.abc import Callable
from io import BytesIO
from typing import Any

import numpy as np

from examples.vlm.game_player.pacman.tools.harness import PacmanHarness
from examples.vlm.game_player.pacman.tools.visual_planner import (
    StandaloneVisualPlanner,
    VisualActionSpace,
    detect_pacman_cell,
)
from examples.vlm.game_player.protocols import Decision, EpisodeStop, Observation

SYSTEM_PROMPT = """Play Pacman from the screenshot and pixel-derived hint only.
- Clear every small pellet to win. Large pellets make ghosts blue and edible briefly.
  Touching a colored ghost costs a life; the third death ends the game.
- Blue walls and screen boundaries are impassable. U/D/L/R mean screen up/down/left/right.
  Choose an adjacent open corridor, favor pellets, and avoid nearby colored ghosts.
- Each command moves one game step; inspect the new screenshot before every command.
- Reset shows the full map; later views may be Pacman-centered crops at the same tile
  scale. Black crop padding is unknown space, not a wall or map boundary.
- At reset, the spawn corridor permits only L or R; U and D are walls.
- Reply with exactly one <answer>MOVE X</answer>, where X is U, D, L, or R. Never output
  multiple moves, WAIT, or option IDs. Bad format ends the episode with zero reward.
Use a separate reasoning channel only for private reasoning; keep final content exact."""

ACTION_REQUEST = "Choose from the current screenshot. Return exactly <answer>MOVE X</answer> with X in U,D,L,R."
RESET_ACTION_REQUEST = "Reset frame: U/D are walls. Return exactly <answer>MOVE L</answer> or <answer>MOVE R</answer>."
STANDALONE_BLOCKED_ACTION_RULE = "A blocked move consumes a step; the next hint reports it so choose another direction."
PROXY_BLOCKED_ACTION_RULE = "A blocked training move ends the episode with zero reward."
_PROMPT_CROP_TILES = 13
_TILE_PIXELS = 16


def _crop_prompt_image(
    image: Any,
    pacman_cell: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a fixed tile crop with black padding, preserving source resolution."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("prompt cropping requires an RGB image")
    radius = _PROMPT_CROP_TILES // 2
    top = (pacman_cell[0] - radius) * _TILE_PIXELS
    left = (pacman_cell[1] - radius) * _TILE_PIXELS
    bottom = top + _PROMPT_CROP_TILES * _TILE_PIXELS
    right = left + _PROMPT_CROP_TILES * _TILE_PIXELS
    source_top, source_left = max(0, top), max(0, left)
    source_bottom, source_right = min(rgb.shape[0], bottom), min(rgb.shape[1], right)
    target_top, target_left = source_top - top, source_left - left
    target_bottom = target_top + source_bottom - source_top
    target_right = target_left + source_right - source_left
    crop = np.zeros(
        (
            _PROMPT_CROP_TILES * _TILE_PIXELS,
            _PROMPT_CROP_TILES * _TILE_PIXELS,
            rgb.shape[2],
        ),
        dtype=rgb.dtype,
    )
    crop[target_top:target_bottom, target_left:target_right] = rgb[
        source_top:source_bottom, source_left:source_right
    ]
    metadata = {
        "center_cell_row_col": list(pacman_cell),
        "requested_bounds_pixels_top_left_bottom_right": [
            top,
            left,
            bottom,
            right,
        ],
        "source_bounds_pixels_top_left_bottom_right": [
            source_top,
            source_left,
            source_bottom,
            source_right,
        ],
        "padding_pixels_top_left_bottom_right": [
            target_top,
            target_left,
            crop.shape[0] - target_bottom,
            crop.shape[1] - target_right,
        ],
        "output_height_width": [crop.shape[0], crop.shape[1]],
    }
    return crop, metadata


def _split_standalone_thinking(content: str, reasoning_content: Any) -> tuple[str, str]:
    """Normalize provider-separated and Qwen template-prefilled thinking."""
    reasoning = reasoning_content if isinstance(reasoning_content, str) else ""
    if "</think>" not in content:
        return content, reasoning

    inline_reasoning, final_content = content.split("</think>", 1)
    inline_reasoning = inline_reasoning.strip()
    if inline_reasoning.startswith("<think>"):
        inline_reasoning = inline_reasoning[len("<think>") :].lstrip()
    if not reasoning:
        reasoning = inline_reasoning
    return final_content.strip(), reasoning


def _completion_usage(completion: Any) -> dict[str, int | None] | None:
    """Copy provider token counts without retaining an SDK response object."""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    copied = {field: getattr(usage, field, None) for field in fields}
    return copied if any(value is not None for value in copied.values()) else None


class PacmanPolicyCodec:
    def __init__(
        self,
        owner: Any,
        *,
        generation_seed: int,
        proxy_session: bool = True,
        event_observer: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.owner = owner
        self.event_observer = event_observer
        self.generation_seed, self.proxy_session = generation_seed, proxy_session
        blocked_action_rule = (
            PROXY_BLOCKED_ACTION_RULE
            if proxy_session
            else STANDALONE_BLOCKED_ACTION_RULE
        )
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n{blocked_action_rule}"}
        ]
        if proxy_session and owner.gconfig.max_tokens is None:
            raise ValueError("Training proxy sessions require max_tokens")
        self._previous_visual_cell: tuple[int, int] | None = None
        self._previous_parsed_action: str | None = None
        self._visual_actions = VisualActionSpace()
        self._visual_planner = None if proxy_session else StandaloneVisualPlanner()

    @staticmethod
    def _detect_pacman_visual_cell(image: Any) -> tuple[int, int] | None:
        """Keep the public detector hook while sharing visual planner extraction."""
        return detect_pacman_cell(image)

    def decide(
        self,
        observation: Observation,
        context: Any,
        generate: Callable[[dict[str, Any]], Any],
        decision_index: int,
    ) -> Decision:
        from PIL import Image

        # Observation state/info and harness context are execution-only. Never
        # serialize them into the policy conversation, even when available.
        image = observation.value
        visual_cell = self._detect_pacman_visual_cell(image)
        visual_blocked_action = (
            not self.proxy_session
            and visual_cell is not None
            and visual_cell == self._previous_visual_cell
            and self._previous_parsed_action is not None
        )
        visual_plan = (
            self._visual_planner.plan(image)
            if self._visual_planner is not None
            else None
        )
        prompt_image, prompt_image_scope = image, "reset"
        prompt_image_crop = None
        prompt_image_fallback_reason = None
        if decision_index == 0:
            # Cache topology for portal-aware jump checks. Harness validation has
            # its own independent RGB cache.
            try:
                self._visual_actions.available_actions(image)
            except ValueError:
                pass
        elif visual_cell is None:
            prompt_image_scope = "fallback"
            prompt_image_fallback_reason = "ambiguous_pacman"
        elif self._previous_visual_cell is None or self._previous_parsed_action is None:
            prompt_image_scope = "fallback"
            prompt_image_fallback_reason = "missing_previous_visual_action"
        elif not self._visual_actions.is_expected_transition(
            self._previous_visual_cell,
            visual_cell,
            self._previous_parsed_action,
        ):
            prompt_image_scope = "fallback"
            prompt_image_fallback_reason = "nonportal_jump_or_respawn"
        else:
            prompt_image, prompt_image_crop = _crop_prompt_image(image, visual_cell)
            prompt_image_scope = "crop"
        with Image.fromarray(prompt_image) as pil_image:
            with BytesIO() as buffer:
                pil_image.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        request_text = RESET_ACTION_REQUEST if decision_index == 0 else ACTION_REQUEST
        if visual_blocked_action:
            request_text = (
                f"{request_text} Previous MOVE {self._previous_parsed_action} did not "
                f"move Pacman; {self._previous_parsed_action} is blocked. Choose a "
                "different adjacent direction."
            )
        if visual_plan is not None:
            request_text = f"{request_text} {visual_plan.prompt_text()}"
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": request_text,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ],
        }
        # Standalone calls are independent current-frame decisions. Stateful visual
        # tracking stays local and enters only as the concise current hint. Training
        # proxy calls retain the full append-only conversation for concat.
        messages = [*self.messages, user_message]
        generation_seed = (self.generation_seed + decision_index) % 0x80000000
        standalone_reasoning = not self.proxy_session and self.owner.gconfig.reasoning
        template = {"chat_template_kwargs": {"enable_thinking": standalone_reasoning}}
        extra = (
            {
                "max_total_tokens": self.owner.gconfig.max_tokens,
                "top_k": -1,
                "extra_body": template,
            }
            if self.proxy_session
            else template
        )
        if self.event_observer is not None:
            self.event_observer({"kind": "request", "decision": decision_index})
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
        # Standalone thinking is shown/audited separately. Concat history keeps
        # only the normalized final assistant content.
        raw_assistant = choice.message.model_dump(exclude_none=True)
        raw_text = choice.message.content
        # Missing textual output is a model format error, not a synthetic token.
        raw_text = raw_text if isinstance(raw_text, str) else ""
        text, reasoning = raw_text, ""
        assistant = raw_assistant
        if standalone_reasoning:
            text, reasoning = _split_standalone_thinking(
                raw_text, raw_assistant.get("reasoning_content")
            )
            assistant = dict(raw_assistant)
            assistant["content"] = text
            if reasoning:
                assistant["reasoning_content"] = reasoning
        else:
            raw_reasoning = raw_assistant.get("reasoning_content")
            reasoning = raw_reasoning if isinstance(raw_reasoning, str) else ""
        if self.event_observer is not None:
            self.event_observer(
                {"kind": "decision", "text": text, "reasoning": reasoning}
            )
        evidence = {
            "completion_id": completion.id,
            "completion": text,
            "finish_reason": choice.finish_reason,
            "model_version": None,
            "inference_evidence_status": "proxy_owned" if self.proxy_session else "sdk",
            "prompt_sha256": hashlib.sha256(
                json.dumps(messages, sort_keys=True).encode()
            ).hexdigest(),
            # Preserve the historical full-frame hash while separately auditing
            # the possibly cropped image sent to the model.
            "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "full_image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "prompt_image_sha256": hashlib.sha256(prompt_image.tobytes()).hexdigest(),
            "prompt_image_scope": prompt_image_scope,
            "prompt_image_crop": prompt_image_crop,
            "prompt_image_fallback_reason": prompt_image_fallback_reason,
            "generation_seed": generation_seed,
            "reasoning_content_allowed": standalone_reasoning,
            "visual_pacman_cell_row_col": list(visual_cell)
            if visual_cell is not None
            else None,
            "previous_visual_pacman_cell_row_col": (
                list(self._previous_visual_cell)
                if self._previous_visual_cell is not None
                else None
            ),
            "previous_parsed_action": self._previous_parsed_action,
            "visual_blocked_action_detected": visual_blocked_action,
            "visual_plan": visual_plan.as_evidence()
            if visual_plan is not None
            else None,
            "parse_valid": False,
            "strict_format_valid": False,
            "format_valid": False,
            "action_legal": False,
        }
        token_usage = _completion_usage(completion)
        if token_usage is not None:
            evidence["token_usage"] = token_usage
        if raw_text != text:
            evidence["raw_completion"] = raw_text
        decision = Decision(
            completion_id=completion.id,
            choice=assistant,
            text=text,
            finish_reason=choice.finish_reason,
            evidence=evidence,
        )
        if self.proxy_session:
            # Training retains the exact append-only SDK conversation for concat.
            self.messages = [*messages, assistant]
        # Both modes retain only compact RGB tracker state. Proxy messages still
        # preserve the complete trainable concat, now with reset-full then crops.
        try:
            action = PacmanHarness.parse(decision)
        except EpisodeStop:
            pass
        else:
            if not self.proxy_session:
                assert self._visual_planner is not None
                self._visual_planner.record_model_action(action)
            self._previous_visual_cell = visual_cell
            self._previous_parsed_action = action
        return decision
