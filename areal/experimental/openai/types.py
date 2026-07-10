# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # noqa

import base64
import re
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from typing import Any

import torch
from openai.types.chat import ChatCompletion
from openai.types.responses.response import Response
from openai.types.responses.response_input_param import ResponseInputParam

from areal.api import ModelResponse
from areal.utils import logging

logger = logging.getLogger("TokenLogpReward")

_DATA_URL_RE = re.compile(r"^data:image/[^;]+;base64,")


def _decode_data_url_image(url: str) -> Any:
    from PIL import Image

    m = _DATA_URL_RE.match(url)
    if not m:
        return None
    try:
        raw = base64.b64decode(url[m.end() :])
        return Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        logger.warning("Failed to decode data-URL image", exc_info=True)
        return None


def _extract_images_from_messages(messages: list[dict] | None) -> list:
    """Walk OpenAI-format messages and return PIL images from image_url parts."""
    images: list = []
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = (part.get("image_url") or {}).get("url", "")
            if not isinstance(url, str) or not url:
                continue
            img = _decode_data_url_image(url)
            if img is not None:
                images.append(img)
    return images


def _image_pad_token_id(tokenizer) -> int | None:
    """Return the image-placeholder token id for Qwen-VL / gemma-style tokenizers."""
    for tok in ("<|image_pad|>", "<image>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            return int(tid)
    return None


def _expand_image_pad_tokens(
    input_tokens: list[int],
    image_grid_thw: Any,
    merge_size: int,
    image_pad_id: int,
) -> list[int]:
    """Replace each single ``<|image_pad|>`` with N copies of the same token.

    Why: the gateway tokenises messages via ``tokenizer.apply_chat_template``,
    which emits one ``<|image_pad|>`` per image slot in the prompt. The
    inference server (SGLang / vLLM) then expands that placeholder to
    ``N = T*H*W / merge_size**2`` real image tokens before running the model,
    but ``ModelResponse.input_tokens`` is set from ``req.input_ids`` — i.e.
    the *unexpanded* sequence. When the training-side model forward consumes
    that unexpanded sequence together with our per-image ``pixel_values``, HF's
    ``get_placeholder_mask`` finds fewer image tokens than image features and
    raises ``Image features and image tokens do not match``. Expanding here
    keeps the training-side view of the prompt aligned with what the model
    actually needs.
    """
    per_image_counts = [
        int(row.prod().item()) // (merge_size * merge_size) for row in image_grid_thw
    ]
    result: list[int] = []
    image_idx = 0
    for token in input_tokens:
        tid = int(token)
        if tid == image_pad_id and image_idx < len(per_image_counts):
            result.extend([image_pad_id] * per_image_counts[image_idx])
            image_idx += 1
        else:
            result.append(tid)
    return result


def _compute_vision_augmentation(
    messages: list[dict],
    resp: ModelResponse,
    processor: Any,
) -> tuple[list[int], list[dict[str, Any]], list[int] | None] | None:
    """Extract images and derive expanded ``input_tokens`` + multi-modal tensors.

    Returns ``(expanded_input_tokens, multi_modal_input, mm_type_ids_input)``
    where ``mm_type_ids_input`` covers only the prompt segment; the caller
    concatenates ``[0] * output_len`` after it. Returns ``None`` when the
    interaction has no images (text-only path unchanged).
    """
    images = _extract_images_from_messages(messages)
    if not images:
        return None

    processed = processor.image_processor(images=images, return_tensors="pt")
    mm_dict: dict[str, Any] = {"pixel_values": processed["pixel_values"]}
    image_grid_thw = processed.get("image_grid_thw")
    if image_grid_thw is not None:
        mm_dict["image_grid_thw"] = image_grid_thw
    multi_modal_input = [mm_dict]

    image_pad_id = _image_pad_token_id(processor.tokenizer)
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))

    if image_grid_thw is None or image_pad_id is None:
        # Not enough info to expand — keep the unexpanded prompt; the model
        # will likely error, but at least the failure mode is clear.
        return list(resp.input_tokens), multi_modal_input, None

    expanded_input = _expand_image_pad_tokens(
        list(resp.input_tokens), image_grid_thw, merge_size, image_pad_id
    )
    mm_type_ids_input = [1 if int(t) == image_pad_id else 0 for t in expanded_input]
    return expanded_input, multi_modal_input, mm_type_ids_input


class ApiType(str, Enum):
    """API type for interaction."""

    COMPLETION = "completion"
    RESPONSE = "response"
    NONE = "none"


class InputName(str, Enum):
    """Input name used for logging."""

    MESSAGES = "messages"
    INPUT_DATA = "input_data"
    NONE = "none"


@dataclass
class InteractionWithTokenLogpReward:
    """Internal structure to store completions/responses with their rewards."""

    # Common
    model_response: ModelResponse | None = None
    reward: float | None = None
    parent: InteractionWithTokenLogpReward | None = None
    chat_template_type: str = "hf"
    _cache: dict[str, torch.Tensor] | None = None

    # Fields used for parent-child relationship resolving
    messages: list[dict] = field(default_factory=list)
    output_message_list: list[dict] | None = None

    # Completion fields (optional for response)
    completion: ChatCompletion | None = None

    # Response fields (optional for completion)
    response: Response | None = None
    input_data: str | ResponseInputParam = field(default_factory=lambda: "")

    # Interaction ID cache (used for deserialization)
    _interaction_id: str | None = None

    @property
    def has_tensor_data(self) -> bool:
        return self.model_response is not None or self._cache is not None

    @property
    def is_completion(self) -> bool:
        return self.completion is not None

    @property
    def is_response(self) -> bool:
        return self.response is not None

    @property
    def api_type(self) -> ApiType:
        """API type (completion/response)."""
        if self.is_completion:
            return ApiType.COMPLETION
        elif self.is_response:
            return ApiType.RESPONSE
        else:
            return ApiType.NONE

    @property
    def input_name_for_logging(self) -> InputName:
        """Input name used for logging."""
        if self.is_completion:
            return InputName.MESSAGES
        elif self.is_response:
            return InputName.INPUT_DATA
        else:
            return InputName.NONE

    @property
    def current_data(self) -> list[dict] | str | ResponseInputParam | None:
        if self.is_completion:
            return self.messages
        elif self.is_response:
            return self.input_data
        else:
            return None

    @property
    def parent_data(self) -> list[dict] | str | ResponseInputParam | None:
        if self.parent is None:
            return None
        return self.parent.current_data

    @property
    def interaction_id(self) -> str | None:
        if self.is_completion:
            return self.completion.id
        elif self.is_response:
            return self.response.id
        elif self._interaction_id is not None:
            return self._interaction_id
        else:
            return None

    @interaction_id.setter
    def interaction_id(self, value):
        if self.is_completion or self.is_response:
            raise ValueError("Cannot set ID for completion or responses")
        self._interaction_id = value

    @property
    def created_at(self) -> float | None:
        if self.is_completion:
            return float(self.completion.created)
        elif self.is_response:
            return float(self.response.created_at)
        else:
            return None

    @property
    def remaining_messages(self) -> list[dict]:
        if self.parent is None:
            return self.messages
        assert self.parent.output_message_list is not None, (
            "Parent output message is not set."
        )
        parent_len = len(self.parent.messages + self.parent.output_message_list)
        return self.messages[parent_len:]

    def to_tensor_dict(self, processor: Any = None) -> dict[str, torch.Tensor]:
        if self._cache is not None:
            return self._cache
        resp = self.model_response
        assert resp is not None, "Model response is not set."

        # For VLM: expand single ``<|image_pad|>`` placeholders in resp.input_tokens
        # into per-patch tokens so the training-side model forward sees a prompt
        # of the same length as the multi-modal features it will inject.
        vision_aug = (
            _compute_vision_augmentation(self.messages, resp, processor)
            if processor is not None
            else None
        )
        if vision_aug is not None:
            expanded_input, multi_modal_input, mm_type_ids_input = vision_aug
        else:
            expanded_input = list(resp.input_tokens)
            multi_modal_input = None
            mm_type_ids_input = None
        input_len = len(expanded_input)
        self.seq_tokens = seq = expanded_input + list(resp.output_tokens)

        if self.chat_template_type == "concat" and self.parent is not None:
            parent_res = self.parent.to_tensor_dict(processor=processor)
            parent_logprobs = parent_res["logprobs"].squeeze(0).tolist()
            parent_loss_mask = parent_res["loss_mask"].squeeze(0).tolist()
            parent_versions = parent_res["versions"].squeeze(0).tolist()
            parent_len = len(parent_logprobs)
            assert parent_len == len(parent_loss_mask) == len(parent_versions)
            if input_len > parent_len:
                logprobs = (
                    parent_logprobs
                    + [0.0] * (input_len - parent_len)
                    + resp.output_logprobs
                )
                loss_mask = (
                    parent_loss_mask
                    + [0] * (input_len - parent_len)
                    + [1] * resp.output_len
                )
                versions = (
                    parent_versions
                    + [-1] * (input_len - parent_len)
                    + resp.output_versions
                )
            else:
                # FIXME: Find out why this happens occasionally
                api_type = self.api_type
                input_name = self.input_name_for_logging
                logger.warning(
                    f"The input length of the child {api_type} ({input_len}) is less than or "
                    f"equal to the length of the parent {api_type} {parent_len}. "
                    f"This should not happen if the {input_name}s are constructed properly. \n"
                    f"Parent input token ids: {self.parent.model_response.input_tokens}\n"
                    f"Parent output token ids: {self.parent.model_response.output_tokens}\n"
                    f"Child input token ids: {resp.input_tokens}\n"
                    f"Parent input {input_name}: {self.parent_data}\n"
                    f"Child input {input_name}: {self.current_data}",
                )
                logprobs = [0.0] * input_len + resp.output_logprobs
                loss_mask = [0] * input_len + [1] * resp.output_len
                versions = [-1] * input_len + resp.output_versions
        else:
            logprobs = [0.0] * input_len + resp.output_logprobs
            loss_mask = [0] * input_len + [1] * resp.output_len
            versions = [-1] * input_len + resp.output_versions
        reward = self.reward if self.reward is not None else 0.0
        result = dict(
            # unsqueeze to add an additional batch dimension
            input_ids=torch.tensor(seq).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            # reward
            rewards=torch.tensor([float(reward)]),
        )
        if multi_modal_input is not None:
            result["multi_modal_input"] = multi_modal_input
            if mm_type_ids_input is not None:
                mm_type = mm_type_ids_input + [0] * resp.output_len
                result["mm_token_type_ids"] = torch.tensor(
                    mm_type, dtype=torch.long
                ).unsqueeze(0)
        self._cache = result
        return result


def concat_string_interactions(
    interactions: dict[str, InteractionWithTokenLogpReward],
) -> dict[str, list[dict]]:
    """Concat interactions that lack tensor data (e.g. external API mode).

    Returns a dict with an ``"interactions"`` key containing a list of
    ``{"request": ..., "response": ..., "reward": ...}`` dicts, one per
    interaction.  This is the counterpart of
    :func:`~areal.utils.data.concat_padded_tensors` for string-only
    trajectories.
    """
    return {
        "interactions": [
            {
                "request": v.messages,
                "response": (
                    v.output_message_list[0]["content"] if v.output_message_list else ""
                ),
                "reward": v.reward,
            }
            for v in interactions.values()
        ]
    }
