# SPDX-License-Identifier: Apache-2.0

import copy
import os
import re

from mathruler.grader import extract_boxed_content, grade_answer
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from areal.api import AsyncRewardWrapper
from areal.utils.image import image2base64

_FORMAT_RE = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
_FORMAT_WEIGHT = 0.1


def _format_reward(predict_str: str) -> float:
    return 1.0 if _FORMAT_RE.fullmatch(predict_str) else 0.0


def _acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def geometry3k_reward_fn(completions: str, answer: str) -> float:
    """Blend an accuracy check with a small format bonus.

    Mirrors the reward used by ``examples/vlm/geometry3k_grpo.py``: pull the
    final answer from ``\\boxed{...}`` and compare via ``mathruler.grader``,
    plus a 0.1-weight bonus when the completion also wraps its reasoning in
    ``<think>...</think>`` tags — matching the prompt template the dataset
    emits.
    """
    format_val = _format_reward(completions)
    acc_val = _acc_reward(completions, answer)
    return (1.0 - _FORMAT_WEIGHT) * acc_val + _FORMAT_WEIGHT * format_val


def _fill_image_urls(messages_chat: list, images_b64: list[str]) -> list:
    """Populate empty ``image_url.url`` entries with base64 PNG data URLs.

    The RL dataset (see ``areal.dataset.geometry3k``) leaves
    ``image_url.url`` blank because the v1 engine used to fill it at
    request time. In v2 agent mode the agent owns that step.
    """
    filled = copy.deepcopy(messages_chat)
    it = iter(images_b64)
    for msg in filled:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "image_url"):
                continue
            url_obj = part.setdefault("image_url", {})
            if url_obj.get("url"):
                continue
            b64 = next(it, None)
            if b64 is None:
                return filled
            url_obj["url"] = f"data:image/png;base64,{b64}"
    return filled


class VisionGeometry3KAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()
        self.kwargs.pop("max_tokens", None)
        self.kwargs.pop("max_turns", None)
        self._reward_fn = AsyncRewardWrapper(geometry3k_reward_fn)

    async def run(self, data: dict, **extra_kwargs):
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")

        images_b64 = image2base64(data["images"])
        messages = _fill_image_urls(data["messages_chat"], images_b64)

        client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0
        )
        comp: ChatCompletion = await client.chat.completions.create(
            messages=messages, model="default", **self.kwargs
        )
        return await self._reward_fn(
            completions=comp.choices[0].message.content,
            answer=data["answer"],
        )
