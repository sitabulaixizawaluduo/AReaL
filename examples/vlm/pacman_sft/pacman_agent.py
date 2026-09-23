# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from PIL.Image import Image as ImageObject

from areal.api.io_struct import detect_image_mime

from .pacman_dataset import SYSTEM_PROMPT, render_question


def pacman_reward_fn(
    completions: str,
    expected_letter: str,
    **_: Any,
) -> float:
    """Reward an exact teacher-action match after option shuffling."""
    return float(str(completions).strip() == str(expected_letter).strip())


def _image_to_data_uri(image: bytes | str | ImageObject) -> str:
    if isinstance(image, str):
        if image.startswith(("data:image/", "http://", "https://")):
            return image
        encoded = image
    elif isinstance(image, bytes):
        encoded = base64.b64encode(image).decode("utf-8")
    elif isinstance(image, ImageObject):
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise TypeError(f"Unsupported Pacman image type: {type(image).__name__}")
    return f"data:{detect_image_mime(encoded)};base64,{encoded}"


def _build_agent_input(data: dict[str, Any]) -> list[dict[str, Any]]:
    images = data["images"]
    if len(images) != 1:
        raise ValueError(
            f"Pacman agent expects exactly one image, received {len(images)}"
        )

    question = render_question(data["actions"])
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{SYSTEM_PROMPT}\n\n<state>\n",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_to_data_uri(images[0]),
                        "detail": "auto",
                    },
                },
                {
                    "type": "text",
                    "text": f"\n</state>\n\n{question}",
                },
            ],
        }
    ]


class PacmanAgent:
    """Make one agentic Chat Completions request and verify the chosen action."""

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs.copy()
        self.kwargs.pop("max_turns", None)

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float:
        http_client = extra_kwargs.get("http_client")
        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )
        completion: ChatCompletion = await client.chat.completions.create(
            messages=_build_agent_input(data),
            model="default",
            **self.kwargs,
        )
        output = completion.choices[0].message.content
        if output is None:
            raise ValueError("The Pacman completion did not contain text output.")
        return pacman_reward_fn(
            completions=output,
            expected_letter=data["expected_letter"],
        )
