# SPDX-License-Identifier: Apache-2.0

"""One environment decision with a visual-state tool and hidden-state verifier."""

from __future__ import annotations

import base64
import json
from typing import Any

from openai import AsyncOpenAI

from .harness import OneStepPacmanHarness
from .policy import ACTIONS

POLICY_TOOL = {
    "type": "function",
    "function": {
        "name": "pacman_policy",
        "description": (
            "Compute a Pacman action distribution from YOUR visual reading of the "
            "screenshot. Supply the complete map and all character positions. "
            "This tool cannot read the game or its hidden state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "map": {
                    "type": "string",
                    "description": (
                        "All map rows top to bottom joined by |. Each tile is # wall, "
                        "= ghost house, . ordinary pellet, o power pill, or space empty."
                    ),
                },
                "pac": {"$ref": "#/$defs/entity"},
                "ghosts": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/entity"},
                    "minItems": 4,
                    "maxItems": 4,
                },
            },
            "required": ["map", "pac", "ghosts"],
            "additionalProperties": False,
            "$defs": {
                "entity": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "dir": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right", "none"],
                        },
                    },
                    "required": ["x", "y", "dir"],
                    "additionalProperties": False,
                }
            },
        },
    },
}

SYSTEM_PROMPT = (
    "You play Pacman from the screenshot only. The maze has 22 rows and 19 columns; "
    "coordinates are grid cells, x rightward and y downward. Pacman and ghosts "
    "can be between cells, so use one decimal place for their coordinates. "
    "Read the full map and current positions from the image, then call "
    "pacman_policy with your inferred state. The tool plans safe routes toward "
    "pellets and avoids predicted ghost collisions. After the tool reply, "
    "answer with exactly one lowercase action: up, down, left, or right."
)


class PacmanH1Agent:
    """Agentic AReaL workflow: one tool call, one game action, one verifier score."""

    def __init__(
        self,
        max_env_steps: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 1024,
    ):
        if max_env_steps != 1:
            raise ValueError("PacmanH1Agent currently supports max_env_steps=1")
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float:
        """Keep oracle_info outside both requests and the callable policy tool."""
        client = AsyncOpenAI(
            base_url=extra_kwargs["base_url"],
            api_key=extra_kwargs["api_key"],
            http_client=extra_kwargs.get("http_client"),
            max_retries=0,
        )
        harness = OneStepPacmanHarness(data, max_env_steps=1)
        image_uri = "data:image/jpeg;base64," + base64.b64encode(harness.image).decode()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {
                        "type": "text",
                        "text": "Read this game frame and make one decision. Call pacman_policy first.",
                    },
                ],
            },
        ]
        first = await client.chat.completions.create(
            model="default",
            messages=messages,
            tools=[POLICY_TOOL],
            tool_choice="required",
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )
        call_list = first.choices[0].message.tool_calls or []
        if len(call_list) != 1 or call_list[0].function.name != "pacman_policy":
            return 0.0
        try:
            inferred_state = json.loads(call_list[0].function.arguments)
            tool_result = harness.call_policy_tool(inferred_state)
        except (ValueError, KeyError, TypeError, IndexError):
            return 0.0

        messages.append(first.choices[0].message.model_dump(exclude_none=True))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_list[0].id,
                "name": "pacman_policy",
                "content": json.dumps(tool_result, separators=(",", ":")),
            }
        )
        second = await client.chat.completions.create(
            model="default",
            messages=messages,
            tools=[POLICY_TOOL],
            tool_choice="none",
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=32,
        )
        answer = second.choices[0].message.content
        if answer is None or answer.strip() not in ACTIONS:
            return 0.0
        return harness.step(answer.strip(), inferred_state)["reward"]
