# SPDX-License-Identifier: Apache-2.0

"""Tests for the two-generation, one-decision Pacman agent workflow."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from examples.vlm.playjev_pacman_h1 import workflow
from examples.vlm.playjev_pacman_h1.harness import OneStepPacmanHarness
from examples.vlm.playjev_pacman_h1.policy import policy_decision


def _state() -> dict:
    rows = ["#" * 19] + ["#" + "." * 17 + "#" for _ in range(20)] + ["#" * 19]
    return {
        "map": "|".join(rows),
        "pac": {"x": 9.0, "y": 12.0, "dir": "left"},
        "ghosts": [
            {"x": 5.0, "y": 5.0, "dir": "up"},
            {"x": 13.0, "y": 5.0, "dir": "down"},
            {"x": 5.0, "y": 15.0, "dir": "left"},
            {"x": 13.0, "y": 15.0, "dir": "right"},
        ],
    }


def _oracle(state: dict) -> dict:
    return {
        **state,
        "ghosts": [
            {**ghost, "vulnerable": False, "eaten": False} for ghost in state["ghosts"]
        ],
        "tick": 200,
        "level": 1,
        "pellets": state["map"].count("."),
        "pills": 0,
    }


def test_harness_allows_exactly_one_scored_decision():
    state = _state()
    harness = OneStepPacmanHarness({"image": b"jpeg", "oracle_info": _oracle(state)})
    action = harness.call_policy_tool(state)["action"]

    outcome = harness.step(action, state)

    assert outcome["done"] is True
    assert outcome["reward"] == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="already ended"):
        harness.step(action, state)


class _Message:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls or []
            ],
        }


class _Client:
    def __init__(self, state: dict, action: str, *, make_tool_call: bool = True):
        self.state = state
        self.action = action
        self.make_tool_call = make_tool_call
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **request):
        self.requests.append(request)
        if len(self.requests) == 1:
            calls = (
                [
                    SimpleNamespace(
                        id="tool-1",
                        function=SimpleNamespace(
                            name="pacman_policy",
                            arguments=json.dumps(self.state),
                        ),
                    )
                ]
                if self.make_tool_call
                else []
            )
            message = _Message(tool_calls=calls)
        else:
            message = _Message(content=self.action)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.asyncio
async def test_workflow_uses_visual_tool_then_scores_private_oracle(monkeypatch):
    state = _state()
    action = policy_decision(state)["action"]
    client = _Client(state, action)
    monkeypatch.setattr(workflow, "AsyncOpenAI", lambda **_: client)
    data = {"image": b"jpeg", "oracle_info": _oracle(state)}

    score = await workflow.PacmanH1Agent().run(
        data, base_url="http://localhost", api_key="session", http_client=None
    )

    assert score == pytest.approx(1.0)
    assert len(client.requests) == 2
    assert client.requests[0]["tool_choice"] == "required"
    assert client.requests[1]["tool_choice"] == "none"
    first_messages = client.requests[0]["messages"]
    assert all("oracle_info" not in message for message in first_messages)
    assert first_messages[1]["content"][0]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert client.requests[1]["messages"][-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_workflow_without_policy_tool_call_gets_zero(monkeypatch):
    state = _state()
    client = _Client(state, "left", make_tool_call=False)
    monkeypatch.setattr(workflow, "AsyncOpenAI", lambda **_: client)

    score = await workflow.PacmanH1Agent().run(
        {"image": b"jpeg", "oracle_info": _oracle(state)},
        base_url="http://localhost",
        api_key="session",
    )

    assert score == 0.0
    assert len(client.requests) == 1
