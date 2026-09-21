# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from argparse import Namespace
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest
from PIL import Image

from examples.vlm.game_player.pacman.tools import evaluate as evaluate_module
from examples.vlm.game_player.pacman.tools.dashboard import (
    DashboardServer,
    DashboardState,
)
from examples.vlm.game_player.pacman.tools.encoding import PacmanPolicyCodec
from examples.vlm.game_player.pacman.tools.evaluate import PacmanEvaluation
from examples.vlm.game_player.pacman.tools.game import PacmanHarnessAdapter
from examples.vlm.game_player.pacman.tools.harness import PacmanHarness
from examples.vlm.game_player.pacman.tools.player import PacmanPlayer
from examples.vlm.game_player.pacman.tools.prepare import GamePreparation
from examples.vlm.game_player.protocols import Decision, Observation


def test_dashboard_keeps_isolated_latest_frames_and_bounded_output():
    """Repeated updates replace per-game pixels and text without retaining history."""
    state = DashboardState()
    state.plan([{"episode_id": "a", "seed": 1}, {"episode_id": "b", "seed": 2}])
    state.begin("a")
    state.begin("b")
    state.frame("b", np.full((8, 9, 3), 255, dtype=np.uint8), {"score": 40})
    other = state.png("b")
    for index in range(30):
        state.frame("a", np.full((8, 9, 3), index, dtype=np.uint8), {"score": index})
        state.event(
            "a",
            {"kind": "decision", "text": "x" * 20000, "reasoning": "r" * 20000},
        )
    snapshot = state.snapshot()
    first, second = snapshot["episodes"]
    assert first["score"] == 29 and second["score"] == 40
    assert first["frame_version"] == 30
    assert state.png("b") == other
    assert len(state._frames) == 2
    assert sum(map(len, state._frames.values())) < 2048
    assert len(first["raw_output"]) == DashboardState.TEXT_LIMIT
    assert first["output_truncated"]
    assert len(first["reasoning"]) == DashboardState.TEXT_LIMIT
    assert first["reasoning_truncated"]
    with Image.open(BytesIO(state.png("a"))) as picture:
        np.testing.assert_array_equal(np.asarray(picture), np.full((8, 9, 3), 29))
    first["score"] = -100
    assert state.snapshot()["episodes"][0]["score"] == 29


def test_dashboard_http_serves_latest_png_and_redacts_error_details():
    """HTTP exposes game evidence only, including completed/error outcomes."""
    state = DashboardState()
    state.plan([{"episode_id": "a", "seed": 1}, {"episode_id": "b", "seed": 2}])
    state.frame("a", np.zeros((8, 9, 3), dtype=np.uint8), {"lives": 3})
    state.finish(
        "a",
        {
            "status": "complete",
            "reward": 0.8,
            "win": True,
            "elapsed_seconds": 12.5,
            "terminal_reason": "all_normal_pellets",
        },
    )
    state.finish(
        "b",
        {
            "status": "error",
            "error_type": "APIError",
            "error": "Authorization: secret-api-key",
            "api_key": "secret-api-key",
        },
    )
    state.complete()
    server = DashboardServer(state, port=0)
    server.start()
    base = f"http://127.0.0.1:{server.server.server_port}"
    try:
        with urlopen(base + "/api/episodes", timeout=2) as response:
            raw = response.read()
            snapshot = json.loads(raw)
        assert b"secret-api-key" not in raw
        assert snapshot["status"] == "partial"
        assert [row["status"] for row in snapshot["episodes"]] == ["terminal", "error"]
        assert snapshot["episodes"][0]["reward"] == 0.8
        assert snapshot["episodes"][1]["error_type"] == "APIError"
        with urlopen(base + "/frames/a.png", timeout=2) as response:
            assert response.headers["Content-Type"] == "image/png"
            assert response.read() == state.png("a")
        with urlopen(base + "/", timeout=2) as response:
            assert b"Pacman live" in response.read()
        with pytest.raises(HTTPError) as caught:
            urlopen(base + "/frames/missing.png", timeout=2)
        assert caught.value.code == 404
    finally:
        server.close()


def test_dashboard_observer_does_not_change_policy_requests():
    """Observer callbacks publish model output without feeding state into prompts."""
    player = PacmanPlayer("server-model", {}, {})
    observation = Observation(np.zeros((8, 9, 3), dtype=np.uint8), {"score": 42})
    state = DashboardState()
    state.plan([{"episode_id": "a", "seed": 1}])
    requests = []

    def generate(request):
        requests.append(deepcopy(request))
        text = "<answer>MOVE U</answer>"
        return SimpleNamespace(
            id="completion",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=text,
                        model_dump=lambda **kwargs: {
                            "role": "assistant",
                            "content": text,
                        },
                    ),
                )
            ],
        )

    plain = PacmanPolicyCodec(player, generation_seed=1, proxy_session=False)
    watched = PacmanPolicyCodec(
        player,
        generation_seed=1,
        proxy_session=False,
        event_observer=lambda event: state.event("a", event),
    )
    plain.decide(observation, None, generate, 0)
    watched.decide(observation, None, generate, 0)
    assert requests[0] == requests[1]
    assert plain.messages == watched.messages
    assert state.snapshot()["episodes"][0]["raw_output"] == "<answer>MOVE U</answer>"
    assert state.snapshot()["episodes"][0]["reasoning"] == ""


def test_dashboard_reasoning_is_display_only_and_isolated_from_concat_history():
    """Dashboard receives bounded reasoning but never sends it in a later request."""
    state = DashboardState()
    state.plan([{"episode_id": "a", "seed": 1}])
    player = PacmanPlayer("server-model", {}, {})
    codec = PacmanPolicyCodec(
        player,
        generation_seed=1,
        proxy_session=False,
        event_observer=lambda event: state.event("a", event),
    )
    requests = []

    def generate(request):
        requests.append(deepcopy(request))
        return SimpleNamespace(
            id="completion",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="<answer>MOVE U</answer>",
                        model_dump=lambda **kwargs: {
                            "role": "assistant",
                            "content": "<answer>MOVE U</answer>",
                            "reasoning_content": "inspect upper lane",
                        },
                    ),
                )
            ],
        )

    observation = Observation(np.zeros((8, 9, 3), dtype=np.uint8), {})
    codec.decide(observation, None, generate, 0)
    codec.decide(observation, None, generate, 1)
    snapshot = state.snapshot()["episodes"][0]
    assert snapshot["reasoning"] == "inspect upper lane"
    assert snapshot["raw_output"] == "<answer>MOVE U</answer>"
    assert "inspect upper lane" not in repr(requests[1]["messages"])


def test_dashboard_action_hooks_distinguish_parsed_and_executed_moves():
    """A legal parse is visible before the environment confirms execution."""
    state = DashboardState()
    state.plan([{"episode_id": "a", "seed": 1}])
    harness = PacmanHarnessAdapter(
        PacmanHarness(), event_observer=lambda event: state.event("a", event)
    )
    harness._visual_actions = SimpleNamespace(
        available_actions=lambda image: ("R",),
    )
    frame = np.zeros((8, 9, 3), dtype=np.uint8)
    context = harness.prepare(Observation(frame, {}))
    action = harness.start(
        Decision("id", {}, "<answer>MOVE R</answer>", "stop"), context
    )
    assert state.snapshot()["episodes"][0]["parsed_action"] == "R"
    assert state.snapshot()["episodes"][0].get("executed_action") is None
    harness.after_step(
        SimpleNamespace(
            action=action,
            terminated=False,
            truncated=False,
            previous=Observation(frame, {}),
            current=Observation(frame, {}),
            evidence={},
        )
    )
    state.event("a", {"kind": "request", "decision": 1})
    row = state.snapshot()["episodes"][0]
    assert row["executed_action"] == "R"
    assert row["phase"] == "thinking"


@pytest.mark.parametrize(
    "outcomes,expected,exit_code",
    [
        (["complete", "complete"], "complete", 0),
        (["complete", "error"], "partial", 2),
        (["error", "error"], "error", 2),
    ],
)
def test_dashboard_global_status_reflects_all_episode_outcomes(
    outcomes, expected, exit_code
):
    """Finishing the evaluator does not turn partial/failed batches into success."""
    state = DashboardState()
    state.plan([{"episode_id": str(i), "seed": i} for i in range(2)])
    for i, status in enumerate(outcomes):
        state.finish(str(i), {"status": status})
    state.complete(
        {
            "planned_episodes": 2,
            "recorded_episodes": 2,
            "completed_episodes": outcomes.count("complete"),
            "pending_episodes": 0,
        }
    )
    assert state.snapshot()["status"] == expected
    assert state.exit_code() == exit_code


def test_dashboard_incomplete_summary_and_early_interrupt_are_not_success():
    """Both the persisted summary and live per-game outcomes gate completion."""
    state = DashboardState()
    state.plan([{"episode_id": "a", "seed": 1}, {"episode_id": "b", "seed": 2}])
    assert state.exit_code() == 130
    state.finish("a", {"status": "complete"})
    state.finish("b", {"status": "complete"})
    state.complete(
        {
            "planned_episodes": 2,
            "recorded_episodes": 1,
            "completed_episodes": 1,
            "pending_episodes": 1,
        }
    )
    assert state.snapshot()["status"] == "partial"
    assert state.exit_code() == 2


def test_dashboard_snapshot_exposes_lifecycle_and_finalizing_phase():
    """Clients can distinguish retained pages from default auto-close runs."""
    state = DashboardState(keep_open=False)
    state.plan([{"episode_id": "a", "seed": 1}])
    state.finish("a", {"status": "complete"})
    state.finalizing()
    snapshot = state.snapshot()
    assert snapshot["status"] == "finalizing"
    assert snapshot["lifecycle"] == "auto_close"
    assert snapshot["keep_open"] is False
    assert state.exit_code() == 130

    retained = DashboardState(keep_open=True).snapshot()
    assert retained["lifecycle"] == "keep_open"
    assert retained["keep_open"] is True


@pytest.mark.asyncio
async def test_evaluation_persists_episodes_before_finalizing_dashboard(
    monkeypatch, tmp_path
):
    """Finalization begins after episode writes and spans client/summary teardown."""
    events = []

    class Client:
        async def __aenter__(self):
            events.append("client-open")
            return self

        async def __aexit__(self, *_):
            events.append("client-closed")

    class Dashboard:
        def plan(self, _):
            events.append("plan")

        def begin(self, _):
            events.append("begin")

        def finish(self, *_):
            events.append("finish")

        def finalizing(self):
            events.append("finalizing")

    async def collect(self, row, **_):
        events.append("collect")
        return SimpleNamespace(
            summary={"status": "complete", "episode_id": row["episode_id"]}
        )

    def write(path, _):
        events.append(f"write-{path.name}")

    def read(_):
        events.append("metrics")
        return {
            "planned_episodes": 1,
            "recorded_episodes": 1,
            "completed_episodes": 1,
            "pending_episodes": 0,
        }

    monkeypatch.setattr(evaluate_module, "AsyncOpenAI", lambda **_: Client())
    monkeypatch.setattr(PacmanPlayer, "collect", collect)
    monkeypatch.setattr(GamePreparation, "source_identity", lambda: {})
    monkeypatch.setattr(
        evaluate_module.SplitManifest, "read", lambda _: SimpleNamespace(dev=(7,))
    )
    monkeypatch.setattr(evaluate_module.ArtifactIdentity, "sha256", lambda _: "digest")
    monkeypatch.setattr(PacmanEvaluation, "_write", write)
    monkeypatch.setattr(evaluate_module.EpisodeMetrics, "read", read)
    monkeypatch.setenv("MAAPACMAN_PACMAN_ROOT", str(tmp_path))
    args = Namespace(
        output=tmp_path / "result",
        manifest=tmp_path / "manifest.json",
        split="dev",
        seed=None,
        limit=1,
        repeats=1,
        generation_seed=11,
        endpoint="http://localhost:30000",
        model="model",
        ghost_reward_target=4,
        temperature=0.2,
        top_p=0.9,
        max_steps=10,
        max_new_tokens=16,
        reasoning=True,
        planner_assisted=True,
        concurrency=1,
        api_key=None,
    )

    await PacmanEvaluation.run(args, dashboard=Dashboard())

    episode_write = events.index("write-dev-7-0.json")
    assert episode_write < events.index("finish") < events.index("finalizing")
    assert events.index("finalizing") < events.index("client-closed")
    assert events.index("client-closed") < events.index("metrics")
    assert events.index("metrics") < events.index("write-summary.json")


@pytest.mark.asyncio
async def test_dashboard_closes_when_evaluation_has_final_results(monkeypatch):
    """Default dashboard lifecycle must not retain server threads after evaluation."""

    async def complete_evaluation(args, dashboard):
        assert dashboard.snapshot()["lifecycle"] == "auto_close"
        assert dashboard.snapshot()["keep_open"] is False
        dashboard.plan([{"episode_id": "a", "seed": 1}])
        dashboard.finish("a", {"status": "complete"})
        return {
            "planned_episodes": 1,
            "recorded_episodes": 1,
            "completed_episodes": 1,
            "pending_episodes": 0,
        }

    monkeypatch.setattr(PacmanEvaluation, "run", complete_evaluation)
    args = Namespace(host="127.0.0.1", port=0, keep_open=False)
    assert await DashboardServer.run(args) == 0


@pytest.mark.asyncio
async def test_dashboard_closes_after_default_evaluation_failure(monkeypatch):
    """A failed default run returns promptly with a failing status."""

    async def fail_evaluation(args, dashboard):
        dashboard.plan([{"episode_id": "a", "seed": 1}])
        raise RuntimeError("provider failed")

    monkeypatch.setattr(PacmanEvaluation, "run", fail_evaluation)
    args = Namespace(host="127.0.0.1", port=0, keep_open=False)
    assert await DashboardServer.run(args) == 2


@pytest.mark.asyncio
async def test_dashboard_keep_open_waits_for_interruption(monkeypatch):
    """Retained viewing is explicit and exits with the finalized result code."""

    async def complete_evaluation(args, dashboard):
        dashboard.plan([{"episode_id": "a", "seed": 1}])
        dashboard.finish("a", {"status": "complete"})
        return {
            "planned_episodes": 1,
            "recorded_episodes": 1,
            "completed_episodes": 1,
            "pending_episodes": 0,
        }

    class InterruptedEvent:
        waited = False

        async def wait(self):
            self.waited = True
            raise asyncio.CancelledError

    interrupted = InterruptedEvent()
    monkeypatch.setattr(PacmanEvaluation, "run", complete_evaluation)
    monkeypatch.setattr(asyncio, "Event", lambda: interrupted)
    args = Namespace(host="127.0.0.1", port=0, keep_open=True)
    assert await DashboardServer.run(args) == 0
    assert interrupted.waited
