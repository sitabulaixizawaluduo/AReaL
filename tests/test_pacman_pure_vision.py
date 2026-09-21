# SPDX-License-Identifier: Apache-2.0

import base64
import builtins
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from examples.vlm.game_player.pacman.tools.cli import PacmanCommands
from examples.vlm.game_player.pacman.tools.encoding import (
    ACTION_REQUEST,
    RESET_ACTION_REQUEST,
    SYSTEM_PROMPT,
    PacmanPolicyCodec,
    _crop_prompt_image,
)
from examples.vlm.game_player.pacman.tools.game import PacmanHarnessAdapter
from examples.vlm.game_player.pacman.tools.harness import PacmanHarness
from examples.vlm.game_player.pacman.tools.player import PacmanPlayer
from examples.vlm.game_player.pacman.tools.rewards import EpisodeReward
from examples.vlm.game_player.pacman.tools.visual_planner import (
    StandaloneVisualPlanner,
    VisualActionSpace,
    VisualPlan,
    reconstruct_visual_level,
)
from examples.vlm.game_player.pacman.train.config import (
    validate_step_efficiency_penalty_weight,
)
from examples.vlm.game_player.protocols import Decision, EpisodeStop, Observation


def test_system_prompt_is_compact_and_preserves_game_contract():
    assert "Blue walls and screen boundaries are impassable" in SYSTEM_PROMPT
    assert "Clear every small pellet to win" in SYSTEM_PROMPT
    assert "third death ends the game" in SYSTEM_PROMPT
    assert "spawn corridor permits only L or R" in SYSTEM_PROMPT
    assert "Each command moves one game step" in SYSTEM_PROMPT
    assert "<answer>MOVE X</answer>" in SYSTEM_PROMPT
    assert len(SYSTEM_PROMPT.split()) < 170


def test_visual_pacman_detector_excludes_pellets_and_hud_lives():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[30:42, 40:52] = (255, 220, 0)  # Pacman-sized maze sprite.
    image[10:13, 10:13] = (255, 220, 0)  # Normal pellet.
    image[94:100, 8:20] = (255, 220, 0)  # Bottom HUD life.
    assert PacmanPolicyCodec._detect_pacman_visual_cell(image) == (2, 2)

    image[50:62, 60:72] = (255, 220, 0)  # Same-sized competing maze object.
    assert PacmanPolicyCodec._detect_pacman_visual_cell(image) is None


def test_visual_pacman_detector_uses_fixed_row_col_tile_grid():
    image = np.zeros((400, 336, 3), dtype=np.uint8)
    image[80:92, 144:156] = (255, 220, 0)
    assert PacmanPolicyCodec._detect_pacman_visual_cell(image) == (5, 9)


def _synthetic_visual_level_frame() -> np.ndarray:
    image = np.full((400, 336, 3), (0, 200, 200), dtype=np.uint8)
    open_cells = [(16, col) for col in range(3, 18)]
    for row, col in open_cells:
        top, left = row * 16, col * 16
        image[top + 4 : top + 13, left + 4 : left + 13] = 0
    for (row, left_col), (_, right_col) in zip(open_cells, open_cells[1:], strict=True):
        y = row * 16 + 8
        image[y - 4 : y + 5, left_col * 16 + 8 : right_col * 16 + 9] = 0
    image[16 * 16 : 16 * 16 + 12, 10 * 16 : 10 * 16 + 12] = (255, 255, 0)
    image[16 * 16 : 16 * 16 + 12, 4 * 16 : 4 * 16 + 12] = (255, 0, 0)
    image[16 * 16 + 7 : 16 * 16 + 9, 13 * 16 + 7 : 13 * 16 + 9] = (
        255,
        255,
        180,
    )
    # Two three-pixel-separated red runs are the visible ghost-pen gate. The
    # component geometry distinguishes them from the red ghost sprite.
    image[184, 152:184] = (255, 0, 0)
    image[187, 152:184] = (255, 0, 0)
    return image


def test_visual_planner_reconstructs_graph_and_advises_from_pixels_only():
    image = _synthetic_visual_level_frame()
    level = reconstruct_visual_level(image)
    assert level is not None
    assert level.pacman_start.row == 16
    assert level.pacman_start.col == 10
    assert level.is_wall(type(level.pacman_start)(11, 10), actor="pacman")

    plan = StandaloneVisualPlanner().plan(image)
    assert plan is not None
    assert plan.action in {"L", "R"}
    assert plan.open_actions == ("L", "R")
    assert plan.pacman_cell == (16, 10)
    assert plan.as_evidence()["source"] == "rgb_pixels_only"
    assert "Pixel plan:" in plan.prompt_text()
    assert len(plan.prompt_text().split()) < 10


def test_visual_planner_fails_closed_when_gate_or_actor_is_ambiguous():
    image = _synthetic_visual_level_frame()
    image[184, 152:184] = 0
    image[187, 152:184] = 0
    assert reconstruct_visual_level(image) is None
    assert StandaloneVisualPlanner().plan(image) is None


def test_visual_plan_hint_and_evidence_are_standalone_only():
    plan = VisualPlan(
        action="R",
        strategy="COLLECT",
        pacman_cell=(16, 10),
        open_actions=("L", "R"),
        normal_ghost_cells=((10, 10),),
        vulnerable_ghost_cells=(),
        visible_normal_pellets=190,
        visible_power_pellets=4,
        planner="edward_visual",
    )
    requests = []

    def generate(request):
        requests.append(deepcopy(request))
        content = "<answer>MOVE R</answer>"
        return SimpleNamespace(
            id="completion",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=content,
                        model_dump=lambda **kwargs: {
                            "role": "assistant",
                            "content": content,
                        },
                    ),
                )
            ],
        )

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for proxy_session in (False, True):
        generation = {"reasoning": False}
        if proxy_session:
            generation["max_tokens"] = 1024
        codec = PacmanPolicyCodec(
            PacmanPlayer("server-model", generation, {}),
            generation_seed=1,
            proxy_session=proxy_session,
        )
        if not proxy_session:
            codec._visual_planner = SimpleNamespace(
                plan=lambda pixels: plan,
                record_model_action=lambda action: None,
            )
        decision = codec.decide(Observation(image), None, generate, 0)
        prompt = requests[-1]["messages"][-1]["content"][0]["text"]
        if proxy_session:
            assert "Pixel-derived visual plan" not in prompt
            assert decision.evidence["visual_plan"] is None
        else:
            assert "recommend=R mode=C" in prompt
            assert decision.evidence["visual_plan"]["source"] == "rgb_pixels_only"


def test_prompt_crop_preserves_tile_resolution_and_black_pads_edges():
    image = np.full((400, 336, 3), 17, dtype=np.uint8)
    crop, metadata = _crop_prompt_image(image, (1, 1))

    assert crop.shape == (208, 208, 3)
    assert metadata["requested_bounds_pixels_top_left_bottom_right"] == [
        -80,
        -80,
        128,
        128,
    ]
    assert metadata["padding_pixels_top_left_bottom_right"] == [80, 80, 0, 0]
    assert np.all(crop[:80] == 0)
    assert np.all(crop[:, :80] == 0)
    assert np.all(crop[80:, 80:] == 17)


def _move_synthetic_pacman(image: np.ndarray, source_col: int, target_col: int):
    moved = image.copy()
    row = 16
    moved[row * 16 : row * 16 + 12, source_col * 16 : source_col * 16 + 12] = 0
    moved[row * 16 : row * 16 + 12, target_col * 16 : target_col * 16 + 12] = (
        255,
        255,
        0,
    )
    return moved


@pytest.mark.parametrize("proxy_session", [False, True])
def test_prompt_uses_reset_full_frame_then_fixed_crop(proxy_session):
    generation = {"reasoning": False}
    if proxy_session:
        generation["max_tokens"] = 4096
    codec = PacmanPolicyCodec(
        PacmanPlayer("server-model", generation, {}),
        generation_seed=1,
        proxy_session=proxy_session,
    )
    reset = _synthetic_visual_level_frame()
    moved = _move_synthetic_pacman(reset, 10, 11)
    requests = []

    def generate(request):
        requests.append(deepcopy(request))
        content = "<answer>MOVE R</answer>"
        return SimpleNamespace(
            id=f"completion-{len(requests)}",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=content,
                        model_dump=lambda **kwargs: {
                            "role": "assistant",
                            "content": content,
                        },
                    ),
                )
            ],
        )

    reset_decision = codec.decide(Observation(reset, {}), None, generate, 0)
    crop_decision = codec.decide(Observation(moved, {}), None, generate, 1)

    assert reset_decision.evidence["prompt_image_scope"] == "reset"
    assert (
        reset_decision.evidence["prompt_image_sha256"]
        == reset_decision.evidence["full_image_sha256"]
    )
    assert crop_decision.evidence["prompt_image_scope"] == "crop"
    assert crop_decision.evidence["prompt_image_crop"] == {
        "center_cell_row_col": [16, 11],
        "requested_bounds_pixels_top_left_bottom_right": [160, 80, 368, 288],
        "source_bounds_pixels_top_left_bottom_right": [160, 80, 368, 288],
        "padding_pixels_top_left_bottom_right": [0, 0, 0, 0],
        "output_height_width": [208, 208],
    }
    assert (
        crop_decision.evidence["prompt_image_sha256"]
        != crop_decision.evidence["full_image_sha256"]
    )
    second_users = [
        message for message in requests[1]["messages"] if message["role"] == "user"
    ]
    expected_sizes = [(336, 400), (208, 208)] if proxy_session else [(208, 208)]
    actual_sizes = []
    for message in second_users:
        url = message["content"][1]["image_url"]["url"]
        with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
            actual_sizes.append(image.size)
    assert actual_sizes == expected_sizes


def test_prompt_falls_back_to_full_frame_after_nonportal_jump():
    codec = PacmanPolicyCodec(
        PacmanPlayer("server-model", {"reasoning": False}, {}),
        generation_seed=1,
        proxy_session=False,
    )
    reset = _synthetic_visual_level_frame()
    respawned = _move_synthetic_pacman(reset, 10, 15)
    requests = []

    def generate(request):
        requests.append(deepcopy(request))
        content = "<answer>MOVE R</answer>"
        return SimpleNamespace(
            id=f"completion-{len(requests)}",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=content,
                        model_dump=lambda **kwargs: {
                            "role": "assistant",
                            "content": content,
                        },
                    ),
                )
            ],
        )

    codec.decide(Observation(reset, {}), None, generate, 0)
    decision = codec.decide(Observation(respawned, {}), None, generate, 1)

    assert decision.evidence["prompt_image_scope"] == "fallback"
    assert (
        decision.evidence["prompt_image_fallback_reason"] == "nonportal_jump_or_respawn"
    )
    url = requests[1]["messages"][-1]["content"][1]["image_url"]["url"]
    with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
        assert image.size == (336, 400)


def test_harness_legal_actions_come_from_pixels_not_observation_info():
    frame = _synthetic_visual_level_frame()
    adapter = PacmanHarnessAdapter(PacmanHarness(), strict_legality=True)

    legal = adapter.prepare(Observation(frame, {"legal_actions": ["U", "D"]}))

    assert legal == ["L", "R"]
    with pytest.raises(ValueError, match="RGB pixels"):
        adapter.prepare(
            Observation(np.zeros((8, 8, 3), dtype=np.uint8), {"legal_actions": ["R"]})
        )


def test_visual_available_actions_include_horizontal_portal_entry():
    image = np.full((400, 336, 3), (0, 200, 200), dtype=np.uint8)
    row = 12
    for col in range(21):
        top, left = row * 16, col * 16
        image[top + 4 : top + 13, left + 4 : left + 13] = 0
    for col in range(20):
        y = row * 16 + 8
        image[y - 4 : y + 5, col * 16 + 8 : (col + 1) * 16 + 9] = 0
    image[row * 16 : row * 16 + 12, 16:28] = (255, 255, 0)
    image[row * 16 : row * 16 + 12, 10 * 16 : 10 * 16 + 12] = (255, 0, 0)
    image[row * 16 + 7 : row * 16 + 9, 15 * 16 + 7 : 15 * 16 + 9] = (
        255,
        255,
        180,
    )
    image[184, 152:184] = (255, 0, 0)
    image[187, 152:184] = (255, 0, 0)

    action_space = VisualActionSpace()
    assert action_space.available_actions(image) == ("L", "R")


@pytest.mark.parametrize("proxy_session", [False, True])
def test_visual_blocked_feedback_is_standalone_only(proxy_session):
    generation = {"reasoning": False}
    if proxy_session:
        generation["max_tokens"] = 1024
    codec = PacmanPolicyCodec(
        PacmanPlayer("server-model", generation, {}),
        generation_seed=1,
        proxy_session=proxy_session,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[30:42, 40:52] = (255, 220, 0)
    requests = []

    def generate(request):
        requests.append(deepcopy(request))
        action = "R" if len(requests) == 1 else "L"
        content = f"<answer>MOVE {action}</answer>"
        return SimpleNamespace(
            id=f"completion-{len(requests)}",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=content,
                        model_dump=lambda **kwargs: {
                            "role": "assistant",
                            "content": content,
                        },
                    ),
                )
            ],
        )

    codec.decide(Observation(image, {"legal_actions": ["R"]}), None, generate, 0)
    decision = codec.decide(
        Observation(image, {"legal_actions": ["L"]}), None, generate, 1
    )
    request_text = requests[1]["messages"][-1]["content"][0]["text"]
    if proxy_session:
        assert "did not move Pacman" not in request_text
        assert not decision.evidence["visual_blocked_action_detected"]
    else:
        assert "Previous MOVE R did not move Pacman; R is blocked." in request_text
        assert decision.evidence["visual_blocked_action_detected"]
        assert decision.evidence["previous_parsed_action"] == "R"


def test_standalone_reasoning_flag_defaults_on_and_can_be_disabled():
    parser = PacmanCommands.parser()
    common = [
        "evaluate",
        "--endpoint",
        "http://localhost:30000/v1",
        "--model",
        "model",
        "--manifest",
        "splits.json",
        "--output",
        "output",
    ]
    assert parser.parse_args(common).reasoning is True
    assert parser.parse_args(common).temperature == 0.2
    assert parser.parse_args(common).top_p == 0.9
    assert parser.parse_args([*common, "--no-reasoning"]).reasoning is False
    assert parser.parse_args([*common, "--reasoning"]).reasoning is True
    tuned = parser.parse_args([*common, "--temperature", "0.1", "--top-p", "0.8"])
    assert tuned.temperature == 0.1
    assert tuned.top_p == 0.8


@pytest.mark.parametrize("proxy_session", [False, True])
def test_policy_prompt_hides_state_and_selects_history_contract(proxy_session):
    """Standalone keeps one current frame while proxy retains complete concat."""
    owner = PacmanPlayer(
        model="server-only-model-alias",
        generation={
            "max_new_tokens": 16,
            "temperature": 0.7,
            "top_p": 1.0,
            **({"max_tokens": 100000} if proxy_session else {}),
        },
        options={},
    )
    codec = PacmanPolicyCodec(owner, generation_seed=17, proxy_session=proxy_session)
    hidden = {
        "position": [901, 902],
        "facing": "hidden-facing",
        "normal_pellets_remaining": 991,
        "power_pellets_remaining": 992,
        "deaths": 993,
        "ghosts": ["hidden-ghost"],
        "edible_ticks": 994,
        "legal_moves": ["hidden-direction"],
        "safe_hints": ["hidden-option"],
    }
    pixels = [np.full((13, 17, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
    requests = []

    def generate(request):
        requests.append(deepcopy(request))
        content = "visual note <answer>MOVE R</answer>"
        message = SimpleNamespace(
            content=content,
            model_dump=lambda **kwargs: {"role": "assistant", "content": content},
        )
        return SimpleNamespace(
            id=f"completion-{len(requests)}",
            usage=SimpleNamespace(completion_tokens=8),
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
        )

    decision = codec.decide(Observation(pixels[0], hidden, hidden), hidden, generate, 0)
    first_history = deepcopy(codec.messages)
    codec.decide(
        Observation(pixels[1], {"changed": True}, {"changed": True}),
        {"new": "hidden-context"},
        generate,
        1,
    )
    codec.decide(
        Observation(pixels[2], {"changed_again": True}, {"changed_again": True}),
        {"newer": "hidden-context"},
        generate,
        2,
    )

    assert "prompt_tokens_counted" not in decision.evidence
    assert "context_budget_upper_bound" not in decision.evidence
    assert decision.evidence["completion"] == "visual note <answer>MOVE R</answer>"
    assert decision.evidence["token_usage"] == {
        "prompt_tokens": None,
        "completion_tokens": 8,
        "total_tokens": None,
    }
    for request in requests:
        assert request["model"] == "server-only-model-alias"
        if proxy_session:
            assert request["extra_body"]["max_total_tokens"] == 100000
        else:
            assert request["extra_body"] == {
                "chat_template_kwargs": {"enable_thinking": True}
            }
    assert requests[0]["messages"][0]["role"] == "system"
    assert SYSTEM_PROMPT in requests[0]["messages"][0]["content"]
    assert requests[1]["messages"][:-1] == first_history
    user_counts = [
        sum(message["role"] == "user" for message in request["messages"])
        for request in requests
    ]
    if proxy_session:
        assert user_counts == [1, 2, 3]
        assert len(codec.messages) == 7
        assert codec.messages[2]["content"] == "visual note <answer>MOVE R</answer>"
    else:
        assert user_counts == [1, 1, 1]
        assert [message["role"] for message in codec.messages] == ["system"]
    assert requests[0]["messages"][-1]["content"][0] == {
        "type": "text",
        "text": RESET_ACTION_REQUEST,
    }
    assert requests[1]["messages"][-1]["content"][0] == {
        "type": "text",
        "text": ACTION_REQUEST,
    }
    latest_users = [
        message for message in requests[-1]["messages"] if message["role"] == "user"
    ]
    expected_pixels = pixels if proxy_session else pixels[-1:]
    assert len(latest_users) == len(expected_pixels)
    for message, expected in zip(latest_users, expected_pixels, strict=True):
        assert len(message["content"]) == 2
        url = message["content"][1]["image_url"]["url"]
        with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
            assert image.size == (17, 13)
            np.testing.assert_array_equal(np.asarray(image), expected)


@pytest.mark.parametrize("direction", ["U", "D", "L", "R"])
def test_move_parser_accepts_only_fixed_direction(direction):
    """Each supported direction maps to a single environment move."""
    decision = Decision("id", {}, f"<answer>MOVE {direction}</answer>", "stop")
    assert PacmanHarness.parse(decision) == direction
    assert decision.evidence["parse_valid"]
    assert decision.evidence["strict_format_valid"]
    assert decision.evidence["format_valid"]


@pytest.mark.parametrize(
    "text",
    [
        "<answer>OPTION A0</answer>",
        "<answer>MOVE A0</answer>",
        "<answer>MOVE UL</answer>",
        "<answer>MOVE u</answer>",
        "MOVE U",
        "<answer>MOVE U</answer><answer>MOVE R</answer>",
    ],
)
def test_move_parser_rejects_options_and_nonprotocol_text(text):
    """No option IDs, combined moves or explanations enter execution."""
    decision = Decision("id", {}, text, "stop")
    with pytest.raises(EpisodeStop, match="invalid_format"):
        PacmanHarness.parse(decision)
    assert not decision.evidence["parse_valid"]
    assert not decision.evidence["strict_format_valid"]


@pytest.mark.parametrize(
    "text",
    [
        "I should move right. <answer>MOVE R</answer>",
        "<answer>MOVE R</answer> because the corridor is clear",
        "reasoning before\n<answer>\nMOVE R\n</answer>\nexplanation after",
    ],
)
def test_move_parser_allows_text_outside_one_valid_answer(text):
    decision = Decision("id", {}, text, "stop")
    assert PacmanHarness.parse(decision) == "R"
    assert decision.evidence["parse_valid"]
    assert not decision.evidence["strict_format_valid"]
    assert not decision.evidence["format_valid"]


def test_harness_tracks_all_strict_across_the_episode():
    harness = PacmanHarness()
    adapter = PacmanHarnessAdapter(harness, strict_legality=True)
    strict = Decision("strict", {}, "  <answer>MOVE R</answer>\n", "stop")
    prose = Decision("prose", {}, "go <answer>MOVE R</answer>", "stop")

    assert adapter.start(strict, ["R"]) == "R"
    assert harness.all_strict
    assert adapter.start(prose, ["R"]) == "R"
    assert not harness.all_strict
    assert harness.format_decisions == 2


def test_episode_reward_combines_game_score_and_all_strict():
    initial = {
        "normal_pellets_remaining": 100,
        "power_pellets_remaining": 4,
        "step": 0,
    }
    final = {
        "normal_pellets_remaining": 50,
        "power_pellets_remaining": 2,
        "death_count": 1,
        "step": 256,
    }
    strict_reward = EpisodeReward(initial)
    strict_total, strict_components = strict_reward.finish(
        "budget_exhausted", final, all_strict=True
    )
    loose_reward = EpisodeReward(initial)
    loose_total, loose_components = loose_reward.finish(
        "budget_exhausted", final, all_strict=False
    )

    # Existing bounded game reward: 0.5*0.5 + 0.05*0.5 - 0.02 = 0.255.
    assert strict_reward.bounded_game_reward == pytest.approx(0.255)
    assert strict_total == pytest.approx(0.9 * 0.255 + 0.1)
    assert loose_total == pytest.approx(0.9 * 0.255)
    assert strict_components["strict_serialization"] == pytest.approx(0.1)
    assert loose_components["strict_serialization"] == 0.0
    assert strict_components["completion_step_penalty"] == 0.0
    assert sum(strict_components.values()) == pytest.approx(strict_total)
    assert sum(loose_components.values()) == pytest.approx(loose_total)


@pytest.mark.parametrize("reason", ["invalid_format", "invalid_action"])
def test_invalid_policy_endings_cancel_game_and_strict_rewards(reason):
    reward = EpisodeReward(
        {
            "normal_pellets_remaining": 100,
            "power_pellets_remaining": 4,
            "step": 7,
        }
    )
    total, components = reward.finish(
        reason,
        {
            "normal_pellets_remaining": 50,
            "power_pellets_remaining": 2,
            "death_count": 1,
            "step": 128,
        },
        all_strict=True,
    )
    assert total == 0.0
    assert reward.strict_format_bonus == 0.0
    assert components["strict_serialization"] == 0.0
    assert sum(components.values()) == pytest.approx(0.0)


def _finish_win_at_step(
    step: int,
    *,
    weight: float = 0.05,
    all_strict: bool = False,
):
    reward = EpisodeReward(
        {
            "normal_pellets_remaining": 100,
            "power_pellets_remaining": 4,
            "step": 7,
        },
        max_steps=512,
        step_efficiency_penalty_weight=weight,
    )
    total, components = reward.finish(
        "all_normal_pellets",
        {
            "normal_pellets_remaining": 0,
            "power_pellets_remaining": 4,
            "death_count": 0,
            "step": step + 7,
        },
        all_strict=all_strict,
    )
    return reward, total, components


def test_faster_completion_receives_a_higher_reward():
    fast, fast_total, fast_components = _finish_win_at_step(100)
    slow, slow_total, slow_components = _finish_win_at_step(800)

    assert fast_total > slow_total
    assert fast.env_steps == 100
    assert fast.completion_step_ratio == pytest.approx(100 / 512)
    assert fast.completion_step_penalty == pytest.approx(-0.05 * 100 / 512)
    assert slow.completion_step_ratio == 1.0
    assert slow.completion_step_penalty == pytest.approx(-0.05)
    assert fast_components["completion_step_penalty"] == pytest.approx(
        0.9 * fast.completion_step_penalty
    )
    assert sum(fast_components.values()) == pytest.approx(fast_total)
    assert sum(slow_components.values()) == pytest.approx(slow_total)


def test_failure_at_the_same_step_has_no_efficiency_penalty():
    win, _, _ = _finish_win_at_step(100)
    failure = EpisodeReward(
        {
            "normal_pellets_remaining": 100,
            "power_pellets_remaining": 4,
            "step": 0,
        },
        max_steps=512,
        step_efficiency_penalty_weight=0.05,
    )
    _, components = failure.finish(
        "budget_exhausted",
        {
            "normal_pellets_remaining": 10,
            "power_pellets_remaining": 4,
            "death_count": 0,
            "step": 100,
        },
        all_strict=False,
    )

    assert win.completion_step_penalty < 0
    assert failure.completion_step_ratio is None
    assert failure.completion_step_penalty == 0.0
    assert components["completion_step_penalty"] == 0.0


def test_zero_efficiency_weight_preserves_the_previous_reward_formula():
    reward, total, components = _finish_win_at_step(512, weight=0.0, all_strict=True)
    assert reward.bounded_game_reward == pytest.approx(0.9)
    assert reward.completion_step_penalty == 0.0
    assert components["completion_step_penalty"] == 0.0
    assert total == pytest.approx(0.9 * 0.9 + 0.1)


@pytest.mark.parametrize("value", [-0.1, 0.100001, float("nan"), float("inf")])
def test_step_efficiency_weight_validation_rejects_invalid_values(value):
    with pytest.raises(ValueError, match=r"finite and in \[0, 0\.1\]"):
        validate_step_efficiency_penalty_weight(value)


@pytest.mark.parametrize("value", [0.0, 0.05, 0.1])
def test_step_efficiency_weight_validation_accepts_bounded_values(value):
    validate_step_efficiency_penalty_weight(value)


def test_proxy_rejects_blocked_move_but_standalone_executes_and_audits_it():
    """Standalone wall collisions are natural no-ops; proxy semantics stay strict."""
    context = ["R"]
    proxy = PacmanHarnessAdapter(PacmanHarness(), strict_legality=True)
    with pytest.raises(EpisodeStop, match="invalid_action"):
        proxy.start(Decision("id", {}, "<answer>MOVE U</answer>", "stop"), context)

    standalone = PacmanHarnessAdapter(PacmanHarness(), strict_legality=False)
    decision = Decision("id", {}, "<answer>MOVE U</answer>", "stop")
    action = standalone.start(decision, context)
    frame = _synthetic_visual_level_frame()
    result = standalone.after_step(
        SimpleNamespace(
            action=action,
            previous=Observation(frame, {"legal_actions": ["R"]}),
            current=Observation(frame, {"legal_actions": ["R"]}),
            evidence={},
            terminated=False,
            truncated=False,
        )
    )
    assert action == "U"
    assert decision.evidence["action_legal"] is False
    assert decision.evidence["blocked_action"] is True
    assert result.action is None
    assert result.status == "blocked"


def test_player_construction_does_not_import_model_libraries(monkeypatch):
    """An opaque server model name works without Transformers or torch locally."""
    original_import = builtins.__import__

    def reject_model_import(name, *args, **kwargs):
        if name.split(".")[0] in {"transformers", "torch", "huggingface_hub"}:
            raise AssertionError(f"Player tried to import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_model_import)
    player = PacmanPlayer("remote-only-alias", generation={}, options={})
    assert player.model == "remote-only-alias"
    assert player.gconfig.max_tokens is None
    assert not hasattr(player, "processor")
    assert not hasattr(player, "tokenizer")
    assert not hasattr(player, "processor_lock")


@pytest.mark.parametrize("proxy_session", [False, True])
def test_remote_context_error_propagates_without_synthetic_game_result(proxy_session):
    """The server's context rejection must not become EpisodeStop or a reward."""
    player = PacmanPlayer(
        "remote-only-alias", {"max_tokens": 1024} if proxy_session else {}, {}
    )
    codec = PacmanPolicyCodec(player, generation_seed=1, proxy_session=proxy_session)
    error = RuntimeError("remote context limit exceeded")

    def reject(request):
        raise error

    with pytest.raises(RuntimeError) as caught:
        codec.decide(
            Observation(np.zeros((8, 8, 3), dtype=np.uint8), {}), None, reject, 0
        )
    assert caught.value is error
    assert len(codec.messages) == 1
    assert codec.messages[0]["role"] == "system"
    assert SYSTEM_PROMPT in codec.messages[0]["content"]


def test_standalone_reasoning_is_accepted_without_entering_next_prompt():
    """Standalone enables provider thinking without retaining assistant history."""
    player = PacmanPlayer("remote-only-alias", {}, {})
    codec = PacmanPolicyCodec(player, generation_seed=1, proxy_session=False)
    requests = []

    def generate(request):
        requests.append(deepcopy(request))
        text = "<answer>MOVE R</answer>"
        reasoning = "The right corridor is clear."
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
                            "reasoning_content": reasoning,
                        },
                    ),
                )
            ],
        )

    decision = codec.decide(
        Observation(np.zeros((8, 8, 3), dtype=np.uint8), {}), None, generate, 0
    )
    assert PacmanHarness.parse(decision) == "R"
    codec.decide(
        Observation(np.zeros((8, 8, 3), dtype=np.uint8), {}), None, generate, 1
    )
    assert requests[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
    assert codec.messages == [codec.messages[0]]
    assert "reasoning_content" not in repr(requests[1]["messages"])


def test_standalone_splits_qwen_prefilled_thinking_from_final_content():
    """Qwen may put the opening think tag in its prompt and only emit its close."""
    player = PacmanPlayer("remote-only-alias", {}, {})
    events = []
    codec = PacmanPolicyCodec(
        player,
        generation_seed=1,
        proxy_session=False,
        event_observer=events.append,
    )
    raw = "inspect the upper lane\n</think>\n\n<answer>MOVE R</answer>"

    def generate(request):
        return SimpleNamespace(
            id="completion",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=raw,
                        model_dump=lambda **kwargs: {
                            "role": "assistant",
                            "content": raw,
                        },
                    ),
                )
            ],
        )

    decision = codec.decide(
        Observation(np.zeros((8, 8, 3), dtype=np.uint8), {}), None, generate, 0
    )

    assert PacmanHarness.parse(decision) == "R"
    assert decision.text == "<answer>MOVE R</answer>"
    assert decision.choice["reasoning_content"] == "inspect the upper lane"
    assert decision.evidence["raw_completion"] == raw
    assert [message["role"] for message in codec.messages] == ["system"]
    assert events[-1] == {
        "kind": "decision",
        "text": "<answer>MOVE R</answer>",
        "reasoning": "inspect the upper lane",
    }


def test_standalone_no_reasoning_keeps_visible_prose_parseable_but_not_strict():
    """Without splitting, visible reasoning prose loses only the strict bonus."""
    player = PacmanPlayer("remote-only-alias", {"reasoning": False}, {})
    codec = PacmanPolicyCodec(player, generation_seed=1, proxy_session=False)
    raw = "inspect the upper lane\n</think>\n<answer>MOVE R</answer>"
    requests = []

    def generate(request):
        requests.append(deepcopy(request))
        return SimpleNamespace(
            id="completion",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=raw,
                        model_dump=lambda **kwargs: {
                            "role": "assistant",
                            "content": raw,
                            "reasoning_content": "inspect the upper lane",
                        },
                    ),
                )
            ],
        )

    decision = codec.decide(
        Observation(np.zeros((8, 8, 3), dtype=np.uint8), {}), None, generate, 0
    )
    assert requests[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert decision.text == raw
    assert decision.choice["reasoning_content"] == "inspect the upper lane"
    assert not decision.evidence["reasoning_content_allowed"]
    assert PacmanHarness.parse(decision) == "R"
    assert decision.evidence["parse_valid"]
    assert not decision.evidence["strict_format_valid"]


def test_separate_reasoning_content_does_not_change_visible_strict_format():
    """Strictness is defined only over visible content."""
    decision = Decision(
        "id",
        {"reasoning_content": "not a policy token"},
        "<answer>MOVE U</answer>",
        "stop",
        evidence={"reasoning_content_allowed": False},
    )
    assert PacmanHarness.parse(decision) == "U"
    assert decision.evidence["parse_valid"]
    assert decision.evidence["strict_format_valid"]


@pytest.mark.parametrize("field", ["tool_calls", "refusal"])
def test_standalone_rejects_tool_calls_and_refusal_with_reasoning(field):
    """Thinking never relaxes the strict single-move response protocol."""
    choice = {"reasoning_content": "private", field: "unsupported"}
    decision = Decision(
        "id",
        choice,
        "<answer>MOVE U</answer>",
        "stop",
        evidence={"reasoning_content_allowed": True},
    )
    with pytest.raises(EpisodeStop, match="invalid_format"):
        PacmanHarness.parse(decision)
