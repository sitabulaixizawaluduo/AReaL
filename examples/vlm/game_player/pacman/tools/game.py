# SPDX-License-Identifier: Apache-2.0

"""Pacman game-specific adapters and the existing episode artifact contract."""

import asyncio
import hashlib
import json
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from examples.vlm.game_player.pacman.tools.encoding import (
    PacmanPolicyCodec,
    VisualTransitionTracker,
)
from examples.vlm.game_player.pacman.tools.harness import PacmanHarness
from examples.vlm.game_player.pacman.tools.rewards import (
    DEFAULT_STEP_EFFICIENCY_PENALTY_WEIGHT,
    EpisodeReward,
)
from examples.vlm.game_player.pacman.tools.storage import JsonEpisodeStore
from examples.vlm.game_player.pacman.tools.visual_planner import (
    StandaloneVisualPlanner,
    VisualActionSpace,
    detect_ghost_mode_signature,
    detect_pacman_cell,
)
from examples.vlm.game_player.protocols import (
    ActionResult,
    Decision,
    EpisodeContext,
    EpisodeRecord,
    EpisodeStop,
    GameStep,
    Observation,
    RewardValue,
    Session,
    Transition,
)


class PacmanEpisodeArtifacts:
    """Preserve the published Pacman audit schema independently of the runtime."""

    def __init__(
        self, owner: Any, context: EpisodeContext, source_model_version: int | None
    ):
        self.owner = owner
        self.context = context
        self.source_model_version = source_model_version
        self.attempt_header: dict[str, Any] = {}

    def begin(self, data: dict[str, Any], *, training: bool) -> dict[str, Any]:
        options = self.owner.options
        seed = int(data["seed"] if "seed" in data else data["env"]["seed"])
        is_eval = self.context.is_eval or bool(options.get("is_eval")) or not training
        split = str(data.get("split", "validation" if is_eval else "train"))
        # Dataset IDs identify cases, not sampled games. UUID also distinguishes
        # recovery sessions, where the controller's task counter starts over.
        attempt_id = self.context.attempt_id or uuid.uuid4().hex
        case_id = str(data.get("episode_id", f"{split}-{seed}"))
        episode_id = (
            f"{case_id}-{self.context.task_id}-{self.context.sample_idx}-{attempt_id}"
        )
        seed_material = (
            int(data.get("generation_seed", self.owner.gconfig.seed or 1)),
            seed,
            self.context.task_id,
            self.context.sample_idx,
        )
        generation_seed = (
            int(data["generation_seed"])
            if "generation_seed" in data and self.context.sample_idx is None
            else int.from_bytes(
                hashlib.sha256(repr(seed_material).encode()).digest()[:4], "big"
            )
            & 0x7FFFFFFF
        )
        self.attempt_header = {
            "schema_version": "pacman-community-episode-v1",
            "attempt_id": attempt_id,
            "episode_id": episode_id,
            "case_id": case_id,
            "session_namespace": self.owner.session_namespace,
            "task_id": self.context.task_id,
            "sample_idx": self.context.sample_idx,
            "seed": seed,
            "generation_seed": generation_seed,
            "split": split,
            "planner_assisted": bool(options.get("planner_assisted", True)),
            "planner_source": (
                "rgb_pixels_only"
                if bool(options.get("planner_assisted", True))
                else None
            ),
            "requested_model_version": self.source_model_version,
        }
        return self.attempt_header

    def finish(
        self,
        *,
        env: Any,
        initial_info: dict[str, Any],
        initial_state_hash: str,
        info: dict[str, Any],
        harness: Any,
        reward: Any,
        decisions: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        max_steps: int,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        options = self.owner.options
        header = self.attempt_header
        attempt_id = header["attempt_id"]
        episode_id = header["episode_id"]
        case_id = header["case_id"]
        seed = header["seed"]
        generation_seed = header["generation_seed"]
        split = header["split"]
        won = info.get("terminal_reason") == "all_normal_pellets"
        deaths = int(info["death_count"])
        summary = {
            "schema_version": "pacman-community-episode-v1",
            "status": "complete",
            "attempt_id": attempt_id,
            "session_namespace": self.owner.session_namespace,
            "experiment_name": options.get("experiment_name"),
            "trial_name": options.get("trial_name"),
            "episode_id": episode_id,
            "episode_int_id": int.from_bytes(
                hashlib.blake2b(episode_id.encode(), digest_size=8).digest(), "big"
            )
            & ((1 << 63) - 1),
            "case_id": case_id,
            "task_id": self.context.task_id,
            "sample_idx": self.context.sample_idx,
            "seed": seed,
            "generation_seed": generation_seed,
            "split": split,
            "initial_state_sha256": initial_state_hash,
            "provenance": env.provenance,
            "harness": (
                "rgb_visual_planner_interruptible_options_v4"
                if bool(options.get("planner_assisted", True))
                else "pure_vision_single_moves_v2"
            ),
            "planner_assisted": bool(options.get("planner_assisted", True)),
            "planner_source": header["planner_source"],
            "reward_objective_contract": "completion_only_step_penalty_strict_format_v3",
            "ghost_mode": "normal",
            "episode_life_mode": "third_death_ends_episode",
            "death_limit": 3,
            "max_steps": max_steps,
            "trainable": bool(decisions),
            "collection_scope": "all_attempts",
            "training_export_status": "proxy_owned"
            if self.context.training
            else "not_requested",
            "inference_evidence_status": "proxy_owned"
            if self.context.proxy_session_id
            else "sdk",
            "win": won,
            "zero_death_win": won and deaths == 0,
            "death_count": deaths,
            "normal_pellet_clear_rate": 1
            - int(info["normal_pellets_remaining"]) / reward.initial_normal,
            "normal_pellets_initial": reward.initial_normal,
            "normal_pellets_remaining": int(info["normal_pellets_remaining"]),
            "game_score": int(info["score"]),
            "total_shaped_reward": reward.total,
            "reward": reward.total,
            "bounded_game_reward": reward.bounded_game_reward,
            "weighted_game_reward": reward.weighted_game_reward,
            "strict_serialization_bonus": reward.strict_format_bonus,
            "all_strict": reward.all_strict,
            "step_efficiency_penalty_weight": reward.step_efficiency_penalty_weight,
            "completion_step_ratio": reward.completion_step_ratio,
            "completion_step_penalty": reward.completion_step_penalty,
            "weighted_completion_step_penalty": reward.components.get(
                "completion_step_penalty", 0.0
            ),
            "special_pellet_clear_rate": 1
            - int(info["power_pellets_remaining"]) / reward.initial_power
            if reward.initial_power
            else 0.0,
            "ghosts_eaten": reward.ghosts_eaten,
            "ghost_reward_target": reward.ghost_reward_target,
            "reward_components": reward.components,
            "env_steps": reward.env_steps,
            "logic_frames": int(info["logic_frame"]) - int(initial_info["logic_frame"]),
            "decisions": len(decisions),
            "terminal_reason": info["terminal_reason"],
            "safety_refusal_triggers": 0,
            "invalid_format": info["terminal_reason"] == "invalid_format",
            "invalid_action": info["terminal_reason"] == "invalid_action",
            "parse_valid_rate": sum(row["parse_valid"] for row in decisions)
            / len(decisions)
            if decisions
            else 0.0,
            # Compatibility metric now measures the strict visible wire format.
            "format_valid_rate": sum(row["strict_format_valid"] for row in decisions)
            / len(decisions)
            if decisions
            else 0.0,
            "strict_format_valid_rate": sum(
                row["strict_format_valid"] for row in decisions
            )
            / len(decisions)
            if decisions
            else 0.0,
            "action_legal_rate": sum(row["action_legal"] for row in decisions)
            / len(decisions)
            if decisions
            else 0.0,
            "planner_hint_rate": sum(
                row.get("planner_recommended_action") is not None for row in decisions
            )
            / len(decisions)
            if decisions
            else 0.0,
            "planner_recommendation_match_rate": (
                sum(
                    bool(row.get("planner_recommendation_match"))
                    for row in decisions
                    if row.get("planner_recommended_action") is not None
                )
                / sum(
                    row.get("planner_recommended_action") is not None
                    for row in decisions
                )
                if any(
                    row.get("planner_recommended_action") is not None
                    for row in decisions
                )
                else None
            ),
            "wall_collision_count": sum(
                bool(row.get("wall_collision")) for row in decisions
            ),
            "wall_collision_rate": sum(
                bool(row.get("wall_collision")) for row in decisions
            )
            / len(decisions)
            if decisions
            else 0.0,
            # Legacy safety-advice metric is distinct from the visual planner.
            "safe_advice_match_rate": None,
            "elapsed_seconds": elapsed_seconds,
            "model_versions": sorted(
                {
                    decision["model_version"]
                    for decision in decisions
                    if decision.get("model_version") is not None
                }
            ),
            "requested_model_version": self.source_model_version,
            "mixed_model_versions": len(
                {
                    decision["model_version"]
                    for decision in decisions
                    if decision.get("model_version") is not None
                }
            )
            > 1,
            "decision_records": decisions,
            "trajectory": trajectory,
        }
        return summary

    def summarize(
        self,
        record: EpisodeRecord,
        game: "PacmanGame",
        harness: PacmanHarness,
        reward: "PacmanRewardAdapter",
    ) -> dict[str, Any]:
        if record.final is None:
            raise ValueError(
                "A complete Pacman episode requires an initial observation"
            )
        decisions = []
        for decision in record.decisions:
            last = (
                record.transitions[decision.end_step - 1]
                if decision.end_step > decision.start_step
                else None
            )
            decisions.append(
                {
                    **decision.evidence,
                    "decision": decision.index,
                    "start_env_step": decision.start_step,
                    "end_env_step": decision.end_step,
                    "option_return": decision.total_reward,
                    "option_status": last.status if last is not None else record.reason,
                    "executed": last is not None,
                    "wall_collision": bool(
                        last is not None
                        and last.evidence.get("visual_move", {}).get("blocked")
                    ),
                }
            )
        trajectory = []
        for transition in record.transitions:
            info = transition.current.info
            trajectory.append(
                {
                    "env_step": int(info["step"]),
                    "logic_frame": int(info["logic_frame"]),
                    "logic_frames": int(info["logic_frames"]),
                    "action": transition.action,
                    "decision": transition.decision_index,
                    "pacman_position": info["pacman_position"],
                    "score": int(info["score"]),
                    "normal_pellets_remaining": int(info["normal_pellets_remaining"]),
                    "death_count": int(info["death_count"]),
                    "lives": int(info["lives_after_step"]),
                    "lives_before": int(info["lives"]),
                    "lives_after": int(info["lives_after_step"]),
                    "respawned": bool(info.get("respawned")),
                    "logic_frame_events": info["logic_frame_events"],
                    "reward": transition.reward,
                    "reward_components": dict(transition.components),
                    "option_status": transition.status,
                    "terminal_reason": info.get("terminal_reason"),
                }
            )
        info = dict(record.final.info)
        info["terminal_reason"] = record.reason
        return self.finish(
            env=game.env,
            initial_info=game.initial_info,
            initial_state_hash=game.initial_state_hash,
            info=info,
            harness=harness,
            reward=reward.reward,
            decisions=decisions,
            trajectory=trajectory,
            max_steps=game.max_steps,
            elapsed_seconds=(
                game.play_elapsed_seconds
                if game.play_elapsed_seconds is not None
                else record.elapsed_seconds
            ),
        )

    def error_summary(
        self,
        record: EpisodeRecord,
        error: BaseException,
    ) -> dict[str, Any]:
        return {
            **self.attempt_header,
            "collection_scope": "all_attempts",
            "terminal_reason": record.reason,
            "status": record.status,
            "error_type": type(error).__name__,
            "error": str(error),
            "trainable": False,
            "total_shaped_reward": None,
        }

    def _persist(self, summary: dict[str, Any]) -> None:
        root = self.owner.options.get("artifact_root")
        if root is None:
            raise ValueError("artifact_root is required to audit every game attempt")
        JsonEpisodeStore(Path(root) / summary["split"])(summary)

    def persist_error(self, error: BaseException) -> None:
        if self.attempt_header:
            self._persist(
                {
                    **self.attempt_header,
                    "collection_scope": "all_attempts",
                    "terminal_reason": "technical_error",
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "trainable": False,
                    "total_shaped_reward": None,
                }
            )


class PacmanGame:
    """Adapt the pinned environment without retaining screenshots in audit records."""

    def __init__(
        self,
        options: dict[str, Any],
        seed: int,
        *,
        frame_observer: Callable[[Any, dict[str, Any]], None] | None = None,
        env_factory: Callable[[Any], Any] | None = None,
    ):
        from maapacman.env import PygamePacmanEnv, PygamePacmanEnvConfig

        self.max_steps = int(options.get("environment_max_steps", 512))
        config = PygamePacmanEnvConfig(
            pacman_python_root=options.get("pacman_python_root"),
            max_steps=self.max_steps,
            ghost_mode="normal",
            episode_life_mode="original_three_lives",
            worker_base_dir=options.get("worker_base_dir"),
        )
        self.started = time.monotonic()
        self.play_elapsed_seconds: float | None = None
        self.env = (env_factory or PygamePacmanEnv)(config)
        self.seed = seed
        self.frame_observer = frame_observer
        self.cancelled = threading.Event()
        self.initial_info: dict[str, Any] = {}
        self.initial_state_hash = ""

    def _check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise asyncio.CancelledError

    def _observation(self, image: Any, info: dict[str, Any]) -> Observation:
        # Native info includes full state copies for every rendered frame. Reward
        # uses the event ledger; those large snapshots must not enter the trace.
        audit_info = {
            key: value
            for key, value in info.items()
            if key not in {"state", "atomic_substeps"}
        }
        return Observation(image, audit_info, self.env.snapshot())

    def reset(self) -> Observation:
        self._check_cancelled()
        image, info = self.env.reset(seed=self.seed)
        self._check_cancelled()
        observation = self._observation(image, info)
        self.initial_info = dict(observation.info)
        self.initial_state_hash = hashlib.sha256(
            json.dumps(observation.state, sort_keys=True).encode()
        ).hexdigest()
        if self.frame_observer is not None:
            self.frame_observer(image, info)
        return observation

    def step(self, action: Any) -> GameStep:
        from maapacman.env import Action

        self._check_cancelled()
        image, reward, terminated, truncated, info = self.env.step(Action(action))
        self._check_cancelled()
        if int(info["death_count"]) >= 3:
            # The pinned game otherwise permits a fourth death. Stop before any
            # further native step, without modifying the external game package.
            terminated, truncated = True, False
            info = {
                **info,
                "terminal_reason": "death_limit",
                "terminated": True,
                "truncated": False,
            }
        observation = self._observation(image, info)
        if self.frame_observer is not None:
            self.frame_observer(image, info)
        return GameStep(observation, reward, terminated, truncated)

    def cancel(self) -> None:
        # Native reset/step have bounded IPC timeouts. The runtime awaits their
        # completion and close; never race env.close against its IPC reader.
        self.cancelled.set()

    def close(self) -> None:
        if self.play_elapsed_seconds is None:
            self.play_elapsed_seconds = time.monotonic() - self.started
        self.env.close()


class PacmanHarnessAdapter:
    """Parse moves and derive legality/continuation only from transient RGB pixels."""

    def __init__(
        self,
        harness: PacmanHarness,
        event_observer=None,
        *,
        visual_planner: StandaloneVisualPlanner | None = None,
        transition_tracker: VisualTransitionTracker | None = None,
    ):
        self.harness = harness
        self.event_observer = event_observer
        self._visual_actions = VisualActionSpace()
        self._visual_planner = visual_planner
        self._transition_tracker = transition_tracker or VisualTransitionTracker()
        self._active_route: tuple[str, ...] = ()
        self._active_index = 0
        self._active_target: tuple[int, int] | None = None
        self._ghost_mode_signature: tuple[int, int] | None = None
        self._last_blocked_action: str | None = None

    def prepare(self, observation: Observation) -> Any:
        legal = list(self._visual_actions.available_actions(observation.value))
        if not legal:
            raise EpisodeStop("no_legal_moves")
        visual_plan = (
            self._visual_planner.plan(
                observation.value,
                blocked_action=self._last_blocked_action,
            )
            if self._visual_planner is not None
            else None
        )
        return {"legal_actions": legal, "visual_plan": visual_plan}

    def start(self, decision: Decision, context: Any) -> Any:
        try:
            command = self.harness.parse(decision)
        finally:
            self.harness.record_format(decision)
        legal = context["legal_actions"]
        visual_plan = context.get("visual_plan")
        self._active_route = ()
        self._active_index = 0
        self._active_target = None
        self._ghost_mode_signature = None
        if command.kind == "OPTION":
            advertised = (
                visual_plan is not None and command.value == visual_plan.option_id
            )
            route = tuple(visual_plan.action_sequence) if advertised else ()
            action = route[0] if route else None
            if advertised and route:
                self._active_route = route[: visual_plan.commit_moves]
                self._active_target = visual_plan.target_cell
        else:
            advertised = True
            action = command.value
        # The prompt evidence is the authoritative RGB snapshot for the selected
        # option. Its ghost mode is already recorded by the pixel planner.
        if command.kind == "OPTION" and visual_plan is not None:
            self._ghost_mode_signature = (
                len(visual_plan.normal_ghost_cells),
                len(visual_plan.vulnerable_ghost_cells),
            )
        action_legal = advertised and action in legal
        if self.event_observer is not None:
            self.event_observer(
                {"kind": "parsed_action", "action": action, "legal": action_legal}
            )
        decision.evidence.update(
            action_legal=action_legal,
            blocked_action=not action_legal,
            selected_action=action,
            selected_option=command.value if command.kind == "OPTION" else None,
            selected_option_route=list(self._active_route),
            planner_recommendation_match=(
                action == decision.evidence.get("planner_recommended_action")
                if decision.evidence.get("planner_recommended_action") is not None
                else None
            ),
        )
        if action is None or (command.kind == "OPTION" and not action_legal):
            raise EpisodeStop("invalid_action")
        return action

    def before_step(self, action: Any) -> None:
        if self._visual_planner is not None:
            self._visual_planner.record_model_action(action)

    def after_step(self, transition: Transition) -> ActionResult:
        if self.event_observer is not None:
            self.event_observer(
                {"kind": "executed_action", "action": transition.action}
            )
        previous_cell = detect_pacman_cell(transition.previous.value)
        current_cell = detect_pacman_cell(transition.current.value)
        blocked = previous_cell is not None and current_cell == previous_cell
        expected = bool(
            previous_cell is not None
            and current_cell is not None
            and not blocked
            and self._visual_actions.is_expected_transition(
                previous_cell,
                current_cell,
                transition.action,
            )
        )
        if blocked:
            self._last_blocked_action = str(transition.action)
        elif expected:
            self._last_blocked_action = None
        transition.evidence["visual_move"] = {
            "source": "rgb_pixels_only",
            "previous_pacman_cell_row_col": list(previous_cell)
            if previous_cell is not None
            else None,
            "current_pacman_cell_row_col": list(current_cell)
            if current_cell is not None
            else None,
            "blocked": blocked,
            "expected_transition": expected,
        }
        self._transition_tracker.record(
            current_cell,
            blocked=blocked,
            expected=expected,
        )
        if transition.terminated or transition.truncated:
            return ActionResult(status="terminal")
        if not self._active_route:
            return ActionResult(status="blocked" if blocked else "move_complete")
        if current_cell is None or not expected:
            return ActionResult(
                status="blocked" if blocked else "visual_option_ambiguous"
            )
        self._active_index += 1
        if current_cell == self._active_target:
            return ActionResult(status="option_completed")
        if self._active_index >= len(self._active_route):
            return ActionResult(status="max_commit")
        current_ghost_mode = detect_ghost_mode_signature(transition.current.value)
        if (
            current_ghost_mode is None
            or current_ghost_mode != self._ghost_mode_signature
        ):
            return ActionResult(status="ghost_state_interrupt")
        next_action = self._active_route[self._active_index]
        try:
            next_legal = self._visual_actions.available_actions(
                transition.current.value
            )
        except ValueError:
            return ActionResult(status="visual_option_ambiguous")
        if next_action not in next_legal:
            return ActionResult(status="option_no_longer_legal")
        if self._visual_planner is None:
            return ActionResult(status="safety_interrupt")
        refreshed_plan = self._visual_planner.plan(transition.current.value)
        transition.evidence["visual_move"]["refreshed_safe_actions"] = (
            list(refreshed_plan.safe_actions) if refreshed_plan is not None else None
        )
        if refreshed_plan is None or next_action not in refreshed_plan.safe_actions:
            return ActionResult(status="safety_interrupt")
        return ActionResult(action=next_action, status="active")


class PacmanRewardAdapter:
    def __init__(
        self,
        game: PacmanGame,
        harness: PacmanHarness,
        options: dict[str, Any],
    ):
        self.game, self.harness, self.options = game, harness, options
        self._reward: EpisodeReward | None = None

    @property
    def reward(self) -> EpisodeReward:
        if self._reward is None:
            self._reward = EpisodeReward(
                self.game.initial_info,
                self.options.get("ghost_reward_target", 4),
                max_steps=self.game.max_steps,
                step_efficiency_penalty_weight=self.options.get(
                    "step_efficiency_penalty_weight",
                    DEFAULT_STEP_EFFICIENCY_PENALTY_WEIGHT,
                ),
            )
        return self._reward

    def step(self, transition: Transition) -> RewardValue:
        value, components = self.reward.step(
            transition.previous.info,
            transition.current.info,
            transition.base_reward,
        )
        return RewardValue(value, components)

    def finish(self, reason: str, observation: Observation) -> RewardValue:
        value, components = self.reward.finish(
            reason,
            observation.info,
            all_strict=self.harness.all_strict,
        )
        return RewardValue(value, components)


class PacmanSessionFactory:
    """Assemble a fresh game and game-specific plugins for each sampled episode."""

    def __init__(self, owner: Any):
        self.owner = owner

    def __call__(
        self,
        data: dict[str, Any],
        context: EpisodeContext,
        *,
        frame_observer: Callable[[Any, dict[str, Any]], None] | None = None,
        event_observer: Callable[[dict[str, Any]], None] | None = None,
        env_factory: Callable[[Any], Any] | None = None,
    ) -> Session:
        artifacts = PacmanEpisodeArtifacts(self.owner, context, context.model_version)
        header = artifacts.begin(data, training=context.training)
        game = None
        try:
            game = PacmanGame(
                self.owner.options,
                header["seed"],
                frame_observer=frame_observer,
                env_factory=env_factory,
            )
            harness = PacmanHarness()
            reward = PacmanRewardAdapter(game, harness, self.owner.options)
            planner_assisted = bool(self.owner.options.get("planner_assisted", True))
            visual_planner = StandaloneVisualPlanner() if planner_assisted else None
            transition_tracker = VisualTransitionTracker()
            policy = PacmanPolicyCodec(
                self.owner,
                generation_seed=header["generation_seed"],
                proxy_session=context.proxy_session_id is not None,
                planner_assisted=planner_assisted,
                visual_planner=visual_planner,
                transition_tracker=transition_tracker,
                event_observer=event_observer,
            )
            return Session(
                game=game,
                harness=PacmanHarnessAdapter(
                    harness,
                    event_observer,
                    visual_planner=visual_planner,
                    transition_tracker=transition_tracker,
                ),
                policy=policy,
                reward=reward,
                max_steps=game.max_steps,
                summarize=lambda record: artifacts.summarize(
                    record, game, harness, reward
                ),
                persist=artifacts._persist,
                error_summary=artifacts.error_summary,
            )
        except BaseException as error:
            if game is not None:
                try:
                    game.close()
                except Exception as cleanup_error:
                    error.add_note(
                        f"Failed to close Pacman environment: {cleanup_error}"
                    )
            try:
                artifacts.persist_error(error)
            except Exception as artifact_error:
                error.add_note(
                    f"Failed to persist Pacman initialization error: {artifact_error}"
                )
            raise
