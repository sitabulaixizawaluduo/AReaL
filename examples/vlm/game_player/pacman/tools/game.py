# SPDX-License-Identifier: Apache-2.0

"""Pacman game-specific adapters and the existing episode artifact contract."""

import asyncio
import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from examples.vlm.game_player.pacman.tools.encoding import PacmanPolicyCodec
from examples.vlm.game_player.pacman.tools.harness import PacmanHarness
from examples.vlm.game_player.pacman.tools.rewards import EpisodeReward
from examples.vlm.game_player.pacman.tools.storage import JsonEpisodeStore
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
            "harness": "advisory_options_and_legal_moves_v1",
            "reward_objective_contract": "bounded_episode_return_v1",
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
            "special_pellet_clear_rate": 1
            - int(info["power_pellets_remaining"]) / reward.initial_power
            if reward.initial_power
            else 0.0,
            "ghosts_eaten": reward.ghosts_eaten,
            "ghost_reward_target": reward.ghost_reward_target,
            "reward_components": reward.components,
            "env_steps": int(info["step"]),
            "logic_frames": int(info["logic_frame"]) - int(initial_info["logic_frame"]),
            "decisions": len(decisions),
            "terminal_reason": info["terminal_reason"],
            "safety_refusal_triggers": harness.refusal_triggers,
            "invalid_format": info["terminal_reason"] == "invalid_format",
            "invalid_action": info["terminal_reason"] == "invalid_action",
            "context_budget_exhausted": info["terminal_reason"]
            == "context_budget_exhausted",
            "format_valid_rate": sum(row["format_valid"] for row in decisions)
            / len(decisions)
            if decisions
            else 0.0,
            "action_legal_rate": sum(row["action_legal"] for row in decisions)
            / len(decisions)
            if decisions
            else 0.0,
            "safe_advice_match_rate": sum(row["safe_advice_match"] for row in decisions)
            / len(decisions)
            if decisions
            else 0.0,
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
    """Strict syntax/physical legality; safe hints never constrain sampling."""

    def __init__(self, harness: PacmanHarness):
        self.harness = harness
        self.initialized = False
        self.option = None
        self.remaining = 0

    def prepare(self, observation: Observation) -> Any:
        if not self.initialized:
            self.harness.planner.observe(observation.state)
            self.initialized = True
        legal = [
            action
            for action in ("U", "D", "L", "R")
            if action in observation.info["legal_actions"]
        ]
        if not legal:
            raise EpisodeStop("no_legal_moves")
        return self.harness.candidates(observation.state), legal

    def start(self, decision: Decision, context: Any) -> Any:
        candidates, legal = context
        match = re.fullmatch(
            r"\s*<answer>(OPTION|MOVE) ([A-Za-z0-9]+)</answer>\s*", decision.text
        )
        # A native reasoning/tool output is not part of this action protocol.
        if (
            match is None
            or decision.choice.get("reasoning_content")
            or decision.choice.get("tool_calls")
            or decision.choice.get("refusal")
        ):
            raise EpisodeStop("invalid_format")
        decision.evidence["format_valid"] = True
        kind, value = match.groups()
        self.option = None
        if kind == "OPTION":
            self.option = next(
                (candidate for candidate in candidates if candidate.option_id == value),
                None,
            )
            action = self.option.first_action if self.option is not None else None
            self.remaining = self.option.commit_moves if self.option is not None else 0
        else:
            action, self.remaining = value, 1
        if action not in legal:
            raise EpisodeStop("invalid_action")
        decision.evidence.update(
            action_legal=True,
            selected_option=value if kind == "OPTION" else None,
            selected_action=action,
            safe_advice_match=any(
                candidate.first_action == action for candidate in candidates
            ),
        )
        return action

    def before_step(self, action: Any) -> None:
        self.harness.planner.record_action(action)

    def after_step(self, transition: Transition) -> ActionResult:
        self.remaining -= 1
        current, previous = transition.current, transition.previous
        self.harness.observe_transition(current.info, current.state)
        if transition.terminated or transition.truncated:
            return ActionResult(status="terminal")
        if self.option is None:
            return ActionResult(status="move_complete")
        action, status = self.harness.continue_option(
            self.option,
            previous.info,
            current.info,
            current.state,
            self.remaining,
        )
        return ActionResult(action, status)


class PacmanRewardAdapter:
    def __init__(self, game: PacmanGame, options: dict[str, Any]):
        self.game, self.options = game, options
        self._reward: EpisodeReward | None = None

    @property
    def reward(self) -> EpisodeReward:
        if self._reward is None:
            self._reward = EpisodeReward(
                self.game.initial_info,
                self.options.get("ghost_reward_target", 4),
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
        value, components = self.reward.finish(reason, observation.info)
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
            reward = PacmanRewardAdapter(game, self.owner.options)
            policy = PacmanPolicyCodec(
                self.owner,
                harness,
                generation_seed=header["generation_seed"],
                proxy_session=context.proxy_session_id is not None,
            )
            return Session(
                game=game,
                harness=PacmanHarnessAdapter(harness),
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
