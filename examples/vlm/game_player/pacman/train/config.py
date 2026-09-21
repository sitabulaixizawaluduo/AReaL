"""Example-only configuration for free-generation Pacman GRPO."""

import math
from dataclasses import dataclass

from examples.vlm.game_player.pacman.tools.rewards import (
    DEFAULT_STEP_EFFICIENCY_PENALTY_WEIGHT,
)

from areal.api.cli_args import GRPOConfig


def validate_step_efficiency_penalty_weight(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 0.1:
        raise ValueError(
            "step_efficiency_penalty_weight must be finite and in [0, 0.1]"
        )


@dataclass
class PacmanConfig(GRPOConfig):
    artifact_root: str = ""
    run_artifact_root: str = ""
    split_manifest: str = ""
    pacman_python_root: str = ""
    worker_base_dir: str = ""
    environment_max_steps: int = 512
    ghost_reward_target: int = 4
    step_efficiency_penalty_weight: float = DEFAULT_STEP_EFFICIENCY_PENALTY_WEIGHT
    require_recovery: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not self.run_artifact_root or self.run_artifact_root == self.artifact_root:
            raise ValueError(
                "run_artifact_root must isolate this trial from prepared sources"
            )
        if self.environment_max_steps < 1 or self.ghost_reward_target < 1:
            raise ValueError(
                "Game step budget and ghost_reward_target must be positive"
            )
        validate_step_efficiency_penalty_weight(self.step_efficiency_penalty_weight)
        if self.critic is not None or self.teacher is not None:
            raise ValueError("This recipe uses critic-free GRPO without a teacher")
        if self.actor.reward_norm is not None or self.actor.adv_norm is not None:
            raise ValueError(
                "Normalize rewards once in rollout groups; actor reward_norm/adv_norm must be null"
            )
        if not self.gconfig.reward_normalization or self.gconfig.n_samples < 2:
            raise ValueError(
                "GRPO needs gconfig.reward_normalization=true and n_samples>=2"
            )
        if self.actor.discount != 1 or self.actor.gae_lambda != 1:
            raise ValueError("Whole-game outcomes require actor discount=gae_lambda=1")
        if self.actor.mask_no_eos_with_zero:
            raise ValueError(
                "Length-limited invalid outputs must remain trainable; mask_no_eos_with_zero must be false"
            )
        if (
            self.actor.optimizer is None
            or self.actor.optimizer.type != "adam"
            or self.actor.optimizer_dtype != "float32"
        ):
            raise ValueError("Use Adam with optimizer_dtype=float32")
        for name in ("main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
            if getattr(self.actor.megatron, name) != "float32":
                raise ValueError(f"Megatron {name} must be float32")
        if self.ref is None or self.ref.path != self.actor.path:
            raise ValueError(
                "Actor and fixed reference must initialize from the same model"
            )
        if any(
            engine.temperature != self.gconfig.temperature
            for engine in (self.actor, self.ref)
        ):
            raise ValueError("Actor, reference and rollout temperatures must match")
        if any(
            engine.mb_spec.max_tokens_per_mb is None
            or engine.mb_spec.max_tokens_per_mb < self.gconfig.max_tokens
            for engine in (self.actor, self.ref)
        ):
            raise ValueError(
                "Concat episode rows require actor/ref max_tokens_per_mb >= gconfig.max_tokens"
            )
        agent = self.rollout.agent
        if (
            agent is None
            or agent.mode != "inline"
            or agent.chat_template_type != "concat"
            or agent.export_style != "concat"
        ):
            raise ValueError(
                "Use inline agent with concat chat template and concat export"
            )
        for generation in (self.gconfig, self.eval_gconfig):
            if (
                generation is None
                or generation.greedy
                or generation.temperature <= 0
                or generation.top_p != 1
                or generation.top_k not in (-1, 100000000)
            ):
                raise ValueError(
                    "Use positive-temperature full-vocabulary sampling: greedy=false, top_p=1, default top_k"
                )
            if (
                generation.max_new_tokens < 2
                or generation.max_tokens <= generation.max_new_tokens
            ):
                raise ValueError(
                    "Generation and context budgets must support complete multi-token answers"
                )

    def workflow_options(self):
        return {
            "artifact_root": self.run_artifact_root,
            "experiment_name": self.experiment_name,
            "trial_name": self.trial_name,
            "environment_max_steps": self.environment_max_steps,
            "ghost_reward_target": self.ghost_reward_target,
            "step_efficiency_penalty_weight": self.step_efficiency_penalty_weight,
            "pacman_python_root": self.pacman_python_root,
            "worker_base_dir": self.worker_base_dir,
        }
