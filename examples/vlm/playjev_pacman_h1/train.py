# SPDX-License-Identifier: Apache-2.0

"""Train the one-decision PlayJev Pacman agent with AReaL GRPO."""

from __future__ import annotations

import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config

from .dataset import PacmanH1Dataset


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GRPOConfig)
    train_dataset = PacmanH1Dataset(config.train_dataset.path, "train")
    valid_dataset = PacmanH1Dataset(config.valid_dataset.path, "valid")
    workflow = "examples.vlm.playjev_pacman_h1.workflow.PacmanH1Agent"
    workflow_kwargs = {
        "max_env_steps": 1,
        "temperature": config.gconfig.temperature,
        "top_p": config.gconfig.top_p,
        "max_completion_tokens": config.gconfig.max_new_tokens,
    }
    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow,
            eval_workflow_kwargs={**workflow_kwargs, "temperature": 0.0},
            dynamic_filter_fn=(
                "examples.vlm.playjev_pacman_h1.filters.reward_has_variance"
            ),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
