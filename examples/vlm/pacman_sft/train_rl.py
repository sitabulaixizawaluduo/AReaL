# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

if __package__:
    from .pacman_dataset import get_pacman_rl_dataset
else:
    from pacman_dataset import get_pacman_rl_dataset

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.utils.hf_utils import load_hf_processor_and_tokenizer


def _build_dataset(dataset_config, processor):
    if dataset_config.path is None:
        raise ValueError("Pacman GRPO requires a dataset path")
    return get_pacman_rl_dataset(
        path=dataset_config.path,
        split=dataset_config.split,
        processor=processor,
        **dataset_config.dataset_kwargs,
    )


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GRPOConfig)
    processor, _ = load_hf_processor_and_tokenizer(config.tokenizer_path)

    train_dataset = _build_dataset(config.train_dataset, processor)
    valid_dataset = (
        _build_dataset(config.valid_dataset, processor)
        if config.valid_dataset is not None
        else None
    )

    workflow_kwargs = {
        "temperature": config.gconfig.temperature,
        "top_p": config.gconfig.top_p,
        "max_tokens": config.gconfig.max_new_tokens,
        "stop": config.gconfig.stop,
    }
    eval_workflow_kwargs = {
        "temperature": 0.0,
        "top_p": config.eval_gconfig.top_p,
        "max_tokens": config.eval_gconfig.max_new_tokens,
        "stop": config.eval_gconfig.stop,
    }
    workflow = "examples.vlm.pacman_sft.pacman_agent.PacmanAgent"

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
