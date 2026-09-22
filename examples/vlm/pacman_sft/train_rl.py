# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import Any

if __package__:
    from .pacman_dataset import get_pacman_rl_dataset
else:
    from pacman_dataset import get_pacman_rl_dataset

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.utils.hf_utils import load_hf_processor_and_tokenizer


def pacman_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids,
    completion_ids,
    expected_letter: str,
    **kwargs: Any,
) -> float:
    """Reward an exact teacher-action match after deterministic option shuffling."""
    del prompt, prompt_ids, completion_ids, kwargs
    return float(str(completions).strip() == str(expected_letter).strip())


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
        "reward_fn": "examples.vlm.pacman_sft.train_rl.pacman_reward_fn",
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "processor": config.tokenizer_path,
        "enable_thinking": False,
    }
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.eval_gconfig

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.vision_rlvr.VisionRLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.vision_rlvr.VisionRLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
