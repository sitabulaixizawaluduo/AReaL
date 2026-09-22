# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

from pacman_dataset import get_pacman_sft_dataset

from areal import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.utils.hf_utils import load_hf_processor_and_tokenizer


def _build_dataset(dataset_config, processor):
    if dataset_config.path is None:
        raise ValueError("Pacman SFT requires a dataset path")
    return get_pacman_sft_dataset(
        path=dataset_config.path,
        split=dataset_config.split,
        processor=processor,
        max_length=dataset_config.max_length,
        **dataset_config.dataset_kwargs,
    )


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, SFTConfig)
    processor, _ = load_hf_processor_and_tokenizer(config.tokenizer_path)

    train_dataset = _build_dataset(config.train_dataset, processor)
    valid_dataset = (
        _build_dataset(config.valid_dataset, processor)
        if config.valid_dataset is not None
        else None
    )

    with SFTTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
