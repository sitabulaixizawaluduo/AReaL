# SPDX-License-Identifier: Apache-2.0

import sys

from areal import SFTTrainer
from areal.api import ModelAllocation
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset.playjev import PlayJevPacmanDataset
from areal.utils.hf_utils import load_hf_processor_and_tokenizer


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, SFTConfig)
    mode = ModelAllocation.from_str(config.actor.backend)
    if not config.actor.backend.startswith("megatron:"):
        raise ValueError("Pacman soft-label SFT requires Megatron")
    if config.actor.temperature != 1.0:
        raise ValueError("Pacman soft-label SFT requires actor.temperature=1")
    if (
        config.actor.enable_tree_training
        or config.actor.megatron.enable_mtp_training
        or config.actor.megatron.lm_head_loss_chunk_size > 0
    ):
        raise ValueError("Disable tree training, MTP training and chunked LM head")
    # Backend validates CP on every pipeline stage as well.
    if mode.parallel.context_parallel_size != 1:
        raise ValueError("The first Pacman SFT recipe requires CP=1")
    processor, _ = load_hf_processor_and_tokenizer(config.tokenizer_path)
    if processor is None:
        raise ValueError("Pacman SFT requires a vision-language model processor")
    datasets = []
    for split, ds_config in (
        ("train", config.train_dataset),
        ("validation", config.valid_dataset),
    ):
        datasets.append(
            PlayJevPacmanDataset(
                path=ds_config.path,
                split=split,
                processor=processor,
                max_length=ds_config.max_length,
                **ds_config.dataset_kwargs,
            )
        )
    with SFTTrainer(
        config, train_dataset=datasets[0], valid_dataset=datasets[1]
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
