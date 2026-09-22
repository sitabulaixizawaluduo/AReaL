#!/usr/bin/env bash

set -euo pipefail

if [[ ! -f pyproject.toml ]]; then
  echo "Run this script from the AReaL repository root." >&2
  exit 2
fi

exec python3 examples/vlm/pacman_sft/train.py \
  --config examples/vlm/pacman_sft/pacman_sft.yaml \
  +scheduler.type=local \
  experiment_name=pacman-vlm-sft \
  trial_name=qwen3_5_0_8b_8gpu \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=8 \
  cluster.fileroot=/storage/openpsi/users/ljl/workspace/game_player/outputs \
  cluster.name_resolve.nfs_record_root=/storage/openpsi/users/ljl/workspace/game_player/outputs/name_resolve \
  actor.backend=megatron:d8p1t1 \
  actor.path=/storage/openpsi/models/Qwen__Qwen3.5-0.8B \
  tokenizer_path=/storage/openpsi/models/Qwen__Qwen3.5-0.8B \
  train_dataset.path=/storage/openpsi/users/ljl/workspace/game_player/pacman \
  valid_dataset.path=/storage/openpsi/users/ljl/workspace/game_player/pacman
