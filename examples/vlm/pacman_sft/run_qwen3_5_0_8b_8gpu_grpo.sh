#!/usr/bin/env bash

set -euo pipefail

if [[ ! -f pyproject.toml ]]; then
  echo "Run this script from the AReaL repository root." >&2
  exit 2
fi

if [[ ! -d /storage/openpsi/models/Qwen__Qwen3.5-0.8B ]]; then
  echo "Qwen3.5-0.8B base model is missing." >&2
  exit 2
fi

exec python3 examples/vlm/pacman_sft/train_rl.py \
  --config examples/vlm/pacman_sft/pacman_grpo.yaml \
  scheduler.type=local \
  experiment_name=ljl-pacman-vlm-grpo \
  trial_name=qwen3_5_0_8b_8gpu-offline \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=8 \
  cluster.fileroot=/storage/openpsi/experiments \
  cluster.name_resolve.nfs_record_root=/storage/openpsi/experiments/name_resolve \
  actor.path=/storage/openpsi/models/Qwen__Qwen3.5-0.8B \
  tokenizer_path=/storage/openpsi/models/Qwen__Qwen3.5-0.8B \
  sglang.model_path=/storage/openpsi/models/Qwen__Qwen3.5-0.8B \
  train_dataset.path=/storage/openpsi/users/ljl/workspace/game_player/pacman \
  valid_dataset.path=/storage/openpsi/users/ljl/workspace/game_player/pacman
