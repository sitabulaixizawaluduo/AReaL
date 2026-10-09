#!/usr/bin/env bash
set -euo pipefail

: "${PLAYJEV_ROOT:?Set PLAYJEV_ROOT}"
: "${MODEL_PATH:?Set MODEL_PATH to a supported small VLM base checkpoint}"
export PLAYJEV_DATA_ROOT="${PLAYJEV_DATA_ROOT:-$PLAYJEV_ROOT/data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
"$PYTHON_BIN" -m areal.infra.launcher.local \
  examples/game_player/pacman_sft/train.py \
  --config examples/game_player/pacman_sft/megatron.yaml \
  trial_name="smoke-$(date +%Y%m%d-%H%M%S)" \
  total_train_steps=2 train_dataset.batch_size=2 valid_dataset.batch_size=2 \
  train_dataset.num_workers=0 valid_dataset.num_workers=0 \
  +train_dataset.dataset_kwargs.limit=4 +valid_dataset.dataset_kwargs.limit=2 \
  actor.optimizer.warmup_steps=0 saver.freq_steps=2 evaluator.freq_steps=2
