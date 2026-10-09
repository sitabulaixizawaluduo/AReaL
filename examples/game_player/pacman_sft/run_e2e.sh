#!/usr/bin/env bash
set -euo pipefail

: "${PLAYJEV_ROOT:?Set PLAYJEV_ROOT}"
: "${MODEL_PATH:?Set MODEL_PATH to a supported VLM base checkpoint}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export SHARD="${SHARD:-pacman-e2e-$(date +%Y%m%d-%H%M%S)}"
export PLAYJEV_DATA_ROOT="$PLAYJEV_ROOT/data"
bash "$SCRIPT_DIR/collect_data.sh"
cd "$REPO_ROOT"
"$PYTHON_BIN" -m areal.infra.launcher.local \
  examples/game_player/pacman_sft/train.py \
  --config examples/game_player/pacman_sft/megatron.yaml \
  trial_name="$SHARD" \
  "+train_dataset.dataset_kwargs.shards=[$SHARD]" \
  "+valid_dataset.dataset_kwargs.shards=[$SHARD]" "$@"
