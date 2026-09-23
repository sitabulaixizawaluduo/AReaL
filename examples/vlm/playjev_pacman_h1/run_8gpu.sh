#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash examples/vlm/playjev_pacman_h1/run_8gpu.sh MODEL_PATH DATA_ROOT EXPERIMENT_ROOT" >&2
  exit 2
fi

if [[ ! -f examples/vlm/playjev_pacman_h1/recipe.yaml ]]; then
  echo "Run this script from the AReaL repository root." >&2
  exit 2
fi
if [[ ! -f "$2/snapshot.json" || ! -f "$2/train/records.jsonl" || ! -f "$2/valid/records.jsonl" ]]; then
  echo "DATA_ROOT must be a frozen Pacman H1 snapshot with separate train/ and valid/." >&2
  exit 2
fi

exec python3 -m examples.vlm.playjev_pacman_h1.train \
  --config examples/vlm/playjev_pacman_h1/recipe.yaml \
  actor.path="$1" \
  tokenizer_path="$1" \
  sglang.model_path="$1" \
  train_dataset.path="$2/train" \
  valid_dataset.path="$2/valid" \
  cluster.fileroot="$3" \
  cluster.name_resolve.nfs_record_root="$3/name_resolve"
