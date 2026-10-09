#!/usr/bin/env bash
set -euo pipefail

: "${PLAYJEV_ROOT:?Set PLAYJEV_ROOT to your PlayJev checkout}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SHARD="${SHARD:-pacman-sft-$(date +%Y%m%d-%H%M%S)}"
if [[ ! "$SHARD" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "SHARD must be a single directory name" >&2
  exit 1
fi
cd "$PLAYJEV_ROOT"
if [[ -e "data/pacman/$SHARD" ]]; then
  echo "Refusing to overwrite existing shard: data/pacman/$SHARD" >&2
  exit 1
fi
# Use the original teacher, screenshots, epsilon exploration and record schema.
"$PYTHON_BIN" -m playjev.collect pacman \
  --actor teacher --steps "${COLLECT_STEPS:-100000}" --pages "${COLLECT_PAGES:-8}" \
  --epsilon "${COLLECT_EPSILON:-0.1}" --seed0 "${COLLECT_SEED0:-1000}" \
  --max-steps "${COLLECT_MAX_STEPS:-3000}" --shard "$SHARD"
echo "Collected $PLAYJEV_ROOT/data/pacman/$SHARD/records.jsonl"
