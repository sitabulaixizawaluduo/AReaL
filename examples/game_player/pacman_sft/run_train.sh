#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"

# 主要修改这两个路径；也可以在启动时通过同名环境变量覆盖。
export MODEL_PATH="${MODEL_PATH:-$WORKSPACE_ROOT/Data/Qwen__Qwen3.5-2B}"
export PLAYJEV_DATA_ROOT="${PLAYJEV_DATA_ROOT:-$WORKSPACE_ROOT/Data/playjev}"

# PlayJev checkout 提供原始 prompt、选项描述和数据读取逻辑。
export PLAYJEV_ROOT="${PLAYJEV_ROOT:-$REPO_ROOT/../PlayJev}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "模型目录不存在或缺少 config.json：$MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$PLAYJEV_DATA_ROOT/pacman" ]] || \
   ! compgen -G "$PLAYJEV_DATA_ROOT/pacman/*/records.jsonl" > /dev/null; then
  echo "数据集路径应包含 pacman/<shard>/records.jsonl：$PLAYJEV_DATA_ROOT" >&2
  exit 1
fi
if [[ ! -f "$PLAYJEV_ROOT/playjev/data.py" ]]; then
  echo "找不到 PlayJev checkout：$PLAYJEV_ROOT" >&2
  exit 1
fi

echo "MODEL_PATH=$MODEL_PATH"
echo "PLAYJEV_DATA_ROOT=$PLAYJEV_DATA_ROOT"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m areal.infra.launcher.local \
  examples/game_player/pacman_sft/train.py \
  --config examples/game_player/pacman_sft/megatron.yaml \
  "trial_name=train-$(date +%Y%m%d-%H%M%S)" "$@"
