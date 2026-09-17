# Independent Pacman tools

These modules have no imports from AReaL or `pacman.train`. They prepare the game,
generate datasets, call a standard OpenAI-compatible API, summarize outcomes and record
video. These are ordinary Python scripts, with no package installation step. Each
entrypoint locates its containing source checkout, so an absolute script path works from
any working directory. Having the source checkout does not require installing AReaL as a
package.

## Install and prepare

Use Python 3.12+ and a separate environment. Install only player/game dependencies:

```bash
python3 -m venv .venv-player
source .venv-player/bin/activate
python -m pip install numpy Pillow pygame==2.6.1 openai 'transformers>=5.0.0,<=5.3.0'
export GAME_PLAYER_SOURCE_ROOT="${PWD}"  # Run this initial setup from the source checkout.
export PACMAN_ARTIFACT_ROOT="${PWD}/pacman-artifacts"
python "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/tools/cli.py" prepare \
  --root "$PACMAN_ARTIFACT_ROOT" --install-game
source "$PACMAN_ARTIFACT_ROOT/environment.sh"
python "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/tools/cli.py" preflight
```

Preparation clones and pins the public `luzai/areal-pacman` and `luzai/pacman-python`
repositories, installs the game package with `--no-deps`, and writes the fixed disjoint
80/24/96 train/dev/test seed manifest. It uses uv if available, otherwise pip. The
preflight checks game imports and source revisions; it never imports torch, SGLang,
Megatron or AReaL. Source changes require a new preparation root.

The player loads the selected model's HF tokenizer/processor locally to construct and
count its multimodal context. Its counting path requests NumPy outputs. Some model
processors may still need optional CPU dependencies documented by that model; this is
separate from installing a training or CUDA stack. A tokenizer/processor compatible with
the endpoint's actual model/template is required.

`datasets.py` owns fixed tasks, `metrics.py` owns outcome accounting, `storage.py` owns
JSON persistence, and `prepare.py` owns game source/dependency setup. Training-specific
preflight lives exclusively under `../train`.

## Evaluate a model

Provide an existing endpoint and its model identifier:

```bash
export PACMAN_ENDPOINT='https://your-provider.example/v1'
export PACMAN_MODEL='Qwen/Qwen3.5-4B'
# Set OPENAI_API_KEY when the endpoint requires authentication.
python "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/tools/cli.py" evaluate \
  --endpoint "$PACMAN_ENDPOINT" --model "$PACMAN_MODEL" \
  --split dev --output "$PACMAN_ARTIFACT_ROOT/eval-dev"
```

The tool directly uses `AsyncOpenAI(..., max_retries=0)`. It does not start an AReaL
proxy, trainer or GPU server, nor call SGLang-specific model-discovery endpoints. The
endpoint must support image messages, the selected model, and the
`chat_template_kwargs.enable_thinking=false` extension used by this player. Rejection of
an unsupported extension is a visible failure, not a silent change to thinking mode. The
recorded model identity is caller-declared; it is not independently verified.

For a local SGLang deployment, install/start SGLang in a **separate server
environment**. In terminal 1, run the server with tokenization enabled (omit
`--skip-tokenizer-init`):

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B \
  --host 127.0.0.1 --port 31000 --context-length 32768 --enable-multimodal
```

In terminal 2, activate the player environment, source its `environment.sh`, and pass
`--endpoint http://127.0.0.1:31000/v1 --model Qwen/Qwen3.5-4B` to the evaluation
command. No custom logit processor is needed. Local unauthenticated servers use an
unused SDK key; hosted keys come from `OPENAI_API_KEY` or the explicit `--api-key`
option.

Each output directory must be new. `--limit 4 --concurrency 1` runs four fixed seeds
serially; `--seed` selects one seed belonging to the split. `--max-steps`,
`--max-tokens` and `--max-new-tokens` set explicit budgets. Every planned attempt
remains in the win-rate denominator, including errors; no hidden success retry occurs.

## Metrics and recording

```bash
python "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/tools/cli.py" summarize \
  "$PACMAN_ARTIFACT_ROOT/eval-dev"
```

The same command can read a training artifact directory without importing a trainer.
Metrics include overall/zero-death wins, raw reward, normal/power pellet clearance,
native game score, deaths, ghosts eaten, illegal format/action rates and safe-advice
adherence. Field coverage is reported. `--scope completed` filters completed training
attempts; it does not prove optimizer acceptance. Version labels distinguish requested,
verified and unknown; SDK text does not imply actual training token/logprob evidence.

Install `ffmpeg` and `ffprobe` locally, then record one fixed dev/test attempt:

```bash
python "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/tools/cli.py" record_video \
  --endpoint "$PACMAN_ENDPOINT" --model "$PACMAN_MODEL" --split dev \
  --output "$PACMAN_ARTIFACT_ROOT/video-dev"
```

The recorder preserves initial/final frames, consecutive logical frames and actual
score/life events. It does not splice attempts or include inference waiting frames. The
showcase encoder requires an authentic zero-death win under normal ghosts and the
third-death episode rule. Other attempts retain their frames and audit evidence.
Independent endpoint/game execution and video generation have not been run locally.
