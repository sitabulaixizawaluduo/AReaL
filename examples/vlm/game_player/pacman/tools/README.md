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
python -m pip install numpy Pillow pygame==2.6.1 openai
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

The player sends `--model` unchanged to the endpoint. It requires no local model files,
tokenizer, processor, Transformers installation or model download. The server owns all
model processing and context limits. The reset request contains the full 336×400 frame;
later requests use a Pacman-centered 13×13-tile (208×208) crop at the original pixel
resolution. Crops receive black edge padding. Ambiguous visual detection or a non-portal
jump/respawn falls back to the full frame. Standalone sends only the current image;
AReaL proxy sessions retain full-reset then cropped images in append-only concat. Remote
context errors remain technical failures, with no synthetic game result.

The policy receives no dynamic source-state text. The fixed prompt preserves the win,
ghost, power-pellet, wall, death and output-format rules in under 170 words. A local
tracker derives the concise current cell, open moves, visible lethal ghosts and
suggested move from RGB pixels. Coordinates never come from source state or enter final
content.

Both modes compare Pacman's current large yellow component with the compact cell stored
before the previous action. If unchanged, the current request reports the blocked
direction without resending the old screenshot. Small pellets and HUD lives are
excluded; ambiguous detections add no feedback.

The shared codec builds the 16-pixel maze topology and excludes the visible red ghost
gate using only the reset RGB frame. It detects visible actors and pellets from later
frames and adds a short `rgb_pixels_only` plan with open moves and one recommendation.
Edward supplies the normal route choice; an RGB-derived deterministic safety fallback
handles Edward refusal. A just-blocked direction is excluded from the next plan. The
text is bounded to the current request and omitted on ambiguous extraction. Training and
standalone evaluation enable it by default. It advises the model but never bypasses the
response or executes an action. Pass `--no-planner-assisted` for standalone ablation;
set `planner_assisted: false` for the matching training ablation.

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
`chat_template_kwargs.enable_thinking=true` extension used by standalone play. The
provider may return private reasoning through SDK `reasoning_content`. For Qwen
templates that prefill `<think>` in the generation prompt, the provider may instead put
the reasoning and emitted `</think>` boundary in `content`; the standalone player splits
that boundary. Only the final content is executed and it must remain an exact move
response. Every standalone call contains one current screenshot; earlier screenshots,
actions and reasoning do not enter the next request. Compact RGB-derived tracker state
may appear only in the current pixel hint. Rejection of an unsupported extension is a
visible failure, not a silent change to thinking mode. The recorded model identity is
caller-declared; it is not independently verified.

Standalone commands enable reasoning by default. Pass `--no-reasoning` when an endpoint
does not support a separate reasoning channel; this sends
`chat_template_kwargs.enable_thinking=false` and does not split Qwen `</think>` text.
Separate SDK `reasoning_content` remains audit-only: it does not enter visible-content
strictness or independently cause `invalid_format`. `--reasoning` restores the default
explicitly. This flag applies only to standalone commands: AReaL training proxy sessions
always disable reasoning generation. Use `--no-reasoning` for a strict train/evaluate
comparison of generation settings.

At the fresh-game spawn point Pacman is in a horizontal corridor: the first action must
be `MOVE L` or `MOVE R`. The first user request states this reset-only constraint;
subsequent requests use the generic four-direction request. In both modes, a well-formed
direction blocked by a wall is the game's natural one-step no-op and play continues from
the next screenshot. Its failed visual legality and collision remain in audit evidence,
and the next request reports the blocked direction.

Every parsed move executes one environment step and then requests a new model decision.
The harness derives legal moves from the full RGB frame and cached pixel-reconstructed
topology. Ambiguous extraction is a technical error and never falls back to source-state
legal actions.

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
serially; `--seed` selects one seed belonging to the split. `--max-steps` and
`--max-new-tokens` set the game-step and per-response output budgets. The latter
defaults to 512 so a provider can reason before returning a move. The endpoint controls
the total context limit; there is no standalone `--max-tokens` option. Sampling defaults
to `--temperature 0.2 --top-p 0.9`; both can be overridden explicitly and are recorded
in `plan.json`. Every planned attempt remains in the win-rate denominator, including
errors; no hidden success retry occurs.

When the provider returns usage metadata, each decision records `prompt_tokens`,
`completion_tokens` and `total_tokens` under `evidence.token_usage`. Evidence also
stores the full-frame and prompt-image hashes, image scope, crop bounds and padding.
These audit values are never included in a subsequent prompt.

## Live dashboard

Run independent games and watch their latest frames in a local browser:

```bash
python "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/tools/cli.py" dashboard \
  --endpoint "$PACMAN_ENDPOINT" --model "$PACMAN_MODEL" \
  --split dev --limit 4 --concurrency 4 \
  --output "$PACMAN_ARTIFACT_ROOT/dashboard-dev" --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/`. Every planned game has a separate tile showing its latest
reset/step screenshot, private reasoning, final model output, parsed and executed move,
score, lives, pellet progress, elapsed time and terminal outcome/reward. Queued games
become active according to `--concurrency`; the responsive grid adapts to the number of
games and window size. A tile becomes terminal only after its episode JSON is safely
written. Once all tiles are terminal, the header reports `finalizing` while the shared
client closes and metrics plus `summary.json` are written. The default command then
reports its final result, closes the HTTP server and exits automatically. Pass
`--keep-open` to retain the finalized page until Ctrl+C. The API snapshot exposes this
choice as `keep_open` and `lifecycle`. Evaluation artifacts are written exactly as with
`evaluate`. The header reports `complete` only when every planned game completed without
a technical error, `partial` when some completed but the batch is incomplete or has
errors, and `error` when none completed. A normal game loss still counts as a completed
game. A retained page closed with Ctrl+C returns exit code 0 for `complete`, 2 for
`partial`/`error`, and 130 if evaluation was interrupted before its result was
finalized.

The dashboard keeps one PNG and bounded status per game, with no frame or output
history. Latest reasoning and final output longer than 8192 characters are independently
truncated only for display; neither dashboard value enters the next policy prompt. The
original model response and normal evaluation artifacts remain unchanged. It uses
Python's standard-library HTTP server and existing image dependencies, with no added
packages. Observer data never enters policy messages or changes rewards, actions or
training. Technical errors display their type, not SDK error messages or API keys. The
server defaults to loopback and has no authentication; keep it local or use SSH
forwarding when viewing remote runs. Ctrl+C cancels active games and closes their
workers and the dashboard. This is a live preview, not a full-frame recording; use
`record_video` when you need a replay.

## Metrics and recording

```bash
python "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/tools/cli.py" summarize \
  "$PACMAN_ARTIFACT_ROOT/eval-dev"
```

The same command can read a training artifact directory without importing a trainer.
Metrics include overall/zero-death wins, raw reward, normal/power pellet clearance,
native game score, deaths, ghosts eaten and illegal format/action rates. Legacy advice
metrics are unavailable for pure-vision runs. Field coverage is reported.
`--scope completed` filters completed training attempts; it does not prove optimizer
acceptance. Version labels distinguish requested, verified and unknown; SDK text does
not imply actual training token/logprob evidence.

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
