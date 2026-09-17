# Train Pacman with AReaL

This directory contains the only AReaL-dependent layer: [agent.py](agent.py) supplies
native workflow context to the independent [PacmanPlayer](../tools/player.py),
[train.py](train.py) uses native PPOTrainer, and [prepare.py](prepare.py) checks
training dependencies and worker images. Gameplay, dataset generation and independent
evaluation remain in [tools](../tools/README.md).

The single [pacman_rl.yaml](pacman_rl.yaml) starts from public Qwen3.5-4B before Pacman
RL, using SGLang, Megatron and critic-free GRPO. There is no custom engine,
distribution, logit processor, tensor-export hook or algorithm implementation. No core
patch or not-yet-merged sequence-mean reduction is assumed.

## Native training semantics

Both native agent chat template and export style are `concat`. An unbranched real
multi-turn history exports one whole-episode row, with loss masks on actual assistant
tokens across turns. The adapter returns one episode reward, or an empty mapping when no
model completion exists. Invalid model output is a real zero-reward training sample;
`mask_no_eos_with_zero=false` keeps length-limited invalid completions trainable.

`gconfig.reward_normalization=true` normalizes one return per episode within each seed's
12-game group, using mean and population standard deviation. Raw game rewards remain in
\[0,1\]; normalized training rewards can be negative or exceed 1. This is not batch
min-max normalization. Actor reward/advantage normalization are disabled to avoid a
second normalization. **The native loss is token-mean, not equal weight per episode.**
One concatenated row per episode does not change that reduction.

The recipe keeps KL 0.01, a fixed reference initialized from the actor's starting model,
one PPO minibatch, LR `5e-7`, clip 0.05, BF16 model weights, FP32 optimizer states and
`attn_impl: flash_attention_2`. Full-episode rows must fit actor/reference
`max_tokens_per_mb`, which follows the context budget. Long conversations can consume
substantial memory. 80 train seeds × batch size 8 gives 10 updates per epoch; with group
size 12, each update uses 96 accepted games. Two epochs give 20 updates.

## Install and prepare before allocating workers

From the source checkout in a compatible Linux CUDA environment:

```bash
uv sync --extra cuda
source .venv/bin/activate
export PACMAN_WORKER_PYTHON="$(command -v python)"
export GAME_PLAYER_SOURCE_ROOT="${PWD}"
export PACMAN_ARTIFACT_ROOT="${PWD}/pacman-artifacts"
python "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/tools/cli.py" prepare \
  --root "$PACMAN_ARTIFACT_ROOT" --install-game
source "$PACMAN_ARTIFACT_ROOT/environment.sh"
python "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/train/prepare.py"
```

The entrypoints are ordinary Python scripts and locate the source checkout themselves;
the absolute paths above also work from another working directory. No examples package
installation is required.

The `cuda` extra includes SGLang and Megatron via `cuda-train`; resolve these pinned
dependencies before starting workers. Independent game preparation never upgrades the
training stack. For Slurm/SIF, install the dependencies in the image beforehand, set
`PACMAN_WORKER_IMAGE` and its absolute `PACMAN_WORKER_PYTHON`, and make code/model/
sources/artifacts visible through existing scheduler mounts. Training preflight checks
the actual image before allocation. No site-specific host/image/partition is embedded.

## Train and run a small acceptance check

```bash
export PACMAN_MODEL_PATH='Qwen/Qwen3.5-4B'
export PACMAN_FILEROOT="$PACMAN_ARTIFACT_ROOT/experiments"
export PACMAN_TRIAL="free-grpo-$(date +%Y%m%dT%H%M%S)"
python "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/train/train.py" \
  --config "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/train/pacman_rl.yaml"
```

The default is local, one GPU, and is **not a fit guarantee**. Override
`PACMAN_SCHEDULER`, `PACMAN_N_NODES`, `PACMAN_GPUS_PER_NODE`, `PACMAN_ACTOR_BACKEND` and
`PACMAN_ROLLOUT_BACKEND` for the allocated hardware. For a separately obtained 16-GPU
allocation, an example is actor `megatron:d8p2t1c1e1`, rollout `sglang:d16p1t1`, and
`actor.mb_spec.n_mbs_divisor=2`; validate the actual topology and memory usage.

A small check explicitly chooses local single-GPU DP1. These commands launch real
training only when you execute them:

```bash
python "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/train/train.py" \
  --config "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/train/pacman_rl.yaml" \
  trial_name=acceptance environment_max_steps=8 total_train_steps=2 \
  scheduler.type=local cluster.n_nodes=1 cluster.n_gpus_per_node=1 \
  actor.backend=megatron:d1p1t1c1e1 rollout.backend=sglang:d1p1t1 \
  train_dataset.batch_size=2 valid_dataset.batch_size=2 \
  gconfig.n_samples=2 rollout.max_concurrent_rollouts=4 \
  gconfig.max_tokens=4096 eval_gconfig.max_tokens=4096 \
  evaluator.freq_epochs=null evaluator.freq_steps=1
```

Inspect valid/invalid decisions, invalid outputs receiving zero reward with no action,
third-death endings, whole-history loss masks, finite updates and repeated memory usage.
For other allocations, scale batch size and microbatch divisibility for DP/PP. Context
budget endings are distinct from game wins or technical failures. No UT, training,
GPU/runtime acceptance or learning-quality validation was run as part of delivery.

## Recovery and metrics

Recovery saves model, optimizer and training state every update. To resume, retain the
same trial, game/dataset settings and allocation-compatible configuration:

```bash
python "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/train/train.py" \
  --config "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/train/pacman_rl.yaml" \
  require_recovery=true total_train_epochs=3 total_train_steps=30
```

`total_train_steps` is the absolute stopping step. `require_recovery=true` refuses a
fresh restart without a complete checkpoint. HF weights alone are not recovery state;
SGLang KV cache is recreated. Copy selected HF checkpoints outside rolling retention.

```bash
python "$GAME_PLAYER_SOURCE_ROOT/examples/vlm/game_player/pacman/tools/cli.py" summarize \
  "$PACMAN_ARTIFACT_ROOT/runs/pacman-grpo/$PACMAN_TRIAL"
```

These are raw game outcomes, not normalized training rewards. Artifact versions may be
`requested` at episode start rather than verified per token; use native training dumps
for actual token evidence. Completed game summaries do not prove optimizer acceptance.
For final checkpoint selection, use the independent
[evaluation tools](../tools/README.md) with identical rules, seeds and budgets.
