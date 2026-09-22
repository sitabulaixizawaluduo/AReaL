# Train Pacman with AReaL

This directory contains the only AReaL-dependent layer: [agent.py](agent.py) supplies
native workflow context to the independent [PacmanPlayer](../tools/player.py),
[train.py](train.py) uses native PPOTrainer, and [prepare.py](prepare.py) checks
training dependencies and worker images. Gameplay, dataset generation and independent
evaluation remain in [tools](../tools/README.md).

The single [pacman_rl.yaml](pacman_rl.yaml) starts from public Qwen3.5-4B before Pacman
RL, using SGLang, Megatron and critic-free GRPO. There is no custom engine,
distribution, logit processor or Pacman-specific tensor-export hook. It uses AReaL's
generic `rollout_mean` loss aggregation to keep logical episodes equally weighted; no
game-specific behavior is added to core training code.

The policy receives original-pixel-scale screenshots, a fixed MOVE/OPTION answer
contract and the same current-turn RGB-only planner hint used by independent evaluation.
The planner reconstructs visible topology and actors from pixels, uses Edward with a
visual fallback and advertises one bounded `OPTION A0` route. The model selects that
option or one corrective move; the harness revalidates every option step from fresh RGB
frames. Set `planner_assisted: false` only for an explicit train/evaluate ablation.

## Native training semantics

The native agent uses the standard `hf` chat template and `individual` export. Every
decision row contains only the fixed system prompt, current screenshot/hint and that
turn's assistant answer. The adapter returns one terminal episode reward, or an empty
mapping when no model completion exists. With `turn_discount=1`, the proxy propagates
that outcome backward to every decision row from the same episode. Actor `discount=1`
and `gae_lambda=1` preserve the shared whole-game signal without decay. Invalid model
output terminates the episode and earns no format bonus, but keeps game progress earned
before termination. `mask_no_eos_with_zero=false` keeps length-limited invalid
completions trainable.

The harness separates executable parsing from strict serialization. Exactly one valid
`<answer>MOVE X</answer>` or `<answer>OPTION A0</answer>` tag executes even with
surrounding visible prose, while the strict flag requires the entire visible response to
be that tag plus optional outer whitespace. Separate `reasoning_content` does not affect
strictness. For normal endings, the raw objective is
`0.9 * bounded_game_reward + 0.1 * all_strict`; one non-strict turn removes the
episode's format bonus. Unparseable output terminates immediately and keeps only the
weighted game progress earned before termination. A parseable wall collision executes as
a one-step environment no-op in both training and evaluation; the next turn receives the
same pixel-derived blocked hint. Artifacts expose parse/strict rates, `all_strict`,
planner hint/match rates, wall collisions, the unscaled game reward, strict bonus and
additive reward components.

Completed games also receive
`step_efficiency = -step_efficiency_penalty_weight * clamp(env_steps/max_steps, 0, 1)`
inside the bounded game reward; the recipe default is `0.05`. Failures receive no step
term, because rewarding shorter failures would favor early death, invalid termination or
giving up instead of collecting more pellets. Setting the weight to zero restores the
previous game-reward formula. Artifacts record the budget, observed steps, configured
weight and actual raw/weighted step contribution.

`gconfig.reward_normalization=true` normalizes one logical episode return within each
seed's 12-game group, using mean and population standard deviation. Every decision row
from an episode receives the same normalized episode signal. Raw game rewards remain in
\[0,1\]; normalized training rewards can be negative or exceed 1. This is not batch
min-max normalization. Actor reward/advantage normalization are disabled to avoid a
second normalization. The entrypoint rejects groups whose `original_rewards` are all
equal because they contain no relative GRPO signal; rejected trajectories can still be
written to rollout artifacts for audit. `actor.loss_aggregation=rollout_mean` makes the
policy objective equal-weight by episode: token losses are averaged within each decision
row, decision rows are averaged within their logical episode, then episode losses are
averaged across the batch. Thus a 400-decision episode does not outweigh a 40-decision
episode, and answer token length does not change a row's weight. The Pacman agent does
not load a local tokenizer/processor; the native proxy and training backend continue
owning tokenization and multimodal training tensors. Proxy requests carry
`gconfig.max_tokens` as `max_total_tokens` per independent turn, without player-side
token estimation or truncation. Remote context-limit failures are rejected as technical
errors.

The recipe keeps KL 0.01, a fixed reference initialized from the actor's starting model,
one PPO minibatch, LR `5e-7`, clip 0.05, BF16 model weights, FP32 optimizer states and
`attn_impl: flash_attention_2`. Each independent decision row must fit actor/reference
`max_tokens_per_mb`, which follows the per-request context budget. 80 train seeds ×
batch size 8 gives 10 updates per epoch; with group size 12, each update uses 96
accepted games. Two epochs give 20 updates.

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
third-death endings, independent decision-row loss masks, finite updates and repeated
memory usage. For other allocations, scale batch size and microbatch divisibility for
DP/PP. Remote context-limit errors are technical failures, not completed games. No UT,
training, GPU/runtime acceptance or learning-quality validation was run as part of
delivery.

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
