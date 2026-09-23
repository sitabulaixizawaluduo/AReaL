# PlayJev Pacman H1 agentic RL

This example trains one visual game decision per rollout (`max_env_steps=1`). The model
sees a Pacman screenshot, infers the maze and character positions, calls `pacman_policy`
with **its inferred state**, and then emits an action. The tool uses PlayJev's Pacman
teacher policy, but cannot access the source-state label. A private verifier compares
the final action against the teacher's decision on the source state and measures how
accurately the tool arguments describe the frame. This is a two-generation agent
workflow, not a single-answer RLVR task.

The teacher source is included in `pacman_teacher.py` to make the rollout tool available
on AReaL workers without importing the PlayJev browser environment. The data collector
independently checks the wrapper's action probabilities against the original PlayJev
teacher for every saved frame.

## Data

Collect from the PlayJev repository in an environment with its Playwright and Chromium
dependencies installed:

```bash
python3 -m examples.vlm.playjev_pacman_h1.data \
  --playjev-root /path/to/PlayJev \
  --output /path/to/pacman_h1 \
  --train-steps 100000 --valid-steps 2000
```

The collector writes `train/records.jsonl`, `valid/records.jsonl`, and JPEGs under each
split's `frames/` directory. The splits use disjoint seed ranges. Each JSONL record
contains an image path, source-state oracle label, teacher action and probabilities,
seed, step, and replayable action prefix. The `PacmanH1Dataset` reads JSONL lazily; only
the screenshot reaches model input. Do not serve the manifest itself to the model.

H1 excludes frames where a ghost is vulnerable or eaten. The original teacher tracks
hidden power-pill and ghost timers across steps, so the stateless tool would otherwise
disagree with the source teacher. Those states require a stateful extension in a later
multi-step stage.

## Train on one 8-GPU node

From the AReaL repository root, with AReaL's Megatron and SGLang dependencies installed,
run:

```bash
bash examples/vlm/playjev_pacman_h1/run_8gpu.sh \
  /path/to/Qwen3.5-0.8B \
  /path/to/pacman_h1 \
  /path/to/experiment_root
```

The script takes three positional paths and sets no environment variables. The recipe
uses Megatron `d8p1t1`, SGLang AWEX colocation on the same eight GPUs, eight GRPO
samples per prompt, `concat` export, group-level actor reward normalization, and a
filter for identical-reward groups. Validation samples one deterministic rollout per
screenshot every 100 steps. The full default training epoch covers the 100,000 train
decisions; checkpoints are saved every 100 steps.

The verifier reward is `0.4 * visual_state_accuracy + 0.6 * teacher_action_quality`.
Action quality is the source teacher's probability for the chosen action divided by its
highest probability. Scores are in `[0, 1]`; invalid tool calls or final answers score
zero. No format-only bonus is added. An all-zero group cannot contribute to
group-normalized GRPO, so it is rejected before PPO.

This H1 harness evaluates the one decision against a recorded source state. It does
**not** advance the browser game during training and does not measure pellets, deaths,
or level completion. Those outcomes belong to the later multi-step (`max_env_steps > 1`)
phase. Before a full run, smoke-test the target Qwen3.5 checkpoint's multimodal tool
calls under the deployed SGLang version; the local CPU tests here do not validate that
GPU runtime path. If almost all groups are rejected at initialization, a short
tool-call-format cold start may be needed before GRPO can learn from the dense verifier
reward.
