# Pacman VLM SFT

This example cold-starts a vision-language policy from teacher-labelled Pacman frames
before online RL. It is intentionally isolated under `examples/`: the standard AReaL
`SFTTrainer`, Megatron engine, and loss implementation are unchanged.

Each `records.jsonl` row becomes one causal-LM SFT sample:

- input: one game frame and a deterministically shuffled list of actions;
- target: the uppercase letter corresponding to `teacher_action`;
- loss: the answer letter and EOS only; the visual prompt is masked;
- split: seeds satisfying `seed % 10 == 0` are validation-only.

The first baseline uses the hard `teacher_action`. The collected `teacher_probs` remain
available in the raw data but are not consumed by AReaL's standard token-level SFT loss.

## Run

From the AReaL repository root, launch the fixed Qwen3.5-0.8B, single-node, eight-GPU
recipe directly:

```bash
bash examples/vlm/pacman_sft/run_qwen3_5_0_8b_8gpu.sh
```

The script uses `megatron:d8p1t1` (DP=8, PP=1, TP=1) with `megatron-bridge` and supplies
the model, dataset, output, and eight-GPU settings as literal CLI overrides. It does not
require, declare, or export custom environment variables.

Dataset preprocessing is lazy: the example scans JSONL metadata at startup and
decodes/processes JPEGs in DataLoader workers. It does not materialize another copy of
the 100,000 images.

## Evaluate a served checkpoint

Use the held-out seed split to benchmark a checkpoint exposed through an
OpenAI-compatible Chat Completions endpoint:

```bash
python examples/vlm/pacman_sft/evaluate_service.py \
  --data-root ~/Workspace/Data/game_player/pacman \
  --base-url http://REMOTE_HOST:PORT/v1 \
  --model SERVED_MODEL_NAME \
  --max-samples 1000 \
  --concurrency 32 \
  --output-dir pacman_sft_eval
```

The evaluator reproduces the training split and option permutation. It reports strict
single-letter teacher accuracy, tolerant parsed accuracy, output-format validity,
request errors, latency percentiles, throughput, label distributions, and a confusion
matrix. `results.jsonl` retains each response and `summary.json` contains aggregate
metrics. The API key defaults to `EMPTY`; use `--api-key-file` when authentication is
required.

The default `train-like` prompt keeps the SFT text immediately around the image while
still using the service's chat template. This is a service-level benchmark, so it is not
bit-identical to the raw token sequence used during training. Use the same prompt mode
when comparing checkpoints. The benchmark measures one-step teacher imitation, not
closed-loop game score.
