# Pacman VLM SFT

This example cold-starts a vision-language policy from teacher-labelled Pacman frames
before online RL. It is intentionally isolated under `examples/`: the standard AReaL
`SFTTrainer`, FSDP engine, and loss implementation are unchanged.

Each `records.jsonl` row becomes one causal-LM SFT sample:

- input: one game frame and a deterministically shuffled list of actions;
- target: the uppercase letter corresponding to `teacher_action`;
- loss: the answer letter and EOS only; the visual prompt is masked;
- split: seeds satisfying `seed % 10 == 0` are validation-only.

The first baseline uses the hard `teacher_action`. The collected `teacher_probs` remain
available in the raw data but are not consumed by AReaL's standard token-level SFT loss.

## Run

The example reads the model, data, and output paths from environment variables. For the
0.8B base model and the Pacman data already installed on the shared filesystem:

```bash
export MODEL_PATH=/storage/openpsi/models/Qwen__Qwen3.5-0.8B
export PACMAN_DATA_ROOT=/storage/openpsi/users/ljl/workspace/game_player/pacman
export AREAL_OUTPUT_ROOT=/storage/openpsi/users/ljl/workspace/game_player/outputs

python examples/vlm/pacman_sft/train.py \
  --config examples/vlm/pacman_sft/pacman_sft.yaml
```

The default recipe uses one node with eight GPUs and `fsdp:d8p1t1`. Override
`ACTOR_BACKEND` and the cluster fields when using a different allocation.

For a short end-to-end smoke run, retain the same distributed setup but limit the
dataset and optimizer steps:

```bash
TRIAL_NAME=smoke python examples/vlm/pacman_sft/train.py \
  --config examples/vlm/pacman_sft/pacman_sft.yaml \
  total_train_steps=2 \
  train_dataset.dataset_kwargs.max_samples=256 \
  valid_dataset.dataset_kwargs.max_samples=64
```

Dataset preprocessing is lazy: the example scans JSONL metadata at startup and
decodes/processes JPEGs in DataLoader workers. It does not materialize another copy of
the 100,000 images.
