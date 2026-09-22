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

From the AReaL repository root, launch the fixed Qwen3.5-0.8B, single-node, eight-GPU
recipe directly:

```bash
bash examples/vlm/pacman_sft/run_qwen3_5_0_8b_8gpu.sh
```

The script uses `fsdp:d8p1t1` and supplies the model, dataset, output, and eight-GPU
settings as literal CLI overrides. It does not require, declare, or export custom
environment variables.

Dataset preprocessing is lazy: the example scans JSONL metadata at startup and
decodes/processes JPEGs in DataLoader workers. It does not materialize another copy of
the 100,000 images.
