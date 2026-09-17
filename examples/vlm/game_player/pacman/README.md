# Pacman player and training example

The same Pacman player supports two independent entrypoints:

- [tools/](tools/README.md): prepare public game sources/data, play or evaluate through
  a standard OpenAI-compatible endpoint, summarize outcomes and record video. No AReaL
  installation is required.
- [train/](train/README.md): adapt the player to AReaL's native SDK-agent GRPO training
  with SGLang and Megatron. The sole recipe is
  [train/pacman_rl.yaml](train/pacman_rl.yaml).

Shared game rules, harness, prompts, reward and player live under `tools/`; the
[framework](../README.md) owns generic execution. Training imports these tools. Tools
never import the training layer or create an AReaL proxy.

## Rules and actions

Normal ghosts are enabled. The episode stops at the third death, regardless of bonus
lives awarded by the original game. Clearing every normal pellet wins; power pellets are
not required. The default budget is 512 environment steps.

The harness supplies safe-option suggestions and legal directions, but does not mask
model logits. The model freely chooses a suggested multi-step option or any physically
legal single move, including a direction with no safety proof:

```text
<answer>OPTION A0</answer>
```

```text
<answer>MOVE U</answer>
```

Directions are `U`, `D`, `L`, `R`; the prompt states currently available option IDs.
Native thinking is disabled with `enable_thinking=false`. Real multi-turn screenshots
and assistant responses remain in the conversation. The player checks the expanded
prompt length and remaining budget before the next request; it never silently drops
history. `context_budget_exhausted` settles the current outcome normally.

Malformed answers, unknown options and physically illegal moves end the episode with
**reward zero, no executed action and no retry**. The actual invalid completion remains
available for training. A first-prompt context overflow has no model completion and
therefore no training sample, but its attempt is audited. Technical failures remain
errors rather than fabricated game outcomes.

## Reward

Let `p` be normal-pellet clearance, `s` power-pellet clearance, `g` ghosts eaten and `d`
deaths. The default ghost target is `K=4`:

```text
R = clip((win ? 0.9 : 0.5*p) + 0.05*s + 0.05*min(g/K, 1) - 0.02*d, 0, 1)
```

An invalid output overrides this formula with zero. Game/step/context-budget endings use
actual progress. There is no distance bonus, step penalty or safety-refusal penalty. The
`reward` and historical `total_shaped_reward` fields both contain this raw bounded
episode return. Transition rewards are zero before final settlement; summing them is not
a substitute for the episode outcome.

Compare checkpoints with identical game rules, harness, seeds, sampling and budgets. Use
dev for selection and untouched test seeds for final acceptance. Primary metrics are
zero-death win rate, overall win rate, ordinary-pellet clearance, game score and raw
reward; also inspect deaths, illegal outputs/actions and safe-suggestion adherence.
Adherence is measured, not directly rewarded.

The code has passed static checks only. No claim of GPU fit, training effectiveness,
endpoint compatibility or full-game runtime acceptance is made by this delivery.
