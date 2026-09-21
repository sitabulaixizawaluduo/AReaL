# Pacman player and training example

The same Pacman player supports two independent entrypoints:

- [tools/](tools/README.md): prepare public game sources/data, play or evaluate through
  a standard OpenAI-compatible endpoint, view parallel games in a live dashboard,
  summarize outcomes and record video. No AReaL installation is required.
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

The policy sees only original-resolution screenshots and fixed action requests. No
source-state position, facing, pellet counts, deaths, ghost state, timers, legal-move
lists or source-state planner suggestions are supplied as text. The screenshot may
naturally contain the game's visible HUD. Environment state remains available only for
execution checks, reward, termination and audit.

Each standalone request contains the fixed system prompt, one current image and a short
pixel-derived hint. The reset request sends the original 336×400 frame. Later requests
send a 13×13-tile (208×208) crop centered on Pacman at the original pixel resolution,
with black padding at screen edges. Previous images and assistant messages are not
resent. Ambiguous Pacman detection or a non-portal position jump/respawn falls back to
the full frame. The visual planner and blocked-move detector always inspect full RGB.

Standalone recovery compares Pacman's current pixel-derived cell with the compact cell
stored before the previous model action. If it is unchanged, the current request reports
that direction as blocked without resending the old frame. Small pellets and HUD lives
are excluded. Training proxy sessions never receive this standalone recovery prompt.

Standalone play also reconstructs the fixed 16-pixel maze grid, walkable graph and red
ghost-pen gate from the reset screenshot. On each turn it detects Pacman, visible
pellets and confidently identified normal/vulnerable ghosts from RGB colors, then uses
the Edward route planner with a deterministic pixel-only safety fallback to add one
concise recommended move to the request. The model still returns and owns the executed
action; the planner never executes a move. The hint and its audit evidence are labeled
`rgb_pixels_only`, omitted when extraction is ambiguous, bounded to the current turn,
and never enabled for training proxy sessions.

The four model outputs are fixed moves:

```text
<answer>MOVE U</answer>
<answer>MOVE D</answer>
<answer>MOVE L</answer>
<answer>MOVE R</answer>
```

Every model move executes exactly one environment step, followed by a new visual
decision. Physical legality is derived from the RGB-reconstructed maze graph and fails
closed on ambiguous extraction; it never falls back to source-state legal actions.

At a newly reset game's spawn point, Pacman is in a horizontal corridor: the first move
must be `L` or `R`; the reset request explicitly offers only those two actions.

Standalone tools enable native thinking with `enable_thinking=true`; SDK
`reasoning_content` is shown/audited but removed before the next request. Standalone
requests retain only the system prompt and current screenshot; the previous screenshot,
action and reasoning are not resent. The original output remains decision evidence.
Standalone `--no-reasoning` sends `enable_thinking=false` and leaves visible content
unsplit. Training proxy requests always keep `enable_thinking=false`. In both modes,
separate SDK `reasoning_content` is audit-only: it is not visible content, does not
affect strictness and does not independently cause `invalid_format`. The proxy retains
the configured `gconfig.max_tokens` as `max_total_tokens`; its original-resolution
reset-frame/cropped screenshots and assistant responses remain in the complete
append-only concat conversation across model decisions. Provider-reported prompt,
completion and total token counts are copied into decision evidence when present, but
never enter a later prompt. Full-frame and prompt-image hashes, scope, crop bounds and
padding are also audited. The player does not load a model, tokenizer or processor. The
remote server owns tokenization and context-limit enforcement.

Malformed answers end the episode with **reward zero, no executed action and no retry**.
For standalone play, a well-formed U/D/L/R blocked by a wall is sent to the environment:
Pacman stays in place, the step is consumed, and play continues from the next
screenshot. The illegal/blocked outcome is audited. Training proxy sessions retain the
original strict behavior, where a physically illegal direction ends the episode with
zero reward before execution. Remote context-limit rejections are technical failures:
they are propagated and audited as errors, not converted into game rewards. Other
technical failures follow the same path.

## Reward

Let `p` be normal-pellet clearance, `s` power-pellet clearance, `g` ghosts eaten, `d`
deaths, `n` environment steps and `M` the step budget. The default ghost target is `K=4`
and completion-only efficiency weight is `w=0.05`:

```text
E = win ? -w*clamp(n/M, 0, 1) : 0
G = clip((win ? 0.9 : 0.5*p) + 0.05*s + 0.05*min(g/K, 1) - 0.02*d + E, 0, 1)
R = 0.9*G + 0.1*I(every visible response is exactly <answer>MOVE X</answer>)
```

A response with exactly one valid answer tag remains executable even when visible text
appears outside the tag, but it loses the whole-episode strict-format bonus. Separate
SDK `reasoning_content` is not visible response text and does not affect strictness. A
fully unparseable response or illegal action overrides the formula with zero.
Game/step-budget endings use actual progress. Step efficiency applies only after a win:
penalizing failures would reward early death, invalid output or giving up over longer
attempts that collect more pellets. `env_steps`, `max_steps`, the configured weight,
actual efficiency penalty, `bounded_game_reward`, `all_strict`, strict bonus and
additive `reward_components` make the final bounded return auditable. Transition rewards
are zero before final settlement.

Compare checkpoints with identical game rules, harness, seeds, sampling and budgets. Use
dev for selection and untouched test seeds for final acceptance. Primary metrics are
zero-death win rate, overall win rate, ordinary-pellet clearance, game score and raw
reward; also inspect deaths and illegal outputs/actions. The legacy advice metric is
null because no safety suggestions are supplied.

The code has passed static checks only. No claim of GPU fit, training effectiveness,
endpoint compatibility or full-game runtime acceptance is made by this delivery.
