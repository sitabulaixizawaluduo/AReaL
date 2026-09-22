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

The policy sees original-resolution screenshots, fixed action requests and an RGB-only
visual-planner hint. No source-state position, facing, pellet counts, deaths, ghost
state, timers, legal-move lists or source-state planner suggestions are supplied as
text. The screenshot may naturally contain the game's visible HUD. Environment state
remains available only for reward, termination and audit.

Every model request is an independent decision containing only the fixed system prompt,
the current image and a short pixel-derived hint. The reset request sends the original
336×400 frame. Later requests send a 13×13-tile (208×208) crop centered on Pacman at the
original pixel resolution, with black padding at screen edges. Previous images, answers
and reasoning are not resent. Ambiguous Pacman detection or a non-portal position
jump/respawn falls back to the full frame. The visual planner and blocked-move detector
always inspect full RGB.

Both modes compare Pacman's current pixel-derived cell with the compact cell stored
before the previous model action. If it is unchanged, the current request reports that
direction as blocked without resending the old frame. Small pellets and HUD lives are
excluded.

Training and standalone play reconstruct the same fixed 16-pixel maze grid, walkable
graph and red ghost-pen gate from the reset screenshot. On each turn they detect Pacman,
visible pellets and confidently identified normal/vulnerable ghosts from RGB colors,
then use the Edward route planner with a deterministic pixel-only safety fallback to add
one bounded visual option. The model owns the high-level choice:
`<answer>OPTION A0</answer>` selects its advertised route, while
`<answer>MOVE X</answer>` remains a one-step correction fallback. During an option, the
harness checks every new screenshot and interrupts on a blocked or unexpected
transition, ambiguous Pacman detection, target arrival, ghost-mode change or the
advertised commit limit. The refreshed RGB/Edward plan exposes all currently safe first
actions; the harness continues the originally advertised route when its next move
remains in that set. It never replaces the route with the refreshed recommendation. No
continuation check reads source state, `info` or source legal actions. The hint and
evidence record `safe_actions`, are labeled `rgb_pixels_only` and are omitted when
extraction is ambiguous. `planner_assisted` defaults to true and supports an explicit
false ablation in both modes.

The model chooses the advertised option or one fixed move:

```text
<answer>MOVE U</answer>
<answer>MOVE D</answer>
<answer>MOVE L</answer>
<answer>MOVE R</answer>
<answer>OPTION A0</answer>
```

For compatibility with checkpoints that emit only the advertised identifier,
`<answer>A0</answer>` executes the same option. It is deliberately non-strict and cannot
earn the strict-format bonus; the canonical option form remains
`<answer>OPTION A0</answer>`.

An option may execute up to eight environment steps inside one model decision. Physical
legality and continuation are derived from fresh RGB frames and the reconstructed maze
graph and fail closed on ambiguous extraction; they never fall back to source-state
legal actions.

At a newly reset game's spawn point, Pacman is in a horizontal corridor: the first move
must be `L` or `R`; the reset request explicitly offers only those two actions.

Standalone tools enable native thinking with `enable_thinking=true`; SDK
`reasoning_content` is shown/audited but removed before the next request. Standalone and
training use the same independent current-frame requests. The original output remains
decision evidence. Standalone `--no-reasoning` sends `enable_thinking=false` and leaves
visible content unsplit. Training proxy requests always keep `enable_thinking=false`. In
both modes, separate SDK `reasoning_content` is audit-only: it is not visible content,
does not affect strictness and does not independently cause `invalid_format`. The proxy
retains the configured `gconfig.max_tokens` as `max_total_tokens` for each request.
Provider- reported prompt, completion and total token counts are copied into decision
evidence when present, but never enter a later prompt. Full-frame and prompt-image
hashes, scope, crop bounds and padding are also audited. The player does not load a
model, tokenizer or processor. The remote server owns tokenization and context-limit
enforcement.

Malformed answers end the episode with **no format bonus, no executed action and no
retry**. Game progress earned before the malformed answer remains in the episode return.
In both modes, a well-formed U/D/L/R blocked by a wall is sent to the environment:
Pacman stays in place, the step is consumed, and the next request reports the visually
detected collision. This is audited as a wall collision rather than an `invalid_action`
ending. Remote context-limit rejections are technical failures: they are propagated and
audited as errors, not converted into game rewards. Other technical failures follow the
same path.

## Reward

Let `p` be normal-pellet clearance, `s` power-pellet clearance, `g` ghosts eaten, `d`
deaths, `n` environment steps and `M` the step budget. The default ghost target is `K=4`
and completion-only efficiency weight is `w=0.05`:

```text
E = win ? -w*clamp(n/M, 0, 1) : 0
G = clip((win ? 0.9 : 0.5*p) + 0.05*s + 0.05*min(g/K, 1) - 0.02*d + E, 0, 1)
R = 0.9*G + 0.1*I(every visible response uses the canonical strict answer form)
```

A response with exactly one valid answer tag remains executable even when visible text
appears outside the tag, but it loses the whole-episode strict-format bonus. Separate
SDK `reasoning_content` is not visible response text and does not affect strictness. A
fully unparseable response terminates immediately and earns no format bonus, while the
`0.9*G` game term keeps prior progress. Game/step-budget endings use actual progress.
Step efficiency applies only after a win: penalizing failures would reward early death,
invalid output or giving up over longer attempts that collect more pellets. `env_steps`,
`max_steps`, the configured weight, actual efficiency penalty, `bounded_game_reward`,
`all_strict`, strict bonus and additive `reward_components` make the final bounded
return auditable. Transition rewards are zero before final settlement.

Compare checkpoints with identical game rules, harness, seeds, sampling and budgets. Use
dev for selection and untouched test seeds for final acceptance. Primary metrics are
zero-death win rate, overall win rate, ordinary-pellet clearance, game score and raw
reward; also inspect deaths, invalid formats, planner recommendation matches and wall
collisions. The legacy safe-advice metric remains null because it is distinct from the
RGB planner recommendation.

The code has passed static checks only. No claim of GPU fit, training effectiveness,
endpoint compatibility or full-game runtime acceptance is made by this delivery.
