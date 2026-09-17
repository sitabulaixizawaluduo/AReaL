# Game player framework

This directory contains ordinary example scripts, not an installable package. It
separates an independent game player from its training adapters:

```text
game_player/
  protocols.py       # Game, harness, policy, reward and episode contracts
  runtime.py         # Blocking game loop and async SDK bridge
  player.py          # Independent GamePlayer
  pacman/
    tools/           # Play, evaluate, prepare data, inspect metrics, record video
    train/           # AReaL adapter, native trainer entrypoint and one recipe
```

The root modules and `pacman/tools` do not import AReaL or the training layer. They can
run a game against an OpenAI-compatible endpoint without installing a trainer,
scheduler, inference server or GPU stack locally. `pacman/train` depends on AReaL and
adapts the same player to native agent training. Dependency direction is **train →
tools/player**, never the reverse.

## Plugin contracts

[protocols.py](protocols.py) defines replaceable components:

| Component      | Responsibility                                                       |
| -------------- | -------------------------------------------------------------------- |
| Task row       | Seed, level/map payload, initial state or other game-specific inputs |
| GameSession    | `reset`, `step`, interruptible `cancel`, resource-releasing `close`  |
| Harness        | Context, action parsing/execution and multi-step option continuation |
| Policy         | SDK requests and parsing the actual returned completion              |
| Reward         | Actual transition events and final episode outcome                   |
| SessionFactory | Assemble fresh components, summary and persistence callbacks         |

[GamePlayer](player.py) runs the same [EpisodeRunner](runtime.py) for independent play
and training. A decision is one actual model completion, potentially with many tokens.
An option may execute several environment steps under that decision; automatic option
steps are not invented policy outputs. The runtime distinguishes game endings from
technical errors and closes resources before completion or cancellation returns.

`cancel()` must be thread-safe and nonblocking. Native calls need bounded timeouts or
must respond to cancellation; Python cannot safely kill an arbitrary blocked thread.
Factories must clean up partial construction failures. Do not put per-episode state in a
shared `last_record` attribute when episodes run concurrently.

The runtime retains explicit JSON evidence after consuming observations, not live
screenshots or simulator state. Record images separately when needed. SDK response text
alone does not establish token IDs, behavior logprobs or training weight versions.
Training evidence belongs to the training adapter/backend, not the game protocol.

## Add another game

Implement these protocols and assemble a Session, then reuse GamePlayer. Dataset rows
may carry full level payloads rather than only seeds. Place game rules, rendering,
harness, reward and tools under that game's directory. If training is needed, add a
separate adapter that supplies trainer context and returns the episode reward.

AReaL owns model parallelism, grouping, loss reduction, weight updates, checkpointing
and scheduling. The player framework does not override these mechanisms. One exported
episode row does not automatically imply equal episode loss weight.

See the [Pacman overview](pacman/README.md), [independent tools](pacman/tools/README.md)
and [training instructions](pacman/train/README.md). Only static checks have been
performed; GPU training and independent endpoint/game execution remain unvalidated.
