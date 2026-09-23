# SPDX-License-Identifier: Apache-2.0

"""Collect source-state labels and load image/oracle pairs for H1 RL."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import stat
import sys
import tempfile
from pathlib import Path

from .policy import ACTIONS, policy_decision, project_oracle_info


def compact_manifest(path: Path) -> int:
    """Convert older list-form replay prefixes to compact action-index strings."""
    count = 0
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        with path.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                prefix = record["action_prefix"]
                if isinstance(prefix, list):
                    record["action_prefix"] = "".join(
                        str(ACTIONS.index(action)) for action in prefix
                    )
                temporary.write(json.dumps(record, separators=(",", ":")) + "\n")
                count += 1
    os.chmod(temporary_path, stat.S_IMODE(path.stat().st_mode))
    os.replace(temporary_path, path)
    return count


async def _collect_split(
    playjev_root: Path,
    output_root: Path,
    split: str,
    count: int,
    seed0: int,
    pages: int,
    epsilon: float,
) -> None:
    if str(playjev_root) not in sys.path:
        sys.path.insert(0, str(playjev_root))
    from playjev.env import VecGame
    from playjev.teachers import make_teacher

    destination = output_root / split
    frames = destination / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed0)
    written = 0
    skipped_special = 0
    next_seed = seed0 + pages
    seeds = list(range(seed0, next_seed))
    steps = [0] * pages
    episodes = [0] * pages
    actions_taken: list[str] = [""] * pages

    async with VecGame("pacman", n=pages) as env:
        obs = await env.reset(seeds)
        if tuple(action["name"] for action in env.actions) != ACTIONS:
            raise RuntimeError("Pacman action indices differ from the teacher wrapper")
        original_teachers = [make_teacher("pacman", env.actions) for _ in range(pages)]
        for teacher in original_teachers:
            teacher.reset()
        with (destination / "records.jsonl").open("w", encoding="utf-8") as stream:
            while written < count:
                actions = []
                for i, observation in enumerate(obs):
                    info = observation["info"]
                    source_probs = original_teachers[i].act(observation)
                    if not any(g["vulnerable"] or g["eaten"] for g in info["ghosts"]):
                        oracle_info = project_oracle_info(info)
                        label = policy_decision(oracle_info)
                        if any(
                            abs(label["probabilities"][name] - source_probs[j]) > 1e-5
                            for j, name in enumerate(ACTIONS)
                        ):
                            raise RuntimeError(
                                f"Single-state teacher diverged at seed={seeds[i]} step={steps[i]}"
                            )
                        if written < count:
                            image_name = f"frames/{written:07d}.jpg"
                            (destination / image_name).write_bytes(observation["frame"])
                            stream.write(
                                json.dumps(
                                    {
                                        "id": f"{split}-{seeds[i]}-{episodes[i]}-{steps[i]}",
                                        "seed": seeds[i],
                                        "episode": episodes[i],
                                        "step": steps[i],
                                        "image": image_name,
                                        "oracle_info": oracle_info,
                                        "teacher_action": label["action"],
                                        "teacher_probabilities": label["probabilities"],
                                        # Action indices are compact and replayable when
                                        # a later multi-step stage restores this state.
                                        "action_prefix": actions_taken[i],
                                    },
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            written += 1
                    else:
                        skipped_special += 1
                    chosen = max(range(len(source_probs)), key=source_probs.__getitem__)
                    if rng.random() < epsilon:
                        chosen = rng.randrange(len(ACTIONS))
                    actions.append(chosen)
                    actions_taken[i] += str(chosen)
                    steps[i] += 1
                obs = await env.step(actions)
                for i, observation in enumerate(obs):
                    if observation.get("errors"):
                        raise RuntimeError(observation["errors"][:2])
                    if observation["done"] or steps[i] >= 3000:
                        seeds[i] = next_seed
                        next_seed += 1
                        episodes[i] += 1
                        steps[i] = 0
                        actions_taken[i] = ""
                        obs[i] = await env.pages[i].reset(seeds[i])
                        original_teachers[i].reset()
    print(f"{split}: {written} samples; skipped {skipped_special} special-mode frames")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Pacman H1 image/oracle pairs")
    parser.add_argument("--playjev-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact-existing", type=Path)
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--valid-steps", type=int, default=2_000)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed0", type=int, default=100_000)
    args = parser.parse_args()
    if args.compact_existing is not None:
        for split in ("train", "valid"):
            manifest = args.compact_existing / split / "records.jsonl"
            print(f"{split}: compacted {compact_manifest(manifest)} records")
        return
    if args.playjev_root is None or args.output is None:
        parser.error("--playjev-root and --output are required for collection")
    if not (args.playjev_root / "playjev" / "env.py").is_file():
        parser.error("--playjev-root must point to the PlayJev repository")
    if args.train_steps < 1 or args.valid_steps < 1 or args.pages < 1:
        parser.error("step counts and pages must be positive")
    if not 0 <= args.epsilon <= 1:
        parser.error("--epsilon must be between 0 and 1")
    if any(
        (args.output / split / "records.jsonl").exists() for split in ("train", "valid")
    ):
        parser.error("output already contains a train or valid manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(
        _collect_split(
            args.playjev_root.resolve(),
            args.output.resolve(),
            "train",
            args.train_steps,
            args.seed0,
            args.pages,
            args.epsilon,
        )
    )
    asyncio.run(
        _collect_split(
            args.playjev_root.resolve(),
            args.output.resolve(),
            "valid",
            args.valid_steps,
            args.seed0 + 10_000_000,
            args.pages,
            args.epsilon,
        )
    )


if __name__ == "__main__":
    main()
