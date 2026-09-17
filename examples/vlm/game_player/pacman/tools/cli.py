"""Independent game preparation, SDK play/evaluation, metrics and video tools."""

import argparse
import asyncio
import json
import os
import sys
from functools import partial
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from examples.vlm.game_player.pacman.tools.metrics import EpisodeMetrics
from examples.vlm.game_player.pacman.tools.prepare import GamePreparation


class PacmanCommands:
    @staticmethod
    def parser():
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        prepare = commands.add_parser("prepare")
        prepare.add_argument("--root", type=Path, required=True)
        prepare.add_argument("--install-game", action="store_true")
        commands.add_parser("preflight")
        for name in ("evaluate", "record_video"):
            command = commands.add_parser(name)
            command.add_argument(
                "--endpoint",
                required=True,
                help="OpenAI-compatible base URL, normally ending in /v1",
            )
            command.add_argument(
                "--model",
                required=True,
                help="Model identifier accepted by the endpoint and available to local HF processing",
            )
            command.add_argument(
                "--api-key",
                default=None,
                help="Defaults to OPENAI_API_KEY; omitted for unauthenticated local servers",
            )
            command.add_argument(
                "--manifest",
                default=os.getenv("PACMAN_SPLIT_MANIFEST"),
                required=not os.getenv("PACMAN_SPLIT_MANIFEST"),
            )
            command.add_argument("--output", required=True)
            command.add_argument("--split", choices=["dev", "test"], default="dev")
            command.add_argument(
                "--seed",
                type=int,
                help="Select a seed belonging to the requested split",
            )
            command.add_argument(
                "--limit", type=int, default=1 if name == "record_video" else None
            )
            command.add_argument("--repeats", type=int, default=1)
            command.add_argument("--generation-seed", type=int, default=0)
            command.add_argument(
                "--concurrency", type=int, default=1 if name == "record_video" else 4
            )
            command.add_argument("--max-steps", type=int, default=512)
            command.add_argument("--max-tokens", type=int, default=32768)
            command.add_argument("--max-new-tokens", type=int, default=64)
            command.add_argument("--ghost-reward-target", type=int, default=4)
        summarize = commands.add_parser("summarize")
        summarize.add_argument("directory", type=Path)
        summarize.add_argument(
            "--scope",
            choices=["all_attempts", "completed"],
            default="all_attempts",
        )
        return parser

    @classmethod
    def main(cls):
        parser = cls.parser()
        args, overrides = parser.parse_known_args()
        if overrides:
            parser.error(f"Unrecognized arguments: {overrides}")
        if args.command == "prepare":
            print(GamePreparation.prepare(args.root, args.install_game))
        elif args.command == "preflight":
            print(json.dumps(GamePreparation.verify_environment(), indent=2))
        elif args.command in ("evaluate", "record_video"):
            from examples.vlm.game_player.pacman.tools.evaluate import PacmanEvaluation

            GamePreparation.verify_environment()
            if (
                args.repeats < 1
                or args.concurrency < 1
                or (args.limit is not None and args.limit < 1)
                or min(
                    args.max_steps,
                    args.max_tokens,
                    args.max_new_tokens,
                    args.ghost_reward_target,
                )
                < 1
            ):
                parser.error(
                    "Episode counts, token budgets, step budget and ghost reward target must be positive"
                )
            factory = None
            if args.command == "record_video":
                from examples.vlm.game_player.pacman.tools.recording import (
                    RecordingEnv,
                    VideoAudit,
                )

                if (args.limit, args.repeats, args.concurrency) != (1, 1, 1):
                    parser.error(
                        "Record one complete attempt at a time: limit=repeats=concurrency=1"
                    )
                factory = partial(
                    RecordingEnv, recording_root=Path(args.output).resolve() / "frames"
                )
            result = asyncio.run(PacmanEvaluation.run(args, env_factory=factory))
            if factory is not None:
                episode = json.loads(
                    next((Path(args.output) / "episodes").glob("*.json")).read_text()
                )
                result = VideoAudit.encode(Path(args.output) / "frames", episode)
            print(json.dumps(result, indent=2))
            if result.get("pending_episodes", 0) or result.get(
                "completed_episodes", 0
            ) < result.get("planned_episodes", 0):
                sys.exit(2)
        elif args.command == "summarize":
            print(json.dumps(EpisodeMetrics.read(args.directory, args.scope), indent=2))


if __name__ == "__main__":
    PacmanCommands.main()
