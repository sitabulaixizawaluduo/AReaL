"""Fixed-plan Pacman evaluation against a standard OpenAI-compatible endpoint."""

import asyncio
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from openai import AsyncOpenAI

from examples.vlm.game_player.pacman.tools.datasets import (
    ArtifactIdentity,
    SplitManifest,
)
from examples.vlm.game_player.pacman.tools.metrics import EpisodeMetrics


class PacmanEvaluation:
    @staticmethod
    def _write(path: Path, value) -> None:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")

    @staticmethod
    async def run(args, frame_observer=None, env_factory=None):
        from examples.vlm.game_player.pacman.tools.player import PacmanPlayer
        from examples.vlm.game_player.pacman.tools.prepare import GamePreparation

        root = Path(args.output).resolve()
        await asyncio.to_thread(root.mkdir, parents=True, exist_ok=False)
        await asyncio.to_thread((root / "episodes").mkdir)
        split = await asyncio.to_thread(SplitManifest.read, args.manifest)
        seeds = getattr(split, args.split)
        if args.seed is not None:
            if args.seed not in seeds:
                raise ValueError("Requested seed does not belong to the chosen split")
            seeds = (args.seed,)
        if args.limit:
            seeds = seeds[: args.limit]
        planned = [
            {
                "episode_id": f"{args.split}-{seed}-{repeat}",
                "seed": seed,
                "generation_seed": args.generation_seed + repeat,
                "split": args.split,
            }
            for seed in seeds
            for repeat in range(args.repeats)
        ]
        url = args.endpoint.rstrip("/")
        endpoint = urlparse(url)
        if (
            endpoint.scheme not in {"http", "https"}
            or not endpoint.netloc
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError(
                "Provide an OpenAI-compatible base URL without credentials, query or fragment"
            )
        if not endpoint.path:
            url += "/v1"
        plan = {
            "schema": "pacman-evaluation-v1",
            "episodes": planned,
            "runtime_sources": await asyncio.to_thread(GamePreparation.source_identity),
            "endpoint": url,
            "model_identity": "caller-declared; no provider-specific discovery API required",
            "model_path": args.model,
            "split_manifest_sha256": await asyncio.to_thread(
                ArtifactIdentity.sha256, args.manifest
            ),
            "harness_contract": "free_actions_with_safe_advice",
            "reward_contract": "bounded_episode_v1",
            "death_limit": 3,
            "ghost_reward_target": args.ghost_reward_target,
            "temperature": 0.7,
            "top_p": 1.0,
            "max_environment_steps": args.max_steps,
            "max_tokens": args.max_tokens,
            "max_new_tokens": args.max_new_tokens,
            "selection": "fixed manifest prefix" if args.limit else "whole split",
            "retry_policy": "none; failures remain in denominator",
            "transport": "standard OpenAI SDK directly to the supplied endpoint",
        }
        await asyncio.to_thread(PacmanEvaluation._write, root / "plan.json", plan)
        player = await asyncio.to_thread(
            PacmanPlayer,
            model=args.model,
            generation={
                "max_new_tokens": args.max_new_tokens,
                "max_tokens": args.max_tokens,
                "temperature": 0.7,
                "top_p": 1.0,
                "seed": args.generation_seed,
            },
            options={
                "environment_max_steps": args.max_steps,
                "ghost_reward_target": args.ghost_reward_target,
                "context_safety_margin": 256,
                "artifact_root": str(root / "artifacts"),
                "pacman_python_root": os.environ["MAAPACMAN_PACMAN_ROOT"],
                "worker_base_dir": os.environ.get("PACMAN_WORKER_BASE_DIR", ""),
            },
        )
        semaphore = asyncio.Semaphore(args.concurrency)
        async with AsyncOpenAI(
            base_url=url,
            api_key=args.api_key or os.getenv("OPENAI_API_KEY") or "unused",
            max_retries=0,
        ) as client:

            async def episode(row):
                async with semaphore:
                    started = time.monotonic()
                    try:
                        collected = await player.collect(
                            row,
                            client=client,
                            factory_kwargs={
                                "frame_observer": frame_observer,
                                "env_factory": env_factory,
                            },
                        )
                        result = {**collected.summary, **row}
                    except Exception as error:
                        result = {
                            **row,
                            "status": "error",
                            "terminal_reason": "technical_error",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    await asyncio.to_thread(
                        PacmanEvaluation._write,
                        root / "episodes" / f"{row['episode_id']}.json",
                        result,
                    )

            await asyncio.gather(*(episode(row) for row in planned))
        summary = await asyncio.to_thread(EpisodeMetrics.read, root)
        await asyncio.to_thread(PacmanEvaluation._write, root / "summary.json", summary)
        return summary
