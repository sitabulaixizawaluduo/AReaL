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
from examples.vlm.game_player.pacman.tools.rewards import (
    DEFAULT_STEP_EFFICIENCY_PENALTY_WEIGHT,
)


class PacmanEvaluation:
    @staticmethod
    def _write(path: Path, value) -> None:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")

    @staticmethod
    async def run(args, frame_observer=None, env_factory=None, dashboard=None):
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
        if dashboard is not None:
            dashboard.plan(planned)
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
            "harness_contract": "pure_vision_single_moves_v1",
            "reward_contract": "completion_only_step_penalty_strict_format_v3",
            "death_limit": 3,
            "ghost_reward_target": args.ghost_reward_target,
            "step_efficiency_penalty_weight": (DEFAULT_STEP_EFFICIENCY_PENALTY_WEIGHT),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_environment_steps": args.max_steps,
            "max_new_tokens": args.max_new_tokens,
            "reasoning_enabled": args.reasoning,
            "selection": "fixed manifest prefix" if args.limit else "whole split",
            "retry_policy": "none; failures remain in denominator",
            "transport": "standard OpenAI SDK directly to the supplied endpoint",
        }
        await asyncio.to_thread(PacmanEvaluation._write, root / "plan.json", plan)
        player = PacmanPlayer(
            model=args.model,
            generation={
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.generation_seed,
                "reasoning": args.reasoning,
            },
            options={
                "environment_max_steps": args.max_steps,
                "ghost_reward_target": args.ghost_reward_target,
                "step_efficiency_penalty_weight": (
                    DEFAULT_STEP_EFFICIENCY_PENALTY_WEIGHT
                ),
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
                    episode_id = row["episode_id"]
                    if dashboard is not None:
                        dashboard.begin(episode_id)

                    def observe_frame(image, info):
                        if frame_observer is not None:
                            frame_observer(image, info)
                        if dashboard is not None:
                            dashboard.frame(episode_id, image, info)

                    def observe_event(event):
                        if dashboard is not None:
                            dashboard.event(episode_id, event)

                    try:
                        collected = await player.collect(
                            row,
                            client=client,
                            factory_kwargs={
                                "frame_observer": observe_frame
                                if dashboard is not None or frame_observer is not None
                                else None,
                                "event_observer": observe_event
                                if dashboard is not None
                                else None,
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
                    if dashboard is not None:
                        dashboard.finish(episode_id, result)

            await asyncio.gather(*(episode(row) for row in planned))
            if dashboard is not None:
                dashboard.finalizing()
        summary = await asyncio.to_thread(EpisodeMetrics.read, root)
        await asyncio.to_thread(PacmanEvaluation._write, root / "summary.json", summary)
        return summary
