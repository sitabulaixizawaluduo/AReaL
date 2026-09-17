# SPDX-License-Identifier: Apache-2.0

"""Independent asynchronous SDK player with bounded game-worker cleanup."""

import asyncio
import os
import uuid
from dataclasses import replace
from typing import Any

from examples.vlm.game_player.protocols import (
    EpisodeContext,
    EpisodeResult,
    SessionFactory,
)
from examples.vlm.game_player.runtime import EpisodeRunner


class GamePlayer:
    """Run a game through an ordinary SDK client, without a training dependency."""

    def __init__(self, factory: SessionFactory):
        self.factory = factory
        self.session_namespace = uuid.uuid4().hex

    async def collect(
        self,
        data: dict[str, Any],
        *,
        client: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
        http_client: Any = None,
        context: EpisodeContext | None = None,
        factory_kwargs: dict[str, Any] | None = None,
    ) -> EpisodeResult:
        owned_transport = client is None and http_client is None
        if client is None:
            from openai import AsyncOpenAI

            endpoint = base_url or os.getenv("OPENAI_BASE_URL")
            if not endpoint:
                raise ValueError(
                    "Supply an SDK client or explicit OpenAI-compatible endpoint"
                )
            client = AsyncOpenAI(
                base_url=endpoint,
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                http_client=http_client,
                max_retries=0,
            )
        elif getattr(client, "max_retries", 0) != 0:
            raise ValueError(
                "Game agents require SDK max_retries=0 to avoid hidden extra decisions"
            )
        try:
            context = context or EpisodeContext(training=False, is_eval=True)
            context = replace(
                context,
                session_namespace=context.session_namespace or self.session_namespace,
                attempt_id=context.attempt_id or uuid.uuid4().hex,
            )
            runner = EpisodeRunner(
                self.factory, client, asyncio.get_running_loop(), context
            )
            worker = asyncio.create_task(
                asyncio.to_thread(runner.run, data, **(factory_kwargs or {}))
            )
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError as cancellation:

                async def cleanup() -> None:
                    await asyncio.gather(worker, return_exceptions=True)
                    await runner.drain()

                try:
                    runner.cancel()
                except Exception as cancel_error:
                    cancellation.add_note(
                        f"Game cancellation hook failed: {type(cancel_error).__name__}: {cancel_error}"
                    )
                finally:
                    cleanup_task = asyncio.create_task(cleanup())
                    while not cleanup_task.done():
                        try:
                            await asyncio.shield(cleanup_task)
                        except asyncio.CancelledError:
                            continue
                    cleanup_task.result()
                raise
            except BaseException:
                await runner.drain()
                raise
            await runner.drain()
            return result
        finally:
            # A caller may share its transport across concurrent episodes.
            # Only close transports created and owned by this invocation.
            if owned_transport:
                await client.close()

    async def aevaluate_episode(
        self, data: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        return (await self.collect(data, **kwargs)).summary
