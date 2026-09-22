# SPDX-License-Identifier: Apache-2.0

"""Adapt the independent Pacman player to AReaL's native SDK proxy workflow."""

import inspect
import uuid
from typing import Any

from examples.vlm.game_player.pacman.tools.player import PacmanPlayer
from examples.vlm.game_player.protocols import EpisodeContext

from areal.api.cli_args import GenerationHyperparameters
from areal.infra import workflow_context
from areal.utils import stats_tracker


class PacmanAgent:
    """Plain async agent; the proxy exports independent decision rows.

    All gameplay and SDK resource management live in the independent player.
    This adapter supplies training context and returns one whole-game reward. With
    individual export and unit turn discount, the proxy propagates that terminal
    outcome to every decision row from the same episode.
    """

    def __init__(
        self,
        *,
        gconfig: GenerationHyperparameters,
        model: str,
        options: dict[str, Any],
    ):
        if gconfig.greedy:
            raise ValueError("Pacman training requires sampled generation")
        self.player = PacmanPlayer(
            model=model,
            generation={
                "temperature": gconfig.temperature,
                "top_p": gconfig.top_p,
                "max_new_tokens": gconfig.max_new_tokens,
                "max_tokens": gconfig.max_tokens,
                "seed": gconfig.seed,
            },
            options=dict(options),
        )

    @staticmethod
    def classify_proxy_failure(error: Exception, **kwargs: Any) -> str:
        # Technical failures must not become synthetic zero-reward games.
        return "unknown_failure_reject"

    async def run(
        self, data: dict[str, Any], **kwargs: Any
    ) -> float | dict[str, float]:
        source = workflow_context.get()
        version = None
        worker = kwargs.get("worker_runtime")
        if worker is not None:
            version = worker.get_version()
            if inspect.isawaitable(version):
                version = await version
        context = EpisodeContext(
            training=not source.is_eval,
            is_eval=source.is_eval,
            task_id=source.task_id,
            sample_idx=source.sample_idx,
            group_size=source.group_size,
            model_version=None if version is None else int(version),
            session_namespace=self.player.session_namespace,
            proxy_session_id=kwargs.get("session_id"),
            attempt_id=uuid.uuid4().hex,
        )
        result = await self.player.collect(
            data,
            client=kwargs.get("client"),
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            http_client=kwargs.get("http_client"),
            context=context,
            factory_kwargs=kwargs.get("factory_kwargs"),
        )
        summary = result.summary
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            pacman_shaped_reward=summary["total_shaped_reward"],
            pacman_game_score=summary["game_score"],
            pacman_win=summary["win"],
            pacman_zero_death_win=summary["zero_death_win"],
            pacman_normal_pellet_clear_rate=summary["normal_pellet_clear_rate"],
            pacman_death_count=summary["death_count"],
            pacman_invalid_format=summary["invalid_format"],
            pacman_invalid_action=summary["invalid_action"],
        )
        # A terminal initial game state has no model completion to train.
        # Returning {} lets the native proxy reject the empty export normally.
        if not result.record.decisions:
            return {}
        return float(result.record.total_reward)
