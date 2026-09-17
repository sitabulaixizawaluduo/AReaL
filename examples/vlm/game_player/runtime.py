# SPDX-License-Identifier: Apache-2.0

"""One interruptible blocking game loop, bridged to asynchronous inference."""

import asyncio
import math
import threading
import time
from concurrent.futures import Future
from typing import Any

from examples.vlm.game_player.protocols import (
    Decision,
    EpisodeContext,
    EpisodeRecord,
    EpisodeResult,
    EpisodeStop,
    RewardValue,
    Session,
    SessionFactory,
    Transition,
    copy_evidence,
)


class EpisodeRunner:
    """Own the session and close it before completion or cancellation returns.

    Factories execute in this thread. A factory that fails partway through
    acquisition must close its own partially constructed resources.
    """

    def __init__(
        self,
        factory: SessionFactory,
        client: Any,
        loop: asyncio.AbstractEventLoop,
        context: EpisodeContext,
    ):
        self.factory, self.client, self.loop = factory, client, loop
        self.record = EpisodeRecord(context)
        self.session: Session | None = None
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._pending: Future | None = None
        # Accessed exclusively by the inference event loop.
        self._generation_tasks: set[asyncio.Task] = set()

    def cancel(self) -> None:
        self.cancelled.set()
        with self._lock:
            pending, session = self._pending, self.session
        if pending is not None:
            pending.cancel()
        if session is not None:
            session.game.cancel()

    def _check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise asyncio.CancelledError

    async def _agenerate(self, request: dict[str, Any]) -> Any:
        task = asyncio.current_task()
        assert task is not None
        self._generation_tasks.add(task)
        try:
            self._check_cancelled()
            return await self.client.chat.completions.create(**request)
        finally:
            self._generation_tasks.discard(task)

    async def drain(self) -> None:
        """Called after the worker exits, while its event loop is still alive."""
        # Let call_soon_threadsafe submissions become tasks before inspection.
        await asyncio.sleep(0)
        tasks = list(self._generation_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def generate(self, request: dict[str, Any]) -> Any:
        self._check_cancelled()
        pending = asyncio.run_coroutine_threadsafe(self._agenerate(request), self.loop)
        with self._lock:
            self._pending = pending
            cancelled = self.cancelled.is_set()
        if cancelled:
            pending.cancel()
        try:
            result = pending.result()
            self._check_cancelled()
            return result
        finally:
            with self._lock:
                self._pending = None

    @staticmethod
    def _validate_decision(decision: Decision) -> None:
        if not isinstance(decision.completion_id, str) or not decision.completion_id:
            raise ValueError("A decision requires the real SDK completion ID")
        if decision.finish_reason == "abort":
            raise RuntimeError("Aborted inference is not a completed game decision")
        decision.evidence = copy_evidence(decision.evidence)

    def _add_reward(self, reward: RewardValue) -> None:
        if not math.isfinite(reward.value) or any(
            not math.isfinite(value) for value in reward.components.values()
        ):
            raise ValueError("Reward and reward components must be finite")
        self.record.total_reward += reward.value
        for key, value in reward.components.items():
            self.record.components[key] = self.record.components.get(key, 0.0) + value

    def _play(self, session: Session) -> None:
        record = self.record
        self._check_cancelled()
        observation = session.game.reset()
        self._check_cancelled()
        record.initial = observation.audit()
        record.final = record.initial
        if observation.terminated or observation.truncated:
            record.terminated = observation.terminated
            record.truncated = observation.truncated
            record.end_kind = (
                "environment_terminated"
                if observation.terminated
                else "environment_truncated"
            )
            record.reason = str(
                observation.info.get("terminal_reason")
                or ("terminated" if observation.terminated else "truncated")
            )
        while (
            record.end_kind == "running" and len(record.transitions) < session.max_steps
        ):
            self._check_cancelled()
            try:
                context = session.harness.prepare(observation)
                decision = session.policy.decide(
                    observation, context, self.generate, len(record.decisions)
                )
            except EpisodeStop as stop:
                record.reason, record.end_kind = stop.reason, stop.end_kind
                record.truncated = True
                break
            self._check_cancelled()
            self._validate_decision(decision)
            if any(
                item.completion_id == decision.completion_id
                for item in record.decisions
            ):
                raise ValueError(
                    "Each game decision must use a distinct SDK completion"
                )
            decision.index = len(record.decisions)
            decision.start_step = len(record.transitions)
            decision.end_step = decision.start_step
            record.decisions.append(decision)
            try:
                action = session.harness.start(decision, context)
            except EpisodeStop as stop:
                # The invalid SDK completion is still part of the trainable game.
                record.reason, record.end_kind = stop.reason, stop.end_kind
                record.truncated = True
                break
            while True:
                self._check_cancelled()
                session.harness.before_step(action)
                step = session.game.step(action)
                self._check_cancelled()
                at_budget = len(record.transitions) + 1 >= session.max_steps
                budget_truncated = at_budget and not (step.terminated or step.truncated)
                terminated, truncated = (
                    bool(step.terminated),
                    bool(step.truncated) or budget_truncated,
                )
                terminal = terminated or truncated
                transition = Transition(
                    index=len(record.transitions),
                    decision_index=decision.index,
                    action=action,
                    previous=observation,
                    current=step.observation,
                    base_reward=step.reward,
                    terminated=terminated,
                    truncated=truncated,
                )
                reward = session.reward.step(transition)
                # Plugins may annotate evidence, but cannot change lifecycle flags.
                transition.terminated, transition.truncated = terminated, truncated
                self._add_reward(reward)
                transition.reward, transition.components = (
                    reward.value,
                    reward.components,
                )
                decision.total_reward += reward.value
                continuation = session.harness.after_step(transition)
                transition.terminated, transition.truncated = terminated, truncated
                transition.status = "terminal" if terminal else continuation.status
                observation = step.observation
                # Plugins have consumed pixels/state; retain only audit metadata.
                transition.previous = transition.previous.audit()
                transition.current = transition.current.audit()
                transition.evidence = copy_evidence(transition.evidence)
                record.transitions.append(transition)
                record.final = transition.current
                decision.end_step = len(record.transitions)
                if terminal:
                    record.terminated = transition.terminated
                    record.truncated = transition.truncated
                    record.end_kind = (
                        "budget"
                        if budget_truncated
                        else "environment_terminated"
                        if terminated
                        else "environment_truncated"
                    )
                    record.reason = (
                        "budget_exhausted"
                        if budget_truncated
                        else str(
                            observation.info.get("terminal_reason")
                            or ("terminated" if terminated else "truncated")
                        )
                    )
                    break
                if continuation.status != "active":
                    break
                if continuation.action is None:
                    raise ValueError("An active option must supply its next action")
                action = continuation.action
            if record.end_kind != "running":
                break
        if record.end_kind == "running":
            record.reason = "budget_exhausted"
            record.end_kind, record.truncated = "budget", True
        adjustment = session.reward.finish(record.reason, observation)
        self._add_reward(adjustment)
        record.stop_reward, record.stop_components = (
            adjustment.value,
            adjustment.components,
        )
        if record.decisions:
            record.decisions[-1].total_reward += adjustment.value
        record.status = "complete"

    def run(self, data: dict[str, Any], **factory_kwargs: Any) -> EpisodeResult:
        started = time.monotonic()
        session = None
        try:
            session = self.factory(data, self.record.context, **factory_kwargs)
            with self._lock:
                self.session = session
            try:
                self._play(session)
            except BaseException as error:
                try:
                    session.game.close()
                except Exception as close_error:
                    error.add_note(f"Game cleanup also failed: {close_error}")
                raise
            else:
                session.game.close()
            self.record.elapsed_seconds = time.monotonic() - started
            self._check_cancelled()
            summary = session.summarize(self.record)
            self._check_cancelled()
            session.persist(summary)
            self._check_cancelled()
            return EpisodeResult(self.record, summary)
        except BaseException as error:
            cancelled = self.cancelled.is_set() or isinstance(
                error, asyncio.CancelledError
            )
            self.record.status = "cancelled" if cancelled else "error"
            self.record.reason = "cancelled" if cancelled else "technical_error"
            self.record.end_kind = self.record.reason
            self.record.elapsed_seconds = time.monotonic() - started
            # Never turn a technical failure/cancellation into a game penalty.
            if session is not None:
                try:
                    summary = (
                        session.error_summary(self.record, error)
                        if session.error_summary is not None
                        else {
                            "attempt_id": self.record.context.attempt_id,
                            "status": self.record.status,
                            "terminal_reason": self.record.reason,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "trainable": False,
                        }
                    )
                    session.persist(summary)
                except Exception as artifact_error:
                    error.add_note(f"Failed to persist episode error: {artifact_error}")
            raise
