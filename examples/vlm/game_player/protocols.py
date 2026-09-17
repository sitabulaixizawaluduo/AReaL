# SPDX-License-Identifier: Apache-2.0

"""Worker-local game plugins; observations are transient, evidence is JSON data."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


def copy_evidence(value: Any) -> Any:
    """Copy strict JSON evidence without retaining game state or image tensors."""
    return json.loads(json.dumps(value, allow_nan=False))


@dataclass
class Observation:
    value: Any
    info: dict[str, Any]
    state: Any = None
    # Used by reset() for terminal initial states. step() uses GameStep flags.
    terminated: bool = False
    truncated: bool = False

    def audit(self) -> "Observation":
        """Retain scalar/event metadata, never pixels or simulator state."""
        return Observation(
            None,
            copy_evidence(self.info),
            terminated=self.terminated,
            truncated=self.truncated,
        )


@dataclass
class GameStep:
    """Flags are authoritative for a step, independently of reset-only flags."""

    observation: Observation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False


@dataclass
class RewardValue:
    value: float = 0.0
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class Decision:
    completion_id: str
    choice: Any
    text: str
    finish_reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    index: int = 0
    start_step: int = 0
    end_step: int = 0
    total_reward: float = 0.0


@dataclass
class Transition:
    index: int
    decision_index: int
    action: Any
    previous: Observation
    current: Observation
    base_reward: float
    terminated: bool
    truncated: bool
    reward: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    status: str = "complete"


@dataclass
class ActionResult:
    action: Any = None
    status: str = "complete"


class EpisodeStop(Exception):
    """An intentional game-policy end, never an infrastructure failure.

    Raise before generation for a budget stop, or from Harness.start after a
    genuine decision has been recorded for an invalid model action.
    """

    def __init__(self, reason: str, end_kind: str = "policy_stop"):
        super().__init__(reason)
        self.reason, self.end_kind = reason, end_kind


class GameSession(Protocol):
    """Blocking operations run off the event loop and must be interruptible.

    ``cancel`` is thread-safe and nonblocking; reset/step must either respond to
    it or have a bounded native timeout. ``close`` is idempotent and joins/releases
    game workers. Python cannot safely kill a blocked arbitrary worker thread.
    """

    def reset(self) -> Observation: ...
    def step(self, action: Any) -> GameStep: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...


class Harness(Protocol):
    def prepare(self, observation: Observation) -> Any: ...
    def start(self, decision: Decision, context: Any) -> Any: ...
    def before_step(self, action: Any) -> None: ...
    def after_step(self, transition: Transition) -> ActionResult: ...


class Policy(Protocol):
    def decide(
        self,
        observation: Observation,
        context: Any,
        generate: Callable[[dict[str, Any]], Any],
        decision_index: int,
    ) -> Decision: ...


class Reward(Protocol):
    def step(self, transition: Transition) -> RewardValue: ...
    def finish(self, reason: str, observation: Observation) -> RewardValue: ...


@dataclass
class EpisodeContext:
    training: bool = True
    is_eval: bool = False
    task_id: Any = None
    sample_idx: int | None = None
    group_size: int = 1
    model_version: int | None = None
    session_namespace: str = ""
    proxy_session_id: str | None = None
    attempt_id: str = ""


@dataclass
class EpisodeRecord:
    context: EpisodeContext
    initial: Observation | None = None
    final: Observation | None = None
    decisions: list[Decision] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    reason: str = "running"
    status: str = "running"
    elapsed_seconds: float = 0.0
    total_reward: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    stop_reward: float = 0.0
    stop_components: dict[str, float] = field(default_factory=dict)
    end_kind: str = "running"
    terminated: bool = False
    truncated: bool = False


@dataclass
class Session:
    game: GameSession
    harness: Harness
    policy: Policy
    reward: Reward
    max_steps: int
    summarize: Callable[[EpisodeRecord], dict[str, Any]]
    persist: Callable[[dict[str, Any]], None]
    error_summary: Callable[[EpisodeRecord, BaseException], dict[str, Any]] | None = (
        None
    )

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


@dataclass
class EpisodeResult:
    record: EpisodeRecord
    summary: dict[str, Any]


class SessionFactory(Protocol):
    def __call__(
        self, data: dict[str, Any], context: EpisodeContext, **kwargs: Any
    ) -> Session: ...
