# pyright: reportMissingImports=false

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from areal.engine.awex_sglang_plugin import AwexSchedulerPlugin


class _StopLoop(Exception):
    pass


class _Batch:
    def __init__(self) -> None:
        self.forward_mode = SimpleNamespace(name="DECODE")


class _BaseScheduler:
    def __init__(self) -> None:
        self.gpu_id = 0
        self.forward_ct_decode = 1
        self.current_scheduler_metrics_enabled = True
        self.server_args = SimpleNamespace(decode_log_interval=1)
        self._engine_paused = False
        self.last_batch = None
        self.is_generation = False
        self.result_queue = deque()

        self._loop_step = 0
        self.process_batch_result_calls = 0
        self.process_awex_queue_calls = 0

    def recv_requests(self):
        return []

    def process_input_requests(self, recv_reqs):
        del recv_reqs

    def get_next_batch_to_run(self):
        if self._loop_step == 0:
            self._loop_step += 1
            return _Batch()
        return None

    def run_batch(self, batch):
        del batch
        return SimpleNamespace(can_run_cuda_graph=True, num_accepted_tokens=7)

    def process_batch_result(self, batch, result):
        del batch, result
        self.process_batch_result_calls += 1

    def self_check_during_idle(self):
        raise _StopLoop

    def event_loop_overlap(self):
        raise _StopLoop

    def event_loop_normal(self):
        raise _StopLoop


class _ReportScheduler(_BaseScheduler):
    def __init__(self, *, with_every_iter: bool = True) -> None:
        super().__init__()
        self.report_decode_stats_calls: list[tuple[bool, object]] = []
        self.report_decode_stats_every_iteration_calls: list[tuple[object, int]] = []

    def report_decode_stats(self, can_run_cuda_graph, running_batch=None):
        self.report_decode_stats_calls.append((can_run_cuda_graph, running_batch))
        return "report-main"

    def report_decode_stats_every_iteration(self, batch, num_accepted_tokens=0):
        self.report_decode_stats_every_iteration_calls.append(
            (batch, num_accepted_tokens)
        )
        return "report-every"


class _ReportOnlyMainScheduler(_BaseScheduler):
    def __init__(self) -> None:
        super().__init__()
        self.report_decode_stats_calls: list[tuple[bool, object]] = []

    def report_decode_stats(self, can_run_cuda_graph, running_batch=None):
        self.report_decode_stats_calls.append((can_run_cuda_graph, running_batch))
        return "report-main"


class _LegacyLogScheduler(_BaseScheduler):
    def __init__(self) -> None:
        super().__init__()
        self.log_decode_stats_calls: list[tuple[bool, object]] = []
        self.log_decode_stats_every_iteration_calls: list[tuple[object, int]] = []

    def log_decode_stats(self, can_run_cuda_graph, running_batch=None):
        self.log_decode_stats_calls.append((can_run_cuda_graph, running_batch))
        return "log-main"

    def log_decode_stats_every_iteration(self, batch, num_accepted_tokens=0):
        self.log_decode_stats_every_iteration_calls.append((batch, num_accepted_tokens))
        return "log-every"


class _NoStatsScheduler(_BaseScheduler):
    pass


class _PausedDrainScheduler(_NoStatsScheduler):
    def __init__(self) -> None:
        super().__init__()
        self._engine_paused = True
        self.last_batch = object()
        self._seeded_paused_result = False

    def recv_requests(self):
        if not self._seeded_paused_result and hasattr(self, "result_queue"):
            self.result_queue.append(
                (_Batch(), SimpleNamespace(can_run_cuda_graph=True))
            )
            self._seeded_paused_result = True
        return []


def test_patch_event_loop_report_api_wraps_and_tracks_decode_hooks():
    """Report-style schedulers must be wrapped with tracking and preserve returns."""
    # Arrange
    scheduler = _ReportScheduler()
    plugin = AwexSchedulerPlugin(scheduler)

    # Act
    plugin._patch_event_loop()
    batch = _Batch()
    main_return = scheduler.report_decode_stats(True, running_batch=batch)
    every_return = scheduler.report_decode_stats_every_iteration(
        batch, num_accepted_tokens=9
    )

    # Assert
    assert main_return == "report-main"
    assert every_return == "report-every"
    assert getattr(scheduler, "_areal_awex_last_decode_stats_ct", None) == 1
    assert getattr(scheduler, "_areal_awex_last_decode_stats_every_iter_ct", None) == 1
    assert scheduler.report_decode_stats_calls == [(True, batch)]
    assert scheduler.report_decode_stats_every_iteration_calls == [(batch, 9)]


def test_patch_event_loop_legacy_log_api_wraps_and_tracks_decode_hooks():
    """Legacy log-style schedulers must still be wrapped with tracking and returns."""
    # Arrange
    scheduler = _LegacyLogScheduler()
    plugin = AwexSchedulerPlugin(scheduler)

    # Act
    plugin._patch_event_loop()
    batch = _Batch()
    main_return = scheduler.log_decode_stats(True, running_batch=batch)
    every_return = scheduler.log_decode_stats_every_iteration(
        batch, num_accepted_tokens=11
    )

    # Assert
    assert main_return == "log-main"
    assert every_return == "log-every"
    assert getattr(scheduler, "_areal_awex_last_decode_stats_ct", None) == 1
    assert getattr(scheduler, "_areal_awex_last_decode_stats_every_iter_ct", None) == 1
    assert scheduler.log_decode_stats_calls == [(True, batch)]
    assert scheduler.log_decode_stats_every_iteration_calls == [(batch, 11)]


def test_patch_event_loop_missing_every_iteration_hook_keeps_binding_and_restoration_safe():
    """Missing per-iteration hook must not crash patching or decode metric restoration."""
    # Arrange
    scheduler = _ReportOnlyMainScheduler()
    plugin = AwexSchedulerPlugin(scheduler)

    # Act
    plugin._patch_event_loop()

    try:
        scheduler.event_loop_normal()
    except _StopLoop:
        pass

    # Assert
    assert scheduler.process_batch_result_calls == 1
    assert len(scheduler.report_decode_stats_calls) == 1
    can_run_cuda_graph, running_batch = scheduler.report_decode_stats_calls[0]
    assert can_run_cuda_graph is True
    assert isinstance(running_batch, _Batch)
    assert getattr(scheduler, "_areal_awex_last_decode_stats_ct", None) == 1
    assert not hasattr(scheduler, "_areal_awex_last_decode_stats_every_iter_ct")


def test_patch_event_loop_missing_all_stats_hooks_does_not_crash_binding_or_loop_patch():
    """Missing all decode-stats hooks must not break event-loop patching."""
    # Arrange
    scheduler = _NoStatsScheduler()
    plugin = AwexSchedulerPlugin(scheduler)

    # Act
    plugin._patch_event_loop()

    try:
        scheduler.event_loop_normal()
    except _StopLoop:
        pass

    # Assert
    assert scheduler.process_batch_result_calls == 1


def test_patch_event_loop_paused_branch_drains_result_queue_before_awex_queue_processing():
    """Paused overlap branch must drain local overlap state before processing AWEX queue."""
    # Arrange
    scheduler = _PausedDrainScheduler()
    plugin = AwexSchedulerPlugin(scheduler)

    def _one_shot_process_awex_queue() -> None:
        scheduler.process_awex_queue_calls += 1
        raise _StopLoop

    plugin.process_awex_queue = _one_shot_process_awex_queue

    # Act
    plugin._patch_event_loop()
    try:
        scheduler.event_loop_overlap()
    except _StopLoop:
        pass

    # Assert
    assert scheduler.process_batch_result_calls == 1
    assert scheduler.process_awex_queue_calls == 1
    assert len(scheduler.result_queue) == 0
    assert scheduler.last_batch is None
