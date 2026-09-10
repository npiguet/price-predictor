"""Unit tests for CollectorSupervisor's worker-log wiring (F4).

Coverage and variant workers are always effect-record collectors (records-only
mode, no ``output.file``), so every one of them can trip one of this branch's
five latched effect-record failure reporters -- unlike sealed match-outcomes,
there is no "plain" mode to preserve here: a real log destination is
unconditional.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from effects.infrastructure.collector_connector import CollectorSupervisor
from tests.unit.sealed.conftest import FakeProcess


def _rigged(supervisor: CollectorSupervisor) -> MagicMock:
    """Replace the supervisor's connector with a MagicMock that hands back a
    clean-exit FakeProcess, and install it."""
    fake_connector = MagicMock()
    fake_connector.start.return_value = FakeProcess(pid=1000, returncode=0)
    supervisor._connector = fake_connector
    return fake_connector


class TestCollectorSupervisorWorkerLogs:
    def test_start_worker_passes_a_real_open_log_file(self, tmp_path):
        supervisor = CollectorSupervisor(worker_count=1, effect_records=tmp_path)
        fake_connector = _rigged(supervisor)

        supervisor.start_worker(0)

        log_file = fake_connector.start.call_args.kwargs["log_file"]
        assert log_file is not None
        assert log_file.closed is False
        supervisor.stop()

    def test_the_log_file_lives_under_the_records_directory_named_by_run_and_worker(
        self, tmp_path,
    ):
        records = tmp_path / "records"
        supervisor = CollectorSupervisor(
            worker_count=1, effect_records=records, run_id="run-xyz",
        )
        _rigged(supervisor)

        supervisor.start_worker(2)

        assert (records / "run-xyz.2.log").exists()
        supervisor.stop()

    def test_a_respawned_slot_reuses_the_same_log_file(self, tmp_path):
        """Two lifetimes of one slot -- ForgeWorkerPool restarts many times
        per run -- must share one file so a reporter line from either
        lifetime survives to the end of the run."""
        supervisor = CollectorSupervisor(worker_count=1, effect_records=tmp_path)
        fake_connector = _rigged(supervisor)

        supervisor.start_worker(0)
        supervisor.start_worker(0)  # same slot, a new JVM

        first, second = fake_connector.start.call_args_list
        assert first.kwargs["log_file"] is second.kwargs["log_file"]
        supervisor.stop()

    def test_different_worker_slots_get_different_log_files(self, tmp_path):
        supervisor = CollectorSupervisor(worker_count=2, effect_records=tmp_path)
        fake_connector = _rigged(supervisor)

        supervisor.start_worker(0)
        supervisor.start_worker(1)

        first, second = fake_connector.start.call_args_list
        assert first.kwargs["log_file"] is not second.kwargs["log_file"]
        supervisor.stop()

    def test_stop_closes_the_worker_logs(self, tmp_path):
        supervisor = CollectorSupervisor(worker_count=1, effect_records=tmp_path)
        fake_connector = _rigged(supervisor)
        supervisor.start_worker(0)
        log_file = fake_connector.start.call_args.kwargs["log_file"]

        supervisor.stop()

        assert log_file.closed

    def test_records_only_mode_is_unaffected_by_the_log_file(self, tmp_path):
        """The pre-existing records-only guard (output_file=None) must not be
        disturbed by adding a log destination."""
        supervisor = CollectorSupervisor(worker_count=1, effect_records=tmp_path)
        fake_connector = _rigged(supervisor)

        supervisor.start_worker(0)

        assert fake_connector.start.call_args.args[0] is None
        supervisor.stop()
