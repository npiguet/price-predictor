"""Unit tests for CollectorSupervisor's worker-log wiring (F4).

Coverage and variant workers are always effect-record collectors (records-only
mode, no ``output.file``), so every one of them can trip one of this branch's
five latched effect-record failure reporters -- unlike sealed match-outcomes,
there is no "plain" mode to preserve here: a real log destination is
unconditional.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


class TestPlayRound:
    """The round supervisor: what it hands the workers and when it stops.

    ``play_round`` used to take the weights and the deck budget and log both
    without using either, so a round never ended and the decks were never the
    coverage decks the caller had computed.
    """

    def _supervisor(self, tmp_path):
        return CollectorSupervisor(worker_count=2, effect_records=tmp_path)

    def _decks_file(self, tmp_path) -> Path:
        """A decks file with one (trivial) entry -- valid input for
        ``play_round``, which now refuses an empty or missing one."""
        path = tmp_path / "decks.txt"
        path.write_text("coverage;COVERAGE;Plains\n", encoding="utf-8")
        return path

    def test_the_round_stops_at_its_match_budget(self, tmp_path):
        supervisor = self._supervisor(tmp_path)
        with patch(
            "effects.infrastructure.collector_connector.ForgeWorkerPool"
        ) as pool:
            supervisor.play_round(self._decks_file(tmp_path), matches=500)

        stop = pool.call_args.kwargs["should_stop"]
        assert stop(499) is False
        assert stop(500) is True
        assert stop(501) is True

    def test_progress_is_counted_from_a_file_not_a_directory(self, tmp_path):
        supervisor = self._supervisor(tmp_path)
        with patch(
            "effects.infrastructure.collector_connector.ForgeWorkerPool"
        ) as pool:
            supervisor.play_round(self._decks_file(tmp_path), matches=10)

        output_path = pool.call_args.kwargs["output_path"]
        assert output_path.is_file() or not output_path.exists()
        assert output_path != Path(tmp_path)

    def test_the_decks_file_reaches_the_worker_on_both_sides(self, tmp_path):
        supervisor = self._supervisor(tmp_path)
        decks = tmp_path / "decks.txt"
        with patch.object(supervisor, "_connector") as connector:
            supervisor._decks_file = decks
            supervisor.start_worker(0)

        kwargs = connector.start.call_args.kwargs
        assert kwargs["side_a_decks_path"] == decks
        assert kwargs["side_b_decks_path"] == decks
        assert kwargs["decks_only"] is True

    def test_the_progress_file_reaches_the_worker_without_becoming_output_file(
        self, tmp_path,
    ):
        """Round-1 review, CRITICAL finding's prescribed fix: ``start_worker``
        must pass the progress file through as its own channel
        (``progress_file``), never by smuggling it in as ``output_file``.

        Passing it as ``output_file`` would flip ``recordsOnly()`` to false
        and construct a real ``CardsPlayedWriter`` at
        ``<effect_records>/cards-played.txt`` every match
        (``MatchWorkerMain.java`` ~237-238) -- not a breach of FR-052/FR-059's
        letter (it is not ``output/sealed/cards-played.txt``), but it
        contradicts the records-only invariant the surrounding Java comment
        states, and it is not what fixes the round-never-ends bug. Both
        halves are asserted together because the fix is exactly "add a
        second, independent channel" -- either half regressing independently
        (progress never reaching the worker, or output_file stopping being
        None) reintroduces a defect this round's review found.
        """
        supervisor = self._supervisor(tmp_path)
        with patch.object(supervisor, "_connector") as connector:
            supervisor._decks_file = tmp_path / "decks.txt"
            supervisor.start_worker(0)

        call = connector.start.call_args
        assert call.args[0] is None
        assert call.kwargs["progress_file"] == supervisor.progress_path

    def test_play_round_refuses_a_missing_decks_file(self, tmp_path):
        supervisor = self._supervisor(tmp_path)

        with pytest.raises(ValueError, match="does not exist"):
            supervisor.play_round(tmp_path / "missing.txt", matches=10)

    def test_play_round_refuses_an_empty_decks_file(self, tmp_path):
        supervisor = self._supervisor(tmp_path)
        empty = tmp_path / "decks.txt"
        empty.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="is empty"):
            supervisor.play_round(empty, matches=10)


class TestTheRoundShutsTheRealPoolDown:
    """Final review, CRITICAL 1: ``stop()`` called ``self._pool.shutdown()``,
    a method ``ForgeWorkerPool`` has never had -- it defines
    ``request_shutdown()``. ``stop()`` sits in the ``finally`` of both
    ``collect_coverage.run`` and ``collect_variants.run``, so every round
    ended in ``AttributeError``: the residue report (FR-051) and the return
    code were never reached and an overnight run finished in a traceback
    having reported nothing.

    Every test written before this one mocked either ``CollectorSupervisor``
    or ``ForgeWorkerPool``, and a ``MagicMock`` answers to any method name at
    all -- which is exactly how a wrong one survived to a real run. This one
    constructs the real pool and stubs only ``run()``, the blocking loop that
    would otherwise spawn JVMs.
    """

    def _played_round(self, tmp_path) -> CollectorSupervisor:
        from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool

        supervisor = CollectorSupervisor(worker_count=1, effect_records=tmp_path)
        decks = tmp_path / "decks.txt"
        decks.write_text("coverage;COVERAGE;Plains\n", encoding="utf-8")
        with patch.object(ForgeWorkerPool, "run", autospec=True):
            supervisor.play_round(decks, matches=1)
        return supervisor

    def test_stop_asks_the_real_pool_to_stop(self, tmp_path):
        from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool

        supervisor = self._played_round(tmp_path)

        with patch.object(
            ForgeWorkerPool, "request_shutdown", autospec=True,
        ) as requested:
            supervisor.stop()

        assert requested.call_count == 1, (
            "the pool was never asked to shut down; stop() is calling a "
            "method ForgeWorkerPool does not define"
        )

    def test_a_second_round_can_follow_a_stopped_one(self, tmp_path):
        """The coverage loop plays round after round through one supervisor."""
        from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool

        supervisor = self._played_round(tmp_path)
        supervisor.stop()

        with patch.object(ForgeWorkerPool, "run", autospec=True):
            supervisor.play_round(tmp_path / "decks.txt", matches=1)
        supervisor.stop()


class TestInterruptIsVisibleToTheCaller:
    """Final review, IMPORTANT 3: ``ForgeWorkerPool.run()`` installs its own
    SIGINT handler, sets its own event and never raises, so a round the
    operator interrupted returns exactly like one that hit its budget. The
    pool already records the difference in ``interrupted``; nothing in
    ``effects`` read it, so Ctrl-C degraded to "skip this round, start
    another".

    Latched rather than delegated live, because ``stop()`` drops the pool and
    ``collect_variants.run`` asks after its ``finally`` has run.
    """

    def _round(self, tmp_path, *, interrupted: bool) -> CollectorSupervisor:
        from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool

        supervisor = CollectorSupervisor(worker_count=1, effect_records=tmp_path)
        decks = tmp_path / "decks.txt"
        decks.write_text("coverage;COVERAGE;Plains\n", encoding="utf-8")

        def _fake_run(pool):
            pool._interrupted = interrupted

        with patch.object(ForgeWorkerPool, "run", _fake_run):
            supervisor.play_round(decks, matches=1)
        return supervisor

    def test_an_uninterrupted_round_reports_no_interrupt(self, tmp_path):
        assert self._round(tmp_path, interrupted=False).interrupted is False

    def test_an_interrupted_round_reports_it(self, tmp_path):
        assert self._round(tmp_path, interrupted=True).interrupted is True

    def test_the_interrupt_survives_stop(self, tmp_path):
        """``collect_variants.run`` asks after its ``finally`` has run."""
        supervisor = self._round(tmp_path, interrupted=True)
        supervisor.stop()
        assert supervisor.interrupted is True

    def test_nothing_played_yet_is_not_an_interrupt(self, tmp_path):
        supervisor = CollectorSupervisor(worker_count=1, effect_records=tmp_path)
        assert supervisor.interrupted is False
