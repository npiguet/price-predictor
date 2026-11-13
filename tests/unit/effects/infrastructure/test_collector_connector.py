"""Unit tests for CollectorSupervisor's worker-log wiring (F4).

Coverage and variant workers are always effect-record collectors (records-only
mode, no ``output.file``), so every one of them can trip one of this branch's
five latched effect-record failure reporters -- unlike sealed match-outcomes,
there is no "plain" mode to preserve here: a real log destination is
unconditional.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from effects.domain.collection_caps import CollectionCaps
from effects.infrastructure.collector_connector import CollectorSupervisor
from tests.unit.sealed.conftest import FakeProcess

#: The one deck every round here plays: a decks-only round refuses none.
_DECKS = [["Plains"]]


def _supervisor(tmp_path, *, workers: int = 1, run_id: str = "run") -> CollectorSupervisor:
    return CollectorSupervisor(
        worker_count=workers, effect_records=tmp_path, run_id=run_id,
        caps=CollectionCaps(),
    )


def _rigged(supervisor: CollectorSupervisor) -> MagicMock:
    """Replace the supervisor's connector with a MagicMock that hands back a
    clean-exit FakeProcess, and install it."""
    fake_connector = MagicMock()
    fake_connector.start.return_value = FakeProcess(pid=1000, returncode=0)
    supervisor._connector = fake_connector
    return fake_connector


class TestCollectorSupervisorWorkerLogs:
    def test_start_worker_passes_a_real_open_log_file(self, tmp_path):
        supervisor = _supervisor(tmp_path)
        fake_connector = _rigged(supervisor)

        supervisor.start_worker(0, tmp_path / "decks.txt")

        log_file = fake_connector.start.call_args.kwargs["log_file"]
        assert log_file is not None
        assert log_file.closed is False
        supervisor.stop()

    def test_the_log_file_lives_under_the_run_s_logs_named_by_run_and_worker(
        self, tmp_path,
    ):
        """V4 #20: the artifact layout puts every worker log under the run
        directory's ``logs/``, never beside the shards."""
        records = tmp_path / "records"
        supervisor = _supervisor(records, run_id="run-xyz")
        _rigged(supervisor)

        supervisor.start_worker(2, tmp_path / "decks.txt")

        assert (records / "logs" / "run-xyz.2.log").exists()
        assert not list(records.glob("*.log"))
        supervisor.stop()

    def test_a_respawned_slot_reuses_the_same_log_file(self, tmp_path):
        """Two lifetimes of one slot -- ForgeWorkerPool restarts many times
        per run -- must share one file so a reporter line from either
        lifetime survives to the end of the run."""
        supervisor = _supervisor(tmp_path)
        fake_connector = _rigged(supervisor)

        supervisor.start_worker(0, tmp_path / "decks.txt")
        supervisor.start_worker(0, tmp_path / "decks.txt")  # same slot, a new JVM

        first, second = fake_connector.start.call_args_list
        assert first.kwargs["log_file"] is second.kwargs["log_file"]
        supervisor.stop()

    def test_different_worker_slots_get_different_log_files(self, tmp_path):
        supervisor = _supervisor(tmp_path, workers=2)
        fake_connector = _rigged(supervisor)

        supervisor.start_worker(0, tmp_path / "decks.txt")
        supervisor.start_worker(1, tmp_path / "decks.txt")

        first, second = fake_connector.start.call_args_list
        assert first.kwargs["log_file"] is not second.kwargs["log_file"]
        supervisor.stop()

    def test_stop_closes_the_worker_logs(self, tmp_path):
        supervisor = _supervisor(tmp_path)
        fake_connector = _rigged(supervisor)
        supervisor.start_worker(0, tmp_path / "decks.txt")
        log_file = fake_connector.start.call_args.kwargs["log_file"]

        supervisor.stop()

        assert log_file.closed



class TestPlayRound:
    """The round supervisor: what it hands the workers and when it stops.

    ``play_round`` used to take the weights and the deck budget and log both
    without using either, so a round never ended and the decks were never the
    coverage decks the caller had computed.
    """

    def _supervisor(self, tmp_path):
        return _supervisor(tmp_path, workers=2)

    def test_the_round_stops_at_its_match_budget(self, tmp_path):
        supervisor = self._supervisor(tmp_path)
        with patch(
            "effects.infrastructure.collector_connector.ForgeWorkerPool"
        ) as pool:
            supervisor.play_round(_DECKS, tmp_path / "decks.txt", label="coverage", matches=500)

        stop = pool.call_args.kwargs["should_stop"]
        assert stop(499) is False
        assert stop(500) is True
        assert stop(501) is True

    def test_each_round_counts_its_own_progress_file_under_logs(self, tmp_path):
        """V4 #8: a worker the kill could not end appends to the progress
        file it was started with; a second round sharing that file counted
        the survivor's matches as its own. V4 #20: the file is under ``logs/``."""
        supervisor = self._supervisor(tmp_path)
        with patch(
            "effects.infrastructure.collector_connector.ForgeWorkerPool"
        ) as pool:
            supervisor.play_round(_DECKS, tmp_path / "decks.txt", label="coverage", matches=10)
            first = pool.call_args.kwargs["output_path"]
            first.write_text("a survivor's match\n", encoding="utf-8")
            supervisor.play_round(_DECKS, tmp_path / "decks.txt", label="coverage", matches=10)
            second = pool.call_args.kwargs["output_path"]

        assert first.parent == second.parent == tmp_path / "logs"
        assert first != second
        assert second.read_text(encoding="utf-8") == ""
        assert first.read_text(encoding="utf-8") == "a survivor's match\n"

    def test_the_decks_file_reaches_the_worker_on_both_sides(self, tmp_path):
        """Through the spawn the round hands its pool, the round's own file."""
        supervisor = self._supervisor(tmp_path)
        decks = tmp_path / "decks.txt"
        with patch(
            "effects.infrastructure.collector_connector.ForgeWorkerPool"
        ) as pool:
            supervisor.play_round(_DECKS, decks, label="coverage", matches=1)
        with patch.object(supervisor, "_connector") as connector:
            pool.call_args.kwargs["spawn_worker"](0)

        kwargs = connector.start.call_args.kwargs
        assert connector.start.call_args.args[0] is None, "records-only: no output file"
        assert kwargs["side_a_decks_path"] == decks
        assert kwargs["side_b_decks_path"] == decks
        assert kwargs["decks_only"] is True
        assert kwargs["run_id"] == "run"
        assert kwargs["worker_index"] == 0
        assert kwargs["best_of"] == 1
        assert kwargs["effect_records_dir"] == tmp_path
        assert kwargs["collection_caps"] == CollectionCaps().as_system_properties()
        assert kwargs["variant_scripts"] is None
        assert kwargs["progress_file"] == supervisor.progress_path

    def test_the_progress_file_reaches_the_worker_without_becoming_output_file(
        self, tmp_path,
    ):
        """Round-1 review, CRITICAL finding's prescribed fix: ``start_worker``
        must pass the progress file through as its own channel
        (``progress_file``), never by smuggling it in as ``output_file``.

        Passing it as ``output_file`` would flip ``recordsOnly()`` to false
        and construct a real ``SealedCorpus`` -- a ``CorpusWriter`` per file --
        appending to ``<effect_records>/cards-played.txt`` every match
        (``MatchWorkerMain.runForever``'s
        ``config.outputFile().map(SealedCorpus::beside)``) -- not a breach of FR-052/FR-059's
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
            supervisor.start_worker(0, tmp_path / "decks.txt")

        call = connector.start.call_args
        assert call.args[0] is None
        assert call.kwargs["progress_file"] == supervisor.progress_path

    def test_play_round_writes_the_decks_file_the_workers_read(self, tmp_path):
        supervisor = self._supervisor(tmp_path)
        decks = tmp_path / "decks.txt"
        with patch("effects.infrastructure.collector_connector.ForgeWorkerPool"):
            supervisor.play_round(
                [["Plains", "Plains"], ["Island"]], decks, label="coverage", matches=1,
            )

        assert decks.read_text(encoding="utf-8") == (
            "coverage;COVERAGE;Plains|Plains\ncoverage;COVERAGE;Island\n"
        )

    def test_play_round_refuses_a_round_with_no_deck(self, tmp_path):
        supervisor = self._supervisor(tmp_path)

        with patch(
            "effects.infrastructure.collector_connector.ForgeWorkerPool"
        ) as pool, pytest.raises(ValueError, match="no deck"):
            supervisor.play_round([], tmp_path / "decks.txt", label="coverage", matches=10)
        pool.assert_not_called()


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

        supervisor = _supervisor(tmp_path)
        with patch.object(ForgeWorkerPool, "run", autospec=True):
            supervisor.play_round(_DECKS, tmp_path / "decks.txt", label="coverage", matches=1)
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

        with patch.object(ForgeWorkerPool, "run", autospec=True) as run:
            supervisor.play_round(_DECKS, tmp_path / "decks.txt", label="coverage", matches=1)
        supervisor.stop()

        assert run.call_count == 1
        assert supervisor.rounds == 2


class TestTheRoundReportsItsWorkerLogs:
    """V4 #3 / P1 #3: worker failure lines went to the logs and nowhere else;
    nothing read them, so a run losing its matches reported success."""

    def _round_logging(self, tmp_path, lines_by_slot, *, workers=2, survivors=()):
        from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool

        supervisor = _supervisor(tmp_path, workers=workers)

        def _fake_run(pool):
            for slot, lines in lines_by_slot.items():
                path = supervisor._worker_logs.path_for(slot)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "ab") as handle:
                    handle.write("".join(f"{line}\n" for line in lines).encode("utf-8"))
            # A worker the pool killed that is still running at the round's end.
            pool._kill_attempted.extend(
                MagicMock(pid=pid, poll=MagicMock(return_value=None)) for pid in survivors
            )

        with patch.object(ForgeWorkerPool, "run", _fake_run):
            supervisor.play_round(_DECKS, tmp_path / "decks.txt", label="coverage", matches=1)
        return supervisor

    def test_a_round_s_failure_lines_are_reported_by_kind(self, tmp_path, caplog):
        supervisor = self._round_logging(tmp_path, {
            0: ["Worker: 10 matches generated", "Error generating match: boom"],
            1: ["Error generating match: deck cannot be materialized: 1 of 40"],
        })

        assert supervisor.worker_failures.as_dict() == {"match_error": 1, "unresolved_deck": 1}
        assert "Round 1 worker logs: 1 match_error, 1 unresolved_deck" in caplog.text

    def test_a_later_round_reports_only_its_own_lines(self, tmp_path, caplog):
        import logging

        from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool

        supervisor = self._round_logging(tmp_path, {0: ["Error generating match: a"]})
        with patch.object(ForgeWorkerPool, "run", autospec=True), caplog.at_level(logging.INFO):
            supervisor.play_round(_DECKS, tmp_path / "decks.txt", label="coverage", matches=1)

        assert "Round 2 worker logs: no failure lines" in caplog.text
        assert supervisor.worker_failures.as_dict() == {"match_error": 1}

    def test_a_worker_that_outlived_the_kill_is_reported(self, tmp_path, caplog):
        supervisor = self._round_logging(tmp_path, {}, survivors=[4242])

        assert supervisor.survivors == [4242]
        assert "outlived the kill (PID 4242)" in caplog.text

    def test_the_run_s_totals_reach_its_manifest(self, tmp_path):
        """V4 #3, #4: the run id, the resolved caps and the failures land in
        the running command's entry of ``run.json``."""
        import json

        from price_predictor.infrastructure.run_manifest import run_recorded

        def body():
            supervisor = self._round_logging(tmp_path, {
                1: ["Error generating match: a", "Error generating match: b"],
            })
            supervisor.stop()
            return 0

        run_recorded(tmp_path, body, command="python -m effects collect-coverage")

        entry = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))["invocations"][0]
        assert entry["run_id"] == "run"
        assert entry["caps"]["max_records_per_game"] == CollectionCaps().max_records_per_game
        assert entry["caps"]["aimed_at"] is None
        assert entry["rounds"] == 1
        assert entry["worker_failures"] == {"match_error": 2}
        assert entry["worker_failure_examples"] == {"match_error": "Error generating match: a"}
        assert entry["worker_survivors"] == []


class TestTheDecksFileIsTheRunsOwn:
    """V4 #20: the decks file the workers read sits under the run's
    ``decks/``, named by the run id, so two runs never share one."""

    def test_the_decks_path_is_under_decks_and_names_the_run(self, tmp_path):
        supervisor = _supervisor(tmp_path, run_id="run-123")

        assert supervisor.decks_path("coverage") == (
            tmp_path / "decks" / "coverage-decks.run-123.txt"
        )


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

        supervisor = _supervisor(tmp_path)

        def _fake_run(pool):
            pool._interrupted = interrupted

        with patch.object(ForgeWorkerPool, "run", _fake_run):
            supervisor.play_round(_DECKS, tmp_path / "decks.txt", label="coverage", matches=1)
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
        supervisor = _supervisor(tmp_path)
        assert supervisor.interrupted is False


class TestACtrlCBetweenRoundsIsNotLost:
    """python R9: the pool's SIGINT handler stays installed after its run, but
    the supervisor latched ``interrupted`` right after ``run()`` returned, so a
    Ctrl-C while the round's shards were counted reached a pool nobody asked."""

    def test_sigint_after_play_round_reads_as_interrupted(self, tmp_path):
        import signal

        from tests.unit.sealed.conftest import FakeProcess

        supervisor = _supervisor(tmp_path)
        connector = MagicMock()
        connector.start.return_value = FakeProcess(pid=1000, hang=True)
        supervisor._connector = connector
        saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with patch(
                "price_predictor.infrastructure.forge_jvm.kill_process_tree",
                side_effect=lambda proc: proc.kill(),
            ):
                # A budget of 0 matches: the real pool stops at its first poll.
                supervisor.play_round(_DECKS, tmp_path / "decks.txt", label="coverage", matches=0)
            assert supervisor.interrupted is False

            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)

            assert supervisor.interrupted is True
            supervisor.stop()
            assert supervisor.interrupted is True
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)


class TestTheSupervisorRecordsWhatItIsTold:
    """P1 #11: the application's facts about a run (the cards Forge cannot
    resolve) reach the running command's run.json entry through the supervisor."""

    def test_record_writes_into_the_running_command_s_entry(self, tmp_path):
        import json

        from price_predictor.infrastructure.run_manifest import run_recorded

        def body():
            _supervisor(tmp_path).record(unresolved_cards=3, unresolved_share=0.25)
            return 0

        run_recorded(tmp_path, body, command="python -m effects collect-coverage")

        entry = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))["invocations"][0]
        assert (entry["unresolved_cards"], entry["unresolved_share"]) == (3, 0.25)
