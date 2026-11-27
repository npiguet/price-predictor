"""A collector refuses a degraded engine.

A run on stock Forge collected ``degraded`` records and a partly patched
build collected with an empty channel, both exiting 0: no supervisor read the
hook report. Now ``CollectorSupervisor`` reads each worker's status line at
every status interval and at each round's end, stops the round by the stop
protocol and raises once the engine is not ``patched`` with every hook,
unless ``allow_degraded``; and records the first lifetime's engine in
``run.json``.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest

from effects.domain.collection_caps import CollectionCaps
from effects.infrastructure.collector_connector import CollectorRun, CollectorSupervisor, RoundSpec
from price_predictor.domain.worker_status import WorkerEngineRefused
from price_predictor.infrastructure.forge_worker_pool import ForgeWorkerPool
from price_predictor.infrastructure.run_manifest import RunCommand, run_recorded
from price_predictor.infrastructure.worker_stops import WorkerStopSettings
from tests.unit.worker_status_line import (
    FORGE_COMMIT,
    PINNED_PREFERENCES,
    hooks_document,
    status_line,
)

_DECKS = [["Plains"]]
_QUICK_STOP = WorkerStopSettings(timeout=1, recycle_seconds=3600)

DEGRADED = status_line("a", hooks=hooks_document("degraded"))
#: A partly patched build: ``patched``, one listener callback missing.
GAP = "EffectRecordListener.onReplacement missing"
PATCHED_WITH_GAP = status_line("a", hooks=hooks_document("patched", (GAP,)))


class _Round:
    """A supervisor whose round's pool writes ``lines`` to slot 0's log, then
    calls the status-interval scan the real pool calls, and notes whether
    that scan asked the pool to stop."""

    def __init__(self, tmp_path, *, allow_degraded: bool = False) -> None:
        self.supervisor = CollectorSupervisor(CollectorRun(
            worker_count=1, effect_records=tmp_path, run_id="run",
            caps=CollectionCaps(), stop=_QUICK_STOP, allow_degraded=allow_degraded,
        ))
        self.decks = tmp_path / "decks.txt"
        self.stopped_at_status: list[bool] = []

    def play(self, *lines: str) -> None:
        def fake_run(pool: ForgeWorkerPool) -> None:
            path = self.supervisor._worker_logs.path_for(0)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "ab") as handle:
                handle.write("".join(f"{line}\n" for line in lines).encode("utf-8"))
            pool._on_status()
            self.stopped_at_status.append(pool._shutdown_event.is_set())

        with patch.object(ForgeWorkerPool, "run", fake_run):
            self.supervisor.play_round(
                _DECKS, self.decks, RoundSpec(label="coverage", matches=1))


def _entry(tmp_path) -> dict:
    return json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))["invocations"][-1]


def _recorded(tmp_path, body) -> dict:
    """Run ``body`` under the collector's manifest; its entry."""
    def wrapped():
        body()
        return 0

    try:
        run_recorded(tmp_path, wrapped, RunCommand("python -m effects collect-coverage"))
    except WorkerEngineRefused:
        pass
    return _entry(tmp_path)


class TestADegradedEngineStopsTheRun:
    def test_a_degraded_report_is_refused_at_the_first_status_scan(self, tmp_path):
        round_ = _Round(tmp_path)

        with pytest.raises(WorkerEngineRefused) as refused:
            round_.play(DEGRADED)

        assert round_.stopped_at_status == [True], "the scan asked the pool to stop"
        message = str(refused.value)
        assert "mode degraded" in message
        assert "--allow-degraded" in message
        assert str(tmp_path) in message, "names where the records written so far stay"

    def test_a_patched_report_with_a_gap_is_refused_naming_the_gap(self, tmp_path):
        round_ = _Round(tmp_path)

        with pytest.raises(WorkerEngineRefused, match=GAP):
            round_.play(PATCHED_WITH_GAP)

        assert round_.stopped_at_status == [True]

    def test_a_line_read_only_at_the_round_s_end_is_refused_too(self, tmp_path):
        """A round shorter than one status interval: the end-of-round scan judges."""
        supervisor = CollectorSupervisor(CollectorRun(
            worker_count=1, effect_records=tmp_path, run_id="run",
            caps=CollectionCaps(), stop=_QUICK_STOP,
        ))

        def fake_run(pool):
            path = supervisor._worker_logs.path_for(0)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(DEGRADED + "\n", encoding="utf-8")

        with patch.object(ForgeWorkerPool, "run", fake_run), pytest.raises(WorkerEngineRefused):
            supervisor.play_round(
                _DECKS, tmp_path / "d.txt", RoundSpec(label="coverage", matches=1))

    def test_a_whole_engine_runs_on(self, tmp_path):
        round_ = _Round(tmp_path)
        round_.play(status_line("a"))
        assert round_.stopped_at_status == [False]

    def test_the_refusal_and_the_engine_reach_run_json(self, tmp_path):
        entry = _recorded(tmp_path, lambda: _Round(tmp_path).play(PATCHED_WITH_GAP))

        assert entry["worker_engine"] == {
            "mode": "patched", "gaps": [GAP], "pinned_preferences": PINNED_PREFERENCES,
            "forge_build_commit": FORGE_COMMIT, "forge_build_dirty": False,
            "connector_forge_commit": FORGE_COMMIT}

    def test_an_engine_changing_under_the_run_is_refused_even_when_allowed(self, tmp_path):
        """A JAR reinstalled under a running collection: the next lifetime differs."""
        round_ = _Round(tmp_path, allow_degraded=True)
        round_.play(status_line("a"))

        with pytest.raises(WorkerEngineRefused, match="lifetime b runs another engine"):
            round_.play(status_line("b", hooks=hooks_document("degraded")))


#: A Forge commit other than the one the run converted at.
OTHER_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"


def _record_conversion(run_dir, commit: str) -> None:
    """``run_dir``'s ``run.json`` records a collection against a conversion at ``commit``."""
    (run_dir / "run.json").write_text(json.dumps({"invocations": [{"conversion": {
        "run": "converted", "forge_commit": commit, "forge_dirty": False,
        "sidecar_digest": "d", "checkout_dirty": False}}]}), encoding="utf-8")


class TestTheForgeBuildTheWorkersLoad:
    """A stale ../forge jar reads patched with no gap;
    its build commit, against the connector's and the conversion's, tells it."""

    def test_a_worker_on_another_forge_build_is_refused_naming_both_commits(self, tmp_path):
        _record_conversion(tmp_path, FORGE_COMMIT)
        round_ = _Round(tmp_path)

        with pytest.raises(WorkerEngineRefused) as refused:
            round_.play(status_line("a", hooks=hooks_document(forge_build_commit=OTHER_COMMIT)))

        assert round_.stopped_at_status == [True]
        assert str(refused.value).startswith(
            f"the Forge workers' engine is not whole (worker lifetime a): Forge build "
            f"{OTHER_COMMIT} differs from connector built at Forge {FORGE_COMMIT}")

    def test_a_connector_rebuilt_at_another_commit_than_the_conversion_is_refused(
            self, tmp_path):
        _record_conversion(tmp_path, OTHER_COMMIT)
        round_ = _Round(tmp_path)

        with pytest.raises(WorkerEngineRefused, match=(
                f"connector built at Forge {FORGE_COMMIT} differs from conversion at "
                f"Forge {OTHER_COMMIT}")):
            round_.play(status_line("a"))

    def test_a_worker_whose_build_names_no_commit_is_refused(self, tmp_path):
        _record_conversion(tmp_path, FORGE_COMMIT)
        round_ = _Round(tmp_path)

        with pytest.raises(WorkerEngineRefused, match="Forge build names no commit"):
            round_.play(status_line("a", hooks={"mode": "patched", "gaps": []}))

    def test_a_later_lifetime_on_another_build_is_refused(self, tmp_path):
        """K2-7: ../forge reinstalled during the run; the next lifetime loads it."""
        _record_conversion(tmp_path, FORGE_COMMIT)
        round_ = _Round(tmp_path)
        round_.play(status_line("a"))

        with pytest.raises(WorkerEngineRefused, match="lifetime b runs another engine"):
            round_.play(status_line("b", hooks=hooks_document(forge_build_commit=OTHER_COMMIT)))

    def test_the_build_reaches_run_json(self, tmp_path):
        _record_conversion(tmp_path, FORGE_COMMIT)
        stale = status_line("a", hooks=hooks_document(forge_build_commit=OTHER_COMMIT))

        entry = _recorded(tmp_path, lambda: _Round(tmp_path).play(stale))

        assert entry["worker_engine"]["forge_build_commit"] == OTHER_COMMIT


class TestWorkersThatPrintNoStatusLine:
    """K1-2: a connector built before the status line, or after its shape
    changed: the run used to end 0 with ``worker_engine: null``."""

    def test_a_round_whose_worker_named_its_shard_only_is_refused_at_its_end(self, tmp_path):
        round_ = _Round(tmp_path, allow_degraded=True)

        with pytest.raises(WorkerEngineRefused, match="no worker lifetime printed a status line"):
            round_.play("Effect records: x.jsonl [mode=degraded]",
                        "effect-record hooks: 50 of 51 missing, so these channels stay empty:")

        assert round_.stopped_at_status == [False], "the status line may still be on its way"

    def test_a_round_that_played_a_match_is_refused_at_the_status_scan(self, tmp_path):
        round_ = _Round(tmp_path)
        round_.supervisor.progress_path.parent.mkdir(parents=True, exist_ok=True)

        def fake_run(pool: ForgeWorkerPool) -> None:
            round_.supervisor.progress_path.write_text("match\n", encoding="utf-8")
            pool._on_status()
            round_.stopped_at_status.append(pool._shutdown_event.is_set())

        with patch.object(ForgeWorkerPool, "run", fake_run), pytest.raises(WorkerEngineRefused):
            round_.supervisor.play_round(
                _DECKS, round_.decks, RoundSpec(label="coverage", matches=1))

        assert round_.stopped_at_status == [True]


class TestAllowDegraded:
    def test_it_lets_both_run_and_warns_once_naming_the_gaps(self, tmp_path, caplog):
        round_ = _Round(tmp_path, allow_degraded=True)
        with caplog.at_level(logging.WARNING):
            round_.play(PATCHED_WITH_GAP)
            round_.play(PATCHED_WITH_GAP)

        assert round_.stopped_at_status == [False, False]
        warnings = [r.getMessage() for r in caplog.records if "--allow-degraded" in r.getMessage()]
        assert len(warnings) == 1 and GAP in warnings[0], warnings

    def test_a_degraded_engine_runs_and_run_json_still_records_it(self, tmp_path):
        entry = _recorded(
            tmp_path, lambda: _Round(tmp_path, allow_degraded=True).play(DEGRADED))

        assert entry["worker_engine"]["mode"] == "degraded"
        assert entry["rounds"] == 1

    def test_no_status_line_yet_records_none(self, tmp_path):
        entry = _recorded(tmp_path, lambda: _Round(tmp_path).play())
        assert entry["worker_engine"] is None
