"""A Ctrl-C between rounds ends the run: once an
interrupt is latched, ``play_round`` returns without starting a pool."""

from __future__ import annotations

from unittest.mock import patch

from effects.domain.collection_caps import CollectionCaps
from effects.infrastructure.collector_connector import CollectorRun, CollectorSupervisor, RoundSpec
from price_predictor.infrastructure.forge_worker_pool import ForgeWorkerPool
from price_predictor.infrastructure.worker_stops import WorkerStopSettings

_DECKS = [["Plains"]]


def _supervisor(tmp_path) -> CollectorSupervisor:
    return CollectorSupervisor(CollectorRun(
        worker_count=1, effect_records=tmp_path, run_id="run", caps=CollectionCaps(),
        stop=WorkerStopSettings(timeout=1, recycle_seconds=3600),
    ))


def _play(supervisor: CollectorSupervisor, tmp_path) -> None:
    supervisor.play_round(_DECKS, tmp_path / "decks.txt", RoundSpec(label="coverage", matches=1))


def test_a_round_after_an_interrupted_one_starts_no_pool(tmp_path):
    supervisor = _supervisor(tmp_path)

    def interrupted_run(pool: ForgeWorkerPool) -> None:
        pool._interrupted = True

    with patch.object(ForgeWorkerPool, "run", interrupted_run):
        _play(supervisor, tmp_path)
    assert supervisor.interrupted

    with patch.object(ForgeWorkerPool, "run") as run, \
            patch("effects.infrastructure.collector_connector.ForgeWorkerPool") as pool:
        _play(supervisor, tmp_path)

    pool.assert_not_called()
    run.assert_not_called()
    assert supervisor.rounds == 1


def test_a_round_without_an_interrupt_still_runs(tmp_path):
    supervisor = _supervisor(tmp_path)

    with patch.object(ForgeWorkerPool, "run") as run:
        _play(supervisor, tmp_path)

    run.assert_called_once()
    assert supervisor.rounds == 1
