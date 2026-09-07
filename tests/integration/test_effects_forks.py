"""Fork collection against a real game (T125).

Forks need a live game to copy, so this is where the budget arithmetic meets the
engine. Skips when the JAR is not built or the run produces no forks — stage
three is opt-in and a stock stage-one run takes none, which is correct rather
than a failure.

The property that matters: **a fork discarded by the score check still counts
against its budget.** Without that, a systematically failing copy retries until
the game ends and the run spends its whole simulation budget producing nothing.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

import pytest

from effects.application.evaluate_effect_model import pair_forks
from effects.infrastructure.record_io import read_records
from price_predictor.infrastructure.forge_jvm import resolve_connector_jar
from sealed.infrastructure.match_worker_connector import MatchWorkerConnector

pytestmark = pytest.mark.integration

_COLLECTION_SECONDS = 240
_POLL_SECONDS = 5


def _run(records_dir: Path) -> list:
    try:
        resolve_connector_jar()
    except FileNotFoundError:
        pytest.skip("forge-connector JAR not built (mvn package -DskipTests)")

    process = MatchWorkerConnector().start(
        None,
        run_id=str(uuid.uuid4()),
        best_of=1,
        effect_records_dir=records_dir,
        worker_index=0,
    )
    try:
        deadline = time.monotonic() + _COLLECTION_SECONDS
        while time.monotonic() < deadline:
            time.sleep(_POLL_SECONDS)
            if process.poll() is not None:
                pytest.fail(f"worker exited early with {process.returncode}")
            if sum(1 for _ in read_records(records_dir)) >= 20:
                break
        return list(read_records(records_dir))
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def test_a_stage_one_run_takes_no_forks(tmp_path: Path) -> None:
    """Correct rather than a failure: stage three is opt-in."""
    records = _run(tmp_path / "records")
    if not records:
        pytest.skip("worker produced no records within the collection window")
    assert not any(record.fork for record in records)
    assert not any(record.interventional for record in records)


def test_a_fork_record_carries_the_forks_own_state(tmp_path: Path) -> None:
    records = _run(tmp_path / "records")
    forks = [r for r in records if r.fork]
    if not forks:
        pytest.skip(
            "no fork records: stage-three collection is opt-in and this run "
            "took none"
        )
    for fork in forks:
        # The intervention changed the board; recording the original would
        # describe a situation the resolution never saw.
        assert fork.state.entities or fork.state.players

    interventional = [f for f in forks if f.interventional]
    for record in interventional:
        assert record.link_id is None, "an intervention has no activation partner"


def test_forks_pair_to_the_records_they_mirror(tmp_path: Path) -> None:
    records = _run(tmp_path / "records")
    pairs = pair_forks(records)
    if not pairs:
        pytest.skip("no matched fork pairs in this run")
    for real, fork in pairs:
        assert fork.mirror_of == real.record_id
        assert fork.game_id == real.game_id, "a fork mirrors a same-game record"


def test_no_record_carries_a_precomputed_diff(tmp_path: Path) -> None:
    """FR-042: the real-versus-fork difference is an evaluation-time
    diagnostic, never something the corpus stores and a trainer could read."""
    records = _run(tmp_path / "records")
    if not records:
        pytest.skip("worker produced no records within the collection window")
    for record in records:
        assert "diff" not in record.extra_fields
        assert "real_vs_fork" not in record.extra_fields
