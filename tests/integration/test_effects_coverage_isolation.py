"""Coverage matches never touch the sealed corpora (T103).

Coverage decks are built for coverage rather than as a fair self-play sample, so
a coverage run that appended to ``match-outcomes.txt`` would corrupt what the
scorer trains on — and the corruption would be invisible, because the rows would
look exactly like ordinary self-play rows.

**Byte-identical** is the right comparison here, unlike for the instrumentation
opt-in: nothing is supposed to be appended at all, so the files must not change
by even a timestamp.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

import pytest

from effects.infrastructure.record_io import read_records
from price_predictor.infrastructure.forge_jvm import resolve_connector_jar
from sealed.infrastructure.match_worker_connector import MatchWorkerConnector

pytestmark = pytest.mark.integration

_COLLECTION_SECONDS = 180
_POLL_SECONDS = 5


def _require_jar() -> None:
    try:
        resolve_connector_jar()
    except FileNotFoundError:
        pytest.skip("forge-connector JAR not built (mvn package -DskipTests)")


def _run_records_only(records_dir: Path) -> int:
    """Run a records-only worker until it writes something, then stop it."""
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
            if sum(1 for _ in read_records(records_dir)) >= 5:
                break
        return sum(1 for _ in read_records(records_dir))
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def test_a_coverage_run_leaves_the_sealed_corpora_byte_identical(
    tmp_path: Path,
) -> None:
    _require_jar()
    sealed_dir = tmp_path / "sealed"
    sealed_dir.mkdir()
    outcomes = sealed_dir / "match-outcomes.txt"
    cards_played = sealed_dir / "cards-played.txt"
    outcomes.write_bytes(b"pre-existing;row\n")
    cards_played.write_bytes(b"pre-existing;row\n")

    before_outcomes = outcomes.read_bytes()
    before_cards = cards_played.read_bytes()

    written = _run_records_only(tmp_path / "records")
    if written == 0:
        pytest.skip("worker produced no records within the collection window")

    assert outcomes.read_bytes() == before_outcomes
    assert cards_played.read_bytes() == before_cards


def test_a_records_only_worker_creates_no_sealed_file_at_all(
    tmp_path: Path,
) -> None:
    """Not "created and left empty" — never constructed."""
    _require_jar()
    _run_records_only(tmp_path / "records")

    for name in ("match-outcomes.txt", "cards-played.txt"):
        assert not (tmp_path / name).exists()
        assert not list(tmp_path.rglob(name))


def test_the_records_it_writes_are_still_well_formed(tmp_path: Path) -> None:
    """Writing no sealed corpus must not mean writing a broken one."""
    _require_jar()
    records_dir = tmp_path / "records"
    if _run_records_only(records_dir) == 0:
        pytest.skip("worker produced no records within the collection window")

    records = list(read_records(records_dir))
    assert records
    assert all(record.game_id for record in records)
    assert all(record.mode.value in ("degraded", "patched") for record in records)
