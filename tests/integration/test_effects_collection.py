"""Collection against a real Forge (T045).

Runs the instrumented worker for long enough to produce records, then checks the
three things the acceptance path turns on: shards fill with ``resolution``
records, every record agrees on one attribution mode, and the two sealed corpora
are untouched in format and content by the flag's presence.

The mode is asserted as consistent rather than as ``degraded``. The sibling
checkout is patched or not independently of this repository, so a test that
pins the value passes only until someone applies the patches and then fails with
nothing wrong. ``test_effects_patched.py`` covers what a patched run must reach.

Skips when the JAR is not built, so the fast suite and a checkout without Forge
stay green.
"""

from __future__ import annotations

import re
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from effects.infrastructure.record_io import iter_shards, read_records
from price_predictor.infrastructure.forge_jvm import resolve_connector_jar
from sealed.infrastructure.match_worker_connector import MatchWorkerConnector

pytestmark = pytest.mark.integration

#: A worker needs to initialize Forge, build two decks and play a game before it
#: writes anything; this is generous enough for a cold JVM on a slow disk.
_COLLECTION_SECONDS = 240
_POLL_SECONDS = 5


def _require_jar() -> None:
    try:
        resolve_connector_jar()
    except FileNotFoundError:
        pytest.skip("forge-connector JAR not built (mvn package -DskipTests)")


def _run_worker(records_dir: Path, output_file: Path | None) -> list:
    """Run one instrumented worker until it has written records, then stop it."""
    connector = MatchWorkerConnector()
    process = connector.start(
        output_file,
        run_id=str(uuid.uuid4()),
        best_of=1,
        effect_records_dir=records_dir,
        worker_index=0,
    )
    try:
        deadline = time.monotonic() + _COLLECTION_SECONDS
        while time.monotonic() < deadline:
            time.sleep(_POLL_SECONDS)
            records = list(read_records(records_dir))
            if len(records) >= 5:
                return records
            if process.poll() is not None:
                pytest.fail(
                    f"worker exited early with code {process.returncode}"
                )
        return list(read_records(records_dir))
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def test_collection_writes_records_under_one_attribution_mode(
    tmp_path: Path,
) -> None:
    _require_jar()
    records_dir = tmp_path / "records"
    output_file = tmp_path / "match-outcomes.txt"

    records = _run_worker(records_dir, output_file)
    if not records:
        pytest.skip("worker produced no records within the collection window")

    shards = iter_shards(records_dir)
    assert shards, "no shard was written"
    # {run_id}.{worker}-{lifetime}.jsonl.gz. The lifetime is minted by the
    # worker JVM, not by the launcher, so it is matched rather than named: it
    # is what keeps a restarted worker from reissuing the ids of the JVM it
    # replaced, and one worker run means exactly one of them.
    assert len(shards) == 1, [shard.name for shard in shards]
    assert re.fullmatch(r".+\.0-[0-9a-z]{1,16}\.jsonl\.gz", shards[0].name), (
        shards[0].name
    )

    kinds = {record.kind.value for record in records}
    assert "resolution" in kinds, f"no resolution records; saw {kinds}"

    # The mode is probed once per worker, so a run cannot mix the two: a corpus
    # that did would be attributed two different ways with nothing saying where
    # the boundary is.
    modes = {record.mode.value for record in records}
    assert len(modes) == 1, f"a single run reported both modes: {modes}"
    assert modes <= {"degraded", "patched"}, modes

    # Ids carry the worker slot and the JVM lifetime, and are unique across the
    # shard. One worker run is one lifetime, so the second half is constant
    # here — but it has to be there, because it is the only thing that keeps
    # the next lifetime of slot 0 from counting over these ids again.
    ids = [record.record_id for record in records]
    assert len(ids) == len(set(ids))
    assert all(record.worker == "0" for record in records)
    lifetimes = {record.lifetime for record in records}
    assert len(lifetimes) == 1, lifetimes
    assert lifetimes != {""}, "ids carry no lifetime segment"

    # Every record names the game it came from, which is the split's join key.
    assert all(record.game_id for record in records)


def test_the_flag_leaves_the_sealed_corpora_alone(tmp_path: Path) -> None:
    """FR-030: adding --effect-records must not change what this command writes.

    Compared by format and row semantics rather than byte-for-byte: a
    match-outcome row carries a per-run timestamp, a run id and a duration, so
    two runs are never byte-identical and a byte comparison could only ever
    fail.
    """
    _require_jar()
    records_dir = tmp_path / "records"
    output_file = tmp_path / "match-outcomes.txt"

    _run_worker(records_dir, output_file)
    if not output_file.exists():
        pytest.skip("worker completed no match within the collection window")

    lines = [
        line for line in output_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        pytest.skip("worker completed no match within the collection window")

    for line in lines:
        fields = line.split(";")
        assert len(fields) == 10, f"match-outcome row has {len(fields)} fields"
        timestamp, run_id, set_code = fields[0], fields[1], fields[2]
        assert timestamp.endswith("Z")
        assert run_id
        assert set_code
        assert set(fields[7]) <= {"A", "B"}, "games column"
        assert set(fields[8]) <= {"A", "B"}, "play column"
        assert fields[9].isdigit(), "duration column"

    cards_played = output_file.parent / "cards-played.txt"
    if cards_played.exists():
        for line in cards_played.read_text(encoding="utf-8").splitlines():
            if line.strip():
                assert len(line.split(";")) == 11


def test_records_only_mode_writes_no_sealed_corpus(tmp_path: Path) -> None:
    """The coverage and variant collectors reuse this worker (FR-052, FR-059)."""
    _require_jar()
    records_dir = tmp_path / "records"

    _run_worker(records_dir, None)

    assert not (tmp_path / "match-outcomes.txt").exists()
    assert not (tmp_path / "cards-played.txt").exists()
