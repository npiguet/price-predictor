"""Helpers for the integration tests that drive a real Forge match worker.

Plain functions rather than fixtures: the worker's arguments differ from test
to test, and a skip has to happen before the first JVM starts.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

import pytest

from effects.domain.records import EffectRecord
from effects.infrastructure.shard_io import read_records
from price_predictor.infrastructure.forge_classpath import resolve_connector_jar
from price_predictor.infrastructure.jvm_processes import kill_process_tree
from sealed.infrastructure.match_worker_connector import (
    MatchWorkerConnector,
    WorkerLaunch,
    WorkerRecords,
    WorkerStopProtocol,
)

#: A worker needs to initialize Forge, build two decks and play a game before
#: it writes anything; this is generous enough for a cold JVM on a slow disk.
COLLECTION_SECONDS = 240
POLL_SECONDS = 5
#: How long a worker asked to stop may take to finish the game it is playing.
STOP_SECONDS = 300


#: The worker-failure kinds (``price_predictor.domain.worker_failures``) whose
#: line means a match, a game or a worker's output (its counts, for a status
#: line that does not parse) was lost: a run the stop protocol ended cleanly
#: has none. The others (a fork discarded, a fidelity
#: difference, a hook that threw) cost a record or a fork and are reported,
#: not asserted: what a few games show of them is a reading, not a verdict.
LOST_OUTPUT_FAILURES = (
    "match_error", "random_seat_match_error", "worker_gave_up",
    "game_end_marker_unwritten", "jvm_error", "log_unread", "status_line_unreadable",
)


def assert_no_lost_output(entry: dict) -> None:
    """``run.json``'s entry records no failure line of a lost match, game or
    worker output; the others it records are printed for the reader."""
    failures = entry["worker_failures"]
    lost = {kind: failures.get(kind, 0) for kind in LOST_OUTPUT_FAILURES}
    assert not any(lost.values()), (lost, entry["worker_failure_examples"])
    others = {kind: count for kind, count in failures.items() if count}
    if others:
        print(f"worker failure lines: {others}; first of each: "
              f"{entry['worker_failure_examples']}")


def require_jar() -> None:
    """Skip the test unless the forge-connector fat JAR is built."""
    try:
        resolve_connector_jar()
    except FileNotFoundError:
        pytest.skip("forge-connector JAR not built (mvn install -DskipTests)")


def stop_file_for(records_dir: Path) -> Path:
    """The stop file of the worker writing ``records_dir``: beside it, never in it."""
    return records_dir.parent / f"{records_dir.name}.worker.stop"


def stop_worker(process, stop_file: Path, seconds: int = STOP_SECONDS) -> None:
    """End a worker by the stop protocol (``contracts/cli.md`` § Match worker
    stop protocol): touch its stop file and wait for it to finish the game it
    is playing, write its marker, close its shard and exit 0. A worker that
    exits otherwise, or outlives ``seconds``, fails the test; the caller's
    ``finally`` still kills it."""
    stop_file.touch()
    try:
        code = process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        pytest.fail(f"the worker did not stop within {seconds} s of the request")
    assert code == 0, f"the worker asked to stop exited with {code}"


def run_worker(
    records_dir: Path, output_file: Path | None = None, *,
    seconds: int = COLLECTION_SECONDS, enough: int = 5,
) -> list[EffectRecord]:
    """Run one instrumented worker until it has written ``enough`` records or
    ``seconds`` have passed, then stop it at its game's end; the records it
    wrote, every game in them ended.

    ``output_file`` None is the records-only mode the collectors use, where
    neither sealed writer is constructed. A worker that exits before it is
    asked fails the test.
    """
    stop_file = stop_file_for(records_dir)
    process = MatchWorkerConnector().start(launch=WorkerLaunch(
        output_file,
        run_id=str(uuid.uuid4()),
        best_of=1,
        records=WorkerRecords(directory=records_dir, worker_index=0),
        stop=WorkerStopProtocol(stop_file=stop_file),
    ))
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            if len(list(read_records(records_dir))) >= enough:
                break
            if process.poll() is not None:
                pytest.fail(f"worker exited early with code {process.returncode}")
        stop_worker(process, stop_file)
        return list(read_records(records_dir))
    finally:
        kill_process_tree(process)
