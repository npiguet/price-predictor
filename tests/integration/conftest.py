"""Helpers for the integration tests that drive a real Forge match worker.

Plain functions rather than fixtures: the worker's arguments differ from test
to test, and a skip has to happen before the first JVM starts.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest

from effects.domain.records import EffectRecord
from effects.infrastructure.shard_io import read_records
from price_predictor.infrastructure.forge_jvm import kill_process_tree, resolve_connector_jar
from sealed.infrastructure.match_worker_connector import MatchWorkerConnector

#: A worker needs to initialize Forge, build two decks and play a game before
#: it writes anything; this is generous enough for a cold JVM on a slow disk.
COLLECTION_SECONDS = 240
POLL_SECONDS = 5


#: The worker-failure kinds (``price_predictor.domain.worker_failures``) whose
#: line means a match, a game or a worker's output was lost: a run the stop
#: protocol ended cleanly has none. The others (a fork discarded, a fidelity
#: difference, a hook that threw) cost a record or a fork and are reported,
#: not asserted: what a few games show of them is a reading, not a verdict.
LOST_OUTPUT_FAILURES = (
    "match_error", "random_seat_match_error", "worker_gave_up",
    "game_end_marker_unwritten", "jvm_error", "log_unread",
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


def run_worker(
    records_dir: Path, output_file: Path | None = None, *,
    seconds: int = COLLECTION_SECONDS, enough: int = 5,
) -> list[EffectRecord]:
    """Run one instrumented worker until it has written ``enough`` records or
    ``seconds`` have passed, then stop it; the records it wrote.

    ``output_file`` None is the records-only mode the collectors use, where
    neither sealed writer is constructed. A worker that exits on its own fails
    the test: every caller kills it.
    """
    process = MatchWorkerConnector().start(
        output_file,
        run_id=str(uuid.uuid4()),
        best_of=1,
        effect_records_dir=records_dir,
        worker_index=0,
    )
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            records = list(read_records(records_dir))
            if len(records) >= enough:
                return records
            if process.poll() is not None:
                pytest.fail(f"worker exited early with code {process.returncode}")
        return list(read_records(records_dir))
    finally:
        kill_process_tree(process)
