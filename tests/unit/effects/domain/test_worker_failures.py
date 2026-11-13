"""The failure lines a collection worker logs, by kind (V4 #3, P1 #3): no
reader of the worker logs existed, so a run losing its matches looked like a
run that played fewer of them."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from effects.domain.worker_failures import (
    FAILURE_PATTERNS,
    WorkerFailure,
    WorkerFailures,
    classify,
)

_CONNECTOR_SOURCES = (
    Path(__file__).resolve().parents[4] / "forge-connector" / "src" / "main" / "java"
)


@functools.cache
def _connector_sources() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in _CONNECTOR_SOURCES.rglob("*.java")
    )


#: A line of each kind, as the Java side prints it.
LINES = {
    WorkerFailure.UNRESOLVED_DECK: (
        "Error generating match: deck cannot be materialized: 1 of 40 card names "
        "did not resolve against Forge's card database: zzz. a decks-only round"
    ),
    WorkerFailure.MATCH_ERROR: "Error generating match: null",
    WorkerFailure.WORKER_GAVE_UP: (
        "Worker giving up after 50 consecutive failed attempts with no success"
    ),
    WorkerFailure.HOOK_FAILURE: (
        "effect-record hook RESOLVE: its listener threw, which loses what that call"
    ),
    WorkerFailure.EMITTER_FAILURE: 'ApiEvents: the memo for "damage_dealt" threw and was ignored',
    WorkerFailure.UNKNOWN_OUTCOME_TYPE: (
        "ClauseOutcomeRouter: outcomeHandler() received an outcome type not in"
    ),
    WorkerFailure.BOUNDARY_STEP_FAILED: (
        "Effect records: the phase boundary's PROBES step threw, and lost its records"
    ),
    WorkerFailure.RECORDS_WITHHELD: (
        "Effect records: game r.0-ab.3 reached --max-records-per-game 5000; withheld 12"
    ),
    WorkerFailure.FORK_SCORE_MISMATCH: (
        "Effect records: discarding a fork whose score 12 disagrees with the live game's 13"
    ),
    WorkerFailure.FORK_CHECK_FAILED: "Effect records: fork score check failed (NPE); discarding",
    WorkerFailure.FORK_FIDELITY: "Effect records: fork fidelity: 2 difference(s), phase MAIN1",
    WorkerFailure.STAGING_FAILED: "Could not stage custom cards from x: denied; variant cards",
    WorkerFailure.VARIANT_NAMES_UNREADABLE: "Could not read variant names from x: gone",
    WorkerFailure.DROPPED_EVENTS_UNWRITABLE: "cannot append dropped-event counts to x: full",
    WorkerFailure.STACK_UNREADABLE: "effect records: MagicStack.primaryAbility cannot be read (x)",
    WorkerFailure.JVM_ERROR: 'Exception in thread "main" java.lang.OutOfMemoryError: heap',
}


@pytest.mark.parametrize("kind", [kind for kind in WorkerFailure if kind in LINES])
def test_each_kind_is_recognised_by_the_line_the_worker_prints(kind):
    assert classify(LINES[kind]) is kind


@pytest.mark.parametrize("line", [
    "Worker: 10 matches generated",
    "Staged 12 custom card scripts from x into y",
    "Effect records: score mismatch detail: phase MAIN1",
    "\tat forge.game.Game.foo(Game.java:12)",
    "",
])
def test_an_ordinary_line_is_no_failure(line):
    assert classify(line) is None


@pytest.mark.parametrize(
    "failure", [p for p in FAILURE_PATTERNS if p.source_fragment is not None],
    ids=lambda p: p.kind.value,
)
def test_each_pattern_s_fragment_is_printed_by_the_connector(failure):
    """A reworded Java message would silently stop being counted."""
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in _CONNECTOR_SOURCES.rglob("*.java")
    )
    assert failure.source_fragment in sources


def test_failures_are_counted_by_kind_with_the_first_line_as_example():
    failures = WorkerFailures()
    for line in ("Error generating match: a", "Worker: 10 matches generated",
                 "Error generating match: b", LINES[WorkerFailure.RECORDS_WITHHELD]):
        failures.add_line(line)

    assert failures.as_dict() == {"match_error": 2, "records_withheld": 1}
    assert failures.examples["match_error"] == "Error generating match: a"
    assert failures.render() == "2 match_error, 1 records_withheld"


def test_merging_adds_counts_and_keeps_the_earlier_example():
    first, second = WorkerFailures(), WorkerFailures()
    first.add_line("Error generating match: a")
    second.add_line("Error generating match: b")
    second.add_line(LINES[WorkerFailure.HOOK_FAILURE])

    first.merge(second)

    assert first.as_dict() == {"match_error": 2, "hook_failure": 1}
    assert first.examples["match_error"] == "Error generating match: a"


def test_no_failure_reads_as_empty():
    assert not WorkerFailures()
    assert WorkerFailures().render() == "no failure lines"


@pytest.mark.parametrize("kind", [
    WorkerFailure.MATCH_ERROR, WorkerFailure.RECORDS_WITHHELD, WorkerFailure.HOOK_FAILURE,
    WorkerFailure.WORKER_GAVE_UP, WorkerFailure.JVM_ERROR,
])
def test_a_failure_line_behind_interleaved_output_is_still_found(kind):
    """stdout and stderr share the log and interleave mid-line, so a failure
    line can start after another stream's unfinished text."""
    assert classify("Worker: 10 matches gen" + LINES[kind]) is kind


@pytest.mark.parametrize("line", [
    "Match format: best-of-1",
    "Eligible sets: 120",
    "Loaded 500 coverage decks.",
    "Loading coverage decks from x",
    "Directed forks: none",
    "Random seat: share=0.25 probability=0.5",
    "effect-record hook failures so far, listener calls that threw and lost their records: {}",
    "[Forge] random seat failed to play Shock",
    "Effect records: dropping a probe branch taken on turn 3 that would mirror x",
])
def test_routine_worker_output_is_no_failure(line):
    assert classify(line) is None


def test_an_unread_generation_is_counted_with_where_it_was():
    failures = WorkerFailures()
    failures.add(WorkerFailure.LOG_UNREAD, "logs/run.0.log: rotated twice")

    assert failures.as_dict() == {"log_unread": 1}
    assert failures.examples["log_unread"] == "logs/run.0.log: rotated twice"
