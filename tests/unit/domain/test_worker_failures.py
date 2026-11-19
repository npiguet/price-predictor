"""The failure lines a collection worker logs, by kind (V4 #3, P1 #3): no
reader of the worker logs existed, so a run losing its matches looked like a
run that played fewer of them."""

from __future__ import annotations

import functools
import re
from pathlib import Path

import pytest

from price_predictor.domain.worker_failures import (
    FAILURE_PATTERNS,
    WorkerFailure,
    WorkerFailures,
    classify,
)

_CONNECTOR_SOURCES = (
    Path(__file__).resolve().parents[3] / "forge-connector" / "src" / "main" / "java"
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
    WorkerFailure.RANDOM_SEAT_MATCH_ERROR: (
        "Error generating match: [random-seat match] Cannot invoke \"Card.getName()\""
    ),
    WorkerFailure.MATCH_ERROR: "Error generating match: null",
    WorkerFailure.WORKER_GAVE_UP: (
        "Worker giving up after 50 consecutive failed attempts with no success"
    ),
    WorkerFailure.HOOK_FAILURE: (
        "effect-record hook RESOLVE: its listener threw, which loses what that call"
    ),
    WorkerFailure.DIRECTED_PRECONDITION_THREW: (
        "Effect records: directed rule 704.5f's precondition threw and reads as not met "
        "(later throws are counted in the directed fork tally, not printed): "
        "java.lang.NullPointerException"
    ),
    WorkerFailure.EMITTER_FAILURE: 'ApiEvents: the memo for "damage_dealt" threw and was ignored',
    WorkerFailure.UNKNOWN_OUTCOME_TYPE: (
        "ClauseOutcomeRouter: outcomeHandler() received an outcome type not in"
    ),
    WorkerFailure.BOUNDARY_STEP_FAILED: (
        "Effect records: the phase boundary's PROBES step threw, and lost its records"
    ),
    WorkerFailure.GAME_END_MARKER_UNWRITTEN: (
        "Effect records: the game-end marker of game r.0-ab.3 could not be written, so "
        "the game reads as cut (later failures are counted, not printed): disk full"
    ),
    WorkerFailure.RECORDS_WITHHELD: (
        "Effect records: game r.0-ab.3 reached --max-records-per-game 5000; withheld 12"
    ),
    WorkerFailure.FORK_SCORE_MISMATCH: (
        "Effect records: discarding a fork whose score 12 disagrees with the live game's 13"
    ),
    WorkerFailure.FORK_CHECK_FAILED: "Effect records: fork score check failed (NPE); discarding",
    WorkerFailure.FORK_FIDELITY: "Effect records: fork fidelity: 2 difference(s), phase MAIN1",
    WorkerFailure.FORK_FAILED: (
        "Effect records: fork intervention body threw, and is discarded (later failures are "
        "counted, not printed): java.lang.IllegalStateException: x"
    ),
    WorkerFailure.FORK_MOVED_LIVE_GAME: (
        "Effect records: fork probe moved the live game; before the fork: timestamp 10; "
        "after: timestamp 15 (later ones are counted, not printed)"
    ),
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


@pytest.mark.parametrize("line", [
    "Effect records: fork fidelity: 0 difference(s), phase MAIN1, turn 3, caller x",
    "Effect records: fork copy dropped: phase MAIN1, turn 3, caller x, no counterpart {LINK=2}",
])
def test_a_fork_copy_without_differences_is_no_failure(line):
    assert classify(line) is None


@pytest.mark.parametrize("line", [
    "Effect records: fork fidelity: 2 difference(s), phase MAIN1, turn 3, caller x",
    "Effect records: fork fidelity: 10 difference(s), phase MAIN1, turn 3, caller x",
    "Effect records: fork fidelity audit failed (java.lang.NullPointerException)",
])
def test_a_fork_copy_with_differences_or_a_failed_audit_is_a_fidelity_failure(line):
    assert classify(line) is WorkerFailure.FORK_FIDELITY


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


@pytest.mark.parametrize("line", [
    "# A fatal error has been detected by the Java Runtime Environment:",
    "# There is insufficient memory for the Java Runtime Environment to continue.",
])
def test_the_jvm_s_fatal_error_banners_are_jvm_errors(line):
    """A JVM that crashed in native code (an access violation) prints the
    fatal-error banner and an hs_err file, and no exception line."""
    assert classify(line) is WorkerFailure.JVM_ERROR


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
    "effect-record hook failures so far, listener calls that threw and lost their records: "
    "{game-end marker=2, RESOLVE=1}; contained by Forge, per slot: {resolve=1}",
    "Effect records: C:/out/run.0-ab.jsonl.gz [mode=patched]",
    "[Forge] random seat failed to play Shock",
    "Effect records: dropping a probe branch taken on turn 3 that would mirror x",
    "Effect records: fork refusals so far, expected and no failure: {intervention no choices=2}",
])
def test_routine_worker_output_is_no_failure(line):
    assert classify(line) is None


def test_an_unread_generation_is_counted_with_where_it_was():
    failures = WorkerFailures()
    failures.add(WorkerFailure.LOG_UNREAD, "logs/run.0.log: rotated twice")

    assert failures.as_dict() == {"log_unread": 1}
    assert failures.examples["log_unread"] == "logs/run.0.log: rotated twice"


def test_a_random_seat_match_error_is_counted_apart_from_other_match_errors():
    failures = WorkerFailures()
    failures.add_line(LINES[WorkerFailure.RANDOM_SEAT_MATCH_ERROR])
    failures.add_line(LINES[WorkerFailure.MATCH_ERROR])

    assert failures.as_dict() == {"match_error": 1, "random_seat_match_error": 1}


def test_the_stop_protocol_s_lines_are_no_failure():
    """A worker that stopped at a game end, or that says at its start how it
    will stop, reports nothing wrong. Spelled from the stand-in's literals,
    which are pinned to the Java sources."""
    from tests.unit.infrastructure import stand_in_worker as java

    stopped = (f"{java.STOPPED} ({java.REASON_STOP_FILE}C:/logs/run.0.stop); 4 "
               f"{java.PROGRESS_NOUN}{java.IN_THIS_JVM}5{java.GAMES_TAIL}")
    recycled = (f"{java.STOPPED} ({java.REASON_RECYCLING}600 s); 9 "
                f"{java.PROGRESS_NOUN}{java.IN_THIS_JVM}5{java.GAMES_TAIL}")
    startup = (f"{java.DESCRIBE}C:/logs/run.0.stop{java.DESCRIBE_EXISTS}"
               f"{java.DESCRIBE_OR}{java.DESCRIBE_RUN}600 s")
    for line in (stopped, recycled, startup):
        assert classify(line) is None, line


def test_the_hook_report_and_the_shard_line_are_spelled_as_the_worker_prints_them():
    from price_predictor.domain.worker_failures import HOOK_COUNTS_LINE, JVM_START_LINE

    main = (_CONNECTOR_SOURCES / "com" / "pricepredictor" / "connector"
            / "MatchWorkerMain.java").read_text(encoding="utf-8")
    assert ('"effect-record hook failures so far, listener calls that threw"\n'
            '                        + " and lost their records: " + counts\n'
            '                        + "; contained by Forge, per slot: " + byForge') in main
    assert '"Effect records: " + effectRecords.path()' in main
    assert '+ " [mode=" + AttributionMode.detect().wireName() + "]"' in main
    assert HOOK_COUNTS_LINE.search(
        "effect-record hook failures so far, listener calls that threw and lost their "
        "records: {a=1}; contained by Forge, per slot: {b=2}")
    assert JVM_START_LINE.search("Effect records: C:/x y/r.0-l.jsonl.gz [mode=degraded]")


class TestHookFailureCounts:
    """The report is a running count per worker JVM, read by nothing before
    (WP7): the run's counts are each lifetime's latest, summed."""

    REPORT = ("effect-record hook failures so far, listener calls that threw and lost "
              "their records: {{{own}}}; contained by Forge, per slot: {{{forge}}}")
    START = "Effect records: C:/out/run.0-{life}.jsonl.gz [mode=patched]"

    def test_a_lifetime_s_latest_report_stands_and_lifetimes_add_up(self):
        from price_predictor.domain.worker_failures import (
            HookFailureCounts,
            hook_failure_document,
        )

        slot = HookFailureCounts()
        for line in (
            self.START.format(life="a"),
            self.REPORT.format(own="RESOLVE=1", forge="resolve=1"),
            self.REPORT.format(own="RESOLVE=3, game-end marker=1", forge="resolve=3"),
            self.START.format(life="b"),
            self.REPORT.format(own="RESOLVE=2", forge=""),
        ):
            slot.add_line(line)

        assert hook_failure_document([slot]) == {
            "connector": {"RESOLVE": 5, "game-end marker": 1},
            "forge": {"resolve": 3},
        }

    def test_slots_add_up_and_render_on_one_line(self):
        from price_predictor.domain.worker_failures import (
            HookFailureCounts,
            hook_failure_document,
            render_hook_failures,
        )

        first, second = HookFailureCounts(), HookFailureCounts()
        first.add_line(self.REPORT.format(own="RESOLVE=1", forge=""))
        second.add_line(self.REPORT.format(own="RESOLVE=2, subscriber X.y=1", forge="s=4"))
        document = hook_failure_document([first, second])

        assert document == {"connector": {"RESOLVE": 3, "subscriber X.y": 1},
                            "forge": {"s": 4}}
        assert render_hook_failures(document) == (
            "3 RESOLVE, 1 subscriber X.y; contained by Forge: 4 s")
        assert render_hook_failures(hook_failure_document([])) == "no hook failures"


class TestForkRefusalCounts:
    """The worker prints, at a match end where it changed, how many forks
    declined for an expected reason (``ForkRefusals``): a running map per JVM
    lifetime that the supervisor did not read (WP5). Refusals are no
    failures: they are counted apart from every failure line."""

    LINE = "Effect records: fork refusals so far, expected and no failure: {{{body}}}"
    START = "Effect records: C:/out/run.0-{life}.jsonl.gz [mode=patched]"

    def test_the_line_is_spelled_as_the_worker_prints_it(self):
        from price_predictor.domain.worker_failures import FORK_REFUSALS_LINE, JVM_START_LINE
        from tests.unit.forge_sources import connector_source

        main = connector_source("MatchWorkerMain.java")
        literal = re.search(r'static final String REFUSALS_LINE = "([^"]*)";', main)
        assert literal is not None, "MatchWorkerMain names the line's start in REFUSALS_LINE"
        assert "System.err.println(REFUSALS_LINE + refusals);" in main
        assert "ForkRefusals.counts()" in main
        # A Java map in key order: ``{a b=1, c d=2}`` (TreeMap.toString).
        refusals = connector_source("effects/ForkRefusals.java")
        assert "Map<String, Long> counts = new TreeMap<>();" in refusals
        assert 'kind + " " + reason' in refusals

        line = literal.group(1) + "{directed no counterpart=2, intervention no choices=1}"
        found = FORK_REFUSALS_LINE.search(line)
        assert found is not None
        assert found.group("refusals") == "directed no counterpart=2, intervention no choices=1"
        assert not JVM_START_LINE.search(line), "a refusal line starts no lifetime"

    def test_the_line_is_no_failure(self):
        line = self.LINE.format(body="directed dies=3, probe no combat=1")
        assert classify(line) is None

    def test_a_lifetime_s_latest_tally_stands_and_lifetimes_add_up(self):
        from price_predictor.domain.worker_failures import (
            ForkRefusalCounts,
            fork_refusal_document,
        )

        slot = ForkRefusalCounts()
        for line in (
            self.START.format(life="a"),
            self.LINE.format(body="directed dies=1"),
            self.LINE.format(body="directed dies=2, intervention no choices=1"),
            self.START.format(life="b"),
            self.LINE.format(body="directed dies=4"),
            "Worker: 3 matches generated",
        ):
            slot.add_line(line)

        assert fork_refusal_document([slot]) == {
            "directed dies": 6, "intervention no choices": 1,
        }

    def test_slots_add_up_in_name_order_and_render_on_one_line(self):
        from price_predictor.domain.worker_failures import (
            ForkRefusalCounts,
            fork_refusal_document,
            render_fork_refusals,
        )

        first, second = ForkRefusalCounts(), ForkRefusalCounts()
        first.add_line(self.LINE.format(body="probe no combat=2"))
        second.add_line(self.LINE.format(body="directed dies=1, probe no combat=1"))
        document = fork_refusal_document([first, second])

        assert list(document.items()) == [("directed dies", 1), ("probe no combat", 3)]
        assert render_fork_refusals(document) == "1 directed dies, 3 probe no combat"
        assert render_fork_refusals(fork_refusal_document([])) == "no fork refusals"
