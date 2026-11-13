"""The failure lines a collection worker writes to its log, by kind.

A Forge worker reports what went wrong on stderr and goes on playing: a match
that could not be generated, a hook listener that threw, records withheld past
the per-game cap, a fork discarded for a score mismatch. Each is a line in the
slot's worker log and nothing else, so a run that lost half its records looks,
from its shards, like a run that played half as many games. The collectors
scan their workers' logs after every round and report these lines by kind
(``collector_connector.CollectorSupervisor``).

Each kind is recognised by the text the Java side prints, found anywhere in
a line: a worker's stdout and stderr share one log and interleave mid-line,
so a failure line can begin after another stream's unfinished text. The
fragment each pattern is built from is checked against the connector's
sources by ``tests/unit/effects/domain/test_worker_failures.py``, and the
same tests hold routine worker output to no match.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum


class WorkerFailure(StrEnum):
    """A kind of failure line, by the name ``run.json`` records it under."""

    #: A deck naming a card Forge's database does not hold (decks-only
    #: rounds refuse the whole deck): the match is lost.
    UNRESOLVED_DECK = "unresolved_deck"
    #: Any other exception out of one match: the match is lost.
    MATCH_ERROR = "match_error"
    #: A worker stopped after too many failed matches in a row.
    WORKER_GAVE_UP = "worker_gave_up"
    #: An engine hook's listener threw; that call's records are lost.
    HOOK_FAILURE = "hook_failure"
    #: An event emitter or memo threw; later ones are not printed.
    EMITTER_FAILURE = "emitter_failure"
    #: A resolution outcome the router does not know.
    UNKNOWN_OUTCOME_TYPE = "unknown_outcome_type"
    #: A phase-boundary step threw and lost its records.
    BOUNDARY_STEP_FAILED = "boundary_step_failed"
    #: A game reached ``--max-records-per-game`` and withheld the rest.
    RECORDS_WITHHELD = "records_withheld"
    #: A fork whose score disagreed with the live game's, discarded.
    FORK_SCORE_MISMATCH = "fork_score_mismatch"
    #: A fork's score could not be checked, discarded.
    FORK_CHECK_FAILED = "fork_check_failed"
    #: The fork fidelity audit found differences, or could not run.
    FORK_FIDELITY = "fork_fidelity"
    #: Variant scripts could not be staged; variant cards will not load.
    STAGING_FAILED = "staging_failed"
    #: Variant names could not be read; variant records keyed as card scripts.
    VARIANT_NAMES_UNREADABLE = "variant_names_unreadable"
    #: The dropped-event counts file could not be appended to.
    DROPPED_EVENTS_UNWRITABLE = "dropped_events_unwritable"
    #: The stack's primary ability cannot be read; special actions misfiled.
    STACK_UNREADABLE = "stack_unreadable"
    #: The JVM itself: an uncaught exception, out of memory.
    JVM_ERROR = "jvm_error"
    #: A log generation rotated away before it was scanned; never a line.
    LOG_UNREAD = "log_unread"


@dataclass(frozen=True)
class FailurePattern:
    """How one kind's line starts, and the literal the Java source prints
    it from (None for a line the JVM itself prints)."""

    kind: WorkerFailure
    pattern: re.Pattern[str]
    source_fragment: str | None


def _pattern(kind: WorkerFailure, regex: str, fragment: str | None) -> FailurePattern:
    return FailurePattern(kind, re.compile(regex), fragment)


#: In match order: the first pattern a line matches decides its kind.
FAILURE_PATTERNS: tuple[FailurePattern, ...] = (
    _pattern(WorkerFailure.UNRESOLVED_DECK,
             r"Error generating match: .*deck cannot be materialized",
             "deck cannot be materialized: "),
    _pattern(WorkerFailure.MATCH_ERROR, r"Error generating match: ", "Error generating match"),
    _pattern(WorkerFailure.WORKER_GAVE_UP, r"Worker giving up after ", "Worker giving up after "),
    _pattern(WorkerFailure.HOOK_FAILURE, r"effect-record hook \S+: its listener threw",
             ": its listener threw, which loses what"),
    _pattern(WorkerFailure.EMITTER_FAILURE, r"ApiEvents: the (emitter|memo) for ",
             "ApiEvents: the emitter for "),
    _pattern(WorkerFailure.UNKNOWN_OUTCOME_TYPE,
             r"ClauseOutcomeRouter: outcomeHandler\(\) received",
             "ClauseOutcomeRouter: outcomeHandler() received "),
    _pattern(WorkerFailure.BOUNDARY_STEP_FAILED, r"Effect records: the phase boundary's ",
             "Effect records: the phase boundary's "),
    _pattern(WorkerFailure.RECORDS_WITHHELD,
             r"Effect records: game \S+ reached --max-records-per-game",
             " reached --max-records-per-game "),
    _pattern(WorkerFailure.FORK_SCORE_MISMATCH, r"Effect records: discarding a fork whose score",
             "Effect records: discarding a fork whose score "),
    _pattern(WorkerFailure.FORK_CHECK_FAILED, r"Effect records: fork score check failed",
             "Effect records: fork score check failed ("),
    _pattern(WorkerFailure.FORK_FIDELITY, r"Effect records: fork fidelity",
             "Effect records: fork fidelity"),
    _pattern(WorkerFailure.STAGING_FAILED, r"Could not stage custom cards ",
             "Could not stage custom cards from "),
    _pattern(WorkerFailure.VARIANT_NAMES_UNREADABLE, r"Could not read variant names ",
             "Could not read variant names from "),
    _pattern(WorkerFailure.DROPPED_EVENTS_UNWRITABLE, r"cannot append dropped-event counts",
             "cannot append dropped-event counts to "),
    _pattern(WorkerFailure.STACK_UNREADABLE,
             r"effect records: MagicStack\.primaryAbility cannot be read",
             "effect records: MagicStack.primaryAbility cannot be read ("),
    _pattern(WorkerFailure.JVM_ERROR,
             r"(Exception in thread |java\.lang\.OutOfMemoryError|# There is insufficient memory)",
             None),
)


def classify(line: str) -> WorkerFailure | None:
    """The kind of failure ``line`` reports, or None for any other line."""
    for failure in FAILURE_PATTERNS:
        if failure.pattern.search(line):
            return failure.kind
    return None


#: Characters of a failure line kept as its kind's example.
EXAMPLE_LENGTH = 300


@dataclass
class WorkerFailures:
    """Failure lines counted by kind, with the first line of each kind as
    its example."""

    counts: Counter[str] = field(default_factory=Counter)
    examples: dict[str, str] = field(default_factory=dict)

    def add_line(self, line: str) -> WorkerFailure | None:
        """Count ``line`` when it reports a failure; its kind, or None."""
        kind = classify(line)
        if kind is not None:
            self.add(kind, line)
        return kind

    def add(self, kind: WorkerFailure, example: str) -> None:
        """Count one failure of ``kind``, ``example`` saying what it was."""
        self.counts[kind.value] += 1
        self.examples.setdefault(kind.value, example.rstrip()[:EXAMPLE_LENGTH])

    def merge(self, other: WorkerFailures) -> None:
        self.counts.update(other.counts)
        for kind, example in other.examples.items():
            self.examples.setdefault(kind, example)

    def __bool__(self) -> bool:
        return bool(+self.counts)

    def as_dict(self) -> dict[str, int]:
        """``kind -> lines``, largest first, for ``run.json``."""
        return dict(sorted(self.counts.items(), key=lambda item: (-item[1], item[0])))

    def render(self) -> str:
        """One line: each kind's count, largest first."""
        if not self:
            return "no failure lines"
        return ", ".join(f"{count} {kind}" for kind, count in self.as_dict().items())
