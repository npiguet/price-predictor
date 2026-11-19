"""The failure lines a Forge worker writes to its log, by kind.

A Forge worker reports what went wrong on stderr and goes on playing: a match
that could not be generated, a hook listener that threw, records withheld past
the per-game cap, a fork discarded for a score mismatch. Each is a line in the
slot's worker log and nothing else, so a run that lost half its records looks,
from its shards, like a run that played half as many games. Every supervisor
that gives its workers a log scans it and reports these lines by kind: the
effects collectors after every round, ``sealed match-outcomes
--effect-records`` at every status line
(``price_predictor.infrastructure.worker_log_scan.WorkerLogReport``).

Each kind is recognised by the text the Java side prints, found anywhere in
a line: a worker's stdout and stderr share one log and interleave mid-line,
so a failure line can begin after another stream's unfinished text. The
fragment each pattern is built from is checked against the connector's
sources by ``tests/unit/domain/test_worker_failures.py``, and the
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
    #: An exception out of a match with a random seat (``[random-seat
    #: match]``): the match is lost, and it would have written no outcome row.
    RANDOM_SEAT_MATCH_ERROR = "random_seat_match_error"
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
    #: A game's end-of-game marker could not be written: the game reads as
    #: cut and ``build-corpus`` drops it (printed once per worker; later ones
    #: are counted in the hook-failure report under ``game-end marker``).
    GAME_END_MARKER_UNWRITTEN = "game_end_marker_unwritten"
    #: A game reached ``--max-records-per-game`` and withheld the rest.
    RECORDS_WITHHELD = "records_withheld"
    #: A fork whose score disagreed with the live game's, discarded.
    FORK_SCORE_MISMATCH = "fork_score_mismatch"
    #: A fork's score could not be checked, discarded.
    FORK_CHECK_FAILED = "fork_check_failed"
    #: The fork fidelity audit found differences (a `0 difference(s)` or drop-only line is not
    #: one), or could not run.
    FORK_FIDELITY = "fork_fidelity"
    #: A fork's copy, body or writer threw, the fork discarded (the first
    #: failure of each kind of fork and step; later ones are counted in the
    #: hook-failure report under ``fork <kind> copy|body|write``).
    FORK_FAILED = "fork_failed"
    #: A fork changed the live game it was taken from (printed once per
    #: kind; counted under ``fork <kind> moved the live game``).
    FORK_MOVED_LIVE_GAME = "fork_moved_live_game"
    #: A directed rule's board precondition threw and read as not met: the
    #: rule's forks are lost wherever it throws (the first throw of each rule
    #: is printed; later ones are counted in the directed fork tally).
    DIRECTED_PRECONDITION_THREW = "directed_precondition_threw"
    #: Variant scripts could not be staged; variant cards will not load.
    STAGING_FAILED = "staging_failed"
    #: Variant names could not be read; variant records keyed as card scripts.
    VARIANT_NAMES_UNREADABLE = "variant_names_unreadable"
    #: The dropped-event counts file could not be appended to.
    DROPPED_EVENTS_UNWRITABLE = "dropped_events_unwritable"
    #: The stack's primary ability cannot be read; special actions misfiled.
    STACK_UNREADABLE = "stack_unreadable"
    #: The JVM itself: an uncaught exception, out of memory, a native crash's
    #: fatal-error banner.
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
    _pattern(WorkerFailure.RANDOM_SEAT_MATCH_ERROR,
             r"Error generating match: \[random-seat match\] ", "[random-seat match] "),
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
    _pattern(WorkerFailure.GAME_END_MARKER_UNWRITTEN,
             r"Effect records: the game-end marker of game ",
             "Effect records: the game-end marker of game "),
    _pattern(WorkerFailure.RECORDS_WITHHELD,
             r"Effect records: game \S+ reached --max-records-per-game",
             " reached --max-records-per-game "),
    _pattern(WorkerFailure.FORK_SCORE_MISMATCH, r"Effect records: discarding a fork whose score",
             "Effect records: discarding a fork whose score "),
    _pattern(WorkerFailure.FORK_CHECK_FAILED, r"Effect records: fork score check failed",
             "Effect records: fork score check failed ("),
    _pattern(WorkerFailure.FORK_FIDELITY,
             r"Effect records: fork fidelity(: [1-9]\d* difference| audit failed)",
             "Effect records: fork fidelity"),
    _pattern(WorkerFailure.FORK_FAILED, r"Effect records: fork \S+ (copy|body|write) threw",
             " threw, and is discarded (later failures"),
    _pattern(WorkerFailure.FORK_MOVED_LIVE_GAME, r"Effect records: fork \S+ moved the live game",
             " moved the live game"),
    _pattern(WorkerFailure.DIRECTED_PRECONDITION_THREW,
             r"Effect records: directed rule \S+'s precondition threw",
             "'s precondition threw and"),
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
             r"(Exception in thread |java\.lang\.OutOfMemoryError|# There is insufficient memory"
             r"|# A fatal error has been detected by the Java Runtime Environment)",
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


# ── the hook-failure report ─────────────────────────────────────────────

#: The counts a worker prints at every match end where they changed
#: (``MatchWorkerMain.HookFailureReport``): the connector's own failures by
#: hook, then Forge's contained ones by slot, each a Java map, running totals
#: of that JVM. A count, not a failure line: no pattern above reads it.
HOOK_COUNTS_LINE = re.compile(
    r"effect-record hook failures so far, listener calls that threw and lost their "
    r"records: \{(?P<connector>[^{}]*)\}"
    r"(?:; contained by Forge, per slot: \{(?P<forge>[^{}]*)\})?"
)
#: The line a worker JVM prints as it opens its shard, at its start
#: (``MatchWorkerMain.openEffectRecords``): a new lifetime, counting from zero.
JVM_START_LINE = re.compile(r"Effect records: .* \[mode=")


def _java_map(body: str | None) -> Counter[str]:
    """``{a=1, b c=2}``'s inside as counts; an item that is no ``key=number`` is skipped."""
    counts: Counter[str] = Counter()
    for item in (body or "").split(", "):
        key, sep, value = item.rpartition("=")
        if sep and value.strip().isdigit():
            counts[key.strip()] = int(value)
    return counts


@dataclass
class HookFailureCounts:
    """The hook-failure counts one slot's worker JVMs printed.

    Each JVM prints its own running totals, from zero at its start, so the
    slot's are each lifetime's latest report summed: a lifetime ends where the
    next one's start line is read.
    """

    #: Summed over the lifetimes already over.
    ended_connector: Counter[str] = field(default_factory=Counter)
    ended_forge: Counter[str] = field(default_factory=Counter)
    #: The running lifetime's latest report.
    connector: Counter[str] = field(default_factory=Counter)
    forge: Counter[str] = field(default_factory=Counter)

    def add_line(self, line: str) -> None:
        if JVM_START_LINE.search(line):
            self.ended_connector.update(self.connector)
            self.ended_forge.update(self.forge)
            self.connector, self.forge = Counter(), Counter()
        found = HOOK_COUNTS_LINE.search(line)
        if found is not None:
            self.connector = _java_map(found.group("connector"))
            self.forge = _java_map(found.group("forge"))

    def totals(self) -> tuple[Counter[str], Counter[str]]:
        """The connector's counts and Forge's, over every lifetime."""
        return self.ended_connector + self.connector, self.ended_forge + self.forge


def hook_failure_document(slots: list[HookFailureCounts]) -> dict[str, dict[str, int]]:
    """Every slot's hook-failure counts summed, as ``run.json`` records them:
    ``{"connector": {hook: calls}, "forge": {slot: calls}}``, each key sorted."""
    connector: Counter[str] = Counter()
    forge: Counter[str] = Counter()
    for slot in slots:
        own, contained = slot.totals()
        connector.update(own)
        forge.update(contained)
    return {"connector": dict(sorted(connector.items())), "forge": dict(sorted(forge.items()))}


def render_hook_failures(document: dict[str, dict[str, int]]) -> str:
    """One line: the listener calls each hook lost, then what Forge contained."""
    if not any(document.values()):
        return "no hook failures"
    parts = [f"{n} {hook}" for hook, n in document["connector"].items()]
    contained = ", ".join(f"{n} {slot}" for slot, n in document["forge"].items())
    return ", ".join(parts) + (f"; contained by Forge: {contained}" if contained else "")


# ── the fork refusal tally ──────────────────────────────────────────────

#: The tally a worker prints at every match end where it changed
#: (``MatchWorkerMain.HookFailureReport``, from ``ForkRefusals``): the forks
#: that declined for an expected reason, ``<kind> <reason>=n`` in name order,
#: running totals of that JVM. Expected, not a failure: no pattern above
#: reads it and it counts toward no failure threshold.
FORK_REFUSALS_LINE = re.compile(
    r"Effect records: fork refusals so far, expected and no failure: \{(?P<refusals>[^{}]*)\}"
)


@dataclass
class ForkRefusalCounts:
    """The fork refusal tallies one slot's worker JVMs printed: each
    lifetime's latest tally, summed, as :class:`HookFailureCounts` reads the
    hook-failure report."""

    #: Summed over the lifetimes already over.
    ended: Counter[str] = field(default_factory=Counter)
    #: The running lifetime's latest tally.
    latest: Counter[str] = field(default_factory=Counter)

    def add_line(self, line: str) -> None:
        if JVM_START_LINE.search(line):
            self.ended.update(self.latest)
            self.latest = Counter()
        found = FORK_REFUSALS_LINE.search(line)
        if found is not None:
            self.latest = _java_map(found.group("refusals"))

    def totals(self) -> Counter[str]:
        """The refusals by kind and reason, over every lifetime."""
        return self.ended + self.latest


def fork_refusal_document(slots: list[ForkRefusalCounts]) -> dict[str, int]:
    """Every slot's fork refusals summed, as ``run.json`` records them:
    ``{"<kind> <reason>": forks}``, in name order."""
    refusals: Counter[str] = Counter()
    for slot in slots:
        refusals.update(slot.totals())
    return dict(sorted(refusals.items()))


def render_fork_refusals(document: dict[str, int]) -> str:
    """One line: each kind and reason's count, in name order."""
    if not document:
        return "no fork refusals"
    return ", ".join(f"{n} {key}" for key, n in document.items())
