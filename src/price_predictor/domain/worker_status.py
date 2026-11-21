"""The status line a Forge worker prints, read back (findings 11.1, 16.4).

A match worker prints ``WorkerStatus.LINE`` and one JSON object at start-up
and after each match whose counts changed (``contracts/cli.md`` § Match
worker stop protocol): its JVM lifetime's token, the attribution mode and the
hook inventory's gaps, the Forge preferences it runs with, the hook-failure
counts, the fork refusal tally and the directed fork tally. Every count is a
running total of that JVM, so a slot's totals are each lifetime's latest line
summed, the lifetime told by the line's own ``lifetime`` member.

The line is read as JSON and never split, so a refusal reason, a gap or a
rule id may hold any character. It is the one line the supervisor reads
counts from; none of it is a failure line (``worker_failures``).

The run's first line also names the engine its workers link, judged here
once for every collecting supervisor (:func:`judge_engine`): one not
``patched`` with every hook, one whose Forge build names no commit or
another than the connector's and the run's conversion, or one that changes
under the run, stops the run (:class:`WorkerEngineRefused`), the first
unless ``--allow-degraded``. So does a run whose workers named their shards
or played a match while no lifetime printed a line that parses.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from price_predictor.domain.converted_provenance import is_full_commit

#: ``WorkerStatus.LINE``: how the status line starts; one JSON object follows.
STATUS_PREFIX = "Effect records: status: "

#: The line a collecting worker prints naming its shard, before its status
#: line and its first match (``MatchWorkerMain.openEffectRecords``).
SHARD_LINE = re.compile(r"Effect records: .+ \[mode=\w+\]")

_DECODER = json.JSONDecoder()


@dataclass(frozen=True)
class LoadedBuilds:
    """The builds a worker JVM loaded (``LoadedBuilds`` Java side), each None
    when its jar's manifest names none."""

    #: The commit the loaded Forge engine (its ``forge-game`` jar) was built at.
    forge_build_commit: str | None = None
    #: Whether that build's checkout had uncommitted changes to tracked files.
    forge_build_dirty: bool | None = None
    #: The connector JAR's ``Forge-Commit``: the Forge commit it was built at.
    connector_forge_commit: str | None = None


@dataclass(frozen=True)
class WorkerStatus:
    """One status line: a worker JVM's state and running totals."""

    #: The JVM lifetime's token, the one its shard is named by.
    lifetime: str
    #: ``patched`` or ``degraded`` (``AttributionMode``).
    mode: str
    #: One entry per hook the engine lacks or whose listener is short; empty when whole.
    gaps: tuple[str, ...]
    #: The four Forge preferences the engine runs with, by name.
    pinned_preferences: dict[str, object]
    #: The connector's hook-failure counts, by hook.
    connector_failures: Counter[str]
    #: Forge's own contained failures, by hook slot.
    forge_failures: Counter[str]
    #: Forks that declined for an expected reason, by ``<kind> <reason>``.
    fork_refusals: Counter[str]
    #: Per directed rule, its counts by name (``matched``, ``kept``, ...).
    directed: dict[str, Counter[str]]
    #: The Forge engine and the connector this JVM loaded.
    builds: LoadedBuilds = LoadedBuilds()


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _counts(value: object) -> Counter[str]:
    if not isinstance(value, dict) or not all(map(_is_count, value.values())):
        raise ValueError(f"not a map of counts: {value!r}")
    return Counter(value)


def _optional(value: object, kind: type) -> object:
    if value is not None and not isinstance(value, kind):
        raise ValueError(f"not a {kind.__name__} or null: {value!r}")
    return value


def _builds(hooks: dict) -> LoadedBuilds:
    """The builds ``hooks`` names; a member a line printed before the
    builds existed lacks reads as None, a build that names none."""
    return LoadedBuilds(
        forge_build_commit=_optional(hooks.get("forge_build_commit"), str),
        forge_build_dirty=_optional(hooks.get("forge_build_dirty"), bool),
        connector_forge_commit=_optional(hooks.get("connector_forge_commit"), str),
    )


def _status(document: dict) -> WorkerStatus:
    hooks, failures = document["hooks"], document["hook_failures"]
    return WorkerStatus(
        lifetime=str(document["lifetime"]),
        mode=str(hooks["mode"]),
        gaps=tuple(hooks["gaps"]),
        pinned_preferences=dict(document["pinned_preferences"]),
        connector_failures=_counts(failures["connector"]),
        forge_failures=_counts(failures["forge"]),
        fork_refusals=_counts(document["fork_refusals"]),
        directed={rule: _counts(counts) for rule, counts in document["directed"].items()},
        builds=_builds(hooks),
    )


class UnreadableStatusLine(ValueError):
    """A line carrying the status prefix whose object is cut or malformed."""


def parse_status_line(line: str) -> WorkerStatus | None:
    """The status ``line`` carries, found anywhere in it (stdout and stderr
    share a log and can interleave), or None for another line; a line with
    the prefix whose object is cut or malformed raises
    :class:`UnreadableStatusLine`."""
    start = line.find(STATUS_PREFIX)
    if start < 0:
        return None
    try:
        document, _ = _DECODER.raw_decode(line, start + len(STATUS_PREFIX))
        return _status(document)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as error:
        raise UnreadableStatusLine(f"{error}: {line}") from error


@dataclass
class SlotStatuses:
    """The status lines one slot's worker JVMs printed: each lifetime's
    latest; and whether a worker named its shard."""

    lifetimes: dict[str, WorkerStatus] = field(default_factory=dict)
    #: A worker of the slot printed the line naming its shard (:data:`SHARD_LINE`).
    named_shard: bool = False

    def add_line(self, line: str) -> bool:
        """Follow ``line``; False when it carries the status prefix and does
        not parse, a line the run cannot judge its engine by."""
        try:
            status = parse_status_line(line)
        except UnreadableStatusLine:
            return False
        if status is not None:
            self.lifetimes[status.lifetime] = status
        elif SHARD_LINE.search(line):
            self.named_shard = True
        return True


def _latest(slots: Iterable[SlotStatuses]) -> Iterator[WorkerStatus]:
    for slot in slots:
        yield from slot.lifetimes.values()


def hook_failure_document(slots: Iterable[SlotStatuses]) -> dict[str, dict[str, int]]:
    """Every lifetime's hook-failure counts summed, as ``run.json`` records them:
    ``{"connector": {hook: calls}, "forge": {slot: calls}}``, each key sorted."""
    connector: Counter[str] = Counter()
    forge: Counter[str] = Counter()
    for status in _latest(slots):
        connector.update(status.connector_failures)
        forge.update(status.forge_failures)
    return {"connector": dict(sorted(connector.items())), "forge": dict(sorted(forge.items()))}


def fork_refusal_document(slots: Iterable[SlotStatuses]) -> dict[str, int]:
    """Every lifetime's fork refusals summed, as ``run.json`` records them:
    ``{"<kind> <reason>": forks}``, in name order."""
    refusals: Counter[str] = Counter()
    for status in _latest(slots):
        refusals.update(status.fork_refusals)
    return dict(sorted(refusals.items()))


def directed_fork_tally_document(slots: Iterable[SlotStatuses]) -> dict[str, dict[str, int]]:
    """Every lifetime's directed fork tally summed, as ``run.json`` records it:
    ``{rule: {count name: n}}``, in rule name order."""
    total: dict[str, Counter[str]] = {}
    for status in _latest(slots):
        for rule, counts in status.directed.items():
            total.setdefault(rule, Counter()).update(counts)
    return {rule: dict(total[rule]) for rule in sorted(total)}


def render_hook_failures(document: dict[str, dict[str, int]]) -> str:
    """One line: the listener calls each hook lost, then what Forge contained."""
    if not any(document.values()):
        return "no hook failures"
    parts = [f"{n} {hook}" for hook, n in document["connector"].items()]
    contained = ", ".join(f"{n} {slot}" for slot, n in document["forge"].items())
    return ", ".join(parts) + (f"; contained by Forge: {contained}" if contained else "")


def render_fork_refusals(document: dict[str, int]) -> str:
    """One line: each kind and reason's count, in name order."""
    if not document:
        return "no fork refusals"
    return ", ".join(f"{n} {key}" for key, n in document.items())


# ── The engine a run's workers link (finding 11.1) ──────────────────────

#: ``AttributionMode.PATCHED``: the engine carries the effect-record hooks.
PATCHED = "patched"

#: How the refusal says what to do, the same on every collecting command.
_REMEDY = (
    "A run collecting effect records needs every hook, built at the Forge commit "
    "it converted at: install ../forge on its effect-record-hooks branch, then "
    "the connector JAR (src/effects/CLAUDE.md § The engine hooks), or pass "
    "--allow-degraded to collect anyway"
)


#: ``--allow-degraded``'s help, on every command that collects effect records.
ALLOW_DEGRADED_HELP = (
    "Collect even when the Forge workers' engine is not patched with every "
    "effect-record hook or was not built at the run's Forge commit (stock "
    "Forge, a stale installed Forge jar): the run "
    "warns once, naming what is missing, instead of stopping with exit 1. "
    "An engine that changes under the run still stops it"
)


class WorkerEngineRefused(RuntimeError):
    """A run collecting effect records stopped for the engine its workers link:
    one not ``patched`` with every hook, or one that changed under the run."""


def engine_document(status: WorkerStatus) -> dict[str, object]:
    """The engine a worker lifetime runs, as ``run.json``'s ``worker_engine``
    records it: ``{"mode", "gaps", "pinned_preferences", "forge_build_commit",
    "forge_build_dirty", "connector_forge_commit"}``."""
    return {
        "mode": status.mode,
        "gaps": list(status.gaps),
        "pinned_preferences": dict(status.pinned_preferences),
        "forge_build_commit": status.builds.forge_build_commit,
        "forge_build_dirty": status.builds.forge_build_dirty,
        "connector_forge_commit": status.builds.connector_forge_commit,
    }


def engine_shortfall(
    status: WorkerStatus, conversion_commit: str | None = None,
) -> tuple[str, ...]:
    """What the engine lacks: each build that names no commit or another than
    the connector's and ``conversion_commit``'s (the run's conversion, when
    it names one), its mode when not ``patched``, then each gap
    ``PatchHooks.report()`` named; empty when it is whole."""
    builds = build_shortfall(status.builds, conversion_commit)
    if status.mode == PATCHED:
        return (*builds, *status.gaps)
    return (*builds, f"mode {status.mode} (no effect-record hooks)", *status.gaps)


def build_shortfall(builds: LoadedBuilds, conversion_commit: str | None) -> tuple[str, ...]:
    """Why ``builds`` are not the run's Forge, each commit named first: each
    build that names no commit, then the first of the Forge build, the
    connector and the conversion whose commit differs from the one before."""
    named = [
        (f"Forge build {builds.forge_build_commit}", builds.forge_build_commit, _NO_BUILD_COMMIT),
        (f"connector built at Forge {builds.connector_forge_commit}",
         builds.connector_forge_commit, _NO_CONNECTOR_COMMIT),
    ]
    missing = tuple(reason for _, commit, reason in named if not is_full_commit(commit))
    if conversion_commit is not None:
        named.append((f"conversion at Forge {conversion_commit}", conversion_commit, None))
    present = [(label, commit) for label, commit, _ in named if is_full_commit(commit)]
    return missing + _first_difference(present)


def _first_difference(present: list[tuple[str, str]]) -> tuple[str, ...]:
    """``"<a> differs from <b>"`` for the first of ``present`` whose commit is
    not the one before it; empty when all agree."""
    for (before, commit), (label, later) in zip(present, present[1:]):
        if later != commit:
            return (f"{before} differs from {label}: the workers' engine is "
                    "not the Forge the run's keys were converted at",)
    return ()


#: The shortfall of a worker whose ``forge-game`` jar names no build commit.
_NO_BUILD_COMMIT = (
    "the worker's Forge build names no commit (its forge-game jar's manifest has "
    "no Forge-Build-Commit: a Forge built before its pom wrote one, or without git)"
)
#: The shortfall of a worker whose connector JAR names no ``Forge-Commit``.
_NO_CONNECTOR_COMMIT = "the worker's connector JAR names no Forge-Commit in its manifest"


def engine_change(first: WorkerStatus, later: WorkerStatus) -> tuple[str, ...]:
    """How ``later``'s engine differs from ``first``'s, one entry per member
    that differs; empty when they are the same."""
    before, after = engine_document(first), engine_document(later)
    return tuple(
        f"{name} {before[name]!r} -> {after[name]!r}"
        for name in before if before[name] != after[name]
    )


@dataclass(frozen=True)
class EngineVerdict:
    """The engine a run's status lines show so far: the first lifetime's, and
    every later lifetime whose engine is not the same."""

    #: The run's first status line; None before any worker printed one.
    first: WorkerStatus | None = None
    #: Per lifetime whose engine differs from the first's: how.
    changed: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: The Forge commit of the run's conversion, None when it names none.
    conversion_commit: str | None = None
    #: Workers named their shards or played a match, and no lifetime's line parsed.
    silent: bool = False

    @property
    def shortfall(self) -> tuple[str, ...]:
        """What the first lifetime's engine lacks (:func:`engine_shortfall`)."""
        if self.first is None:
            return ()
        return engine_shortfall(self.first, self.conversion_commit)

    def refusal(self, *, allow_degraded: bool) -> str | None:
        """Why the run must stop, or None: workers that print no status line
        and an engine that changed under the run, always; one that is not
        whole, unless ``allow_degraded``."""
        if self.silent:
            return _SILENT_REFUSAL
        if self.changed:
            return self._change_refusal()
        if self.shortfall and not allow_degraded:
            return (
                f"the Forge workers' engine is not whole (worker lifetime "
                f"{self.first.lifetime}): {'; '.join(self.shortfall)}. {_REMEDY}"
            )
        return None

    def _change_refusal(self) -> str:
        lifetime, change = next(iter(self.changed.items()))
        return (
            f"worker lifetime {lifetime} runs another engine than the run's first "
            f"(lifetime {self.first.lifetime}): {'; '.join(change)}; a run's records "
            "come from one engine (was a Forge or connector jar installed during the run?)"
        )

    def warning(self) -> str | None:
        """Under ``--allow-degraded``, what the run collects without, or None."""
        if not self.shortfall:
            return None
        return (
            f"--allow-degraded: collecting on an engine that is not whole (worker "
            f"lifetime {self.first.lifetime}): {'; '.join(self.shortfall)}"
        )


#: Why a run stops whose workers named their shards or played a match while
#: no lifetime printed a status line that parses: its engine was never judged.
_SILENT_REFUSAL = (
    "no worker lifetime printed a status line that parses (Effect records: status:), "
    "though the workers named their shards or played a match: the connector JAR was "
    "built before the line existed, or the line changed its shape, so the workers' "
    "engine cannot be judged. Rebuild the connector "
    "(mvn -f forge-connector/pom.xml install -DskipTests)"
)


def judge_engine(
    first: WorkerStatus | None, slots: Iterable[SlotStatuses], *,
    conversion_commit: str | None = None, silent: bool = False,
) -> EngineVerdict:
    """The verdict on ``first``, the run's first status line, against the
    run's ``conversion_commit``, and on every lifetime ``slots`` hold, each
    compared with it; ``silent`` when workers that should have printed a
    line printed none that parses (``first`` is then None)."""
    if first is None:
        return EngineVerdict(silent=silent)
    changed = {
        status.lifetime: change
        for status in _latest(slots)
        if (change := engine_change(first, status))
    }
    return EngineVerdict(first, changed, conversion_commit)


@dataclass
class EngineCheck:
    """A run's watch over its workers' engine, at every log scan: the first
    refusal stands, and the ``--allow-degraded`` warning is given once."""

    allow_degraded: bool = False
    #: Why the run must stop, once a scan found a reason.
    refusal: str | None = None
    _warned: bool = False

    def review(self, verdict: EngineVerdict) -> str | None:
        """Take in ``verdict``; the warning to give now, at most once a run."""
        if self.refusal is None:
            self.refusal = verdict.refusal(allow_degraded=self.allow_degraded)
        if self._warned or not self.allow_degraded:
            return None
        warning = verdict.warning()
        self._warned = warning is not None
        return warning
