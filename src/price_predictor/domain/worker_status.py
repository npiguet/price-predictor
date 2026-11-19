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
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

#: ``WorkerStatus.LINE``: how the status line starts; one JSON object follows.
STATUS_PREFIX = "Effect records: status: "

_DECODER = json.JSONDecoder()


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


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _counts(value: object) -> Counter[str]:
    if not isinstance(value, dict) or not all(map(_is_count, value.values())):
        raise ValueError(f"not a map of counts: {value!r}")
    return Counter(value)


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
    )


def parse_status_line(line: str) -> WorkerStatus | None:
    """The status ``line`` carries, found anywhere in it (stdout and stderr
    share a log and can interleave), or None for another line or one whose
    object is cut or malformed."""
    start = line.find(STATUS_PREFIX)
    if start < 0:
        return None
    try:
        document, _ = _DECODER.raw_decode(line, start + len(STATUS_PREFIX))
        return _status(document)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
        return None


@dataclass
class SlotStatuses:
    """The status lines one slot's worker JVMs printed: each lifetime's latest."""

    lifetimes: dict[str, WorkerStatus] = field(default_factory=dict)

    def add_line(self, line: str) -> None:
        status = parse_status_line(line)
        if status is not None:
            self.lifetimes[status.lifetime] = status


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
