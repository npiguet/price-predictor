"""The directed fork tally a worker prints, summed over a run (DF01).

Per rule, the boundaries its precondition held at (``matched``), the draws
made (``drawn``), the forks the game's budget blocked (``budget_blocked``),
the forks taken and kept, and the throws of its precondition. The line is
``DirectedForkTally.LINE`` and one JSON object, read as JSON and never split.
Each JVM prints its own running totals, so a slot's are each lifetime's latest
line summed, as ``HookFailureCounts`` does.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from price_predictor.domain.worker_failures import JVM_START_LINE

#: ``DirectedForkTally.LINE``: a worker prints it at a match end where the
#: tally changed, then the object ``{rule: {count name: n}}``.
DIRECTED_FORK_TALLY_PREFIX = "Effect records: directed fork tally so far: "

_DECODER = json.JSONDecoder()


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_rule_counts(counts: object) -> bool:
    return isinstance(counts, dict) and all(_is_count(n) for n in counts.values())


def _parse(line: str) -> dict[str, Counter[str]] | None:
    """The tally ``line`` carries, or None for another line or a malformed one."""
    start = line.find(DIRECTED_FORK_TALLY_PREFIX)
    if start < 0:
        return None
    try:
        document, _ = _DECODER.raw_decode(line[start + len(DIRECTED_FORK_TALLY_PREFIX):])
    except json.JSONDecodeError:
        return None
    if not isinstance(document, dict) or not all(map(_is_rule_counts, document.values())):
        return None
    return {rule: Counter(counts) for rule, counts in document.items()}


@dataclass
class DirectedForkTallyCounts:
    """The directed fork tallies one slot's worker JVMs printed."""

    #: Summed over the lifetimes already over.
    ended: dict[str, Counter[str]] = field(default_factory=dict)
    #: The running lifetime's latest tally.
    latest: dict[str, Counter[str]] = field(default_factory=dict)

    def add_line(self, line: str) -> None:
        if JVM_START_LINE.search(line):
            _add_into(self.ended, self.latest)
            self.latest = {}
        found = _parse(line)
        if found is not None:
            self.latest = found

    def totals(self) -> dict[str, dict[str, int]]:
        """Counts by rule and count name, over every lifetime."""
        total: dict[str, Counter[str]] = {}
        _add_into(total, self.ended)
        _add_into(total, self.latest)
        return {rule: dict(counts) for rule, counts in total.items()}


def _add_into(target: dict[str, Counter[str]], source: dict[str, Counter[str]]) -> None:
    for rule, counts in source.items():
        target.setdefault(rule, Counter()).update(counts)


def directed_fork_tally_document(
    slots: list[DirectedForkTallyCounts],
) -> dict[str, dict[str, int]]:
    """Every slot's tallies summed, as ``run.json`` records them:
    ``{rule: {count name: n}}``, in rule name order."""
    total: dict[str, Counter[str]] = {}
    for slot in slots:
        _add_into(total, {rule: Counter(c) for rule, c in slot.totals().items()})
    return {rule: dict(total[rule]) for rule in sorted(total)}
