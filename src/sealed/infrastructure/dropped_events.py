"""The dropped-event counts every collecting supervisor prints when its run ends.

Shared by ``sealed match-outcomes`` and the ``effects`` collectors. Loads no
torch.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

#: The file each worker lifetime appends its games' dropped-event counts to,
#: beside its shard: ``{run_id}.{worker}-{lifetime}`` then this. The JVM's
#: ``DroppedEvents.SUFFIX``; a test pins the two.
DROPPED_EVENTS_SUFFIX = ".dropped-events.tsv"


def _count_line(line: str) -> tuple[tuple[str, str], int] | None:
    """One ``event_type<TAB>phase<TAB>count`` line, or None when it does not
    parse."""
    fields = line.split("\t")
    if len(fields) != 3 or not fields[2].isdigit():
        return None
    return (fields[0], fields[1]), int(fields[2])


def _lifetime_counts(path: Path) -> Counter[tuple[str, str]]:
    """One worker lifetime's file, read; an unreadable one counts nothing."""
    counts: Counter[tuple[str, str]] = Counter()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        # Printed from the supervisors' ``finally``: one unreadable file
        # must not hide the run's report and return code.
        return counts
    for line in text.split("\n")[:-1]:
        parsed = _count_line(line)
        if parsed is not None:
            counts[parsed[0]] += parsed[1]
    return counts


def sum_dropped_events(records_dir: Path, run_id: str) -> tuple[Counter[tuple[str, str]], int]:
    """The run's dropped-event counts summed over its worker lifetimes, and
    how many lifetimes' files they came from.

    Each lifetime appends one ``event_type<TAB>phase<TAB>count`` line per pair
    and game, written in one append a kill can cut. Only a newline-terminated
    line counts: a count cut inside its digits (``123`` as ``12``) still
    parses, so an unterminated tail is dropped whole, as a shard's trailing
    partial line is; a terminated line that does not parse is skipped too.
    """
    counts: Counter[tuple[str, str]] = Counter()
    files = sorted(Path(records_dir).glob(f"{run_id}.*{DROPPED_EVENTS_SUFFIX}"))
    for path in files:
        counts.update(_lifetime_counts(path))
    return counts, len(files)


def _plural(lifetimes: int) -> str:
    return "" if lifetimes == 1 else "s"


def _table(counts: Counter[tuple[str, str]], lifetimes: int) -> str:
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    type_width = max(len("event type"), *(len(t) for (t, _), _ in rows))
    phase_width = max(len("phase"), *(len(p) for (_, p), _ in rows))
    total = sum(counts.values())
    count_width = max(len("count"), len(str(total)))
    lines = [
        f"Dropped events (no record open), summed over {lifetimes} worker "
        f"lifetime{_plural(lifetimes)}:",
        f"  {'event type':<{type_width}}  {'phase':<{phase_width}}  {'count':>{count_width}}",
    ]
    lines.extend(
        f"  {event_type:<{type_width}}  {phase:<{phase_width}}  {count:>{count_width}}"
        for (event_type, phase), count in rows
    )
    lines.append(f"  {'total':<{type_width}}  {'':<{phase_width}}  {total:>{count_width}}")
    return "\n".join(lines)


def dropped_events_report(records_dir: Path, run_id: str) -> str:
    """The table a run under ``--count-dropped-events`` prints when it ends
    (FR-041): every (event type, engine phase) pair, most dropped first."""
    counts, lifetimes = sum_dropped_events(records_dir, run_id)
    if not counts:
        return (
            f"Dropped events: none counted over {lifetimes} worker lifetime"
            f"{_plural(lifetimes)}"
        )
    return _table(counts, lifetimes)
