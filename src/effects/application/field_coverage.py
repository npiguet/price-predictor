"""Which record fields a corpus has ever actually carried.

A collector writing a field as a fixed literal produces records byte-identical
to ones where the field is genuinely empty, and every contract test passes. This
walks the record dataclasses rather than a field list, so a field added later is
audited too.

Absence is not proof: a field reads constant whether it is unwired or the corpus
merely never exercised it, which is why :data:`KNOWN_CONSTANT_FIELDS` is
asserted in one direction only.

The walk reaches every gen-3 member too; the table of the members a gen-3
record writes (:mod:`effects.domain.gen3_members`) gives each its own row on a
gen-3 corpus's report, absent ones included, which the walk alone cannot show,
and names the members constant by construction, tagged ``fixed`` rather than
``NEW``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

from effects.domain.event_schema import EVENT_PARAMS, Event, EventType
from effects.domain.gen3_members import FIXED_PATHS, WALK_PATHS
from effects.domain.records import EffectRecord
from price_predictor.domain.dataclass_defaults import field_default

#: A field with no declared default: there is nothing to compare against, so
#: only whether it varies says anything about it.
_REQUIRED = object()
#: An event parameter the event did not write: its cell's default, so a
#: parameter the collector never writes reads populated 0.
_UNWRITTEN = object()


@dataclasses.dataclass(frozen=True, slots=True)
class FieldCoverage:
    """One field's record over a corpus."""

    path: str
    #: Instances whose value differed from the field's declared default;
    #: meaningless for a required field.
    populated: int
    #: Instances examined. Larger than the record count for a repeated field.
    seen: int
    #: Whether two instances ever disagreed.
    varied: bool

    @property
    def constant(self) -> bool:
        """Whether the corpus only ever saw one value here: never varying
        rather than never differing from the default, because a required field
        has none, and a field pinned to a non-default constant is dead too."""
        return not self.varied


#: Fields a collector writes as a fixed literal (deviations log, D59), each
#: confirmed by reading the writer. Fields constant only because one run is one
#: run (``mode``, ``run_id``, ``mirror_of`` and the probe fields under an
#: emptied ``--probe-keywords``) are absent, or a corpus exercising them would
#: fail the test. The list only shrinks: a listed field that carries data fails the
#: coverage test until it is removed.
#:
#: Never list an activation's ``outcome`` or ``costs``, an event's
#: ``attributed_to``, a record's ``ability_unresolved`` or a rewrite's
#: ``result`` and ``replaced_by``: each read constant through a collector
#: defect, and listing one would make a broken channel look like a known one.
#:
#: Empty since the gen-3 pilot (T079): a decision candidate's
#: ``responsible_static`` now names the static of a ``static_prohibition``
#: check, and ``global_.emblems`` carried emblems there.
KNOWN_CONSTANT_FIELDS: frozenset[str] = frozenset()


def field_coverage(records: Iterable[EffectRecord]) -> dict[str, FieldCoverage]:
    """Per-field coverage over a corpus, keyed by dotted path.

    A repeated field is counted per instance, not per record: ``seen`` for
    ``record.state.entities[].tapped`` is the number of entities across every
    snapshot. ``[]`` in a path marks where that repetition happens.
    """
    tally: dict[str, _Tally] = {}
    for record in records:
        _visit(record, "record", tally)
    return {
        path: FieldCoverage(
            path=path, populated=t.populated, seen=t.seen, varied=t.varied,
        )
        for path, t in sorted(tally.items())
    }


def report_lines(
    coverage: dict[str, FieldCoverage], *, all_fields: bool = False,
) -> list[str]:
    """``field-coverage``'s rows, constant fields only unless ``all_fields``.

    A constant field is tagged ``known`` when :data:`KNOWN_CONSTANT_FIELDS`
    lists it, ``fixed`` when the gen-3 table says the writer fixes its value
    (``generation``, the frozen ``tiers``), and ``NEW`` otherwise: only the
    writer says whether a new one is broken or merely rare. A listed field
    that varied is named for removal.
    """
    lines = []
    for row in coverage.values():
        if row.constant:
            lines.append(f"  [{_constant_tag(row.path)}] {row.path}   (seen {row.seen})")
        elif all_fields:
            lines.append(f"  [varies] {row.path}   ({row.populated}/{row.seen})")
    stale = sorted(
        path for path in KNOWN_CONSTANT_FIELDS
        if path in coverage and not coverage[path].constant
    )
    if stale:
        lines += ["", "listed as constant but carrying data — remove from the list:"]
        lines.extend(f"  {path}" for path in stale)
    return lines + gen3_report_lines(coverage) + event_param_lines(coverage)


def _constant_tag(path: str) -> str:
    if path in KNOWN_CONSTANT_FIELDS:
        return "known "
    return "fixed " if path in FIXED_PATHS else "NEW   "


def event_param_lines(coverage: dict[str, FieldCoverage]) -> list[str]:
    """One row per parameter of each event type the corpus holds, over every
    event list it appears in (P3b #14): ``absent`` where no event of the type
    wrote it, ``const`` where it never varied, ``varies`` otherwise. A
    parameter some type writes only at one site (a combat's, a rewrite's)
    reads by its whole."""
    found: dict[tuple[str, str], list[FieldCoverage]] = {}
    for path, row in coverage.items():
        head, marker, name = path.rpartition(".")
        if ".params<" not in head or not head.endswith(">"):
            continue
        event = head.rsplit(".params<", 1)[1][:-1]
        found.setdefault((event, name), []).append(row)
    if not found:
        return []
    rows, tally = [], {"varies": 0, "const": 0, "absent": 0}
    for (event, name), cells in sorted(found.items()):
        written = sum(cell.populated for cell in cells)
        seen = sum(cell.seen for cell in cells)
        if not written:
            status = "absent"
        elif any(cell.varied for cell in cells):
            status = "varies"
        else:
            status = "const"
        tally[status] += 1
        rows.append(f"  [{status:<6}] {event}.{name}   ({written}/{seen})")
    head = (f"event parameters: {tally['varies']} vary, {tally['const']} constant, "
            f"{tally['absent']} never written")
    return ["", head, *rows]


def gen3_report_lines(coverage: dict[str, FieldCoverage]) -> list[str]:
    """One row per walk path of the gen-3 member table; none when no record
    was read.

    ``absent`` is a path no record reached (no record of its kind, or no
    instance of its block); ``fixed`` one constant by construction;
    ``const`` one that never varied, ``known`` when
    :data:`KNOWN_CONSTANT_FIELDS` lists it.
    """
    if "record.generation" not in coverage:
        return []
    rows, tally = [], {"varies": 0, "const": 0, "fixed": 0, "absent": 0}
    for path in WALK_PATHS:
        status, seen = _gen3_status(path, coverage.get(path))
        tally[status] += 1
        rows.append(f"  [{status:<6}] {path}{seen}")
    head = (f"gen-3 record members: {tally['varies']} vary, {tally['const']} constant, "
            f"{tally['fixed']} fixed by the writer, {tally['absent']} absent")
    return ["", head, *rows]


def _gen3_status(path: str, row: FieldCoverage | None) -> tuple[str, str]:
    """A gen-3 row's status and its count, as printed after the path."""
    if row is None:
        return "absent", ""
    if not row.constant:
        return "varies", f"   ({row.populated}/{row.seen})"
    known = ", known" if path in KNOWN_CONSTANT_FIELDS else ""
    return ("fixed" if path in FIXED_PATHS else "const"), f"   (seen {row.seen}{known})"


class _Tally:
    """One path's running counts. ``first`` is the first value's repr,
    compared until the field varies, so the repr cost falls on constant fields.
    """

    __slots__ = ("populated", "seen", "varied", "first")

    def __init__(self) -> None:
        self.populated = 0
        self.seen = 0
        self.varied = False
        self.first: str | None = None

    def add(self, value: Any, default: Any) -> None:
        self.seen += 1
        self.populated += _differs(value, default)
        if self.varied:
            return
        shown = repr(value)
        if self.first is None:
            self.first = shown
        elif shown != self.first:
            self.varied = True


def _visit(value: Any, path: str, tally: dict[str, _Tally]) -> None:
    """Count every field of one dataclass instance, then recurse."""
    if isinstance(value, Event):
        _visit_params(value, path, tally)
    for spec in dataclasses.fields(value):
        if not spec.compare:
            # Bookkeeping kept out of equality, such as whether a payload
            # wrote a key at all: how the record was spelt, not what it says.
            continue
        child = getattr(value, spec.name)
        child_path = f"{path}.{spec.name}"
        if spec.name == "payload" and _is_instance(child):
            # The one union in the schema. Without the type the combat
            # payload's fields and the continuous payload's would share a path.
            child_path = f"{path}.payload<{type(child).__name__}>"
        else:
            default = field_default(spec, otherwise=_REQUIRED)
            tally.setdefault(child_path, _Tally()).add(child, default)
        _descend(child, child_path, tally)


def _visit_params(event: Event, path: str, tally: dict[str, _Tally]) -> None:
    """One cell per parameter the event's schema row lists, and per one it
    wrote beyond it, keyed by the event's type: a single ``params`` cell
    would hide a parameter never written behind the others varying."""
    params = event.params or {}
    kind = event.type.value if isinstance(event.type, EventType) else str(event.type)
    names = dict.fromkeys((*EVENT_PARAMS.get(event.type, ()), *params))
    for name in names:
        tally.setdefault(f"{path}.params<{kind}>.{name}", _Tally()).add(
            params.get(name, _UNWRITTEN), _UNWRITTEN)


def _descend(value: Any, path: str, tally: dict[str, _Tally]) -> None:
    if _is_instance(value):
        _visit(value, path, tally)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _descend(item, f"{path}[]", tally)


def _is_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _differs(value: Any, default: Any) -> bool:
    if default is _REQUIRED:
        return True
    return value != default
