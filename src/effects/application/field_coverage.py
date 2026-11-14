"""Which record fields a corpus has ever actually carried.

A collector writing a field as a fixed literal produces records byte-identical
to ones where the field is genuinely empty, and every contract test passes. This
walks the record dataclasses rather than a field list, so a field added later is
audited too.

Absence is not proof: a field reads constant whether it is unwired or the corpus
merely never exercised it, which is why :data:`KNOWN_CONSTANT_FIELDS` is
asserted in one direction only.

The walk reaches every gen-3 field too; :data:`GEN3_FIELDS` names the ones
root § 8.13 adds so a gen-3 corpus's report rows each one of them, absent
ones included, which the walk alone cannot show.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

from effects.domain.event_schema import EVENT_PARAMS, Event, EventType
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


_ENTITY = "record.state.entities[]"
_DECISION = "record.payload<PlayabilityDecisionPayload>.candidates[]"
_ATTACKERS = "record.payload<PlayabilityAttackersPayload>"

#: The fields of the gen-3 record (root § 8.13, with the snapshot facts of
#: § 8.4 it points to), by walk path. A gen-3 pilot is judged on each one
#: varying, being listed as constant, or being absent for a reason the
#: operator accepts (SC-003), so the report gives every one its own row.
GEN3_FIELDS: tuple[str, ...] = (
    "record.generation", "record.aimed_at", "record.directed",
    *(f"{_ENTITY}.{name}" for name in (
        "owner", "face_state", "mana_cost", "printed_loyalty", "printed_defense",
        "real_face", "designations", "x_paid", "chosen", "costs_paid", "history",
        "order_rank", "role")),
    f"{_ENTITY}.lines[].active", f"{_ENTITY}.lines[].affected",
    *(f"record.state.players[].{name}" for name in (
        "max_land_plays", "land_plays_unlimited", "speed", "city_blessing", "rad",
        "monarch", "initiative", "turn_counts")),
    *(f"record.state.global_.{name}" for name in (
        "day_night", "floating_effects", "delayed_triggers", "dungeons")),
    "record.state.refs.random", "record.state.links",
    "record.payload<ResolutionPayload>.conditions_met",
    "record.payload<ResolutionPayload>.outcome", "record.payload<ActivationPayload>.events",
    f"{_DECISION}.timing", f"{_DECISION}.failed_checks", f"{_DECISION}.land_play",
    *(f"{_ATTACKERS}.{name}" for name in (
        "anchor_attacker", "legal_defenders", "must_attack", "tapped_by_attack")),
    "record.payload<PlayabilityBlockersPayload>.must_block",
    "record.payload<TurnActionPayload>.events",
    # WP7: the game-end marker's withheld count, a rewrite's unknown incoming
    # event and set-aside ability, a blocker a lure requires elsewhere.
    "record.payload<TurnActionPayload>.records_withheld",
    *(f"record.payload<RewritePayload>.{name}" for name in (
        "incoming_unknown", "buffered", "buffered_ran", "buffered_events")),
    "record.payload<PlayabilityBlockersPayload>.forbidden[].required_elsewhere",
    # A forbidden creature's own keyword line, and a responsible line no key names.
    *(f"record.payload<{payload}>.forbidden[].{name}"
      for payload in ("PlayabilityAttackersPayload", "PlayabilityBlockersPayload")
      for name in ("responsible_own", "responsible_unresolved")),
)


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
    lists it and ``NEW`` otherwise: only the writer says whether a new one is
    broken or merely rare. A listed field that varied is named for removal.
    """
    lines = []
    for row in coverage.values():
        if not row.constant:
            if all_fields:
                lines.append(f"  [varies] {row.path}   ({row.populated}/{row.seen})")
        elif row.path in KNOWN_CONSTANT_FIELDS:
            lines.append(f"  [known ] {row.path}   (seen {row.seen})")
        else:
            lines.append(f"  [NEW   ] {row.path}   (seen {row.seen})")
    stale = sorted(
        path for path in KNOWN_CONSTANT_FIELDS
        if path in coverage and not coverage[path].constant
    )
    if stale:
        lines += ["", "listed as constant but carrying data — remove from the list:"]
        lines.extend(f"  {path}" for path in stale)
    return lines + gen3_report_lines(coverage) + event_param_lines(coverage)


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
    """One row per :data:`GEN3_FIELDS` path, when the corpus holds gen-3 records.

    ``absent`` is a path no record reached (no record of its kind, or no
    instance of its block); ``const`` one that never varied, ``known`` when
    :data:`KNOWN_CONSTANT_FIELDS` lists it.
    """
    generation = coverage.get("record.generation")
    if generation is None or not generation.populated:
        return []
    rows, tally = [], {"varies": 0, "const": 0, "absent": 0}
    for path in GEN3_FIELDS:
        row = coverage.get(path)
        if row is None:
            status, seen = "absent", ""
        elif row.constant:
            status = "const"
            seen = f"   (seen {row.seen}{', known' if path in KNOWN_CONSTANT_FIELDS else ''})"
        else:
            status, seen = "varies", f"   ({row.populated}/{row.seen})"
        tally[status] += 1
        rows.append(f"  [{status:<6}] {path}{seen}")
    head = (f"gen-3 fields (root § 8.13): {tally['varies']} vary, {tally['const']} "
            f"constant, {tally['absent']} absent")
    return ["", head, *rows]


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
