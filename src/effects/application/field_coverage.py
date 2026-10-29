"""Which record fields a corpus has ever actually carried.

A collector writing a field as a fixed literal produces records byte-identical
to ones where the field is genuinely empty, and every contract test passes. This
walks the record dataclasses rather than a field list, so a field added later is
audited too.

Absence is not proof: a field reads constant whether it is unwired or the corpus
merely never exercised it, which is why :data:`KNOWN_CONSTANT_FIELDS` is
asserted in one direction only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

from effects.domain.records import EffectRecord

#: A field with no declared default: there is nothing to compare against, so
#: only whether it varies says anything about it.
_REQUIRED = object()


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
KNOWN_CONSTANT_FIELDS: frozenset[str] = frozenset({
    # Forge has no cantBeCastStatic to name what forbids a decision; the
    # attacker and blocker subkinds name theirs.
    "record.payload<PlayabilityDecisionPayload>.candidates[].responsible_static",
    # Collected, but a sealed pool essentially never resolves a planeswalker
    # ultimate, so this reads constant on the format rather than the wiring.
    "record.state.global_.emblems",
})


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
    return lines


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
    for spec in dataclasses.fields(value):
        child = getattr(value, spec.name)
        child_path = f"{path}.{spec.name}"
        if spec.name == "payload" and _is_instance(child):
            # The one union in the schema. Without the type the combat
            # payload's fields and the continuous payload's would share a path.
            child_path = f"{path}.payload<{type(child).__name__}>"
        else:
            tally.setdefault(child_path, _Tally()).add(child, _default_of(spec))
        _descend(child, child_path, tally)


def _descend(value: Any, path: str, tally: dict[str, _Tally]) -> None:
    if _is_instance(value):
        _visit(value, path, tally)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _descend(item, f"{path}[]", tally)


def _is_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _default_of(spec: dataclasses.Field) -> Any:
    if spec.default is not dataclasses.MISSING:
        return spec.default
    if spec.default_factory is not dataclasses.MISSING:
        return spec.default_factory()
    return _REQUIRED


def _differs(value: Any, default: Any) -> bool:
    if default is _REQUIRED:
        return True
    return value != default
