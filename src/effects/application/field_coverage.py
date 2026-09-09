"""Which record fields a corpus has ever actually carried.

A collector that writes a field as a fixed literal produces records that are
byte-identical to records where the field is genuinely empty. Nothing fails, no
test goes red, and the contract tests pass because the *shape* is right and only
the content is missing. Fourteen such fields were found one at a time, each
while working on something else.

This walks the record dataclasses instead of a hand-written field list, so every
field is audited and one added later is audited without being added here. What
it reports is per field: how many instances were examined, and how many carried
something other than the declared default.

Absence is not proof. A field can read as constant because it is written as a
literal, or because the corpus was too small or too ordinary to contain one --
"no Volrath's Shapeshifter was drawn" and "the name channel is unwired" look the
same from here. That asymmetry is why :data:`KNOWN_CONSTANT_FIELDS` is asserted
in one direction only: observing a value is evidence, not observing one is not.
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
    #: Instances whose value differed from the field's declared default.
    #: Meaningless for a required field, which has no default to differ from.
    populated: int
    #: Instances examined. Larger than the record count for a repeated field.
    seen: int
    #: Whether two instances ever disagreed.
    varied: bool

    @property
    def constant(self) -> bool:
        """Whether the corpus only ever saw one value here.

        Never varying rather than never differing from the default, because a
        required field has no default: ``Candidate.ability`` is written as an
        empty list by the collector and declared without one, so comparing
        against a default reported it as carrying data on all 160,843 rows.
        This also catches a field pinned to a non-default constant.
        """
        return not self.varied


#: Fields a collector writes as a fixed literal (deviations log, D59).
#:
#: Membership is a judgement, not a measurement: a field reads as constant here
#: whether it is unwired or merely unexercised, and only the source says which.
#: Everything below was confirmed by reading the writer. Fields that are
#: constant only because one run is one run -- ``mode``, ``run_id``,
#: ``mirror_of``, ``probed_keyword`` and ``probed_entity`` without
#: ``--probe-keywords``, ``synthetic`` before stage four -- are deliberately
#: absent, because listing them would fail the test on the corpus that does
#: exercise them. The probe fields have read constant on every corpus collected
#: so far, and that is a launch flag rather than a dead channel: it is
#: ``validate-corpus``, reporting the probe count as a watched number in the
#: first minutes of a pass, that tells the two apart -- not this list.
#:
#: The list only shrinks. Implementing one makes the coverage test fail until it
#: is removed, which is what stops a fixed field from quietly staying on a list
#: of known problems, and what stops the list itself from drifting.
KNOWN_CONSTANT_FIELDS: frozenset[str] = frozenset({
    # ── deliberately not collected ──
    # A verdict that does not name the static forbidding it. The attacker and
    # blocker subkinds carry theirs, through cantAttackStatic and
    # cantBlockByStatic; the decision subkind would need an equivalent
    # cantBeCastStatic that Forge does not have, and only about a quarter of
    # forbidden creatures find a static even where the hook exists.
    "record.payload<PlayabilityDecisionPayload>.candidates[].responsible_static",
    # Eight cards in Forge write the name layer from a static, the head has no
    # field for a name, and it is open-vocabulary unlike every other channel.
    "record.payload<ContinuousPayload>.contributions[].name",
    # ── collected, but the format cannot exercise it ──
    # An emblem is a planeswalker ultimate's leavings, and a sealed limited pool
    # essentially never resolves one. The collector reads all four trait lists
    # off each command-zone emblem card, so this field being constant is
    # evidence about the format rather than about the wiring — and because the
    # list is asserted in one direction only, the day a corpus does contain an
    # emblem the coverage test fails and this entry comes off.
    "record.state.global_.emblems",
})

#: Deliberately NOT listed above, and they must not be added: an activation's
#: ``outcome``, every field of its ``costs``, an event's ``attributed_to``, and
#: a record's ``ability_unresolved``. All four read as constant in a collected
#: corpus, and all four are collector defects rather than format scarcity — the
#: outcome was a hardcoded literal, the cost lists were read off a Cost object
#: whose parts were never the ones paid, the attribution pointer was set only
#: for the stack root, and ``ability_unresolved`` was declared and never
#: written, so an empty ``ability`` still could not say why it was empty.
#: Listing any of them would have made a broken channel look like a known one,
#: which is the exact failure this list is shaped to avoid.
#:
#: The ``rewrite`` payload's ``result`` and ``replaced_by`` join that sentence.
#: Both read as constant on every corpus collected so far -- ``result`` as
#: ``None`` and ``replaced_by`` as empty, because the hook did not pass either --
#: and both would look like format scarcity from here, since the channel itself
#: held 34 records in 1.88M. It was not scarcity: the payload modelled an
#: edit-the-event mechanism Forge does not have, so ``incoming`` and
#: ``outgoing`` came back byte-identical and 87% of the channel was dropped on
#: the floor. A field that was constant because its channel was empty must not
#: stay excused once the channel fills; excusing these two would have hidden
#: exactly the defect that emptied it.


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


def constant_fields(coverage: dict[str, FieldCoverage]) -> set[str]:
    """Paths that never carried anything but their default."""
    return {path for path, cov in coverage.items() if cov.constant}


class _Tally:
    """One path's running counts.

    ``first`` holds the repr of the first value seen and is compared against
    until the field varies, after which nothing more needs computing — so the
    repr cost is paid on constant fields, whose values are the small ones.
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
