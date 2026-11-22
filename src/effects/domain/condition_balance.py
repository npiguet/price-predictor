"""Condition balance (FR-056): whether a turn or step condition a record's
outcome depends on held, which splits a cell into met and unmet siblings.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from effects.domain.outcome_signature import SIGNATURE_CLASSES, outcome_signature, signature_label
from effects.domain.provenance import ProvenanceKey, SidecarLookup
from effects.domain.records import (
    Candidate,
    EffectRecord,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityCheck,
    PlayabilityDecisionPayload,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.sampling import SamplingClass
from effects.domain.script_text import params, root, segments

#: The classes whose cells split by whether a turn or step condition held:
#: the outcome-signature classes, and the decision and legality classes,
#: whose signature is the condition alone (FR-056).
CONDITION_CLASSES: frozenset[SamplingClass] = SIGNATURE_CLASSES | {
    SamplingClass.PLAYABILITY_DECISION, SamplingClass.PLAYABILITY_LEGALITY,
}

#: An activation restriction's turn and step parameters, those Forge's
#: ``checkTimingRestrictions`` reads as whose turn, phase and combat
#: (``SpellAbilityRestriction``). Casting speed is not one.
_RESTRICTION_PARAMS = frozenset({
    "PlayerTurn", "OpponentTurn", "ActivationPhases", "ActivationFirstCombat",
    "ActivationAfterBlockers",
})
#: A resolving clause's turn and step parameters (``SpellAbilityCondition``).
_CLAUSE_PARAMS = frozenset({
    "ConditionPlayerTurn", "ConditionOpponentTurn", "ConditionPhases",
    "ConditionFirstCombat",
})
#: A static's turn and step parameters, and its ``Condition$`` values of that
#: kind (``StaticAbility.checkConditions``).
_STATIC_PARAMS = frozenset({"Phases", "PlayerTurn"})
_STATIC_CONDITIONS = frozenset({"PlayerTurn", "NotPlayerTurn", "ExtraTurn"})
#: The checks a candidate fails when its turn or step restriction does not hold.
_TIMING_CHECKS = frozenset({
    PlayabilityCheck.WHOSE_TURN, PlayabilityCheck.PHASE, PlayabilityCheck.COMBAT,
})


class ConditionSide(StrEnum):
    MET = "met"
    UNMET = "unmet"


@dataclass(frozen=True, slots=True)
class TurnStepCondition:
    """A turn or step condition a record's outcome depends on: the line
    carrying it, and whether it held."""

    carrier: ProvenanceKey
    side: ConditionSide

    @property
    def label(self) -> str:
        """The condition's part of an outcome signature."""
        return f"condition:{self.side}"


def turn_step_condition(
    record: EffectRecord, lines: SidecarLookup | None,
) -> TurnStepCondition | None:
    """The turn or step condition of a record's acting line, candidate or
    listed static, and whether it held (FR-056). None where none carries one,
    or the record cannot say whether it held.

    Read from ``conditions_met`` on a resolution's effect half, from
    ``failed_checks`` on a decision's candidate, and from the line's
    ``active`` on a listed static: the acting static of a ``continuous``
    record, or a static a decision or legality record names as responsible.
    The first that carries one decides.
    """
    if lines is None:
        return None
    if record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION:
        return _clause_condition(record, record.payload, lines)
    if record.kind is RecordKind.CONTINUOUS:
        return _static_condition(record, record.ability or (), lines)
    return _decision_condition(record, lines)


def _decision_condition(
    record: EffectRecord, lines: SidecarLookup,
) -> TurnStepCondition | None:
    """A decision's first candidate's condition, or a decision or legality
    record's responsible statics'."""
    payload = record.payload
    if isinstance(payload, PlayabilityDecisionPayload) and payload.candidates:
        return _first_candidate_condition(record, payload.candidates[0], lines)
    if isinstance(payload, PlayabilityAttackersPayload | PlayabilityBlockersPayload):
        listed = [key for entry in payload.forbidden for key in entry.responsible_static]
        return _static_condition(record, listed, lines)
    return None


def _first_candidate_condition(
    record: EffectRecord, candidate: Candidate, lines: SidecarLookup,
) -> TurnStepCondition | None:
    """The candidate's own restriction, else the statics it names as
    responsible, its failed checks' included."""
    found = _candidate_condition(candidate, lines)
    if found is not None:
        return found
    listed = [*candidate.responsible_static, *(
        key for check in candidate.failed_checks or ()
        for key in check.responsible_static
    )]
    return _static_condition(record, listed, lines)


def _clause_condition(
    record: EffectRecord, payload, lines: SidecarLookup,
) -> TurnStepCondition | None:
    """The first resolved clause whose segment carries a turn or step condition.

    A clause is named as its events' ``attributed_to`` is: ``root``, or the
    chain index counted from the first sub-ability, which is the segment
    after the root.
    """
    if not isinstance(payload, ResolutionPayload) or not payload.conditions_met:
        return None
    line = lines.first_line(record.ability or ())
    if line is None or not line.script_text:
        return None
    chain = segments(line.script_text)
    for entry in payload.conditions_met:
        if _clause_carries(entry.segment, chain):
            return TurnStepCondition(record.ability[0], _side(entry.met))
    return None


def _clause_carries(segment: str, chain: list[str]) -> bool:
    """Whether the clause ``segment`` names in ``chain`` carries a turn or step
    condition."""
    index = 0 if segment == "root" else _chain_index(segment)
    if index is None or index >= len(chain):
        return False
    return bool(_CLAUSE_PARAMS & params(chain[index]).keys())


def _chain_index(segment: str) -> int | None:
    return int(segment) + 1 if segment.isdigit() else None


def _candidate_condition(
    candidate: Candidate, lines: SidecarLookup,
) -> TurnStepCondition | None:
    """The candidate's own restriction, held unless it failed a timing check."""
    if candidate.failed_checks is None or not candidate.ability:
        return None
    line = lines.first_line(candidate.ability)
    if line is None or not line.script_text:
        return None
    if not _RESTRICTION_PARAMS & params(root(line.script_text)).keys():
        return None
    failed = {check.check for check in candidate.failed_checks}
    return TurnStepCondition(candidate.ability[0], _side(not failed & _TIMING_CHECKS))


def _static_condition(
    record: EffectRecord, keys: Iterable[ProvenanceKey], lines: SidecarLookup,
) -> TurnStepCondition | None:
    for key in keys:
        # Keys read off a record's statics, never validated by the refusal
        # rule: one the sidecars do not describe, or of a tree the run has no
        # root for, is no line (tallied by the cache), never an error.
        line = lines.line_or_none(key)
        if line is None or not line.script_text:
            continue
        found = params(root(line.script_text))
        if not (_STATIC_PARAMS & found.keys() or found.get("Condition") in _STATIC_CONDITIONS):
            continue
        active = _line_active(record, key)
        if active is not None:
            return TurnStepCondition(key, _side(active))
    return None


def _line_active(record: EffectRecord, key: ProvenanceKey) -> bool | None:
    """Forge's answer to whether the static ``key`` holds, read off the snapshot:
    on a card the board holds, never on a last-known copy (no zone)."""
    held = (
        line.active
        for entity in record.state.entities if entity.zone is not None
        for line in entity.lines if key in line.keys and line.active is not None
    )
    return next(held, None)


def _side(met: bool) -> ConditionSide:
    return ConditionSide.MET if met else ConditionSide.UNMET


def cell_signature(
    record: EffectRecord, klass: SamplingClass, lines: SidecarLookup | None,
) -> str:
    """The signature part of a record's cell: the outcome signature on the
    classes FR-049 step 2 splits, then whether a turn or step condition held
    (FR-056), so the met and unmet cells are siblings the allocator gives
    equal budgets, a short one repeating up to the reuse cap.

    A record no condition applies to keeps gen-2's label exactly.
    """
    parts = []
    if klass in SIGNATURE_CLASSES:
        parts.append(signature_label(outcome_signature(record)))
    if klass in CONDITION_CLASSES:
        condition = turn_step_condition(record, lines)
        if condition is not None:
            parts.append(condition.label)
    return "|".join(parts)


def condition_key(
    klass: SamplingClass, condition: TurnStepCondition,
) -> tuple[str, str, str]:
    """``(class, carrier, side)``: what the survey counts a condition under."""
    return (str(klass), condition.carrier.label, str(condition.side))


def one_sided_conditions(
    counts: Mapping[tuple[str, str, str], int],
) -> dict[str, list[str]]:
    """``class -> carriers`` of every condition seen on one side only, which
    no balance can even out; the manifest names them (FR-056).

    ``counts`` is keyed by :func:`condition_key`.
    """
    out: dict[str, list[str]] = {}
    for (klass, carrier), seen in sorted(_sides_seen(counts).items()):
        if len(seen) == 1:
            out.setdefault(klass, []).append(carrier)
    return out


def _sides_seen(
    counts: Mapping[tuple[str, str, str], int],
) -> dict[tuple[str, str], set[str]]:
    """``(class, carrier) -> sides`` with a positive count."""
    sides: dict[tuple[str, str], set[str]] = {}
    for (klass, carrier, side), n in counts.items():
        if n > 0:
            sides.setdefault((klass, carrier), set()).add(side)
    return sides
