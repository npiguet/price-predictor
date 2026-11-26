"""The legality fields a playability record trains, read off its payload
rather than off events: which entities a decision's candidate may target,
which creatures may attack or block, what each attacker may attack, what a
requirement makes it do, and the tap a declared attack made.

They are written on every entity each one is defined for, the affected gate
untouched (root spec § 11.4); :func:`effects.domain.effect_targets.derive_targets`
calls :func:`apply_playability` with its entry maker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from effects.domain.effect_head_layout import HeadField
from effects.domain.records import (
    EffectRecord,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
)

if TYPE_CHECKING:
    from effects.domain.effect_targets import EntityTargets


def apply_playability(record: EffectRecord, slot) -> None:
    """Legality fields from the payload rather than from events, on every
    entity each one is defined for, the gate untouched (root spec § 11.4)."""
    payload = record.payload
    if isinstance(payload, PlayabilityDecisionPayload):
        _decision_legality(record, payload, slot)
    elif isinstance(payload, PlayabilityAttackersPayload):
        _attackers_legality(record, payload, slot)
    elif isinstance(payload, PlayabilityBlockersPayload):
        _blockers_legality(payload, slot)


def target_legality_subjects(record: EffectRecord) -> tuple[str, ...]:
    """Every card and player a decision record writes ``target_legal`` at:
    each of the snapshot's, and each of the candidate's ``legal_targets``,
    when the candidate has a targeting clause (a legal target, or
    ``has_legal_target`` false for want of one); none otherwise."""
    payload = record.payload
    if not isinstance(payload, PlayabilityDecisionPayload) or not payload.candidates:
        return ()
    candidate = payload.candidates[0]
    if not candidate.legal_targets and candidate.has_legal_target:
        return ()
    return tuple(dict.fromkeys((
        *(e.id for e in record.state.entities), *(p.id for p in record.state.players),
        *candidate.legal_targets)))


def _decision_legality(record, payload: PlayabilityDecisionPayload, slot) -> None:
    """``target_legal`` at :func:`target_legality_subjects`: 1 in the
    candidate's ``legal_targets`` (the union over its targeting clauses), 0
    elsewhere."""
    subjects = target_legality_subjects(record)
    if not subjects:
        return
    legal = set(payload.candidates[0].legal_targets)
    for ref in subjects:
        slot(ref).set(HeadField.TARGET_LEGAL, 1 if ref in legal else 0)


def _attackers_legality(record, payload: PlayabilityAttackersPayload, slot) -> None:
    """The declaration's legal and forbidden attackers; on an anchored record
    which players, planeswalkers and battles the anchor may attack, whether
    it must, and on a declared anchor the tap the attack made (FR-037)."""
    _mark_legality(slot, HeadField.ATTACKER_LEGAL, payload.legal_attackers, payload)
    if payload.anchor_attacker is None:
        return
    defenders = set(payload.legal_defenders)
    for ref in _attackable(record):
        slot(ref).set(HeadField.ATTACK_TARGET_LEGAL, 1 if ref in defenders else 0)
    entry = slot(payload.anchor_attacker)
    entry.set(HeadField.COMBAT_REQUIREMENT, 1 if payload.must_attack else 0)
    if payload.tapped_by_attack is not None:
        _tap_by_attack(record, payload, entry)


#: The permanent types a creature may attack (508.1b).
_ATTACKABLE_TYPES = frozenset({"planeswalker", "battle"})


def _attackable(record: EffectRecord) -> list[str]:
    """Every player, planeswalker and battle of the snapshot, the players
    first: what an attack may be declared at, legal for the anchor or not."""
    return [player.id for player in record.state.players] + [
        entity.id for entity in record.state.entities if _is_attackable(entity)
    ]


def _is_attackable(entity) -> bool:
    """A planeswalker or a battle on the battlefield; a phased-out one is no
    defender at all."""
    return entity.on_battlefield and bool(
        _ATTACKABLE_TYPES & {kind.lower() for kind in entity.types})


def _tap_by_attack(record: EffectRecord, payload: PlayabilityAttackersPayload,
                   entry: EntityTargets) -> None:
    """The tap the attack made: what the declaration did to the anchor, so
    the anchor is affected; the record trains no gate, so that is all it
    says."""
    anchor = payload.anchor_attacker
    entity = next((e for e in record.state.entities if e.id == anchor), None)
    before = 1 if entity is not None and entity.tapped else 0
    entry.affected = True
    entry.set(HeadField.TAP_STATE, 1 if payload.tapped_by_attack else before)


def _blockers_legality(payload: PlayabilityBlockersPayload, slot) -> None:
    """The declaration's legal and forbidden blockers, the anchor's
    ``min_blockers``, and which listed creatures a requirement makes block
    the anchor (509.1c, a must-block requirement), where the record says."""
    _mark_legality(slot, HeadField.BLOCKER_LEGAL, payload.legal_blockers, payload)
    # min_blockers is read at the anchored attacker, not at the blockers.
    slot(payload.anchor_attacker).set(HeadField.MIN_BLOCKERS, payload.min_blockers)
    if payload.must_block is not None:
        must = set(payload.must_block)
        for listed in (*payload.legal_blockers, *(f.entity for f in payload.forbidden)):
            slot(listed).set(HeadField.COMBAT_REQUIREMENT, 1 if listed in must else 0)


def _mark_legality(slot, name: str, legal, payload) -> None:
    """One combat declaration's legal entities as 1, its forbidden ones as 0."""
    for entity in legal:
        slot(entity).set(name, 1)
    for forbidden in payload.forbidden:
        slot(forbidden.entity).set(name, 0)
