"""The qualifier of the protection copies (FR-058): what a record could carry.

A pure function of the record and, where given, its sidecars; the copy itself
is :func:`effects.domain.protection_copies.protection_copy`.

Torch-free: ``build-corpus``'s survey and write workers call it.
"""

from __future__ import annotations

from dataclasses import replace

from effects.domain.protection_aura import aura_opportunity
from effects.domain.protection_chain import no_prevention_on_the_board, refuses_damage
from effects.domain.protection_damage import damage_opportunities
from effects.domain.protection_model import (
    Opportunity,
)
from effects.domain.protection_playability import block_opportunities, target_opportunities
from effects.domain.provenance import SidecarLookup
from effects.domain.records import (
    CombatPayload,
    EffectRecord,
    Moment,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    ResolutionPayload,
)
from effects.domain.rule_outcomes import Characteristics


def qualify(record: EffectRecord, lines: SidecarLookup | None = None
            ) -> tuple[Opportunity, ...]:
    """Every protection copy ``record`` can carry, in record order; empty for none.

    ``lines`` reads the sidecars: a resolution's acting chain (``NoPrevention$``,
    a later clause whose amount reads the damage dealt) and the statics on the
    board (an active ``Mode$ CantPreventDamage``). Without ``lines`` neither is
    read (a unit test's record).
    """
    if _not_natural(record):
        return ()
    entities = {entity.id: entity for entity in record.state.entities}
    return tuple(_keeping_attachments(o, record.state.entities)
                 for o in _opportunities(record, lines, entities))


def _not_natural(record: EffectRecord) -> bool:
    """Whether ``record`` is a fork, intervention, directed fork or other
    synthetic record."""
    return bool(record.fork or record.interventional or record.synthetic
                or record.directed is not None)


def _keeping_attachments(opportunity: Opportunity, entities) -> Opportunity:
    """``opportunity`` without the qualities an object attached to the
    protected permanent has: a protection from it would make the Aura or
    Equipment fall off (702.16c-d), which the copy does not show."""
    attached = [Characteristics.of(e) for e in entities
                if e.attached_to == opportunity.protected and e.on_battlefield]
    if not attached:
        return opportunity
    return replace(opportunity, everything=False, qualities=tuple(
        q for q in opportunity.qualities if not any(q.covers(a) for a in attached)))


def _opportunities(record: EffectRecord, lines: SidecarLookup | None,
                   entities) -> tuple[Opportunity, ...]:
    payload = record.payload
    if isinstance(payload, PlayabilityDecisionPayload):
        return target_opportunities(record, payload, entities, lines)
    if isinstance(payload, PlayabilityBlockersPayload):
        return block_opportunities(record, payload, entities)
    if isinstance(payload, CombatPayload):
        return _combat_opportunities(record, payload, entities, lines)
    if isinstance(payload, ResolutionPayload) and record.moment is Moment.RESOLUTION:
        return _resolution_opportunities(record, payload, entities, lines)
    return ()


def _combat_opportunities(record: EffectRecord, payload: CombatPayload, entities,
                          lines: SidecarLookup | None) -> tuple[Opportunity, ...]:
    if lines is not None and no_prevention_on_the_board(record, lines):
        return ()
    return damage_opportunities(payload.events, entities, targets=())


def _resolution_opportunities(record: EffectRecord, payload: ResolutionPayload, entities,
                              lines: SidecarLookup | None) -> tuple[Opportunity, ...]:
    aura = aura_opportunity(record, payload, entities)
    if aura is not None:
        return (aura,)
    refs = record.state.refs
    if refs.targets_at_cast is None:
        # What the record cannot show: the targets before Forge's
        # illegal-target check (a worker without the resolution hook).
        return ()
    if lines is not None and refuses_damage(record, lines):
        return ()
    # Every target of the acting chain, a sub-ability's and a mode's own
    # included, as cast and after the check.
    return damage_opportunities(
        payload.events, entities, targets=(*refs.targets_at_cast, *refs.targets))
