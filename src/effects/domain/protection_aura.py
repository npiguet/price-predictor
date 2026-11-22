"""The Aura-put-into-the-graveyard outcome of a protection copy (702.16c): which
Aura spells a protection on their one target makes fizzle.

Torch-free: ``build-corpus``'s survey and write workers call it.
"""

from __future__ import annotations

from effects.domain.protection_model import (
    Opportunity,
    ProtectionOutcome,
    entering,
    is_aura,
    permanent,
    rare_qualities,
)
from effects.domain.records import ResolutionPayload
from effects.domain.state_snapshot import EntityState, Zone


def aura_opportunity(record, payload: ResolutionPayload, entities) -> Opportunity | None:
    """An Aura spell resolving onto one permanent, its whole record that move."""
    refs = record.state.refs
    aura = entities.get(refs.source)
    if not _is_resolving_aura(aura, refs):
        return None
    if _bestowed(aura):
        return None  # 702.103e: it would resolve as a creature, not fizzle
    target = refs.as_cast[0]
    if permanent(entities, target) is None:
        return None
    if any(set(event.subjects) - {aura.id, target} for event in payload.events):
        return None
    if entering(payload.events, aura.id) is None:
        return None
    return Opportunity(ProtectionOutcome.AURA_TO_GRAVEYARD, target, (aura.id,),
                       rare_qualities(aura))


def _is_resolving_aura(aura: EntityState | None, refs) -> bool:
    """Whether ``aura`` is an Aura spell on the stack with one target as cast."""
    if aura is None or aura.zone != Zone.STACK:
        return False
    return is_aura(aura) and len(refs.as_cast) == 1


def _bestowed(aura: EntityState) -> bool:
    """Whether the Aura spell carries bestow, by a keyword line or a grant."""
    lines = aura.lines
    return ("bestow" in {k.lower() for k in aura.granted_temporary.keywords}
            or any((line.fragment or "").lower().startswith("bestow") for line in lines))
