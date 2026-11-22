"""The outcome signature of a record: the distinct ``(zone outcome, changed)``
pairs over its entities, which splits a sampling class's family into cells
(FR-049 step 2).
"""

from __future__ import annotations

from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField, ZoneOutcome
from effects.domain.effect_targets import derive_targets
from effects.domain.records import RecordKind, ResolutionOutcome
from effects.domain.sampling import SamplingClass
from effects.domain.state_snapshot import Zone

#: The sampling classes whose families split further by outcome signature
#: (FR-049 step 2); the others have no per-entity outcome to sort by.
SIGNATURE_CLASSES: frozenset[SamplingClass] = frozenset({
    SamplingClass.RESOLUTION_EFFECT, SamplingClass.REWRITE, SamplingClass.COMBAT,
    SamplingClass.CONTINUOUS,
})


#: The signature's zone part where a record's targets leave the zone outcome unknown.
UNKNOWN_ZONE_OUTCOME = "unknown"
#: A spell card's own move off the stack, as its zone outcome.
_OFF_THE_STACK = frozenset(ZONE_OUTCOMES.index(o) for o in (ZoneOutcome.DIED, ZoneOutcome.EXILED))


def _spell_on_the_stack(record) -> str | None:
    """The acting source of a resolution's half when it is a card on the stack."""
    if record.kind is not RecordKind.RESOLUTION:
        return None
    source = record.state.refs.source
    entity = record.state.entity(source) if source is not None else None
    return source if entity is not None and entity.zone == Zone.STACK else None


def outcome_signature(record) -> frozenset[tuple[str, bool]]:
    """The distinct ``(zone outcome, changed)`` pairs over a record's entities,
    from the targets the head trains on.

    A set, not a multiset, so a wipe of three and a wipe of seven share a
    signature rather than splitting one rule by board size (FR-049 step 2).
    An entity whose zone outcome is unknown (a move to a destination no
    outcome names: the stack, the sideboard, merged) reads
    :data:`UNKNOWN_ZONE_OUTCOME`, never stayed. The acting spell's own move
    off the stack to a graveyard or exile, which every resolved instant's
    half ends with and a fizzled one holds alone (608.2b), is no outcome; a
    half that did not resolve names its outcome instead (``fizzled``,
    ``declined``, ``partially_fizzled``), so a fizzle never shares a cell
    with a kill.
    """
    pairs = set()
    spell = _spell_on_the_stack(record)
    outcome_of_half = getattr(record.payload, "outcome", None)
    if outcome_of_half not in (None, ResolutionOutcome.RESOLVED):
        pairs.add((str(outcome_of_half), False))
    for subject, entry in derive_targets(record).items():
        zone = entry.fields.get(HeadField.ZONE_OUTCOME)
        if subject == spell and zone in _OFF_THE_STACK:
            continue
        if zone is not None:
            outcome = ZONE_OUTCOMES[zone]
        elif HeadField.ZONE_OUTCOME in entry.unknown:
            outcome = UNKNOWN_ZONE_OUTCOME
        else:
            outcome = ZoneOutcome.STAYED
        pairs.add((outcome, entry.affected))
    return frozenset(pairs)


def signature_label(signature: frozenset[tuple[str, bool]]) -> str:
    """A signature as one stable string, for a cell key and the manifest."""
    if not signature:
        return "(empty)"
    return ",".join(sorted(
        f"{zone}:{'changed' if changed else 'unchanged'}" for zone, changed in signature
    ))
