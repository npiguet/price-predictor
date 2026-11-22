"""The damage-prevented outcome of a protection copy (702.16e): which permanents
a protection would shield, and the rewrite of their damage events.

Torch-free: ``build-corpus``'s survey and write workers call it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from effects.domain.event_schema import Event, EventType
from effects.domain.protection_model import (
    Opportunity,
    ProtectionOutcome,
    is_aura,
    permanent,
    shared_qualities,
)
from effects.domain.rule_outcomes import (
    Characteristics,
    ProtectionQuality,
    damage_lethal,
    damage_prevented,
)
from effects.domain.state_snapshot import EntityState, Zone

#: Source keywords whose damage does more than mark damage: a copy preventing
#: it would have to undo the life gained, the counters or the poison too.
SOURCE_EXCLUSIONS = frozenset({"lifelink", "infect", "wither"})


def damage_opportunities(events: tuple[Event, ...], entities,
                         targets: tuple[str, ...]) -> tuple[Opportunity, ...]:
    """Each permanent the record damaged whose damage alone a protection undoes."""
    found = []
    for subject in dict.fromkeys(
            s for event in events if event.type is EventType.DAMAGE_DEALT for s in event.subjects):
        sources = _prevented_sources(subject, events, entities, targets)
        if sources is None:
            continue
        found.append(Opportunity(
            ProtectionOutcome.DAMAGE_PREVENTED, subject, sources,
            shared_qualities(entities[source] for source in sources),
        ))
    return tuple(found)


def _source_of(event: Event) -> str | None:
    return event.params.get("source") or event.params.get("cause")


def _is_damage_to(event: Event, subject: str) -> bool:
    return event.type is EventType.DAMAGE_DEALT and event.subjects == (subject,)


def _is_death(event: Event, subject: str) -> bool:
    return (event.type is EventType.ZONE_CHANGE and event.subjects == (subject,)
            and str(event.params.get("from_zone", "")).lower() == Zone.BATTLEFIELD
            and str(event.params.get("to_zone", "")).lower() == Zone.GRAVEYARD)


def _is_destroyed(event: Event, subject: str) -> bool:
    return event.type is EventType.DESTROYED and event.subjects == (subject,)


def _death_is_the_damages(entity: EntityState, dealt: int, deathtouch: bool) -> bool:
    """Whether ``dealt`` damage alone kills ``entity``: not already lethal
    before it, and lethal with it or any deathtouch."""
    return not damage_lethal(entity.toughness, entity.damage, 0) and (
        deathtouch or damage_lethal(entity.toughness, entity.damage, dealt))


def _keywords(entity: EntityState) -> set[str]:
    return set(entity.granted_temporary.keywords)


@dataclass(slots=True)
class _Hits:
    """What one permanent's events in a record show of the damage it took."""

    sources: list[str] = field(default_factory=list)
    dealt: int = 0
    deathtouch: bool = False
    died: bool = False
    destroyed: bool = False

    @property
    def distinct_sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.sources))

    def take(self, event: Event, subject: str, entities) -> bool:
        """Count ``event`` among the hits; False where it is none the copy can
        show (another kind of event, a source doing more than marking damage,
        a second death)."""
        if _is_damage_to(event, subject):
            return self._take_damage(event, entities)
        if _is_death(event, subject) and not self.died:
            self.died = True
            return True
        if _is_destroyed(event, subject) and not self.destroyed:
            self.destroyed = True  # every real lethal-damage death carries one
            return True
        return False

    def _take_damage(self, event: Event, entities) -> bool:
        source = entities.get(_source_of(event))
        if source is None or _keywords(source) & SOURCE_EXCLUSIONS:
            return False
        self.sources.append(source.id)
        self.dealt += int(event.params.get("amount", 0) or 0)
        self.deathtouch = self.deathtouch or "deathtouch" in _keywords(source)
        return True


def _hits(subject: str, events: tuple[Event, ...], entities) -> _Hits | None:
    """``subject``'s damage events and its death, or None where any other
    event touches it or a source does more than mark damage."""
    hits = _Hits()
    for event in events:
        if subject in event.subjects and not hits.take(event, subject, entities):
            return None
    return hits


def _unshieldable(entity: EntityState, subject: str, targets: tuple[str, ...]) -> bool:
    """Whether the acting chain targets ``subject``, or it has no power and
    toughness to take damage with."""
    return subject in targets or entity.pt is None


def _prevented_sources(subject: str, events: tuple[Event, ...], entities,
                       targets: tuple[str, ...]) -> tuple[str, ...] | None:
    """The sources that damaged ``subject``, or None where preventing their
    damage would not be all that changes."""
    entity = permanent(entities, subject)
    if entity is None or _unshieldable(entity, subject, targets):
        return None
    hits = _hits(subject, events, entities)
    if hits is None or _death_unexplained(entity, hits):
        return None
    return hits.distinct_sources


def _death_unexplained(entity: EntityState, hits: _Hits) -> bool:
    """Whether the record's death of ``entity`` is not the damage's: a
    ``destroyed`` with no move (a regeneration the copy does not show), or a
    death the damage alone does not cause."""
    if hits.destroyed and not hits.died:
        return True
    return hits.died and not _death_is_the_damages(entity, hits.dealt, hits.deathtouch)


def prevent_damage(payload, opportunity: Opportunity, quality: ProtectionQuality, entities):
    """Each damage a source with the quality dealt the protected permanent is
    prevented (702.16e); once every one is, the death it caused is undone."""
    protected = opportunity.protected
    events = [_prevented(event, protected, quality, entities) for event in payload.events]
    if not any(_is_damage_to(e, protected) for e in events):
        events = _without_the_death(events, protected, entities)
    return replace(payload, events=tuple(events)), (protected,)


def _prevented(event: Event, protected: str, quality: ProtectionQuality, entities) -> Event:
    """``event`` as Forge writes a hit the protection prevents, else itself."""
    if not _is_damage_to(event, protected):
        return event
    source = entities.get(_source_of(event))
    if source is None or not damage_prevented((quality,), Characteristics.of(source)):
        return event
    params = {"amount": event.params.get("amount", 0), "source": source.id}
    if "cause" in event.params:
        params["cause"] = event.params["cause"]
    return replace(event, type=EventType.DAMAGE_PREVENTED, params=params)


def _without_the_death(events: list[Event], protected: str, entities) -> list[Event]:
    """``events`` without the death of ``protected``: its ``destroyed``, its
    move to the graveyard, the ``unattached`` of each Aura or Equipment that
    fell off it and the Aura's own move to the graveyard (704.5m). Left: the
    player counters (``this_turn`` is the state before the half) and every
    event of another permanent."""
    fallen = {e.subjects[0] for e in events if e.type is EventType.UNATTACHED
              and e.params.get("detached_from") == protected}
    auras = {i for i in fallen if is_aura(entities.get(i))}
    gone = (protected, *auras)
    return [e for e in events if not _is_the_death(e, protected, gone, fallen)]


def _is_the_death(event: Event, protected: str, gone: tuple[str, ...],
                  fallen: set[str]) -> bool:
    """Whether ``event`` is part of the death of ``protected`` and of the Auras
    ``gone`` with it: a ``destroyed``, a move to the graveyard, or the
    ``unattached`` of what ``fallen`` off it."""
    return (_is_destroyed(event, protected) or any(_is_death(event, i) for i in gone)
            or (event.type is EventType.UNATTACHED and event.subjects[0] in fallen))
