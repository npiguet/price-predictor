"""Rule-labelled copies of real records granting a rare protection quality (FR-058).

Protection from a colour is common in the corpus; protection from a creature
type, a card type, a condition or everything is not, so a model sees too few
records to learn what those qualities stop. A copy takes a real record, grants
one object such a quality, and rewrites the payload by rule 702.16, so the
labels :func:`effects.domain.effect_targets.derive_targets` reads off the copy
are the rule's (research R5). Written into the corpus by ``build-corpus``'s
write pass, beside the record's natural copies, in the numbers the rule budgets
draw for the protection rules; each carries ``synthetic = true``, its drawn rule
and its curation.

Two steps, as in :mod:`effects.domain.lethality_counterfactuals`: a pure
qualifier (:func:`qualify`) names what a record could carry, then
:func:`protection_copy` draws from a seeded generator and makes the copy. Four
outcomes, each copied only where nothing but the protection changes what the
record shows:

- **damage prevented** (702.16e): a ``resolution`` effect half or a ``combat``
  record dealing damage to a permanent nothing else in the record touches but
  its own death; the protection covers every source that damaged it, and the
  permanent is not a target of the acting ability (a targeted spell would lose
  its target instead), nor can it be: the acting chain targets nowhere but its
  root, whose targets ``refs.targets`` holds. Each ``damage_dealt`` becomes the ``damage_prevented``
  Forge writes for a prevented hit, and a death the damage caused is undone. A
  source with lifelink, infect or wither is refused: what its damage does
  besides marking damage would have to be undone too;
- **target made illegal** (702.16b): a decision whose candidate has at least
  two legal targets loses one permanent from them, the verdict unchanged; the
  candidate acts through the first face its card shows (no Adventure, back
  face or split half, whose characteristics the snapshot does not hold);
- **block made illegal** (702.16f): a real blockers record's anchored
  attacker gains protection from a quality one of its legal blockers has, and
  every legal blocker with the quality becomes forbidden;
- **Aura put into the graveyard** (702.16c): an Aura spell's resolution whose
  one target gains protection from the Aura; the spell has no legal target
  and fizzles (608.2b, ``outcome`` ``fizzled``), so the Aura goes from the
  stack to the graveyard and nothing it would have done happens.

Protection from everything (702.16j) covers every outcome; no quality is
granted that an Aura or Equipment attached to the protected permanent has,
which would make it fall off (702.16c-d), and no protection from everything
then. The rare qualities of the other rules are those of the covered object:
its card types, its creature types where it is a creature or kindred, and the
conditions it meets.
The quality is granted the way a resolved effect grants a keyword: a granted
keyword line plus its kept flag (research M5,
:meth:`~effects.domain.state_snapshot.EntityState.with_granted_keyword`).

Only natural records qualify: no fork, intervention, directed fork or other
synthetic record.

Torch-free: ``build-corpus``'s survey and write workers call it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import StrEnum

from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import SidecarLookup
from effects.domain.records import (
    CombatPayload,
    EffectRecord,
    ForbiddenEntity,
    Moment,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    ResolutionOutcome,
    ResolutionPayload,
)
from effects.domain.rule_outcomes import (
    CARD_TYPES,
    CONDITIONS,
    Characteristics,
    ProtectionQuality,
    QualityKind,
    damage_lethal,
    damage_prevented,
    may_block,
    may_enchant,
    may_target,
)
from effects.domain.script_text import params, segments
from effects.domain.state_snapshot import EntityState, FaceState, Zone


class ProtectionOutcome(StrEnum):
    """What a granted protection changes in a copied record."""

    DAMAGE_PREVENTED = "damage_prevented"
    TARGET_ILLEGAL = "target_illegal"
    BLOCK_ILLEGAL = "block_illegal"
    AURA_TO_GRAVEYARD = "aura_to_graveyard"


#: The rule each outcome shows, from a rare quality other than everything.
OUTCOME_RULES: dict[ProtectionOutcome, str] = {
    ProtectionOutcome.DAMAGE_PREVENTED: "702.16e",
    ProtectionOutcome.TARGET_ILLEGAL: "702.16b",
    ProtectionOutcome.BLOCK_ILLEGAL: "702.16f",
    ProtectionOutcome.AURA_TO_GRAVEYARD: "702.16c",
}
#: Protection from everything, which shows any of the four.
EVERYTHING_RULE = "702.16j"
PROTECTION_RULES: frozenset[str] = frozenset({*OUTCOME_RULES.values(), EVERYTHING_RULE})

#: Source keywords whose damage does more than mark damage: a copy preventing
#: it would have to undo the life gained, the counters or the poison too.
SOURCE_EXCLUSIONS = frozenset({"lifelink", "infect", "wither"})


@dataclass(frozen=True, slots=True)
class Opportunity:
    """One way a record can carry a protection copy."""

    outcome: ProtectionOutcome
    #: The entity granted the protection.
    protected: str
    #: The objects the quality must cover: the damage's sources, the spell or
    #: ability's card, the blocker, the Aura.
    covered: tuple[str, ...]
    #: The rare qualities every covered object has, everything left out; empty
    #: where they share none, which leaves the record to protection from
    #: everything alone.
    qualities: tuple[ProtectionQuality, ...]
    #: Whether protection from everything shows the outcome: not where an
    #: Aura or Equipment is attached to the protected permanent (702.16c-d).
    everything: bool = True

    @property
    def rule(self) -> str:
        return OUTCOME_RULES[self.outcome]


@dataclass(frozen=True, slots=True)
class ProtectionCopy:
    """One copy: the edited record and what the write pass records about it."""

    record: EffectRecord
    #: The drawn rule the copy carries in its curation.
    rule: str
    quality: ProtectionQuality
    outcome: ProtectionOutcome
    protected: str
    #: The curation's ``outcome_entities``: the entities whose outputs the
    #: rule changed. The protected permanent for prevented damage and an
    #: illegal target, the blockers made illegal for a block, the Aura and its
    #: target for an Aura put into the graveyard. The condition entity is the
    #: protected permanent, which carries the protection.
    outcome_entities: tuple[str, ...]

    @property
    def condition_entities(self) -> tuple[str, ...]:
        return (self.protected,)


def rare_qualities(entity: EntityState) -> tuple[ProtectionQuality, ...]:
    """The rare qualities ``entity`` has, in a fixed order: its card types, its
    creature types (on a creature or a kindred card), the conditions it meets."""
    characteristics = Characteristics.of(entity)
    found = [ProtectionQuality(QualityKind.CARD_TYPE, t)
             for t in CARD_TYPES if t in characteristics.types]
    if characteristics.types & {"creature", "kindred"}:
        found += [ProtectionQuality(QualityKind.CREATURE_TYPE, s)
                  for s in sorted(characteristics.subtypes)]
    found += [ProtectionQuality(QualityKind.CONDITION, name)
              for name, condition in CONDITIONS.items() if condition.holds(characteristics)]
    return tuple(found)


def rules_offered(opportunities: tuple[Opportunity, ...]) -> tuple[str, ...]:
    """The protection rules a record's copies can be drawn for, sorted: each
    outcome's rule where a rare quality covers its objects, and 702.16j
    wherever any outcome is open."""
    rules = {o.rule for o in opportunities if o.qualities}
    if any(o.everything for o in opportunities):
        rules.add(EVERYTHING_RULE)
    return tuple(sorted(rules))


# ── the qualifier ────────────────────────────────────────────────────────


def qualify(record: EffectRecord, lines: SidecarLookup | None = None
            ) -> tuple[Opportunity, ...]:
    """Every protection copy ``record`` can carry, in record order; empty for none.

    ``lines`` reads a resolution's acting chain: ``refs.targets`` holds the
    root's targets alone, so a resolution whose chain targets anywhere else (a
    sub-ability's ``ValidTgts$``, a charm mode) offers no damage-prevented
    copy, a protected target there losing the target rather than the damage.
    Without ``lines`` the chain is not read (a unit test's record).
    """
    if record.fork or record.interventional or record.synthetic or record.directed is not None:
        return ()
    entities = {entity.id: entity for entity in record.state.entities}
    return tuple(_keeping_attachments(o, record.state.entities)
                 for o in _opportunities(record, lines, entities))


def _keeping_attachments(opportunity: Opportunity, entities) -> Opportunity:
    """``opportunity`` without the qualities an object attached to the
    protected permanent has: a protection from it would make the Aura or
    Equipment fall off (702.16c-d), which the copy does not show."""
    attached = [Characteristics.of(e) for e in entities
                if e.attached_to == opportunity.protected]
    if not attached:
        return opportunity
    return replace(opportunity, everything=False, qualities=tuple(
        q for q in opportunity.qualities if not any(q.covers(a) for a in attached)))


def _opportunities(record: EffectRecord, lines: SidecarLookup | None,
                   entities) -> tuple[Opportunity, ...]:
    payload = record.payload
    if isinstance(payload, PlayabilityDecisionPayload):
        return _target_opportunities(record, payload, entities)
    if isinstance(payload, PlayabilityBlockersPayload):
        return _block_opportunities(record, payload, entities)
    if isinstance(payload, CombatPayload):
        return _damage_opportunities(payload.events, entities, targets=())
    if isinstance(payload, ResolutionPayload) and record.moment is Moment.RESOLUTION:
        aura = _aura_opportunity(record, payload, entities)
        if aura is not None:
            return (aura,)
        if lines is not None and _targets_beyond_its_root(record, lines):
            return ()
        return _damage_opportunities(payload.events, entities, targets=record.state.refs.targets)
    return ()


#: Parameters that make a chain segment choose targets of its own.
_TARGETING_PARAMS = frozenset({"ValidTgts", "TargetType", "TgtPrompt", "TargetMin", "TargetMax"})


def _targets_beyond_its_root(record: EffectRecord, lines: SidecarLookup) -> bool:
    """Whether the acting chain may target what ``refs.targets`` does not hold:
    a charm mode (whose targets are not traced there), a sub-ability with
    targets of its own, or an acting line the sidecars cannot show."""
    keys = record.ability or ()
    if any(key.option is not None for key in keys):
        return True
    line = lines.first_line(keys)
    if line is None or not line.script_text:
        return bool(keys)
    return any(_TARGETING_PARAMS & params(segment).keys()
               for segment in segments(line.script_text)[1:])


def _permanent(entities: dict[str, EntityState], ref: str | None) -> EntityState | None:
    entity = entities.get(ref) if ref is not None else None
    return entity if entity is not None and entity.on_battlefield else None


def _target_opportunities(record, payload: PlayabilityDecisionPayload,
                          entities) -> tuple[Opportunity, ...]:
    """A decision's legal targets, one of them made illegal while another stays."""
    if not payload.candidates:
        return ()
    candidate = payload.candidates[0]
    source = entities.get(record.state.refs.source)
    if candidate.land_play or source is None or len(candidate.legal_targets) < 2:
        return ()
    if not _shows_the_acting_face(source, candidate.ability or ()):
        return ()
    qualities = rare_qualities(source)
    return tuple(
        Opportunity(ProtectionOutcome.TARGET_ILLEGAL, target, (source.id,), qualities)
        for target in candidate.legal_targets if _permanent(entities, target) is not None
    )


def _shows_the_acting_face(source: EntityState, keys) -> bool:
    """Whether the card's snapshot holds the characteristics of what acts:
    its first face, shown face up, and not a split card, whose card in hand
    shows both halves at once. An Adventure or a back face cast from hand has
    characteristics the snapshot does not hold (Bonecrusher Giant shows the
    Giant while Stomp is cast)."""
    if source.face_state not in (None, FaceState.FRONT) or " // " in (source.name or ""):
        return False
    return all(key.face == 0 for key in keys)


def _block_opportunities(record, payload: PlayabilityBlockersPayload,
                         entities) -> tuple[Opportunity, ...]:
    """A real block declaration's attacker, protected from one legal blocker."""
    if record.what_if or _permanent(entities, payload.anchor_attacker) is None:
        return ()
    return tuple(
        Opportunity(ProtectionOutcome.BLOCK_ILLEGAL, payload.anchor_attacker, (blocker,),
                    rare_qualities(entities[blocker]))
        for blocker in payload.legal_blockers if _permanent(entities, blocker) is not None
    )


def _damage_opportunities(events: tuple[Event, ...], entities,
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
            _shared(entities[source] for source in sources),
        ))
    return tuple(found)


def _source_of(event: Event) -> str | None:
    return event.params.get("source") or event.params.get("cause")


def _is_death(event: Event, subject: str) -> bool:
    return (event.type is EventType.ZONE_CHANGE and event.subjects == (subject,)
            and str(event.params.get("from_zone", "")).lower() == Zone.BATTLEFIELD
            and str(event.params.get("to_zone", "")).lower() == Zone.GRAVEYARD)


def _prevented_sources(subject: str, events: tuple[Event, ...], entities,
                       targets: tuple[str, ...]) -> tuple[str, ...] | None:
    """The sources that damaged ``subject``, or None where preventing their
    damage would not be all that changes."""
    entity = _permanent(entities, subject)
    if entity is None or subject in targets or entity.pt is None:
        return None
    sources: list[str] = []
    dealt = 0
    deathtouch = died = False
    for event in events:
        if subject not in event.subjects:
            continue
        if event.type is EventType.DAMAGE_DEALT and event.subjects == (subject,):
            source = entities.get(_source_of(event))
            keywords = set(source.granted_temporary.keywords) if source is not None else set()
            if source is None or keywords & SOURCE_EXCLUSIONS:
                return None
            sources.append(source.id)
            dealt += int(event.params.get("amount", 0) or 0)
            deathtouch = deathtouch or "deathtouch" in keywords
        elif _is_death(event, subject) and not died:
            died = True
        else:
            return None
    if died and (damage_lethal(entity.toughness, entity.damage, 0)
                 or not (deathtouch or damage_lethal(entity.toughness, entity.damage, dealt))):
        return None  # it died of something besides the damage
    return tuple(dict.fromkeys(sources))


def _aura_opportunity(record, payload: ResolutionPayload, entities) -> Opportunity | None:
    """An Aura spell resolving onto one permanent, its whole record that move."""
    refs = record.state.refs
    aura = entities.get(refs.source)
    if (aura is None or aura.zone != Zone.STACK
            or "aura" not in {s.lower() for s in aura.subtypes} or len(refs.targets) != 1):
        return None
    target = refs.targets[0]
    if _permanent(entities, target) is None:
        return None
    if any(set(event.subjects) - {aura.id, target} for event in payload.events):
        return None
    if _entering(payload.events, aura.id) is None:
        return None
    return Opportunity(ProtectionOutcome.AURA_TO_GRAVEYARD, target, (aura.id,),
                       rare_qualities(aura))


def _entering(events: tuple[Event, ...], aura: str) -> Event | None:
    return next((event for event in events
                 if event.type is EventType.ZONE_CHANGE and event.subjects == (aura,)
                 and str(event.params.get("to_zone", "")).lower() == Zone.BATTLEFIELD), None)


def _shared(objects) -> tuple[ProtectionQuality, ...]:
    """The rare qualities every one of ``objects`` has, in the first's order."""
    lists = [rare_qualities(entity) for entity in objects]
    if not lists:
        return ()
    rest = [set(qualities) for qualities in lists[1:]]
    return tuple(q for q in lists[0] if all(q in other for other in rest))


# ── the copy ─────────────────────────────────────────────────────────────


def protection_copy(record: EffectRecord, opportunities: tuple[Opportunity, ...], *,
                    rule: str, rng: random.Random, index: int) -> ProtectionCopy | None:
    """One copy of ``record`` drawn for ``rule``, or None where no opportunity
    shows it.

    ``rng`` draws the opportunity, then the quality: protection from
    everything for 702.16j, else one of the opportunity's rare qualities.
    ``index`` numbers the copies of one record, which keeps their ids apart.
    """
    if rule == EVERYTHING_RULE:
        pool = [o for o in opportunities if o.everything]
    else:
        pool = [o for o in opportunities if o.rule == rule and o.qualities]
    if not pool:
        return None
    opportunity = rng.choice(pool)
    quality = (ProtectionQuality.everything() if rule == EVERYTHING_RULE
               else rng.choice(opportunity.qualities))

    record, outcome_entities = copied(record, opportunity, quality, index)
    return ProtectionCopy(
        record=record, rule=rule, quality=quality, outcome=opportunity.outcome,
        protected=opportunity.protected, outcome_entities=outcome_entities,
    )


def copied(record: EffectRecord, opportunity: Opportunity, quality: ProtectionQuality,
           index: int) -> tuple[EffectRecord, tuple[str, ...]]:
    """``record`` with ``opportunity.protected`` granted ``quality`` and its
    payload rewritten by rule 702.16, and the entities whose outputs the rule
    changed."""
    entities = tuple(
        entity.with_granted_keyword(quality.keyword, quality.fragment)
        if entity.id == opportunity.protected else entity
        for entity in record.state.entities
    )
    by_id = {entity.id: entity for entity in entities}
    payload, named = _REWRITES[opportunity.outcome](record.payload, opportunity, quality, by_id)
    return replace(
        record,
        record_id=f"{record.record_id}~pc{index}",
        synthetic=True,
        state=replace(record.state, entities=entities),
        payload=payload,
    ), named


def _prevent(payload, opportunity: Opportunity, quality: ProtectionQuality, entities):
    """Each damage a source with the quality dealt the protected permanent is
    prevented (702.16e); once every one is, the death it caused is undone."""
    protected = opportunity.protected
    events = []
    for event in payload.events:
        source = entities.get(_source_of(event))
        if (event.type is EventType.DAMAGE_DEALT and event.subjects == (protected,)
                and source is not None
                and damage_prevented((quality,), Characteristics.of(source))):
            params = {"amount": event.params.get("amount", 0), "source": source.id}
            if "cause" in event.params:
                params["cause"] = event.params["cause"]
            event = replace(event, type=EventType.DAMAGE_PREVENTED, params=params)
        events.append(event)
    if not any(e.type is EventType.DAMAGE_DEALT and e.subjects == (protected,) for e in events):
        events = [e for e in events if not _is_death(e, protected)]
    return replace(payload, events=tuple(events)), (protected,)


def _untarget(payload, opportunity: Opportunity, quality: ProtectionQuality, entities):
    """The protected permanent leaves the candidate's legal targets (702.16b)."""
    candidate = payload.candidates[0]
    source = Characteristics.of(entities[opportunity.covered[0]])
    if may_target((quality,), source):
        return payload, ()
    legal = tuple(t for t in candidate.legal_targets if t != opportunity.protected)
    return replace(payload, candidates=(
        replace(candidate, legal_targets=legal), *payload.candidates[1:],
    )), (opportunity.protected,)


def _forbid(payload, opportunity: Opportunity, quality: ProtectionQuality, entities):
    """Every legal blocker with the quality becomes forbidden (702.16f)."""
    attacker = entities[payload.anchor_attacker]
    forbidden = []
    for blocker_id in payload.legal_blockers:
        blocker = entities.get(blocker_id)
        if blocker is not None and not may_block(
                attacker.granted_temporary.keywords, attacker.colors,
                blocker.granted_temporary.keywords, blocker.colors,
                {t.lower() for t in blocker.types}, attacker_protections=(quality,),
                blocker_subtypes={s.lower() for s in blocker.subtypes},
                blocker_mana_value=blocker.mana_value):
            forbidden.append(blocker_id)
    gone = set(forbidden)
    must_block = payload.must_block
    return replace(
        payload,
        legal_blockers=tuple(b for b in payload.legal_blockers if b not in gone),
        forbidden=(*payload.forbidden, *(ForbiddenEntity(entity=b) for b in forbidden)),
        must_block=None if must_block is None else tuple(b for b in must_block if b not in gone),
    ), tuple(forbidden)


def _fizzle(payload, opportunity: Opportunity, quality: ProtectionQuality, entities):
    """The Aura spell has no legal target left, so it fizzles (rule 608.2b):
    its half's ``outcome`` is ``fizzled`` and its one event the card's move
    from the stack to its owner's graveyard, which is 702.16c's outcome; it
    does nothing else. The move is kept although a collected fizzled half
    writes ``events: []``: the copy exists to show it."""
    aura = opportunity.covered[0]
    if may_enchant((quality,), Characteristics.of(entities[aura])):
        return payload, ()
    moved = replace(_entering(payload.events, aura), params={
        "from_zone": Zone.STACK.value, "to_zone": Zone.GRAVEYARD.value, "cause": aura,
    })
    conditions = None if payload.conditions_met is None else ()
    return replace(payload, events=(moved,), conditions_met=conditions,
                   outcome=ResolutionOutcome.FIZZLED, outcome_written=True), (
        aura, opportunity.protected)


_REWRITES = {
    ProtectionOutcome.DAMAGE_PREVENTED: _prevent,
    ProtectionOutcome.TARGET_ILLEGAL: _untarget,
    ProtectionOutcome.BLOCK_ILLEGAL: _forbid,
    ProtectionOutcome.AURA_TO_GRAVEYARD: _fizzle,
}
