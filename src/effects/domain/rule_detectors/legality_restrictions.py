"""Which rule keeps an entity out of a blockers or targets legality list (D4).

``legality_membership`` without a ``restriction`` reads a list's inside and
outside whatever put an entity there, so every "can't block" rule, from
flying to a tapped creature, tagged the same records. A ``restriction``
names the one rule a declaration speaks for, decided by ``rule_outcomes``:

- on ``legal_blockers``, each listed creature against the anchor attacker:
  a :class:`BlockingRestriction` (``flying``, ``horsemanship``, ``fear``,
  ``intimidate``, ``shadow``, ``protection``, ``unblockable``), read off the
  anchor's and the blocker's keywords, colours and types
  (``rule_outcomes.blocking_restrictions``; a changeling blocker has every
  creature type, and skulk compares powers); ``tapped`` (509.1a: a blocker
  is untapped); ``landwalk``, applying where the defending player controls a
  land the attacker's landwalk line names (``Landwalk:Swamp``,
  ``Land.nonBasic``, ``Swamp.Snow``; where no line names it, a gen-2 entity
  or a chosen type, the attacker's landwalk and no other reason); and
  ``effect``, a creature no static, evasion or tap keeps out;
- on ``legal_targets``, each battlefield entity and player against each
  targeting candidate's card: a :class:`TargetingRestriction`
  (``hexproof``, ``hexproof_from``, ``shroud``, ``protection``), read off
  the entity's keywords, the candidate's card's characteristics and whether
  its controller is the target's opponent
  (``rule_outcomes.targeting_restrictions``); and ``selector``, an entity
  none of those keeps out, which the candidate's own targeting restriction
  (an enchant, an equip, a "target creature") decides.

A restriction *applies* to an entity where the rule would forbid it, and is
*in play* where the rule's keyword is there at all (the attacker flies; the
entity has hexproof). Under ``fires_when = outside`` the ``fires`` side is an
entity outside the list the restriction applies to, and the near miss an
entity inside it with the restriction in play but not applying (a flier
blocked by a creature with reach, a hexproof creature its own controller
targets). Under ``inside`` the ``fires`` side is an entity the restriction
applies to that is inside nonetheless (an effect lifting hexproof), and the
near miss one it applies to, outside. ``selector`` and ``effect`` apply to
every entity no other reason explains, and are in play everywhere.

A player's keywords are in no snapshot, so no keyword restriction reads a
player.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.keyword_names import HEXPROOF_FROM, PROTECTION_FROM
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord
from effects.domain.rule_outcomes import (
    BlockingRestriction,
    Characteristics,
    ProtectionQuality,
    QualityKind,
    TargetingRestriction,
    blocking_restrictions,
    landwalk_lands,
    targeting_restrictions,
)
from effects.domain.state_snapshot import EntityState

TAPPED, LANDWALK, EFFECT, SELECTOR, SICK = "tapped", "landwalk", "effect", "selector", "sick"
#: What keeps a creature out of the legal attackers (508.1a): tapped, or
#: summoning sick without haste.
ATTACKER_RESTRICTIONS = frozenset({TAPPED, SICK})
BLOCKER_RESTRICTIONS = frozenset({*BlockingRestriction, TAPPED, LANDWALK, EFFECT})
TARGET_RESTRICTIONS = frozenset({*TargetingRestriction, SELECTOR})
QUALITY_KINDS = frozenset(QualityKind)
JUDGED = frozenset({"permanent", "player"})

#: The attacker keyword that puts each evasion in play.
_EVASION_KEYWORD = {
    BlockingRestriction.FLYING: "flying", BlockingRestriction.HORSEMANSHIP: "horsemanship",
    BlockingRestriction.FEAR: "fear", BlockingRestriction.INTIMIDATE: "intimidate",
    BlockingRestriction.UNBLOCKABLE: "unblockable",
}


@dataclass(frozen=True, slots=True)
class Judgement:
    """One entity's standing under one restriction."""

    entity: str
    inside: bool
    applies: bool
    in_play: bool


def sides(judged: Iterable[Judgement], fires_when: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(fires, near)`` entity ids, each once, in the order judged."""
    fires, near = [], []
    for j in judged:
        if fires_when == "outside":
            if j.applies and not j.inside:
                fires.append(j.entity)
            elif j.in_play and not j.applies and j.inside:
                near.append(j.entity)
        elif j.applies:
            (fires if j.inside else near).append(j.entity)
    return tuple(dict.fromkeys(fires)), tuple(dict.fromkeys(near))


def _protections(keywords: set[str], quality: str | None) -> list[ProtectionQuality]:
    found = [q for q in map(ProtectionQuality.of_keyword, sorted(keywords)) if q is not None]
    return [q for q in found if quality is None or q.kind == quality]


def _filtered(keywords: set[str], quality: str | None) -> set[str]:
    """``keywords`` with every protection of another quality kind dropped."""
    if quality is None:
        return keywords
    kept = {str(q.keyword) for q in _protections(keywords, quality)}
    return {k for k in keywords if not k.startswith(PROTECTION_FROM) or k in kept}


def blocker_judgements(record: EffectRecord, lines: SidecarLookup | None, restriction: str,
                       quality: str | None) -> list[Judgement]:
    """Each listed creature of a blockers record against its anchor attacker."""
    payload, state = record.payload, record.state
    anchor = state.entity(payload.anchor_attacker)
    if anchor is None:
        return []
    keywords_of = KeywordResolver(lines).keywords_of
    attacker = _filtered(keywords_of(anchor), quality)
    statics = {f.entity for f in payload.forbidden if f.responsible_static}
    listed = [(e, True) for e in payload.legal_blockers]
    listed += [(f.entity, False) for f in payload.forbidden]
    found = []
    for entity_id, inside in dict.fromkeys(listed):
        blocker = state.entity(entity_id)
        if blocker is None:
            continue
        blocker_keywords = keywords_of(blocker)
        evasions = blocking_restrictions(
            attacker, anchor.colors, blocker_keywords, blocker.colors,
            {t.lower() for t in blocker.types},
            blocker_subtypes={s.lower() for s in blocker.subtypes},
            blocker_mana_value=blocker.mana_value,
            attacker_power=anchor.power if anchor.pt is not None else None,
            blocker_power=blocker.power if blocker.pt is not None else None,
            blocker_changeling="changeling" in blocker_keywords)
        explained = bool(evasions) or blocker.tapped or entity_id in statics
        walked = _walked(anchor, blocker, state) if restriction == LANDWALK else None
        found.append(_blocker(restriction, entity_id, inside, attacker, blocker_keywords,
                              evasions, blocker, explained, walked))
    return found


def _walked(anchor: EntityState, blocker: EntityState, state) -> bool | None:
    """Whether the blocker's controller controls a land the anchor's landwalk
    names (702.14c), read off its keyword lines' fragments; None where no
    line says which land (a gen-2 entity, a chosen type)."""
    selectors = [line.fragment.split(":")[1] for line in anchor.lines or ()
                 if line.fragment and line.fragment.lower().startswith("landwalk:")
                 and len(line.fragment.split(":")) > 1]
    if not selectors or any(s.split(".")[0] in ("ChosenType", "YourBasic") for s in selectors):
        return None
    lands = [e for e in state.entities if e.on_battlefield and e.controller == blocker.controller]
    return any(landwalk_lands(selector, land) for selector in selectors for land in lands)


def _blocker(restriction: str, entity_id: str, inside: bool, attacker: set[str],
             blocker_keywords: set[str], evasions, blocker: EntityState,
             explained: bool, walked: bool | None = None) -> Judgement:
    if restriction == TAPPED:
        return Judgement(entity_id, inside, blocker.tapped, True)
    if restriction == LANDWALK:
        walks = "landwalk" in attacker
        if walked is not None:
            # The land the landwalk names is on the defender's battlefield.
            return Judgement(entity_id, inside, walks and walked, walks)
        return Judgement(entity_id, inside, walks and not inside and not explained, walks)
    if restriction == EFFECT:
        return Judgement(entity_id, inside, not inside and not explained
                         and "landwalk" not in attacker, True)
    rule = BlockingRestriction(restriction)
    if rule is BlockingRestriction.SHADOW:
        in_play = "shadow" in attacker or "shadow" in blocker_keywords
    elif rule is BlockingRestriction.PROTECTION:
        in_play = any(k.startswith(PROTECTION_FROM) for k in attacker)
    elif rule is BlockingRestriction.SKULK:
        in_play = "skulk" in attacker
    else:
        in_play = _EVASION_KEYWORD[rule] in attacker
    return Judgement(entity_id, inside, rule in evasions, in_play)


def attacker_judgements(record: EffectRecord, lines: SidecarLookup | None,
                        restriction: str) -> list[Judgement]:
    """Each creature of an attackers record (a gen-3 record's anchor alone)
    against 508.1a's two conditions."""
    payload, state = record.payload, record.state
    listed = [(e, True) for e in payload.legal_attackers]
    listed += [(f.entity, False) for f in payload.forbidden]
    anchor = getattr(payload, "anchor_attacker", None)
    keywords_of = KeywordResolver(lines).keywords_of
    found = []
    for entity_id, inside in dict.fromkeys(listed):
        creature = state.entity(entity_id)
        if creature is None or (anchor is not None and entity_id != anchor):
            continue
        if restriction == TAPPED:
            found.append(Judgement(entity_id, inside, creature.tapped, True))
        else:
            sick = creature.sick
            found.append(Judgement(entity_id, inside,
                                   sick and "haste" not in keywords_of(creature), sick))
    return found


def target_judgements(record: EffectRecord, lines: SidecarLookup | None, restriction: str,
                      quality: str | None, candidates, universe: Iterable[str]) -> list[Judgement]:
    """Each entity of ``universe`` against each targeting candidate's card."""
    state = record.state
    source_entity = state.entity(state.refs.source) if state.refs.source else None
    if source_entity is None:
        return []
    source = Characteristics.of(source_entity)._replace(
        changeling="changeling" in KeywordResolver(lines).keywords_of(source_entity))
    keywords_of = KeywordResolver(lines).keywords_of
    actor = record.actor_player
    players = {p.id for p in state.players}
    found = []
    for candidate in candidates:
        legal = set(candidate.legal_targets)
        for entity_id in universe:
            if entity_id in players:
                keywords: set[str] = set()
                controller = entity_id
            else:
                entity = state.entity(entity_id)
                if entity is None:
                    continue
                keywords = _filtered(keywords_of(entity), quality)
                controller = entity.controller
            rules = targeting_restrictions(keywords, source=source,
                                           opponents=controller != actor)
            found.append(_target(restriction, entity_id, entity_id in legal, keywords, rules))
    return found


def _target(restriction: str, entity_id: str, inside: bool, keywords: set[str],
            rules) -> Judgement:
    if restriction == SELECTOR:
        return Judgement(entity_id, inside, not rules, True)
    rule = TargetingRestriction(restriction)
    in_play = {
        TargetingRestriction.HEXPROOF: "hexproof" in keywords,
        TargetingRestriction.HEXPROOF_FROM: any(k.startswith(HEXPROOF_FROM) for k in keywords),
        TargetingRestriction.SHROUD: "shroud" in keywords,
        TargetingRestriction.PROTECTION: any(k.startswith(PROTECTION_FROM) for k in keywords),
    }[rule]
    return Judgement(entity_id, inside, rule in rules, in_play)
