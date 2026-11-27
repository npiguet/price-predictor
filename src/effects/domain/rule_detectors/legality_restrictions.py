"""Which rule keeps an entity out of a blockers or targets legality list.

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
  ``Land.nonBasic``, ``Swamp.Snow``; where no line names it, a chosen type,
  the attacker's landwalk and no other reason); and
  ``effect``, a forbidden creature nothing the record shows explains: no
  evasion, tap or landwalk, no responsible line (gen-3 names the static
  forbidding the block, Pacifism's included, or the creature's own keyword
  line: ``responsible_static``; or says one exists that no key names:
  ``responsible_unresolved``), no can't-block keyword of its own or hidden
  (Grakk's "can't attack or block", the "alone" forms while its controller
  has no other creature, Panic's ``granted_temporary.hidden_keywords``) and
  no requirement binding it to another attacker (``required_elsewhere``: a
  lure, or another attacker that must be blocked if able). When the
  blockers are declarable no blocker is declared yet, so of the gates
  ``CombatUtil.canBlock(attacker, blocker, combat)`` reads, only a detained
  creature (701.35a, ``Card.isDetained``), which no record marks, is
  refused with none of those, so ``effect`` has no near miss: it is in play
  only where it applies;
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
near miss one it applies to, outside. ``selector`` applies to every entity
no other reason explains, and is in play everywhere; ``effect`` applies to
the same, and is in play only there, as ``tapped`` is in play only on a
tapped creature: neither has a near miss.

A player's keywords are in no snapshot, so no keyword restriction reads a
player.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial

from effects.domain.damage_step_keywords import KeywordsOf
from effects.domain.keyword_names import HEXPROOF_FROM, PROTECTION_FROM
from effects.domain.records import Candidate, EffectRecord, ForbiddenEntity
from effects.domain.rule_outcomes import (
    BlockingRestriction,
    Characteristics,
    Combatant,
    ProtectionQuality,
    QualityKind,
    TargetingRestriction,
    blocking_restrictions,
    landwalk_lands,
    targeting_restrictions,
)
from effects.domain.state_snapshot import CardLine, EntityState, StateSnapshot

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
class Restriction:
    """The one rule a declaration names, with a protection's quality kind
    (``everything``) where it names one."""

    name: str
    quality: str | None = None


@dataclass(frozen=True, slots=True)
class Judgement:
    """One entity's standing under one restriction."""

    entity: str
    inside: bool
    applies: bool
    in_play: bool


def _outside_side(j: Judgement) -> bool | None:
    """``fires_when = outside``: outside where the restriction applies, the
    near miss inside where it is in play and does not."""
    if j.applies:
        return None if j.inside else True
    return False if j.in_play and j.inside else None


def _inside_side(j: Judgement) -> bool | None:
    """``fires_when = inside``: inside nonetheless where the restriction
    applies, the near miss outside."""
    return j.inside if j.applies else None


def sides(judged: Iterable[Judgement], fires_when: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(fires, near)`` entity ids, each once, in the order judged."""
    side_of = _outside_side if fires_when == "outside" else _inside_side
    fires, near = [], []
    for j in judged:
        side = side_of(j)
        if side is not None:
            (fires if side else near).append(j.entity)
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


def _listed_entities(state: StateSnapshot, legal: Sequence[str],
                     forbidden: Sequence[ForbiddenEntity]
                     ) -> Iterator[tuple[str, bool, EntityState]]:
    """Each entity a legality list names, once, inside (legal) or outside
    (forbidden), with its snapshot entity; one the snapshot lacks left out."""
    listed = [(e, True) for e in legal]
    listed += [(f.entity, False) for f in forbidden]
    for entity_id, inside in dict.fromkeys(listed):
        entity = state.entity(entity_id)
        if entity is not None:
            yield entity_id, inside, entity


# ── blockers ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Blocking:
    """One listed creature against the anchor attacker, with what the
    restrictions read of the pair."""

    entity_id: str
    inside: bool
    blocker: EntityState
    blocker_keywords: set[str]
    #: The anchor's keywords, protections of another quality kind dropped.
    attacker: set[str]
    evasions: frozenset[BlockingRestriction]
    #: Whether something the record shows keeps it from blocking.
    explained: bool
    #: Whether its controller controls a land the anchor's landwalk names;
    #: None where no line says which land, or the restriction is no landwalk.
    walked: bool | None


@dataclass(frozen=True, slots=True)
class _Blockers:
    """A blockers record's anchor attacker, as each listed creature is judged against."""

    state: StateSnapshot
    anchor: EntityState
    attacker: set[str]
    keywords_of: KeywordsOf
    #: The forbidden creatures the record names a responsible line for.
    named: set[str]
    landwalk: bool

    def blocking(self, entity_id: str, inside: bool, blocker: EntityState) -> _Blocking:
        blocker_keywords = self.keywords_of(blocker)
        evasions = blocking_restrictions(
            Combatant.of(self.anchor, self.attacker), Combatant.of(blocker, blocker_keywords))
        explained = (bool(evasions) or blocker.tapped or entity_id in self.named
                     or _cant_block_keyword(blocker, blocker_keywords, self.state))
        walked = _walked(self.anchor, blocker, self.state) if self.landwalk else None
        return _Blocking(entity_id, inside, blocker, blocker_keywords, self.attacker,
                         frozenset(evasions), explained, walked)


def blocker_judgements(record: EffectRecord, keywords_of: KeywordsOf,
                       restriction: Restriction) -> list[Judgement]:
    """Each listed creature of a blockers record against its anchor attacker."""
    payload, state = record.payload, record.state
    anchor = state.entity(payload.anchor_attacker)
    if anchor is None:
        return []
    blockers = _Blockers(state, anchor, _filtered(keywords_of(anchor), restriction.quality),
                         keywords_of, {f.entity for f in payload.forbidden if f.explained},
                         restriction.name == LANDWALK)
    judge = _blocker_judge(restriction.name)
    return [judge(blockers.blocking(entity_id, inside, blocker)) for entity_id, inside, blocker
            in _listed_entities(state, payload.legal_blockers, payload.forbidden)]


#: The keywords Forge reads, with no static, to keep a creature from
#: blocking at all (``CombatUtil.canBlock(Card, boolean)``), lower-cased.
_CANT_BLOCK = frozenset({"cardname can't block.", "cardname can't attack or block."})
#: The same while its controller has no other creature in play.
_CANT_BLOCK_ALONE = frozenset({"cardname can't block alone.",
                               "cardname can't attack or block alone."})


def _cant_block_keyword(blocker: EntityState, keywords: set[str], state: StateSnapshot) -> bool:
    """Whether a keyword of the blocker's own, printed or granted
    (``keywords``), or hidden (``granted_temporary.hidden_keywords``), keeps
    it from blocking at all, as Forge reads it: "can't block", "can't attack
    or block", and the "alone" forms while its controller controls no other
    creature."""
    hidden = blocker.granted_temporary.hidden_keywords or ()
    texts = {k.lower().strip() for k in (*keywords, *hidden)}
    if texts & _CANT_BLOCK:
        return True
    return bool(texts & _CANT_BLOCK_ALONE) and _alone(blocker, state)


def _alone(blocker: EntityState, state: StateSnapshot) -> bool:
    """Whether its controller controls no other creature."""
    creatures = [e for e in state.entities if _creature_of(e, blocker.controller)]
    return len(creatures) < 2


def _creature_of(entity: EntityState, controller: str | None) -> bool:
    if not entity.on_battlefield or entity.controller != controller:
        return False
    return "creature" in {t.lower() for t in entity.types}


#: A landwalk selector naming no land the record shows.
_UNNAMED_LANDS = ("ChosenType", "YourBasic")


def _walked(anchor: EntityState, blocker: EntityState, state: StateSnapshot) -> bool | None:
    """Whether the blocker's controller controls a land the anchor's landwalk
    names (702.14c), read off its keyword lines' fragments; None where no
    line says which land (a chosen type)."""
    selectors = [s for s in map(_landwalk_selector, anchor.lines) if s is not None]
    if _names_no_land(selectors):
        return None
    lands = [e for e in state.entities if e.on_battlefield and e.controller == blocker.controller]
    return any(landwalk_lands(selector, land) for selector in selectors for land in lands)


def _landwalk_selector(line: CardLine) -> str | None:
    """The land a ``Landwalk:<selector>`` fragment names; None for another line."""
    fragment = line.fragment
    if not fragment or not fragment.lower().startswith("landwalk:"):
        return None
    parts = fragment.split(":")
    return parts[1] if len(parts) > 1 else None


def _names_no_land(selectors: list[str]) -> bool:
    if not selectors:
        return True
    return any(s.split(".")[0] in _UNNAMED_LANDS for s in selectors)


def _tapped_blocker(b: _Blocking) -> Judgement:
    return _tapped(b.entity_id, b.inside, b.blocker)


def _landwalk(b: _Blocking) -> Judgement:
    walks = "landwalk" in b.attacker
    if b.walked is not None:
        # The land the landwalk names is on the defender's battlefield.
        return Judgement(b.entity_id, b.inside, walks and b.walked, walks)
    return Judgement(b.entity_id, b.inside, walks and not b.inside and not b.explained, walks)


def _effect(b: _Blocking) -> Judgement:
    applies = not b.inside and not b.explained and "landwalk" not in b.attacker
    return Judgement(b.entity_id, b.inside, applies, applies)


def _shadow_in_play(rule: BlockingRestriction, b: _Blocking) -> bool:
    return "shadow" in b.attacker or "shadow" in b.blocker_keywords


def _protection_in_play(rule: BlockingRestriction, b: _Blocking) -> bool:
    return any(k.startswith(PROTECTION_FROM) for k in b.attacker)


def _skulk_in_play(rule: BlockingRestriction, b: _Blocking) -> bool:
    return "skulk" in b.attacker


def _evasion_keyword_in_play(rule: BlockingRestriction, b: _Blocking) -> bool:
    return _EVASION_KEYWORD[rule] in b.attacker


#: Whether each evasion is in play; the others' attacker keyword (``_EVASION_KEYWORD``).
_EVASION_IN_PLAY: dict[BlockingRestriction, Callable[[BlockingRestriction, _Blocking], bool]] = {
    BlockingRestriction.SHADOW: _shadow_in_play,
    BlockingRestriction.PROTECTION: _protection_in_play,
    BlockingRestriction.SKULK: _skulk_in_play,
}


def _evasion(rule: BlockingRestriction, b: _Blocking) -> Judgement:
    in_play = _EVASION_IN_PLAY.get(rule, _evasion_keyword_in_play)(rule, b)
    return Judgement(b.entity_id, b.inside, rule in b.evasions, in_play)


_BLOCKER_JUDGES: dict[str, Callable[[_Blocking], Judgement]] = {
    TAPPED: _tapped_blocker, LANDWALK: _landwalk, EFFECT: _effect}


def _blocker_judge(restriction: str) -> Callable[[_Blocking], Judgement]:
    """How a blocker restriction judges one listed creature: tapped,
    landwalk, effect, else the evasion it names."""
    judge = _BLOCKER_JUDGES.get(restriction)
    return judge if judge is not None else partial(_evasion, BlockingRestriction(restriction))


def _tapped(entity_id: str, inside: bool, creature: EntityState) -> Judgement:
    """A tapped creature can neither attack nor block (508.1a, 509.1a), and
    nothing lets it: the restriction is in play only where it applies, so it
    has no near miss. An untapped creature among the legal ones says nothing
    of the rule."""
    return Judgement(entity_id, inside, creature.tapped, creature.tapped)


# ── attackers ───────────────────────────────────────────────────────────


def attacker_judgements(record: EffectRecord, keywords_of: KeywordsOf,
                        restriction: str) -> list[Judgement]:
    """Each creature of an attackers record (a gen-3 record's anchor alone)
    against 508.1a's two conditions."""
    payload, state = record.payload, record.state
    anchor = getattr(payload, "anchor_attacker", None)
    found = []
    for entity_id, inside, creature in _listed_entities(
            state, payload.legal_attackers, payload.forbidden):
        if anchor is None or entity_id == anchor:
            found.append(_tapped(entity_id, inside, creature) if restriction == TAPPED
                         else _sick(entity_id, inside, creature, keywords_of))
    return found


def _sick(entity_id: str, inside: bool, creature: EntityState,
          keywords_of: KeywordsOf) -> Judgement:
    """Summoning sick without haste: in play on any sick creature."""
    sick = creature.sick
    return Judgement(entity_id, inside, sick and "haste" not in keywords_of(creature), sick)


# ── targets ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TargetPool:
    """The targeting candidates judged, and the entities each is judged against."""

    candidates: Sequence[Candidate]
    universe: Iterable[str]


def _has_hexproof(keywords: set[str]) -> bool:
    return "hexproof" in keywords


def _has_hexproof_from(keywords: set[str]) -> bool:
    return any(k.startswith(HEXPROOF_FROM) for k in keywords)


def _has_shroud(keywords: set[str]) -> bool:
    return "shroud" in keywords


def _has_protection(keywords: set[str]) -> bool:
    return any(k.startswith(PROTECTION_FROM) for k in keywords)


#: Whether each targeting restriction is in play on an entity's keywords.
_TARGET_IN_PLAY: dict[TargetingRestriction, Callable[[set[str]], bool]] = {
    TargetingRestriction.HEXPROOF: _has_hexproof,
    TargetingRestriction.HEXPROOF_FROM: _has_hexproof_from,
    TargetingRestriction.SHROUD: _has_shroud,
    TargetingRestriction.PROTECTION: _has_protection,
}


@dataclass(frozen=True, slots=True)
class _Targets:
    """A decision record's source card, as each entity is judged against it."""

    state: StateSnapshot
    keywords_of: KeywordsOf
    restriction: Restriction
    source: Characteristics
    actor: str | None
    players: frozenset[str]

    def judgements(self, candidate: Candidate, universe: Iterable[str]) -> Iterator[Judgement]:
        legal = set(candidate.legal_targets)
        for entity_id in universe:
            standing = self._standing(entity_id)
            if standing is not None:
                yield self._judge(entity_id, entity_id in legal, *standing)

    def _standing(self, entity_id: str) -> tuple[set[str], str | None] | None:
        """The entity's keywords and controller; a player has no keywords in
        a snapshot. None for an entity the snapshot lacks."""
        if entity_id in self.players:
            return set(), entity_id
        entity = self.state.entity(entity_id)
        if entity is None:
            return None
        return _filtered(self.keywords_of(entity), self.restriction.quality), entity.controller

    def _judge(self, entity_id: str, inside: bool, keywords: set[str],
               controller: str | None) -> Judgement:
        rules = targeting_restrictions(keywords, source=self.source,
                                       opponents=controller != self.actor)
        if self.restriction.name == SELECTOR:
            return Judgement(entity_id, inside, not rules, True)
        rule = TargetingRestriction(self.restriction.name)
        return Judgement(entity_id, inside, rule in rules, _TARGET_IN_PLAY[rule](keywords))


def target_judgements(record: EffectRecord, keywords_of: KeywordsOf, restriction: Restriction,
                      pool: TargetPool) -> list[Judgement]:
    """Each entity of the pool's universe against each targeting candidate's card."""
    state = record.state
    source_entity = state.entity(state.refs.source) if state.refs.source else None
    if source_entity is None:
        return []
    source = Characteristics.of(source_entity)._replace(
        changeling="changeling" in keywords_of(source_entity))
    targets = _Targets(state, keywords_of, restriction, source, record.actor_player,
                       frozenset(p.id for p in state.players))
    return [j for candidate in pool.candidates
            for j in targets.judgements(candidate, pool.universe)]
