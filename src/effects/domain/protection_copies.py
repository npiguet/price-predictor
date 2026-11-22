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
qualifier (:func:`effects.domain.protection_opportunities.qualify`) names what
a record could carry, then :func:`protection_copy` draws from a seeded
generator and makes the copy. Four outcomes, each copied only where nothing
but the protection changes what the record shows:

- **damage prevented** (702.16e): a ``resolution`` effect half or a ``combat``
  record dealing damage to a permanent nothing else in the record touches but
  its own death; the protection covers every source that damaged it, and the
  permanent is no target of the acting chain, any clause or mode, as cast or
  after Forge's illegal-target check (``refs.targets_at_cast``,
  ``refs.targets``; a targeted spell would lose its target instead). Each
  ``damage_dealt`` becomes the ``damage_prevented`` Forge writes for a
  prevented hit, and a death the damage caused is undone whole: its
  ``destroyed``, its move to the graveyard, the ``unattached`` of each Aura
  or Equipment that fell off it and the Aura's own move. Refused: a source
  with lifelink, infect or wither, whose damage does more than mark damage;
  a resolution whose half does not say its targets as cast (no resolution
  hook); damage that can't be prevented (``NoPrevention$`` in the chain, an
  active ``Mode$ CantPreventDamage`` static on the board); and a chain with a
  later clause the prevention would change unseen (one remembering or
  storing what was dealt, or an amount read from a variable, a drain's
  "gain life equal to the damage dealt");
- **target made illegal** (702.16b): a decision whose candidate has at least
  two legal targets loses one permanent from them, the verdict unchanged, the
  acting chain having one targeting clause of minimum one (a second clause,
  a fight spell's, may have had it as its only candidate); the
  candidate acts through the first face its card shows (no Adventure, back
  face or split half, whose characteristics the snapshot does not hold);
- **block made illegal** (702.16f): a real blockers record's anchored
  attacker gains protection from a quality one of its legal blockers has, and
  every legal blocker with the quality becomes forbidden (a changeling has
  every creature type) and stays in ``must_block``;
- **Aura put into the graveyard** (702.16c): an Aura spell's resolution whose
  one target gains protection from the Aura; the spell has no legal target
  and fizzles (608.2b, ``outcome`` ``fizzled``), so the Aura goes from the
  stack to the graveyard and nothing it would have done happens; a bestowed
  Aura, which would resolve as a creature instead (702.103e), is not copied.

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

from effects.domain.protection_damage import prevent_damage
from effects.domain.protection_model import (
    EVERYTHING_RULE,
    Opportunity,
    ProtectionOutcome,
    entering,
)
from effects.domain.records import (
    EffectRecord,
    ForbiddenEntity,
    ResolutionOutcome,
)
from effects.domain.rule_outcomes import (
    Characteristics,
    Combatant,
    ProtectionQuality,
    may_block,
    may_enchant,
    may_target,
)
from effects.domain.state_snapshot import Zone


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


@dataclass(frozen=True, slots=True)
class CopyDraw:
    """What one copy is drawn with: its ``rule``, the generator ``rng`` that
    draws the opportunity and then the quality, and the ``index`` that numbers
    the copies of one record, which keeps their ids apart."""

    rule: str
    rng: random.Random
    index: int


def protection_copy(record: EffectRecord, opportunities: tuple[Opportunity, ...],
                    draw: CopyDraw) -> ProtectionCopy | None:
    """One copy of ``record`` drawn for ``draw.rule``, or None where no
    opportunity shows it.

    ``draw.rng`` draws the opportunity, then the quality: protection from
    everything for 702.16j, else one of the opportunity's rare qualities.
    """
    rule = draw.rule
    if rule == EVERYTHING_RULE:
        pool = [o for o in opportunities if o.everything]
    else:
        pool = [o for o in opportunities if o.rule == rule and o.qualities]
    if not pool:
        return None
    opportunity = draw.rng.choice(pool)
    quality = (ProtectionQuality.everything() if rule == EVERYTHING_RULE
               else draw.rng.choice(opportunity.qualities))

    record, outcome_entities = copied(record, opportunity, quality, draw.index)
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
    refs = record.state.refs
    if opportunity.outcome is ProtectionOutcome.AURA_TO_GRAVEYARD and named:
        # The fizzled spell's refs as the live shape holds them: no target
        # left after the check, the protected one as cast.
        refs = replace(refs, targets=(), targets_at_cast=refs.as_cast)
    return replace(
        record,
        record_id=f"{record.record_id}~pc{index}",
        synthetic=True,
        state=replace(record.state, entities=entities, refs=refs),
        payload=payload,
    ), named


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
                Combatant.of(attacker), Combatant.of(blocker), attacker_protections=(quality,)):
            forbidden.append(blocker_id)
    gone = set(forbidden)
    # ``must_block`` lists the creatures a requirement makes block, legal or
    # forbidden, whether or not they can obey (509.1c): a forbidden one stays.
    return replace(
        payload,
        legal_blockers=tuple(b for b in payload.legal_blockers if b not in gone),
        forbidden=(*payload.forbidden, *(ForbiddenEntity(entity=b) for b in forbidden)),
    ), tuple(forbidden)


def _fizzle(payload, opportunity: Opportunity, quality: ProtectionQuality, entities):
    """The Aura spell has no legal target left, so it fizzles (rule 608.2b):
    its half's ``outcome`` is ``fizzled`` and its one event the card's move
    from the stack to its owner's graveyard, as a collected fizzled spell's
    half holds it (``duration`` instant, ``attributed_to`` unresolved), which
    is 702.16c's outcome; it does nothing else."""
    aura = opportunity.covered[0]
    if may_enchant((quality,), Characteristics.of(entities[aura])):
        return payload, ()
    moved = replace(entering(payload.events, aura), params={
        "from_zone": Zone.STACK.value, "to_zone": Zone.GRAVEYARD.value, "cause": aura,
    }, duration="instant", attributed_to="unresolved")
    conditions = None if payload.conditions_met is None else ()
    return replace(payload, events=(moved,), conditions_met=conditions,
                   outcome=ResolutionOutcome.FIZZLED, outcome_written=True), (
        aura, opportunity.protected)


_REWRITES = {
    ProtectionOutcome.DAMAGE_PREVENTED: prevent_damage,
    ProtectionOutcome.TARGET_ILLEGAL: _untarget,
    ProtectionOutcome.BLOCK_ILLEGAL: _forbid,
    ProtectionOutcome.AURA_TO_GRAVEYARD: _fizzle,
}
