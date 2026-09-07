"""Snapshot to the effect head's one token surface.

    [GLOBAL] [ACT] [PLAYER] [PLAYER] [CARD] e e … [CARD] e …

One token per slot, with **position ids resetting at each ``[CARD]``**. The
reset is what makes an entity's ability tokens a block rather than a sequence:
the head reads "this card, then its abilities" the same way whether the card is
first on the board or twelfth, so a board of nine permanents is not nine
different geometries.

Every record kind enters this one surface as a variant of it rather than getting
a surface of its own — a combat record leaves ``[ACT]`` empty, a ``continuous``
record has the acting static's own contributions masked out of the entity
inputs, a ``rewrite`` carries its pending event as an overlay. That is what lets
one trunk learn from eight kinds of record at once.

Two rules shape the numeric encoding:

- **Nothing is binned.** A scalar enters as its raw value *and* a ``log1p``
  copy, so the head can read both "how much" and "how much, roughly" without
  the modeller having chosen the bucket edges.
- **No perspective is stored.** Controller tags are derived here, relative to
  the record's ``actor_player``, which is why one snapshot can serve records
  with different actors.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum

from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    CombatPayload,
    EffectRecord,
    Moment,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
)
from effects.domain.state_snapshot import COLORS, EntityState, PlayerState


class SlotKind(IntEnum):
    GLOBAL = 0
    ACT = 1
    PLAYER = 2
    CARD = 3
    ABILITY = 4


# ── checked-in vocabularies ─────────────────────────────────────────────
#
# Each ends in an "other" bucket, so a value the corpus grows later is
# unfamiliar rather than unrepresentable. Adding a member shifts nothing above
# it, which keeps an older checkpoint's feature layout readable.

PHASES: tuple[str, ...] = (
    "untap", "upkeep", "draw", "main1", "begin_combat", "declare_attackers",
    "declare_blockers", "first_strike_damage", "combat_damage", "end_combat",
    "main2", "end_of_turn", "cleanup",
)
COMBAT_SUBSTEPS: tuple[str, ...] = ("first_strike", "regular")
ZONES: tuple[str, ...] = (
    "battlefield", "stack", "hand", "graveyard", "library", "exile", "command",
)
CARD_TYPES: tuple[str, ...] = (
    "artifact", "battle", "creature", "enchantment", "instant", "land",
    "planeswalker", "sorcery", "kindred", "emblem",
)
SUPERTYPES: tuple[str, ...] = ("basic", "legendary", "snow", "world")
COUNTER_TYPES: tuple[str, ...] = (
    "P1P1", "M1M1", "LOYALTY", "CHARGE", "TIME", "STUN", "SHIELD", "OIL",
    "POISON", "EXPERIENCE", "KI", "LEVEL", "LORE", "DEFENSE",
)
#: Keywords the overlay tracks individually. The eight damage-step keywords lead
#: the list because gate 2's perturbation removes one of them from exactly this
#: channel when it is not printed on the entity, so each has to be addressable.
OVERLAY_KEYWORDS: tuple[str, ...] = (
    "first_strike", "double_strike", "deathtouch", "lifelink", "trample",
    "indestructible", "wither", "infect",
    "flying", "reach", "vigilance", "haste", "menace", "hexproof", "shroud",
    "defender", "protection", "ward", "flash", "banding", "fear", "intimidate",
    "horsemanship", "skulk", "shadow", "unblockable", "annihilator",
)
THIS_TURN_COUNTERS: tuple[str, ...] = (
    "creatures_died", "spells_cast", "lands_played",
)


def _one_hot(value: str | None, vocabulary: tuple[str, ...]) -> list[float]:
    """One-hot over ``vocabulary`` plus a trailing "other/absent" slot."""
    out = [0.0] * (len(vocabulary) + 1)
    if value is None:
        return out
    normalized = value.strip().lower().replace(" ", "_")
    for index, member in enumerate(vocabulary):
        if member.lower() == normalized:
            out[index] = 1.0
            return out
    out[-1] = 1.0
    return out


def _scalar(value: float) -> list[float]:
    """A scalar as raw plus ``log1p``. Nothing is binned (FR-074).

    ``log1p`` of a negative value is undefined, so the copy is signed:
    ``sign(v) * log1p(|v|)``, which keeps the transform monotone across zero.
    """
    magnitude = math.log1p(abs(float(value)))
    return [float(value), math.copysign(magnitude, value)]


def _multi_hot(present: set[str], vocabulary: tuple[str, ...]) -> list[float]:
    """Multi-hot over ``vocabulary`` plus a count of everything else."""
    normalized = {p.strip().lower().replace(" ", "_") for p in present}
    out = [1.0 if member.lower() in normalized else 0.0 for member in vocabulary]
    known = {member.lower() for member in vocabulary}
    out.append(float(len(normalized - known)))
    return out


# ── slots ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Slot:
    """One token of the input surface."""

    kind: SlotKind
    position: int
    features: tuple[float, ...] = ()
    #: The ability vector an ``ACT`` or ``ABILITY`` slot carries.
    e: tuple[float, ...] | None = None
    #: ``CARD`` slots: token ids of the entity's subtypes, mean-pooled by the
    #: model. Subtypes are an open vocabulary (every set adds creature types),
    #: so they enter through the shared token embedding rather than a one-hot.
    subtype_tokens: tuple[int, ...] = ()
    entity_id: str | None = None
    player_id: str | None = None


@dataclass(frozen=True, slots=True)
class EffectHeadInput:
    """The full token surface for one record."""

    slots: tuple[Slot, ...]
    record_kind: RecordKind
    actor_player: str
    #: Index into ``slots`` of the ``[ACT]`` token, always 1.
    act_index: int = 1
    #: Slot indices of the per-entity head's targets: every CARD and PLAYER.
    entity_slots: tuple[int, ...] = field(default_factory=tuple)

    def positions(self) -> tuple[int, ...]:
        return tuple(slot.position for slot in self.slots)

    def of_kind(self, kind: SlotKind) -> tuple[Slot, ...]:
        return tuple(slot for slot in self.slots if slot.kind is kind)


# ── feature builders ────────────────────────────────────────────────────


def global_features(record: EffectRecord) -> tuple[float, ...]:
    """Turn structure, plus the record's own kind flags."""
    state = record.state
    g = state.global_
    out: list[float] = []
    out += _scalar(g.turn)
    out += _scalar(g.stack_size)
    out += _one_hot(g.phase, PHASES)
    out += _one_hot(g.combat_substep, COMBAT_SUBSTEPS)
    out.append(1.0 if g.active == record.actor_player else 0.0)
    out.append(1.0 if g.priority == record.actor_player else 0.0)
    out += _scalar(len(g.emblems))
    out += _one_hot(record.kind.value, tuple(k.value for k in RecordKind))
    out += _one_hot(
        record.moment.value if record.moment else None,
        tuple(m.value for m in Moment),
    )
    out += _one_hot(
        record.subkind.value if record.subkind else None,
        tuple(s.value for s in PlayabilitySubkind),
    )
    return tuple(out)


def act_features(record: EffectRecord) -> tuple[float, ...]:
    """Announced values and the resolution outcome flag.

    Empty for ``combat`` and for the ``attackers``/``blockers`` subkinds: no
    single line acts, so there is nothing for the slot to describe and filling
    it with the board's averages would invent an actor.
    """
    if _act_is_empty(record):
        return tuple([0.0] * _ACT_FEATURE_WIDTH)
    refs = record.state.refs
    out: list[float] = [1.0]  # slot is populated
    out += _scalar(refs.x if refs.x is not None else 0)
    out.append(1.0 if refs.x is not None else 0.0)
    out += _scalar(len(refs.modes))
    out += _scalar(len(refs.targets))
    out += _scalar(len(refs.choices))
    # The outcome flag comes from the paired cost record; a half with no
    # partner resolved by definition, which is why `resolved` is the default.
    outcome = ResolutionOutcome.RESOLVED
    if record.moment is Moment.ACTIVATION and hasattr(record.payload, "outcome"):
        outcome = record.payload.outcome
    out += _one_hot(outcome.value, tuple(o.value for o in ResolutionOutcome))
    return tuple(out)


_ACT_FEATURE_WIDTH = 1 + 2 + 1 + 2 + 2 + 2 + (len(ResolutionOutcome) + 1)


def _act_is_empty(record: EffectRecord) -> bool:
    if record.kind is RecordKind.COMBAT:
        return True
    return record.subkind in (
        PlayabilitySubkind.ATTACKERS, PlayabilitySubkind.BLOCKERS,
    )


def player_features(
    player: PlayerState, record: EffectRecord,
) -> tuple[float, ...]:
    """One player's resources, plus their relation to the acting player."""
    out: list[float] = []
    for value in (player.life, player.hand, player.library, player.graveyard,
                  player.poison, player.energy):
        out += _scalar(value)
    for key in THIS_TURN_COUNTERS:
        out += _scalar(player.this_turn.get(key, 0))
    for color in COLORS:
        out += _scalar(player.floating_mana.get(color, 0))
    for color in COLORS:
        out += _scalar(player.untapped_production.get(color, 0))
    out.append(1.0 if player.id == record.actor_player else 0.0)
    out.append(0.0 if player.id == record.actor_player else 1.0)
    out.append(1.0 if player.id in record.state.refs.targets else 0.0)
    return tuple(out)


def card_features(
    entity: EntityState,
    record: EffectRecord,
    *,
    masked_keywords: frozenset[str] = frozenset(),
) -> tuple[float, ...]:
    """One entity's structured characteristics plus its overlay.

    ``masked_keywords`` removes keywords from the temporarily-granted channel.
    Gate 2's perturbation uses it to strip one keyword from a combat
    participant that was granted it rather than printing it, and a
    ``continuous`` record uses it to hide the acting static's own contribution.
    """
    state = record.state
    out: list[float] = []

    # ── structured characteristics ──
    out += _multi_hot(set(entity.types), CARD_TYPES)
    out += _multi_hot(set(entity.supertypes), SUPERTYPES)
    out += _multi_hot(set(entity.colors), COLORS)
    out += _scalar(entity.mana_value)
    if entity.pt is not None:
        for pair in (entity.pt.base, entity.pt.boosts, entity.pt.counters):
            out += _scalar(pair[0])
            out += _scalar(pair[1])
        out.append(1.0)
    else:
        out += [0.0] * 12
        out.append(0.0)

    # ── overlay ──
    out += _one_hot(entity.zone, ZONES)
    out.append(1.0 if entity.tapped else 0.0)
    out.append(1.0 if entity.sick else 0.0)
    out.append(1.0 if entity.face_down else 0.0)
    out += _scalar(entity.damage)
    for counter in COUNTER_TYPES:
        out += _scalar(entity.counters.get(counter, 0))
    out += _scalar(sum(
        value for name, value in entity.counters.items()
        if name not in COUNTER_TYPES
    ))
    combat = entity.combat
    out.append(1.0 if combat and combat.attacking else 0.0)
    out.append(1.0 if combat and combat.blocking else 0.0)
    out.append(1.0 if combat and combat.became_blocked else 0.0)
    out += _scalar(len(combat.blocked_by) if combat else 0)
    out.append(1.0 if entity.attached_to else 0.0)
    granted = set(entity.granted_temporary.keywords) - set(masked_keywords)
    out += _multi_hot(granted, OVERLAY_KEYWORDS)
    out += _scalar(len(entity.granted_temporary.abilities))
    out.append(1.0 if state.controller_tag(
        entity.controller, record.actor_player) == "mine" else 0.0)
    out.append(1.0 if entity.id in state.refs.targets else 0.0)
    out.append(1.0 if entity.id == state.refs.source else 0.0)

    # ── stack extras ──
    extras = entity.stack_extras
    out += _scalar(len(extras.targets) if extras else 0)
    out += _scalar(sum(extras.per_target_amounts.values()) if extras else 0)
    out += _scalar(sum(extras.up_to_counts.values()) if extras else 0)

    # ── per-kind overlays ──
    out += _damage_assignment_features(entity, record)
    out += _pending_event_features(entity, record)
    return tuple(out)


def _damage_assignment_features(
    entity: EntityState, record: EffectRecord,
) -> list[float]:
    """A combat record's declared combat and damage-assignment choices."""
    payload = record.payload
    if not isinstance(payload, CombatPayload):
        return [0.0, 0.0, 0.0]
    assigned = payload.assignment_choices.get(entity.id, {})
    return [
        1.0 if entity.id in payload.attackers else 0.0,
        float(len(assigned)),
        float(sum(assigned.values())),
    ]


def _pending_event_features(
    entity: EntityState, record: EffectRecord,
) -> list[float]:
    """The pending-event overlay a ``rewrite`` or ``trigger`` record carries."""
    pending = record.state.pending_event
    if pending is None:
        return [0.0, 0.0]
    return [
        1.0,
        1.0 if entity.id in pending.subjects else 0.0,
    ]


# ── assembly ────────────────────────────────────────────────────────────


def build_effect_head_input(
    record: EffectRecord,
    *,
    e_for,
    e_dim: int,
    candidate_index: int | None = None,
    subtype_tokens_for=None,
    context_dropout: float = 0.0,
    rng: random.Random | None = None,
    masked_keywords: dict[str, frozenset[str]] | None = None,
) -> EffectHeadInput:
    """Lay one record out as the token surface.

    Args:
        e_for: ``ProvenanceKey -> tuple[float, ...] | None``, the ability cache
            lookup. A key the cache cannot resolve contributes a zero vector
            rather than dropping the token, so the entity keeps its shape.
        e_dim: bottleneck width, for the zero vectors above.
        candidate_index: a ``playability``/``decision`` record trains as one
            example per candidate; this selects which candidate's ``e`` sits in
            ``[ACT]``.
        subtype_tokens_for: ``EntityState -> tuple[int, ...]``, optional.
        context_dropout: probability of dropping each *context* ability token
            during training. The acting entity's tokens are never dropped —
            dropping the ability under study would make the record unanswerable.
        masked_keywords: per-entity keywords to hide from the temporary-grant
            channel.
    """
    rng = rng or random.Random()
    masked = masked_keywords or {}
    zero = tuple([0.0] * e_dim)
    slots: list[Slot] = []
    position = 0

    slots.append(Slot(
        kind=SlotKind.GLOBAL, position=position,
        features=global_features(record),
    ))
    position += 1

    slots.append(Slot(
        kind=SlotKind.ACT, position=position,
        features=act_features(record),
        e=_act_vector(record, e_for, zero, candidate_index),
    ))
    position += 1

    entity_slots: list[int] = []
    for player in record.state.players:
        entity_slots.append(len(slots))
        slots.append(Slot(
            kind=SlotKind.PLAYER, position=position,
            features=player_features(player, record),
            player_id=player.id,
        ))
        position += 1

    source_id = record.state.refs.source
    for entity in record.state.entities:
        # Position ids restart here: an entity plus its abilities is one block,
        # read the same way wherever the entity sits on the board.
        position = 0
        entity_slots.append(len(slots))
        slots.append(Slot(
            kind=SlotKind.CARD, position=position,
            features=card_features(
                entity, record,
                masked_keywords=masked.get(entity.id, frozenset()),
            ),
            subtype_tokens=(
                subtype_tokens_for(entity) if subtype_tokens_for else ()
            ),
            entity_id=entity.id,
        ))
        position += 1
        is_context = entity.id != source_id
        for key in _ability_keys(entity):
            if (
                is_context
                and context_dropout > 0.0
                and rng.random() < context_dropout
            ):
                continue
            slots.append(Slot(
                kind=SlotKind.ABILITY, position=position,
                e=e_for(key) or zero,
                entity_id=entity.id,
            ))
            position += 1

    return EffectHeadInput(
        slots=tuple(slots),
        record_kind=record.kind,
        actor_player=record.actor_player,
        entity_slots=tuple(entity_slots),
    )


def _ability_keys(entity: EntityState) -> tuple[ProvenanceKey, ...]:
    """An entity's ability tokens: printed and attachment-granted only (FR-073).

    Temporary grants ride the overlay instead. The two channels differ in what
    they are: a printed or equipped ability is part of what the card *is*, while
    a grant until end of turn is part of what happened to it.
    """
    return (*entity.printed, *entity.granted_attached)


def _act_vector(
    record: EffectRecord, e_for, zero: tuple[float, ...],
    candidate_index: int | None,
) -> tuple[float, ...] | None:
    if _act_is_empty(record):
        return zero
    payload = record.payload
    if isinstance(payload, PlayabilityDecisionPayload):
        if candidate_index is None or candidate_index >= len(payload.candidates):
            return zero
        keys = payload.candidates[candidate_index].ability
        return _first_vector(keys, e_for, zero)
    if record.ability:
        return _first_vector(record.ability, e_for, zero)
    return zero


def _first_vector(keys, e_for, zero: tuple[float, ...]) -> tuple[float, ...]:
    """The first key that resolves; zeros if none do.

    A rendered line merged from several traits carries several keys and they all
    name the same cache row, so the first hit is the line's vector.
    """
    for key in keys:
        vector = e_for(key)
        if vector is not None:
            return vector
    return zero


def anchored_attacker(record: EffectRecord) -> str | None:
    """The attacker a ``blockers`` record is anchored to (FR-071).

    It is carried by the source flag on that entity's ``[CARD]`` slot rather
    than in ``[ACT]``, which is empty for this subkind. ``min_blockers`` is read
    at this entity by the per-entity head.
    """
    if isinstance(record.payload, PlayabilityBlockersPayload):
        return record.payload.anchor_attacker
    return None


def continuous_masked_keywords(record: EffectRecord) -> dict[str, frozenset[str]]:
    """Keywords a ``continuous`` record's own static contributed (FR-020).

    Removed from the entity inputs so the model is not asked to predict a state
    that already contains its answer. Structured features stay: the static
    changed keywords, not the card's printed type line.
    """
    payload = record.payload
    if record.kind is not RecordKind.CONTINUOUS or not hasattr(
        payload, "contributions"
    ):
        return {}
    return {
        contribution.entity: frozenset(contribution.keywords)
        for contribution in payload.contributions
        if contribution.keywords
    }
