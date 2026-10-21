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
from functools import lru_cache

from effects.domain.keyword_names import overlay_keyword
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    CombatPayload,
    ContinuousPayload,
    EffectRecord,
    Moment,
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

# ── input layouts (FR-063c) ──────────────────────────────────────────────
#
# The vocabularies above were written before the collector's spelling was
# checked, and three of them miss it: Forge names five combat phases
# ``combat_*``, writes counters as ``+1/+1``, and spells protection as
# ``protection_from_red``. Each of those landed in its vocabulary's "other"
# slot. The fix is a layout of its own rather than an edit of the tuples,
# because a checkpoint trained on the earlier inputs has to keep reading them
# bit for bit.

#: Where a checkpoint records the layout it trained on, in its
#: ``training_settings`` — beside ``tokenizer_rules``, the precedent. A
#: checkpoint recording none (gen-1, and the gen-2 arms trained before the
#: layout existed) reads :data:`EARLIER_LAYOUT`.
INPUT_LAYOUT_KEY = "input_layout"
INPUT_LAYOUT = "gen-2"
EARLIER_LAYOUT: str | None = None

PHASE_ALIASES: dict[str, str] = {
    "combat_begin": "begin_combat",
    "combat_declare_attackers": "declare_attackers",
    "combat_declare_blockers": "declare_blockers",
    "combat_first_strike_damage": "first_strike_damage",
    "combat_end": "end_combat",
}
COUNTER_ALIASES: dict[str, str] = {"+1/+1": "P1P1", "-1/-1": "M1M1"}
#: Appended rather than inserted, so every earlier member keeps its slot.
OVERLAY_KEYWORDS_GEN2: tuple[str, ...] = (*OVERLAY_KEYWORDS, "landwalk")

#: The largest magnitude a count or a signed delta is scored at. Above about
#: forty an amount is the game state compounding — a counter doubling run to two
#: million in one long game — not what the ability's text says, and one such
#: target under the Poisson loss pulls a step a million times harder than an
#: ordinary one. Every consumer of the magnitudes caps them here: the field
#: losses, the constant-predictor floor and the evaluation's deviance. Defined
#: here, torch-free, so the thermometer below and the gate metrics share it.
MAGNITUDE_CAP = 40.0

#: Positions of a thermometer code. Position k is set when the value is at
#: least k, so the dot product of an amount's code with a one-hot of t is 1
#: exactly when the amount is at least t — the comparison "this damage kills
#: this creature" is one attention score. As many as the magnitude cap: life
#: often climbs past its starting twenty, and a larger value sets every
#: position.
THERMOMETER_STEPS = int(MAGNITUDE_CAP)


def thermometer(value: float) -> list[float]:
    """``value`` as :data:`THERMOMETER_STEPS` positions, position k set when ``value >= k``."""
    return [1.0 if value >= k else 0.0 for k in range(1, THERMOMETER_STEPS + 1)]


_NO_THERMOMETER: tuple[float, ...] = (0.0,) * THERMOMETER_STEPS


def _phase_in(phase: str | None, layout: str | None) -> str | None:
    if layout is EARLIER_LAYOUT or phase is None:
        return phase
    return PHASE_ALIASES.get(phase.strip().lower(), phase)


@lru_cache(maxsize=None)
def _positions(vocabulary: tuple[str, ...]) -> dict[str, int]:
    """``normalized member -> its slot``, built once per vocabulary.

    The one-hots used to lower every member of the vocabulary on every call,
    and the multi-hot lowered them twice — a vocabulary scan per feature,
    twenty-seven of them for the overlay keywords alone, a million times over
    a four-shard run. The vocabularies are checked-in constants, so the scan
    is done once here and every call is a dict lookup per value present.

    First occurrence wins, which is what :func:`_one_hot` did when it returned
    at its first match; no vocabulary has two members that lower to the same
    string, and one that did would have meant different things to the two
    helpers before this as well.
    """
    positions: dict[str, int] = {}
    for index, member in enumerate(vocabulary):
        positions.setdefault(member.lower(), index)
    return positions


def _spelling(value: str) -> str:
    """A value in its vocabulary's spelling: lower case, underscores."""
    return value.strip().lower().replace(" ", "_")


def _one_hot(value: str | None, vocabulary: tuple[str, ...]) -> list[float]:
    """One-hot over ``vocabulary`` plus a trailing "other/absent" slot."""
    out = [0.0] * (len(vocabulary) + 1)
    if value is None:
        return out
    index = _positions(vocabulary).get(_spelling(value))
    out[-1 if index is None else index] = 1.0
    return out


def _push_scalar(out: list[float], value: float) -> None:
    """A scalar as raw plus ``log1p``, onto the caller's list. Nothing is binned (FR-074).

    ``log1p`` of a negative value is undefined, so the copy is signed:
    ``sign(v) * log1p(|v|)``, which keeps the transform monotone across zero.
    Appended rather than returned, because a two-element list per scalar —
    twenty-eight per entity, millions per run — made the allocation, not the
    arithmetic, the cost of a scalar.
    """
    raw = float(value)
    out.append(raw)
    out.append(math.copysign(math.log1p(abs(raw)), raw))


#: Thirteen zeros: the power/toughness block of an entity that has none. Its
#: own constant because most boards carry non-creatures.
_NO_PT: tuple[float, ...] = (0.0,) * 13
#: The counter block of an entity carrying none — one scalar per checked-in
#: counter type plus the catch-all, and a scalar of zero is two zeros.
_NO_COUNTERS: tuple[float, ...] = (0.0,) * (2 * (len(COUNTER_TYPES) + 1))
#: The stack-extras block of an entity that is not on the stack.
_NO_STACK_EXTRAS: tuple[float, ...] = (0.0,) * 6


def _multi_hot(present, vocabulary: tuple[str, ...]) -> list[float]:
    """Multi-hot over ``vocabulary`` plus a count of everything else.

    One dict lookup per value the entity actually carries, rather than one
    membership test per vocabulary entry: an entity has a handful of types
    and keywords, and the vocabularies are ten to twenty-seven long.
    """
    positions = _positions(vocabulary)
    out = [0.0] * (len(vocabulary) + 1)
    seen: set[str] = set()
    other = 0
    for value in present:
        normalized = _spelling(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        index = positions.get(normalized)
        if index is None:
            other += 1
        else:
            out[index] = 1.0
    out[-1] = float(other)
    return out


# ── slots ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Slot:
    """One token of the input surface."""

    kind: SlotKind
    position: int
    features: tuple[float, ...] = ()
    #: The ability vector an ``ACT`` or ``ABILITY`` slot carries — either the
    #: values themselves (inference, reading the cache) or an ``int`` row of a
    #: matrix supplied at collate time (training, where ``e`` is the live
    #: encoder output and must stay a tensor the gradient can flow back
    #: through). A row index is not a vector of one element: the two are told
    #: apart by type, never by length.
    e: tuple[float, ...] | int | None = None
    entity_id: str | None = None
    player_id: str | None = None
    #: ``ABILITY`` slots read from a charm mode's ``option`` line (FR-063a).
    #: They follow their root's slot inside the card's block, and the flag plus
    #: that adjacency is the whole link between a mode and its charm.
    option: bool = False


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


#: The three enum vocabularies the global slot one-hots over. Built here
#: rather than per call: they are as fixed as the checked-in tuples above.
_RECORD_KINDS: tuple[str, ...] = tuple(k.value for k in RecordKind)
_MOMENTS: tuple[str, ...] = tuple(m.value for m in Moment)
_PLAYABILITY_SUBKINDS: tuple[str, ...] = tuple(
    s.value for s in PlayabilitySubkind
)


def global_features(
    record: EffectRecord, *, layout: str | None = INPUT_LAYOUT,
) -> tuple[float, ...]:
    """Turn structure, plus the record's own kind flags."""
    g = record.state.global_
    out: list[float] = []
    _push_scalar(out, g.turn)
    _push_scalar(out, g.stack_size)
    out += _one_hot(_phase_in(g.phase, layout), PHASES)
    out += _one_hot(g.combat_substep, COMBAT_SUBSTEPS)
    out.append(1.0 if g.active == record.actor_player else 0.0)
    out.append(1.0 if g.priority == record.actor_player else 0.0)
    _push_scalar(out, len(g.emblems))
    out += _one_hot(record.kind.value, _RECORD_KINDS)
    out += _one_hot(
        record.moment.value if record.moment else None, _MOMENTS,
    )
    out += _one_hot(
        record.subkind.value if record.subkind else None,
        _PLAYABILITY_SUBKINDS,
    )
    return tuple(out)


#: Width of the record-kind flags that end every ``[GLOBAL]`` slot: the kind,
#: moment and subkind one-hots, each with its trailing "other/absent" slot.
#: The ``no-state`` baseline keeps exactly these (FR-094).
KIND_FLAG_WIDTH = len(_RECORD_KINDS) + len(_MOMENTS) + len(_PLAYABILITY_SUBKINDS) + 3

#: The ``[ACT]`` slot's index in every surface: ``[GLOBAL]`` is first, then it.
ACT_SLOT = 1


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
    _push_scalar(out, refs.x if refs.x is not None else 0)
    out.append(1.0 if refs.x is not None else 0.0)
    _push_scalar(out, len(refs.modes))
    _push_scalar(out, len(refs.targets))
    _push_scalar(out, len(refs.choices))
    # The outcome flag comes from the paired cost record; a half with no
    # partner resolved by definition, which is why `resolved` is the default.
    outcome = ResolutionOutcome.RESOLVED
    if record.moment is Moment.ACTIVATION and hasattr(record.payload, "outcome"):
        outcome = record.payload.outcome
    out += _one_hot(outcome.value, _RESOLUTION_OUTCOMES)
    return tuple(out)


_ACT_FEATURE_WIDTH = 1 + 2 + 1 + 2 + 2 + 2 + (len(ResolutionOutcome) + 1)
_RESOLUTION_OUTCOMES: tuple[str, ...] = tuple(
    o.value for o in ResolutionOutcome
)


def _act_is_empty(record: EffectRecord) -> bool:
    if record.kind is RecordKind.COMBAT:
        return True
    return record.subkind in (
        PlayabilitySubkind.ATTACKERS, PlayabilitySubkind.BLOCKERS,
    )


def player_features(
    player: PlayerState, record: EffectRecord, *,
    layout: str | None = INPUT_LAYOUT,
) -> tuple[float, ...]:
    """One player's resources, plus their relation to the acting player.

    Under the gen-2 layout a thermometer code of life follows, so a burn
    spell's damage reads against a life total the way it reads against a
    toughness.
    """
    out: list[float] = []
    for value in (player.life, player.hand, player.library, player.graveyard,
                  player.poison, player.energy):
        _push_scalar(out, value)
    for key in THIS_TURN_COUNTERS:
        _push_scalar(out, player.this_turn.get(key, 0))
    for color in COLORS:
        _push_scalar(out, player.floating_mana.get(color, 0))
    for color in COLORS:
        _push_scalar(out, player.untapped_production.get(color, 0))
    out.append(1.0 if player.id == record.actor_player else 0.0)
    out.append(0.0 if player.id == record.actor_player else 1.0)
    out.append(1.0 if player.id in record.state.refs.targets else 0.0)
    if layout is not EARLIER_LAYOUT:
        out += thermometer(player.life)
    return tuple(out)


def card_features(
    entity: EntityState,
    record: EffectRecord,
    *,
    masked_keywords: frozenset[str] = frozenset(),
    layout: str | None = INPUT_LAYOUT,
) -> tuple[float, ...]:
    """One entity's structured characteristics plus its overlay.

    ``masked_keywords`` removes keywords from the temporarily-granted channel.
    Gate 2's perturbation uses it to strip one keyword from a combat
    participant that was granted it rather than printing it, and a
    ``continuous`` record uses it to hide the acting static's own contribution.

    Under the gen-2 layout, thermometer codes of total power, total toughness
    and toughness left after marked damage close the vector. The last is
    computed here rather than left to the trunk: subtracting is one more thing
    a transformer would have to learn before it could compare.
    """
    out: list[float] = []
    _push_characteristics(out, entity)
    _push_overlay(out, entity, record, masked_keywords=masked_keywords, layout=layout)
    _push_stack_extras(out, entity)
    out += _damage_assignment_features(entity, record)
    out += _pending_event_features(entity, record)
    if layout is not EARLIER_LAYOUT:
        _push_comparable_amounts(out, entity)
    return tuple(out)


def _push_characteristics(out: list[float], entity: EntityState) -> None:
    """Types, supertypes, colours, mana value and the P/T breakdown."""
    out += _multi_hot(entity.types, CARD_TYPES)
    out += _multi_hot(entity.supertypes, SUPERTYPES)
    out += _multi_hot(entity.colors, COLORS)
    _push_scalar(out, entity.mana_value)
    pt = entity.pt
    if pt is None:
        out += _NO_PT
        return
    for pair in (pt.base, pt.boosts, pt.counters):
        _push_scalar(out, pair[0])
        _push_scalar(out, pair[1])
    out.append(1.0)


def _push_overlay(
    out: list[float], entity: EntityState, record: EffectRecord, *,
    masked_keywords: frozenset[str], layout: str | None,
) -> None:
    """What the game has done to the card: zone, status, counters, combat,
    temporary grants, and its relation to the acting player and the targets."""
    state = record.state
    out += _one_hot(entity.zone, ZONES)
    out.append(1.0 if entity.tapped else 0.0)
    out.append(1.0 if entity.sick else 0.0)
    out.append(1.0 if entity.face_down else 0.0)
    _push_scalar(out, entity.damage)
    _push_counters(out, entity.counters, layout)
    combat = entity.combat
    out.append(1.0 if combat and combat.attacking else 0.0)
    out.append(1.0 if combat and combat.blocking else 0.0)
    out.append(1.0 if combat and combat.became_blocked else 0.0)
    _push_scalar(out, len(combat.blocked_by) if combat else 0)
    out.append(1.0 if entity.attached_to else 0.0)
    granted = entity.granted_temporary.keywords
    if masked_keywords:
        granted = set(granted) - set(masked_keywords)
    if layout is EARLIER_LAYOUT:
        out += _multi_hot(granted, OVERLAY_KEYWORDS)
    else:
        out += _multi_hot(
            [overlay_keyword(keyword) for keyword in granted], OVERLAY_KEYWORDS_GEN2,
        )
    _push_scalar(out, len(entity.granted_temporary.abilities))
    out.append(1.0 if state.controller_tag(
        entity.controller, record.actor_player) == "mine" else 0.0)
    out.append(1.0 if entity.id in state.refs.targets else 0.0)
    out.append(1.0 if entity.id == state.refs.source else 0.0)


def _push_counters(out: list[float], counters, layout: str | None) -> None:
    if not counters:
        # The common case by far, and every one of those scalars is zero.
        out += _NO_COUNTERS
        return
    if layout is not EARLIER_LAYOUT:
        renamed: dict[str, int] = {}
        for name, value in counters.items():
            name = COUNTER_ALIASES.get(name, name)
            renamed[name] = renamed.get(name, 0) + value
        counters = renamed
    for counter in COUNTER_TYPES:
        _push_scalar(out, counters.get(counter, 0))
    _push_scalar(out, sum(
        value for name, value in counters.items() if name not in COUNTER_TYPES
    ))


def _push_stack_extras(out: list[float], entity: EntityState) -> None:
    extras = entity.stack_extras
    if extras is None:
        out += _NO_STACK_EXTRAS
        return
    _push_scalar(out, len(extras.targets))
    _push_scalar(out, sum(extras.per_target_amounts.values()))
    _push_scalar(out, sum(extras.up_to_counts.values()))


def _push_comparable_amounts(out: list[float], entity: EntityState) -> None:
    """The gen-2 layout's thermometers: total power, total toughness, and the
    toughness left after marked damage."""
    if entity.pt is None:
        out += _NO_THERMOMETER * 3
        return
    power, toughness = entity.pt.total
    out += thermometer(power)
    out += thermometer(toughness)
    out += thermometer(toughness - entity.damage)


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
    has_line=None,
    candidate_index: int | None = None,
    context_dropout: float = 0.0,
    rng: random.Random | None = None,
    masked_keywords: dict[str, frozenset[str]] | None = None,
    stripped_keywords: dict[str, frozenset[str]] | None = None,
    keyword_of=None,
    options_for=None,
    layout: str | None = INPUT_LAYOUT,
) -> EffectHeadInput:
    """Lay one record out as the token surface.

    Args:
        e_for: ``ProvenanceKey -> tuple[float, ...] | int | None``, the ability
            lookup. A tuple is the vector itself, read from the cache; an ``int``
            is a row of a matrix supplied at collate time, which is how the
            training loop keeps ``e`` a live encoder output the gradient can
            flow back through. A key that resolves to neither contributes a zero
            vector rather than dropping the token, so the entity keeps its shape.
        e_dim: bottleneck width, for the zero vectors above.
        has_line: ``ProvenanceKey -> bool``, whether the key maps to a rendered
            line at all. An entity key with no line contributes no token:
            Forge attaches an implicit cast-this-permanent object to every
            permanent, the converter renders no line for it, and a token for
            it is board content no rules text produced. Distinct from ``e_for``
            returning None, which keeps a zero token so a variant that masks
            ``e`` keeps the geometry. None keeps every key.
        candidate_index: a ``playability``/``decision`` record trains as one
            example per candidate; this selects which candidate's ``e`` sits in
            ``[ACT]``.
        context_dropout: probability of dropping each *context* ability token
            during training. The acting entity's tokens are never dropped —
            dropping the ability under study would make the record unanswerable.
        masked_keywords: per-entity keywords to hide from the temporary-grant
            channel.
        stripped_keywords: per-entity keywords to remove from the entity's model
            input **entirely** — gate 2's perturbation. It hides the keyword
            from the overlay the way ``masked_keywords`` does *and* drops the
            ability slot whose line is that keyword, because a keyword the card
            prints reaches the model as an ability token and clearing only the
            overlay would leave it in. The two parameters stay separate because
            a ``continuous`` record masks what its own static contributed, which
            is a grant, and must not also silence a printed line that happens to
            name the same keyword.
        keyword_of: ``ProvenanceKey -> str | None``, which keyword a line *is*.
            Required for ``stripped_keywords`` to reach the ability channel;
            without it only the overlay is stripped.
        layout: the input layout (FR-063c) — :data:`INPUT_LAYOUT`, or
            :data:`EARLIER_LAYOUT` for a checkpoint that recorded none.
        options_for: ``ProvenanceKey -> tuple[ProvenanceKey, ...]``, the keys of
            the charm-mode ``option`` lines that follow the key's line in its
            sidecar (FR-063a). A snapshot names an entity's abilities by root
            key only, so each such line gets its own ``ABILITY`` slot right
            after its root's, flagged ``option`` and continuing the block's
            positions. None adds none, which is every gen-1 sidecar anyway.
    """
    rng = rng or random.Random()
    masked = masked_keywords or {}
    stripped = stripped_keywords or {}
    zero = tuple([0.0] * e_dim)
    slots: list[Slot] = []
    position = 0

    slots.append(Slot(
        kind=SlotKind.GLOBAL, position=position,
        features=global_features(record, layout=layout),
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
            features=player_features(player, record, layout=layout),
            player_id=player.id,
        ))
        position += 1

    source_id = record.state.refs.source
    for entity in record.state.entities:
        gone = stripped.get(entity.id, frozenset())
        entity_slots.append(len(slots))
        slots += _entity_block(
            entity, record, e_for=e_for, zero=zero,
            masked=masked.get(entity.id, frozenset()) | gone, layout=layout,
            options_for=options_for,
            keeps=lambda key, entity=entity, gone=gone: _keeps_ability(
                key, has_line=has_line, gone=gone, keyword_of=keyword_of,
                dropout=context_dropout if entity.id != source_id else 0.0, rng=rng,
            ),
        )

    return EffectHeadInput(
        slots=tuple(slots),
        record_kind=record.kind,
        actor_player=record.actor_player,
        entity_slots=tuple(entity_slots),
    )


def _entity_block(
    entity: EntityState, record: EffectRecord, *, e_for, zero, masked, layout,
    options_for, keeps,
) -> list[Slot]:
    """One entity's ``CARD`` slot, then an ``ABILITY`` slot per ability it keeps,
    each charm root followed by its mode lines.

    Position ids restart at the block: an entity plus its abilities is read the
    same way wherever the entity sits on the board.
    """
    block = [Slot(
        kind=SlotKind.CARD, position=0,
        features=card_features(entity, record, masked_keywords=masked, layout=layout),
        entity_id=entity.id,
    )]
    for key in _ability_keys(entity):
        if not keeps(key):
            continue
        modes = options_for(key) if options_for is not None else ()
        for mode_key, option in ((key, False), *((mode, True) for mode in modes)):
            resolved = e_for(mode_key)
            block.append(Slot(
                kind=SlotKind.ABILITY, position=len(block),
                # Explicitly against None: row 0 is a valid matrix row and
                # ``or`` would read it as a miss.
                e=zero if resolved is None else resolved,
                entity_id=entity.id, option=option,
            ))
    return block


def _keeps_ability(key, *, has_line, gone, keyword_of, dropout: float, rng) -> bool:
    """Whether one of an entity's abilities gets a slot.

    ``dropout`` is zero for the acting entity: dropping the ability under study
    would make the record unanswerable.
    """
    if has_line is not None and not has_line(key):
        return False
    if gone and keyword_of is not None and keyword_of(key) in gone:
        # Dropped rather than zeroed: a zero token would still tell the head
        # this creature has an ability here, and gate 2 is asking what the
        # board looks like without the keyword at all.
        return False
    return not (dropout > 0.0 and rng.random() < dropout)


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
):
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


def _first_vector(keys, e_for, zero: tuple[float, ...]):
    """The first key that resolves; zeros if none do.

    A rendered line merged from several traits carries several keys and they all
    name the same cache row, so the first hit is the line's vector.
    """
    for key in keys:
        vector = e_for(key)
        if vector is not None:
            return vector
    return zero


def continuous_masked_keywords(record: EffectRecord) -> dict[str, frozenset[str]]:
    """Keywords a ``continuous`` record's own static contributed (FR-020).

    Removed from the entity inputs so the model is not asked to predict a state
    that already contains its answer. Structured features stay: the static
    changed keywords, not the card's printed type line.
    """
    payload = record.payload
    if record.kind is not RecordKind.CONTINUOUS or not isinstance(
        payload, ContinuousPayload,
    ):
        return {}
    return {
        contribution.entity: frozenset(contribution.keywords)
        for contribution in payload.contributions
        if contribution.keywords
    }


def feature_widths(records, *, layout: str | None = INPUT_LAYOUT) -> dict[SlotKind, int]:
    """Each slot kind's width, measured from real records.

    Measured rather than declared: the vocabularies the features are built from
    live in this module, and a constant elsewhere would be a second place to
    keep them in step. A player's and a card's width need a record that has one.
    """
    probe = records[0]
    widths = {
        SlotKind.GLOBAL: len(global_features(probe, layout=layout)),
        SlotKind.ACT: len(act_features(probe)),
        SlotKind.PLAYER: 0,
        SlotKind.CARD: 0,
    }
    for record in records:
        if record.state.players:
            widths[SlotKind.PLAYER] = len(
                player_features(record.state.players[0], record, layout=layout)
            )
        if record.state.entities:
            widths[SlotKind.CARD] = len(
                card_features(record.state.entities[0], record, layout=layout)
            )
        if widths[SlotKind.PLAYER] and widths[SlotKind.CARD]:
            break
    return widths
