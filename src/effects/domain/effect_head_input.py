"""Snapshot to the effect head's one token surface.

    [GLOBAL] [ACT] [PLAYER] [PLAYER] [CARD] e e … [CARD] e …

Position ids reset at each ``[CARD]``, so an entity and its abilities read the
same wherever the entity sits on the board. Every record kind is a variant of
this one surface. Controller tags are derived here, relative to the record's
``actor_player``, so one snapshot serves records with different actors.
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


# Each vocabulary's one-hot ends in an "other" slot. Never edit these tuples:
# a checkpoint trained on them reads them bit for bit; a new spelling goes in a
# new InputLayout instead.

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
#: Keywords the overlay tracks individually; the eight damage-step keywords
#: must be among them, because gate 2 strips a granted one from this channel.
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

# Input layouts (FR-063c). Forge's own spellings of phases, counters and
# protection miss the vocabularies above and land in "other"; the gen-2 layout
# aliases them.

#: Where a checkpoint's ``training_settings`` record the layout it trained on.
INPUT_LAYOUT_KEY = "input_layout"

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


@dataclass(frozen=True, slots=True)
class InputLayout:
    """One input layout, as the features it switches (FR-063c)."""

    #: Recorded under :data:`INPUT_LAYOUT_KEY`; None for the earlier layout,
    #: which no checkpoint records.
    name: str | None
    forge_spellings: bool
    overlay_keywords: tuple[str, ...]
    #: Whether card slots end with thermometers of power, toughness and
    #: toughness left, and player slots with one of life.
    thermometers: bool

    @classmethod
    def of(cls, training_settings) -> InputLayout:
        """The layout a checkpoint's training settings record; the earlier one
        where they record none (gen-1, and gen-2 arms trained before the
        layout existed)."""
        if (training_settings or {}).get(INPUT_LAYOUT_KEY) is None:
            return EARLIER_LAYOUT
        return INPUT_LAYOUT


#: The layout every training run builds and records.
INPUT_LAYOUT = InputLayout(
    name="gen-2", forge_spellings=True, overlay_keywords=OVERLAY_KEYWORDS_GEN2,
    thermometers=True,
)
EARLIER_LAYOUT = InputLayout(
    name=None, forge_spellings=False, overlay_keywords=OVERLAY_KEYWORDS,
    thermometers=False,
)

#: The largest magnitude a count or a signed delta is scored at, by the field
#: losses, the floor and the evaluation's deviance alike. Above it an amount is
#: the game state compounding, and one such target would dominate the Poisson
#: loss.
MAGNITUDE_CAP = 40.0

THERMOMETER_STEPS = int(MAGNITUDE_CAP)


def thermometer(value: float) -> list[float]:
    """``value`` as :data:`THERMOMETER_STEPS` positions, position k set when
    ``value >= k``, so "amount >= t" is one dot product with a one-hot of t."""
    return [1.0 if value >= k else 0.0 for k in range(1, THERMOMETER_STEPS + 1)]


_NO_THERMOMETER: tuple[float, ...] = (0.0,) * THERMOMETER_STEPS


def _phase_in(phase: str | None, layout: InputLayout) -> str | None:
    if not layout.forge_spellings or phase is None:
        return phase
    return PHASE_ALIASES.get(phase.strip().lower(), phase)


@lru_cache(maxsize=None)
def _positions(vocabulary: tuple[str, ...]) -> dict[str, int]:
    """``normalized member -> its slot``, built once per vocabulary."""
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
    """A scalar as raw plus signed ``log1p``, onto the caller's list; nothing
    is binned (FR-074). Appended rather than returned: a list per scalar made
    the allocation the cost."""
    raw = float(value)
    out.append(raw)
    out.append(math.copysign(math.log1p(abs(raw)), raw))


_NO_PT: tuple[float, ...] = (0.0,) * 13
#: One scalar (two zeros) per counter type plus the catch-all.
_NO_COUNTERS: tuple[float, ...] = (0.0,) * (2 * (len(COUNTER_TYPES) + 1))
_NO_STACK_EXTRAS: tuple[float, ...] = (0.0,) * 6


def _multi_hot(present, vocabulary: tuple[str, ...]) -> list[float]:
    """Multi-hot over ``vocabulary`` plus a count of everything else."""
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


@dataclass(frozen=True, slots=True)
class Slot:
    """One token of the input surface."""

    kind: SlotKind
    position: int
    features: tuple[float, ...] = ()
    #: An ``ACT`` or ``ABILITY`` slot's ability vector: the values (from the
    #: cache) or, in training, an ``int`` row of the batch's live encoder
    #: output. Told apart by type, never by length.
    e: tuple[float, ...] | int | None = None
    entity_id: str | None = None
    player_id: str | None = None
    #: A charm mode's ``option`` line (FR-063a); the flag plus following its
    #: root's slot is the whole link between a mode and its charm.
    option: bool = False


@dataclass(frozen=True, slots=True)
class EffectHeadInput:
    """The full token surface for one record."""

    slots: tuple[Slot, ...]
    #: Slot indices of the per-entity head's targets: every CARD and PLAYER.
    entity_slots: tuple[int, ...] = field(default_factory=tuple)


_RECORD_KINDS: tuple[str, ...] = tuple(k.value for k in RecordKind)
_MOMENTS: tuple[str, ...] = tuple(m.value for m in Moment)
_PLAYABILITY_SUBKINDS: tuple[str, ...] = tuple(
    s.value for s in PlayabilitySubkind
)


def global_features(
    record: EffectRecord, *, layout: InputLayout = INPUT_LAYOUT,
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


#: Width of the kind, moment and subkind one-hots that end every ``[GLOBAL]``
#: slot; the ``no-state`` baseline keeps exactly these (FR-094).
KIND_FLAG_WIDTH = len(_RECORD_KINDS) + len(_MOMENTS) + len(_PLAYABILITY_SUBKINDS) + 3

ACT_SLOT = 1


def act_features(record: EffectRecord) -> tuple[float, ...]:
    """Announced values and the resolution outcome flag; zeros for
    ``combat`` and the ``attackers``/``blockers`` subkinds, where no single
    line acts."""
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
    layout: InputLayout = INPUT_LAYOUT,
) -> tuple[float, ...]:
    """One player's resources, plus their relation to the acting player."""
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
    if layout.thermometers:
        out += thermometer(player.life)
    return tuple(out)


def card_features(
    entity: EntityState,
    record: EffectRecord,
    *,
    masked_keywords: frozenset[str] = frozenset(),
    layout: InputLayout = INPUT_LAYOUT,
) -> tuple[float, ...]:
    """One entity's structured characteristics plus its overlay;
    ``masked_keywords`` removes keywords from the temporarily-granted channel."""
    out: list[float] = []
    _push_characteristics(out, entity)
    _push_overlay(out, entity, record, masked_keywords=masked_keywords, layout=layout)
    _push_stack_extras(out, entity)
    out += _damage_assignment_features(entity, record)
    out += _pending_event_features(entity, record)
    if layout.thermometers:
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
    masked_keywords: frozenset[str], layout: InputLayout,
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
    if layout.forge_spellings:
        granted = [overlay_keyword(keyword) for keyword in granted]
    out += _multi_hot(granted, layout.overlay_keywords)
    _push_scalar(out, len(entity.granted_temporary.abilities))
    out.append(1.0 if state.controller_tag(
        entity.controller, record.actor_player) == "mine" else 0.0)
    out.append(1.0 if entity.id in state.refs.targets else 0.0)
    out.append(1.0 if entity.id == state.refs.source else 0.0)


def _push_counters(out: list[float], counters, layout: InputLayout) -> None:
    if not counters:
        out += _NO_COUNTERS
        return
    if layout.forge_spellings:
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
    toughness left after marked damage, subtracted here so the trunk need not
    learn to."""
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
    layout: InputLayout = INPUT_LAYOUT,
) -> EffectHeadInput:
    """Lay one record out as the token surface.

    Args:
        e_for: ``ProvenanceKey -> tuple[float, ...] | int | None`` (see
            :attr:`Slot.e`). None gives a zero vector rather than dropping the
            token, so a variant that masks ``e`` keeps the geometry.
        has_line: ``ProvenanceKey -> bool``; a key with no rendered line (the
            implicit permanent spell) gets no token. None keeps every key.
        candidate_index: which candidate's ``e`` a ``decision`` record's
            ``[ACT]`` reads.
        context_dropout: per-token drop probability for every entity's
            abilities but the acting one's.
        masked_keywords: per-entity keywords to hide from the temporary-grant
            channel, e.g. a ``continuous`` record's own contribution.
        stripped_keywords: gate 2's perturbation: per-entity keywords removed
            from the overlay *and* as ability slots, since a printed keyword
            reaches the model as a token. Kept apart from ``masked_keywords``,
            which must not silence a printed line of the same keyword. The
            ability channel needs ``keyword_of``.
        keyword_of: ``ProvenanceKey -> str | None``, which keyword a line *is*.
        options_for: ``ProvenanceKey -> tuple[ProvenanceKey, ...]``, the
            charm-mode keys whose slots follow the key's (FR-063a).
        layout: :data:`INPUT_LAYOUT`, or the one a checkpoint recorded
            (:meth:`InputLayout.of`).
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

    return EffectHeadInput(slots=tuple(slots), entity_slots=tuple(entity_slots))


def _entity_block(
    entity: EntityState, record: EffectRecord, *, e_for, zero, masked, layout,
    options_for, keeps,
) -> list[Slot]:
    """One entity's ``CARD`` slot, then an ``ABILITY`` slot per ability it
    keeps, each charm root followed by its mode lines; positions restart."""
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
    """Whether one of an entity's abilities gets a slot."""
    if has_line is not None and not has_line(key):
        return False
    if gone and keyword_of is not None and keyword_of(key) in gone:
        # Dropped rather than zeroed: a zero token still says an ability is
        # here.
        return False
    return not (dropout > 0.0 and rng.random() < dropout)


def _ability_keys(entity: EntityState) -> tuple[ProvenanceKey, ...]:
    """An entity's ability tokens: printed and attachment-granted only;
    temporary grants ride the overlay (FR-073)."""
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
    """The first key that resolves; zeros if none do. A merged line's keys
    all name the same row."""
    for key in keys:
        vector = e_for(key)
        if vector is not None:
            return vector
    return zero


def continuous_masked_keywords(record: EffectRecord) -> dict[str, frozenset[str]]:
    """Keywords a ``continuous`` record's own static contributed (FR-020),
    hidden so the input does not already contain the answer."""
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


def feature_widths(records, *, layout: InputLayout = INPUT_LAYOUT) -> dict[SlotKind, int]:
    """Each slot kind's width, measured from real records; 0 for a kind no
    record has."""
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
