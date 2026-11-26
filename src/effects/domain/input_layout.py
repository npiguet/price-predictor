"""The token surface's layout, and the slots that are no card.

    [GLOBAL] [ACT] [PLAYER] [PLAYER] [CARD] e e … [CARD] e …

Position ids reset at each ``[CARD]``, so an entity and its abilities read the
same wherever the entity sits on the board. Every record kind is a variant of
this one surface. Controller tags are derived here, relative to the record's
``actor_player``, so one snapshot serves records with different actors.

This module holds the slot kinds, the :class:`Slot` and :class:`EffectHeadInput`
values, the scalar encoding every slot's features are written in, and the
``[GLOBAL]``, ``[ACT]`` and ``[PLAYER]`` slots' features; a card's are
:mod:`effects.domain.card_features`'.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

from effects.domain.event_schema import Event, signed_delta
from effects.domain.input_vocabularies import (
    COMBAT_SUBSTEPS,
    DAY_NIGHT,
    EVENT_TYPES,
    MOMENTS,
    PENDING_EVENT_MODES,
    PHASES,
    PLAYABILITY_SUBKINDS,
    RECORD_KINDS,
    THIS_TURN_COUNTERS,
    TURN_COUNTS,
    ZONES,
    one_hot,
    one_hot_width,
    phase_in,
)
from effects.domain.records import (
    EffectRecord,
)
from effects.domain.state_snapshot import (
    COLORS,
    PlayerState,
)


class SlotKind(IntEnum):
    GLOBAL = 0
    ACT = 1
    PLAYER = 2
    CARD = 3
    ABILITY = 4


#: The largest magnitude a count or a signed delta is scored at, by the field
#: losses, the floor and the evaluation's deviance alike. Above it an amount is
#: the game state compounding, and one such target would dominate the count
#: loss.
MAGNITUDE_CAP = 40.0

THERMOMETER_STEPS = int(MAGNITUDE_CAP)


def thermometer(value: float) -> list[float]:
    """``value`` as :data:`THERMOMETER_STEPS` positions, position k set when
    ``value >= k``, so "amount >= t" is one dot product with a one-hot of t."""
    return [1.0 if value >= k else 0.0 for k in range(1, THERMOMETER_STEPS + 1)]


NO_THERMOMETER: tuple[float, ...] = (0.0,) * THERMOMETER_STEPS


#: What :func:`push_scalar` appends per scalar.
SCALAR_WIDTH = 2


def push_scalar(out: list[float], value: float) -> None:
    """A scalar as raw plus signed ``log1p``, onto the caller's list; nothing
    is binned (FR-074). Appended rather than returned: a list per scalar made
    the allocation the cost."""
    raw = float(value)
    out.append(raw)
    out.append(math.copysign(math.log1p(abs(raw)), raw))


def push_optional(out: list[float], value: float | None) -> None:
    """A present flag, then the scalar (0 when absent)."""
    out.append(0.0 if value is None else 1.0)
    push_scalar(out, 0 if value is None else value)


@dataclass(frozen=True, slots=True)
class Slot:
    """One token of the input surface."""

    kind: SlotKind
    position: int
    features: tuple[float, ...] = ()
    #: An ``ACT`` or ``ABILITY`` slot's ability vector as a row of the batch's
    #: live encoder output, in training; None where ``e_vector`` holds it.
    e_row: int | None = None
    #: An ``ACT`` or ``ABILITY`` slot's ability vector as values (from the
    #: cache, or the zero vector); None where ``e_row`` names it. At most one of
    #: the two is set.
    e_vector: tuple[float, ...] | None = None
    entity_id: str | None = None
    player_id: str | None = None
    #: A charm mode's ``option`` line (FR-063a); the flag plus following its
    #: root's slot is the whole link between a mode and its charm.
    option: bool = False
    #: A ``CARD`` slot's subtypes, as rows of the subtype table.
    subtypes: tuple[int, ...] = ()
    #: A ``CARD`` slot's chosen creature types, as rows of the same table.
    chosen_types: tuple[int, ...] = ()
    #: A face-down ``CARD`` slot's real face's subtypes, as rows of the same
    #: table; the model marks them apart from ``subtypes`` by one learned vector.
    real_face_subtypes: tuple[int, ...] = ()
    #: A ``CARD`` slot's named counters other than keyword counters, as
    #: ``(row of the counter table, count)``.
    counters: tuple[tuple[int, float], ...] = ()

    @property
    def subject_id(self) -> str | None:
        """The entity or player a ``CARD`` or ``PLAYER`` slot stands for: the
        id its per-entity targets and outputs are keyed by."""
        return self.entity_id or self.player_id


@dataclass(frozen=True, slots=True)
class EffectHeadInput:
    """The full token surface for one record."""

    slots: tuple[Slot, ...]
    #: Slot indices of the per-entity head's targets: every CARD and PLAYER.
    entity_slots: tuple[int, ...] = field(default_factory=tuple)
    #: ``(from slot, to slot, relation type)`` triples over ``slots`` (root spec § 10.1).
    relations: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)
    #: The entities whose strip left a stripped keyword's statement in place:
    #: an ability slot that states it beside other keywords stays (gate 2
    #: leaves such a carrier unqualified).
    unremovable: tuple[str, ...] = field(default_factory=tuple)


def _event_amount(event: Event) -> float:
    """The amount a pending event carries: its ``amount``, else its ``delta``
    signed by its mode (``event_schema.signed_delta``: a trigger's ``LifeLost 3``
    is -3, its ``LifeGained 3`` +3)."""
    for name in ("amount", "delta"):
        value = event.params.get(name)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value) if name == "amount" else signed_delta(event)
    return 0.0


def _push_pending_event_globals(out: list[float], pending: Event | None) -> None:
    """The pending event's type, amount and mode, and whether it is combat
    damage (``in.pending_event_details``); zeros without one. A trigger's
    fired bit hangs on them ("whenever CARDNAME deals combat damage"), and
    a ``state_flag_change`` is a phase, a room entered or an attacker
    unblocked only by its mode."""
    if pending is None:
        out += _NO_PENDING_EVENT
        return
    out += one_hot(pending.type.value, EVENT_TYPES)
    push_scalar(out, _event_amount(pending))
    out += one_hot(pending.params.get("mode"), PENDING_EVENT_MODES)
    out.append(1.0 if pending.params.get("combat") is True else 0.0)
    out += one_hot(pending.params.get("from_zone"), ZONES)
    out += one_hot(pending.params.get("to_zone"), ZONES)


#: The pending-event block of a record without one.
_NO_PENDING_EVENT: tuple[float, ...] = (0.0,) * (
    one_hot_width(EVENT_TYPES) + SCALAR_WIDTH + one_hot_width(PENDING_EVENT_MODES) + 1
    + 2 * one_hot_width(ZONES)
)


def global_features(record: EffectRecord) -> tuple[float, ...]:
    """Turn structure, day or night, the pending event, plus the record's own
    kind flags, which stay last (:data:`KIND_FLAG_WIDTH`)."""
    state = record.state
    g = state.global_
    out: list[float] = []
    push_scalar(out, g.turn)
    push_scalar(out, g.stack_size)
    out += one_hot(phase_in(g.phase), PHASES)
    out += one_hot(g.combat_substep, COMBAT_SUBSTEPS)
    out.append(1.0 if g.active == record.actor_player else 0.0)
    out.append(1.0 if g.priority == record.actor_player else 0.0)
    push_scalar(out, len(g.emblems))
    out += one_hot(g.day_night or "neither", DAY_NIGHT)
    _push_pending_event_globals(out, state.pending_event)
    out += one_hot(record.kind.value, RECORD_KINDS)
    out += one_hot(
        record.moment.value if record.moment else None, MOMENTS,
    )
    out += one_hot(
        record.subkind.value if record.subkind else None,
        PLAYABILITY_SUBKINDS,
    )
    return tuple(out)


#: Width of the kind, moment and subkind one-hots that end every ``[GLOBAL]``
#: slot; the ``no-state`` baseline keeps exactly these (FR-094).
KIND_FLAG_WIDTH = sum(map(one_hot_width, (RECORD_KINDS, MOMENTS, PLAYABILITY_SUBKINDS)))

ACT_SLOT = 1


def act_features(record: EffectRecord) -> tuple[float, ...]:
    """Announced values and the dice and coins the resolution rolled and
    flipped (``in.random_results``); zeros for ``combat`` and the
    ``attackers``/``blockers`` subkinds, where no single line acts. Whether a
    cost half's spell resolved is an output, not an input (root spec § 11.5,
    FR-052)."""
    if not record.has_act_line:
        return tuple([0.0] * _ACT_FEATURE_WIDTH)
    refs = record.state.refs
    out: list[float] = [1.0]  # slot is populated
    push_scalar(out, refs.x if refs.x is not None else 0)
    out.append(1.0 if refs.x is not None else 0.0)
    push_scalar(out, len(refs.modes))
    push_scalar(out, len(refs.as_cast))
    push_scalar(out, len(refs.choices))
    _push_random_results(out, refs.random)
    return tuple(out)


def _push_random_results(out: list[float], random_results) -> None:
    """The dice rolled (how many, their sum, the highest) and the coin flips
    won and lost; zeros where the record carries none."""
    dice = random_results.dice if random_results is not None else ()
    push_scalar(out, len(dice))
    push_scalar(out, sum(dice))
    push_scalar(out, max(dice, default=0))
    push_scalar(out, random_results.coin_flips_won if random_results is not None else 0)
    push_scalar(out, random_results.coin_flips_lost if random_results is not None else 0)


#: The populated and X-present flags, the X, mode, target and choice scalars,
#: and the five random-result scalars.
_ACT_FEATURE_WIDTH = 2 + 9 * SCALAR_WIDTH


def player_features(player: PlayerState, record: EffectRecord) -> tuple[float, ...]:
    """One player's resources and designations, whether the pending event
    names them as its subject, plus their relation to the acting player and
    to the turn (root spec § 9.4, § 9.5)."""
    out: list[float] = []
    for value in (player.life, player.hand, player.library, player.graveyard,
                  player.poison, player.energy):
        push_scalar(out, value)
    for key in THIS_TURN_COUNTERS:
        push_scalar(out, player.this_turn.get(key, 0))
    # The land plays the turn allows, beside the lands played (FR-044); 0 and
    # the flag where a static lifts the limit.
    push_scalar(out, player.max_land_plays or 0)
    out.append(1.0 if player.land_plays_unlimited else 0.0)
    for color in COLORS:
        push_scalar(out, player.floating_mana.get(color, 0))
    for color in COLORS:
        push_scalar(out, player.untapped_production.get(color, 0))
    push_scalar(out, player.speed)
    push_scalar(out, player.rad)
    out.append(1.0 if player.city_blessing else 0.0)
    out.append(1.0 if player.monarch else 0.0)
    out.append(1.0 if player.initiative else 0.0)
    counts = player.turn_counts
    for name in TURN_COUNTS:
        push_scalar(out, getattr(counts, name) if counts is not None else 0)
    out.append(1.0 if record.state.names_as_pending_subject(player.id) else 0.0)
    pending = record.state.pending_event
    out += pending_param_bits(pending, player.id, ("defender", "cause")) if pending else (0.0, 0.0)
    g = record.state.global_
    out.append(1.0 if player.id == record.actor_player else 0.0)
    out.append(0.0 if player.id == record.actor_player else 1.0)
    out.append(1.0 if player.id == g.active else 0.0)
    out.append(1.0 if player.id == g.priority else 0.0)
    out.append(1.0 if player.id in record.state.refs.as_cast else 0.0)
    out += thermometer(player.life)
    return tuple(out)


def pending_param_bits(pending: Event, ref: str, names: tuple[str, ...]) -> list[float]:
    """Whether the pending event's params ``names`` (each a ref or a list of
    refs) name ``ref``."""
    params = pending.params
    return [1.0 if ref in _refs_of(params.get(name)) else 0.0 for name in names]


def _refs_of(value) -> tuple[str, ...]:
    """The refs a pending-event param holds: one, a list, or none."""
    if isinstance(value, str):
        return (value,)
    return tuple(value) if isinstance(value, list) else ()
