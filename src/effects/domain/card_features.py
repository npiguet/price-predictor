"""A ``[CARD]`` slot's features, part by part (root spec § 9.4).

Each part appends a fixed width; :data:`CARD_PARTS` names them in order, and a
checkpoint records that order.
"""

from __future__ import annotations

from dataclasses import dataclass

from effects.domain.input_layout import (
    NO_THERMOMETER,
    SCALAR_WIDTH,
    pending_param_bits,
    push_optional,
    push_scalar,
    thermometer,
)
from effects.domain.input_vocabularies import (
    CARD_TYPES,
    CHOSEN_DIRECTIONS,
    CHOSEN_PARITIES,
    COLOR_NAMES,
    DESIGNATIONS,
    EFFECT_DURATIONS,
    ENTITY_ROLES,
    FACE_STATES,
    MANA_SYMBOLS,
    SUPERTYPES,
    TIER_ROLE,
    ZONES,
    counter_id,
    is_keyword_counter,
    multi_hot,
    one_hot,
    one_hot_width,
    positions,
    spelling,
    subtype_id,
    vocabulary_slot,
)
from effects.domain.kept_keyword_flags import KEPT_KEYWORD_FLAGS
from effects.domain.keyword_names import overlay_keyword
from effects.domain.records import (
    EffectRecord,
    Payload,
    ResolutionPayload,
)
from effects.domain.state_snapshot import (
    COLORS,
    CardChoices,
    CommandZoneObject,
    CostsPaid,
    EntityState,
    StateSnapshot,
)

#: Power and toughness of the base, boosts and counters, then the present flag.
_NO_PT: tuple[float, ...] = (0.0,) * (3 * 2 * SCALAR_WIDTH + 1)
#: Targets, per-target amounts and up-to counts.
_NO_STACK_EXTRAS: tuple[float, ...] = (0.0,) * (3 * SCALAR_WIDTH)


# ── the card slot ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CardContext:
    """What a ``[CARD]`` slot's parts read, built once per slot: the entity,
    the record's snapshot (its refs, its pending event, the turn and the
    command zone), its acting player and its payload, and the keywords gate
    2's strip removes from the keyword flags."""

    entity: EntityState
    state: StateSnapshot
    actor_player: str
    payload: Payload
    masked_keywords: frozenset[str] = frozenset()

    @classmethod
    def of(
        cls, entity: EntityState, record: EffectRecord,
        masked_keywords: frozenset[str] = frozenset(),
    ) -> CardContext:
        """``entity``'s context in ``record``."""
        return cls(entity=entity, state=record.state, actor_player=record.actor_player,
                   payload=record.payload, masked_keywords=masked_keywords)


def card_features(
    entity: EntityState,
    record: EffectRecord,
    *,
    masked_keywords: frozenset[str] = frozenset(),
) -> tuple[float, ...]:
    """One entity's structured characteristics and state (root spec § 9.4);
    ``masked_keywords`` removes keywords from its keyword flags, gate 2's
    strip (:meth:`SurfaceBuilder.build`)."""
    card = CardContext.of(entity, record, masked_keywords)
    out: list[float] = []
    for _, push in _CARD_PARTS:
        push(out, card)
    return tuple(out)


def card_segments(
    entity: EntityState, record: EffectRecord,
) -> dict[str, slice]:
    """Where each named part of :func:`card_features` sits; every part has a
    fixed width, so any entity gives the same slices."""
    card = CardContext.of(entity, record)
    out: list[float] = []
    segments: dict[str, slice] = {}
    for name, push in _CARD_PARTS:
        start = len(out)
        push(out, card)
        segments[name] = slice(start, len(out))
    return segments


def _push_colors_and_mana_value(out, colors, mana_value) -> None:
    """The colours' multi-hot and the mana value, as a card's characteristics
    and a face-down card's real face read them."""
    out += multi_hot(colors, COLORS)
    push_scalar(out, mana_value)


def _push_characteristics(out: list[float], card: CardContext) -> None:
    """Types, supertypes, colours, mana value and the P/T breakdown; whether
    it is a copy and whether a token (subtypes are the slot's ids)."""
    entity = card.entity
    out += multi_hot(entity.types, CARD_TYPES)
    out += multi_hot(entity.supertypes, SUPERTYPES)
    _push_colors_and_mana_value(out, entity.colors, entity.mana_value)
    pt = entity.pt
    if pt is None:
        out += _NO_PT
    else:
        for pair in (pt.base, pt.boosts, pt.counters):
            push_scalar(out, pair[0])
            push_scalar(out, pair[1])
        out.append(1.0)
    out.append(1.0 if entity.copy_source is not None else 0.0)
    out.append(1.0 if entity.token_script_id is not None else 0.0)


def _push_face_state(out: list[float], card: CardContext) -> None:
    """Face state one-hot (``in.face_state``). A face-down card's real face
    is the next part's; its lines enter as inactive ability slots."""
    out += one_hot(card.entity.face_state, FACE_STATES)


#: What :func:`_push_real_face` writes for a card with no real face.
_NO_REAL_FACE: tuple[float, ...] = (0.0,) * (
    one_hot_width(CARD_TYPES) + one_hot_width(COLORS) + 4 * SCALAR_WIDTH + 1
)


def _push_real_face(out: list[float], card: CardContext) -> None:
    """A face-down card's real face: its types, colours, mana value, printed
    power and toughness (a present flag, then two scalars) and how many
    subtypes it has (``in.face_state``). Turning the card face up makes it
    this, so ``face_after``, the base power and toughness and the type
    changes depend on it; the face-down characteristics stay in
    ``characteristics``. Zeros on every other card. Which subtypes they are
    is the slot's ``real_face_subtypes`` (:func:`real_face_subtype_rows`),
    not a feature."""
    entity = card.entity
    face = entity.real_face
    if face is None:
        out += _NO_REAL_FACE
        return
    out += multi_hot(face.types, CARD_TYPES)
    _push_colors_and_mana_value(out, face.colors, face.mana_value)
    power, toughness = face.base_pt if face.base_pt is not None else (0, 0)
    out.append(0.0 if face.base_pt is None else 1.0)
    push_scalar(out, power)
    push_scalar(out, toughness)
    push_scalar(out, len(face.subtypes))


def _push_mana_cost(out: list[float], card: CardContext) -> None:
    """A scalar per mana symbol kind and one for every other symbol, after a
    flag for having a mana cost at all (``in.mana_cost``)."""
    entity = card.entity
    cost = entity.mana_cost
    out.append(0.0 if cost is None else 1.0)
    cost = cost or {}
    for symbol in MANA_SYMBOLS:
        push_scalar(out, cost.get(symbol, 0))
    push_scalar(out, sum(n for symbol, n in cost.items() if symbol not in _MANA_SYMBOL_SET))


_MANA_SYMBOL_SET = frozenset(MANA_SYMBOLS)


def _push_printed_loyalty_defense(out: list[float], card: CardContext) -> None:
    """Printed loyalty and defense, each a scalar with a thermometer, as
    toughness has (``in.printed_loyalty_defense``)."""
    entity = card.entity
    for value in (entity.printed_loyalty, entity.printed_defense):
        push_optional(out, value)
        out += NO_THERMOMETER if value is None else thermometer(value)


def _push_status(out: list[float], card: CardContext) -> None:
    """Zone, tapped, summoning sick, marked damage, combat. Summoning sickness
    reads on a permanent only (:attr:`EntityState.on_battlefield`): Forge
    writes it in every zone, where it would say "is a creature card"."""
    entity = card.entity
    out += one_hot(entity.zone, ZONES)
    out.append(1.0 if entity.tapped else 0.0)
    out.append(1.0 if entity.sick and entity.on_battlefield else 0.0)
    push_scalar(out, entity.damage)
    combat = entity.combat
    out.append(1.0 if combat and combat.attacking else 0.0)
    out.append(1.0 if combat and combat.blocking else 0.0)
    out.append(1.0 if combat and combat.became_blocked else 0.0)
    push_scalar(out, len(combat.blocked_by) if combat else 0)


def _push_role(out: list[float], card: CardContext) -> None:
    """Why the entity is in the snapshot, over :data:`ENTITY_ROLES`
    (``in.role``). A last-known copy keeps the zone it last had, so without
    its role one with zone ``battlefield`` reads as a permanent."""
    entity = card.entity
    out += one_hot(entity.role or TIER_ROLE, ENTITY_ROLES)


def keywords_of(entity: EntityState) -> set[str]:
    """Every keyword the card has now, in :func:`overlay_keyword`'s spelling:
    its keyword list, its keyword lines in force and its keyword counters."""
    found = {overlay_keyword(keyword) for keyword in entity.granted_temporary.keywords}
    found.update(overlay_keyword(line.keyword) for line in entity.keyword_lines
                 if line.in_force and line.keyword)
    found.update(overlay_keyword(name) for name, count in entity.counters.items()
                 if count > 0 and is_keyword_counter(name))
    return found


def _push_keyword_flags(out: list[float], card: CardContext) -> None:
    """One flag per kept keyword (root spec § 9.6), and nothing for any other:
    a keyword without a flag is read through its line."""
    entity = card.entity
    flags = [0.0] * len(KEPT_KEYWORD_FLAGS)
    slots = positions(KEPT_KEYWORD_FLAGS)
    hidden = {overlay_keyword(keyword) for keyword in card.masked_keywords}
    for keyword in keywords_of(entity) - hidden:
        index = slots.get(spelling(keyword))
        if index is not None:
            flags[index] = 1.0
    out += flags


#: Forge's hidden keywords, as it writes them, that forbid blocking and
#: attacking outright: every unconditional spelling the review pilots'
#: ``granted_temporary.hidden_keywords`` carry.
CANT_BLOCK_KEYWORDS = frozenset({"CARDNAME can't block.", "CARDNAME can't attack or block."})
CANT_ATTACK_KEYWORDS = frozenset({"CARDNAME can't attack.", "CARDNAME can't attack or block."})


def _push_hidden_keywords(out: list[float], card: CardContext) -> None:
    """Can't block, can't attack: Forge's hidden keywords no line carries, a
    pump's or a static's "can't block this turn", which the blocker and
    attacker legality outputs turn on. No other hidden keyword is read."""
    entity = card.entity
    hidden = {keyword.strip() for keyword in entity.granted_temporary.hidden_keywords or ()}
    out.append(1.0 if hidden & CANT_BLOCK_KEYWORDS else 0.0)
    out.append(1.0 if hidden & CANT_ATTACK_KEYWORDS else 0.0)


def _push_relation_to_act(out: list[float], card: CardContext) -> None:
    """Its relation to the acting player, the targets and the source, then the
    turn, priority and control inputs of root spec § 9.5."""
    entity = card.entity
    state = card.state
    g = state.global_
    out.append(1.0 if entity.controller == card.actor_player else 0.0)
    out.append(1.0 if state.targets_card(entity.id) else 0.0)
    out.append(1.0 if entity.id == state.refs.source else 0.0)
    out.append(1.0 if entity.controller == g.active else 0.0)
    out.append(1.0 if entity.controller == g.priority else 0.0)
    out.append(1.0 if entity.owner is not None and entity.owner == card.actor_player else 0.0)
    out.append(1.0 if entity.owner is not None and entity.owner != entity.controller else 0.0)


def _push_designations(out: list[float], card: CardContext) -> None:
    entity = card.entity
    out += multi_hot(entity.designations, DESIGNATIONS)


def _push_x_paid(out: list[float], card: CardContext) -> None:
    entity = card.entity
    push_optional(out, entity.x_paid)


def _push_chosen(out: list[float], card: CardContext) -> None:
    """What the card remembers choosing (``in.chosen_values``): colours, card
    types, a number, whether a player, cards or names, a direction, even or
    odd, a sector or a mode, and stored die results. A creature type is a row
    of the subtype table, and a player, card or name a relation."""
    entity = card.entity
    chosen = entity.chosen
    if chosen is None:
        out += _NO_CHOSEN
        return
    _push_chosen_kinds(out, chosen)
    push_optional(out, chosen.number)
    out.append(1.0 if chosen.player is not None else 0.0)
    push_scalar(out, len(chosen.cards or ()))
    push_scalar(out, len(chosen.names or ()))
    out += one_hot(chosen.direction, CHOSEN_DIRECTIONS)
    out += one_hot(chosen.even_odd, CHOSEN_PARITIES)
    out.append(1.0 if chosen.sector else 0.0)
    out.append(1.0 if chosen.mode else 0.0)
    rolls = chosen.stored_rolls or ()
    push_scalar(out, len(rolls))
    push_scalar(out, sum(rolls))


def _push_chosen_kinds(out: list[float], chosen: CardChoices) -> None:
    """The chosen colours and the chosen types that are card types."""
    colors = [COLOR_NAMES.get(spelling(color), color) for color in chosen.colors or ()]
    out += multi_hot(colors, COLORS)
    card_types = [t for t in (chosen.type, chosen.type2)
                  if t is not None and vocabulary_slot(t, CARD_TYPES) is not None]
    out += multi_hot(card_types, CARD_TYPES)


#: The chosen block of a card that chose nothing.
_NO_CHOSEN: tuple[float, ...] = (0.0,) * (
    one_hot_width(COLORS) + one_hot_width(CARD_TYPES) + 1 + SCALAR_WIDTH + 1
    + 2 * SCALAR_WIDTH + one_hot_width(CHOSEN_DIRECTIONS)
    + one_hot_width(CHOSEN_PARITIES) + 2 + 2 * SCALAR_WIDTH
)


def chosen_subtypes(entity: EntityState) -> tuple[int, ...]:
    """The subtype rows of a card's chosen types that are no card type."""
    chosen = entity.chosen
    if chosen is None:
        return ()
    return tuple(
        subtype_id(t) for t in (chosen.type, chosen.type2)
        if t is not None and vocabulary_slot(t, CARD_TYPES) is None
    )


def real_face_subtype_rows(entity: EntityState) -> tuple[int, ...]:
    """The subtype rows of a face-down card's real face; none on any other."""
    face = entity.real_face
    return () if face is None else tuple(subtype_id(subtype) for subtype in face.subtypes)


def _paid_for(card: CardContext) -> CostsPaid | None:
    """What paid for the card, as the input reads it: nothing for the acting
    source on an effect half. A forced resolution skips the payment, so there
    "paid no mana" would tell a forced half from a live one (D16)."""
    if isinstance(card.payload, ResolutionPayload) and card.entity.id == card.state.refs.source:
        return None
    return card.entity.costs_paid


def _push_costs_paid(out: list[float], card: CardContext) -> None:
    """The mana spent per colour, the zone cast from and the mana rider, after
    a flag for having been cast (``in.costs_paid``); all zero for the acting
    source on an effect half (:func:`_paid_for`)."""
    paid = _paid_for(card)
    out.append(0.0 if paid is None else 1.0)
    mana = paid.mana if paid is not None else {}
    for color in COLORS:
        push_scalar(out, mana.get(color, 0))
    out += one_hot(paid.cast_from if paid is not None else None, ZONES)
    out.append(1.0 if paid is not None and paid.mana_rider else 0.0)


def _push_history(out: list[float], card: CardContext) -> None:
    """What the card did this turn (``in.history``)."""
    entity = card.entity
    history = entity.history
    for done in (
        history.entered_this_turn if history else False,
        history.attacked_this_turn if history else False,
        history.blocked_this_turn if history else False,
        history.loyalty_activated if history else False,
    ):
        out.append(1.0 if done else 0.0)


def _push_order_rank(out: list[float], card: CardContext) -> None:
    entity = card.entity
    push_optional(out, entity.order_rank)


def _push_stack_extras(out: list[float], card: CardContext) -> None:
    entity = card.entity
    extras = entity.stack_extras
    if extras is None:
        out += _NO_STACK_EXTRAS
        return
    push_scalar(out, len(extras.targets))
    push_scalar(out, sum(extras.per_target_amounts.values()))
    push_scalar(out, sum(extras.up_to_counts.values()))


def _push_pending_event(out: list[float], card: CardContext) -> None:
    """Whether a ``rewrite`` or ``trigger`` record has a pending event,
    whether this card is its subject and whether its source (the damage
    source: "whenever CARDNAME deals combat damage"); the event's type,
    amount, mode and combat bit are the ``[GLOBAL]`` slot's."""
    entity = card.entity
    state = card.state
    pending = state.pending_event
    if pending is None:
        out += (0.0,) * 6
        return
    out.append(1.0)
    out.append(1.0 if state.names_as_pending_subject(entity.id) else 0.0)
    out.append(1.0 if pending.params.get("source") == entity.id else 0.0)
    out += pending_param_bits(pending, entity.id, ("defender", "blockers", "cause"))


def command_zone_object(state: StateSnapshot, object_id: str) -> CommandZoneObject | None:
    """The floating effect or pending delayed trigger named ``object_id``;
    None for any other id."""
    g = state.global_
    for found in (*g.floating_effects, *g.delayed_triggers):
        if found.id == object_id:
            return found
    return None


def _push_command_zone(out: list[float], card: CardContext) -> None:
    """What a command-zone object's text cannot say (root spec § 9.3): its
    duration over the three values, and the amount left on a counting-down
    shield after a present flag; zeros on any other card. Its links are
    relations, and what it does is its source lines' ``e``."""
    found = command_zone_object(card.state, card.entity.id)
    out += one_hot(found.duration if found is not None else None, EFFECT_DURATIONS)
    push_optional(out, found.shield_left if found is not None else None)


def _push_comparable_amounts(out: list[float], card: CardContext) -> None:
    """Thermometers of total power, total toughness, and the toughness left
    after marked damage, subtracted here so the trunk need not learn to."""
    entity = card.entity
    if entity.pt is None:
        out += NO_THERMOMETER * 3
        return
    power, toughness = entity.pt.total
    out += thermometer(power)
    out += thermometer(toughness)
    out += thermometer(toughness - entity.damage)


#: :func:`card_features`, part by part, in order.
_CARD_PARTS = (
    ("characteristics", _push_characteristics),
    ("face_state", _push_face_state),
    ("real_face", _push_real_face),
    ("mana_cost", _push_mana_cost),
    ("printed_loyalty_defense", _push_printed_loyalty_defense),
    ("status", _push_status),
    ("role", _push_role),
    ("keyword_flags", _push_keyword_flags),
    ("hidden_keywords", _push_hidden_keywords),
    ("relation_to_act", _push_relation_to_act),
    ("designations", _push_designations),
    ("x_paid", _push_x_paid),
    ("chosen", _push_chosen),
    ("costs_paid", _push_costs_paid),
    ("history", _push_history),
    ("order_rank", _push_order_rank),
    ("stack_extras", _push_stack_extras),
    ("pending_event", _push_pending_event),
    ("command_zone", _push_command_zone),
    ("comparable_amounts", _push_comparable_amounts),
)
#: The card slot's parts by name, in order.
CARD_PARTS: tuple[str, ...] = tuple(name for name, _ in _CARD_PARTS)


def counter_rows(entity: EntityState) -> tuple[tuple[int, float], ...]:
    """A card's named counters but its keyword counters, as rows of the
    counter table with their counts; two spellings of one type add up."""
    rows: dict[int, float] = {}
    for name, count in entity.counters.items():
        if count and not is_keyword_counter(name):
            row = counter_id(name)
            rows[row] = rows.get(row, 0.0) + float(count)
    return tuple(rows.items())
