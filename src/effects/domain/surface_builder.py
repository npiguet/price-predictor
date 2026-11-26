"""Snapshot to the effect head's one token surface (:mod:`effects.domain.input_layout`).

The board's cards include the gen-3 snapshot's extras (each library's top card,
cards exiled with a permanent, a mutated permanent's components, the pending
event's last known subject) and, after them, a command-zone card per floating
effect or delayed trigger the snapshot lists no entity for, whose ability
slots are the source lines that made it (root spec § 9.3). Relations are built
last, over the slots as they stand (:mod:`effects.domain.slot_relations`).
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Protocol

from effects.domain.card_features import (
    card_features,
    chosen_subtypes,
    counter_rows,
    real_face_subtype_rows,
)
from effects.domain.input_layout import (
    ACT_SLOT,
    EffectHeadInput,
    Slot,
    SlotKind,
    act_features,
    global_features,
    player_features,
)
from effects.domain.input_vocabularies import (
    subtype_id,
)
from effects.domain.keyword_names import overlay_keyword, overlay_keyword_set
from effects.domain.provenance import ProvenanceKey, first_text
from effects.domain.records import (
    EffectRecord,
    acting_keys,
)
from effects.domain.slot_lines import (
    SlotLine,
    ability_features,
    slot_lines,
)
from effects.domain.slot_relations import SlotRelations, name_key
from effects.domain.state_snapshot import (
    CardLine,
    CommandZoneObject,
    EntityRole,
    EntityState,
    FaceState,
    LineGrant,
    LineRef,
    StateSnapshot,
    Zone,
)

# ── the command zone ──────────────────────────────────────────────────────


def command_zone_entities(record: EffectRecord) -> tuple[EntityState, ...]:
    """A card for each floating effect and pending delayed trigger that is no
    card of the snapshot (a timestamped ``F`` effect, a ``D`` trigger), read
    like a permanent in the command zone (root spec § 9.3): its lines are the
    source lines that made it, granted by an effect, and its controller is that of the card
    its ``created_by`` link ends at. An effect card (``E``) is a snapshot
    entity already, its lines its source lines."""
    state = record.state
    g = state.global_
    if not (g.floating_effects or g.delayed_triggers):
        return ()
    known = {entity.id for entity in state.entities}
    controllers = {entity.id: entity.controller for entity in state.entities}
    makers = _makers(state)
    return tuple(
        _command_zone_card(found, controllers.get(makers.get(found.id, ""), ""))
        for found in (*g.floating_effects, *g.delayed_triggers)
        if found.id not in known
    )


def _makers(state: StateSnapshot) -> dict[str, str]:
    """Each object's maker: the card its first ``created_by`` link ends at."""
    makers: dict[str, str] = {}
    for link in state.links:
        if link.type == "created_by" and isinstance(link.from_, str):
            end = link.to.entity if isinstance(link.to, LineRef) else link.to
            makers.setdefault(link.from_, end)
    return makers


def _command_zone_card(found: CommandZoneObject, controller: str) -> EntityState:
    """A command-zone object as a face-up card: its lines its source lines,
    each granted by an effect."""
    return EntityState(
        id=found.id, name="", zone=Zone.COMMAND.value, controller=controller,
        face_state=FaceState.FRONT.value,
        lines=tuple(
            CardLine(keys=(key,), kind=key.trait_kind, granted=True,
                     grant=LineGrant.EFFECT.value)
            for key in found.source_lines
        ),
    )


def dungeon_room(record: EffectRecord, entity_id: str) -> str | None:
    """The room a player is in, where ``entity_id`` is their dungeon; None
    for any other card."""
    return next((found.room for found in record.state.global_.dungeons
                 if found.dungeon == entity_id), None)


def has_card_slot(entity: EntityState) -> bool:
    """Whether a snapshot entity gets a ``[CARD]`` slot: every one but a
    library card below its library's top (root spec § 9.3).

    The snapshot holds such a card because the record's own events draw,
    mill or search for it, so its slot would show the head the outcome it
    predicts; its player's counts keep what happened to it. Only order rank 0
    is the top: a library card without a rank is not known to be.
    """
    return entity.role != EntityRole.LIBRARY_CARD or entity.order_rank == 0


def slotted_entities(record: EffectRecord) -> tuple[EntityState, ...]:
    """The snapshot's entities that get a ``[CARD]`` slot, in board order."""
    return tuple(entity for entity in record.state.entities if has_card_slot(entity))


def board_entities(record: EffectRecord) -> tuple[EntityState, ...]:
    """Every card the surface gives a ``[CARD]`` slot: the snapshot's
    entities but a library card below the top (:func:`has_card_slot`), then
    :func:`command_zone_entities`."""
    return (*slotted_entities(record), *command_zone_entities(record))


class LineLookup(Protocol):
    """What the surface asks of each ability key's rendered line."""

    def e_for(self, key: ProvenanceKey) -> tuple[float, ...] | int | None:
        """The line's ``e``: a row of the batch's encoder output or the
        values (:attr:`Slot.e_row`, :attr:`Slot.e_vector`). None gives a zero vector
        rather than dropping the token, so a variant that masks ``e`` keeps
        the geometry."""

    def has_line(self, key: ProvenanceKey) -> bool:
        """False for a key with no rendered line (the implicit permanent
        spell), which gets no token."""

    def keyword_of(self, key: ProvenanceKey) -> str | None:
        """Which keyword the line *is*."""

    def options_for(self, key: ProvenanceKey) -> tuple[ProvenanceKey, ...]:
        """The charm-mode keys whose slots follow the key's (FR-063a)."""

    def board_keys(self, key: ProvenanceKey) -> tuple[ProvenanceKey, ...]:
        """The keys a board line of ``key`` reads as slots, one per line: a
        saga's bare keyword key is its chapter lines; any other key itself."""

    def room_of(self, key: ProvenanceKey) -> str | None:
        """The dungeon room the key's line is (its ``ValidRoom$``); None for
        any other line. Asked of a dungeon's lines only."""


def _split_e(e: tuple[float, ...] | int) -> tuple[int | None, tuple[float, ...] | None]:
    """A line's ``e`` as a :class:`Slot`'s ``(e_row, e_vector)``: an ``int``
    row of the batch's encoder output, or values. Told apart by type, never
    by length."""
    return (e, None) if isinstance(e, int) else (None, e)


#: The ``stripped`` map of a record built as collected: no entity loses a keyword.
NO_STRIP: Mapping[str, frozenset[str]] = MappingProxyType({})


class SurfaceBuilder:
    """Lays records out as the token surface, under one batch's constants.

    ``rng`` draws the context dropout: each entity's abilities but the acting
    one's lose their token with probability ``context_dropout``, drawn in
    board order.
    """

    def __init__(
        self, lines: LineLookup, *, e_dim: int, rng: random.Random,
        context_dropout: float = 0.0,
    ) -> None:
        self.lines = lines
        self.rng = rng
        self.context_dropout = context_dropout
        self._zero = tuple([0.0] * e_dim)

    def build(
        self, record: EffectRecord, stripped: Mapping[str, frozenset[str]] = NO_STRIP,
    ) -> EffectHeadInput:
        """One record's surface.

        A ``decision`` record's ``[ACT]`` reads its candidate's ``e``
        (FR-060a). A ``continuous`` record's keyword flags read its board as
        written, which the collector writes without the record's static.
        ``stripped`` is gate 2's perturbation: per-entity keywords removed
        from the flags *and* as ability slots, the keyword's lines whether
        printed or granted.

        A floating effect or delayed trigger that is no snapshot entity gets a
        ``[CARD]`` slot after the board's (:func:`command_zone_entities`), but
        no per-entity target: nothing the head predicts happens to it.
        Relations are built last, over the slots as they stand after context
        dropout and the strip (:class:`SlotRelations`).
        """
        act_row, act_vector = _split_e(self._act_vector(record))
        slots = [
            Slot(
                kind=SlotKind.GLOBAL, position=0,
                features=global_features(record),
            ),
            Slot(
                kind=SlotKind.ACT, position=ACT_SLOT, features=act_features(record),
                e_row=act_row, e_vector=act_vector,
            ),
        ]
        relations = SlotRelations()
        entity_slots: list[int] = []
        unremovable: list[str] = []
        for player in record.state.players:
            entity_slots.append(len(slots))
            relations.subjects[player.id] = len(slots)
            slots.append(Slot(
                kind=SlotKind.PLAYER, position=len(slots),
                features=player_features(player, record),
                player_id=player.id,
            ))

        source_id = record.state.refs.source
        for entity in slotted_entities(record):
            entity_slots.append(len(slots))
            block, keys, unremoved = self._entity_block(
                entity, record, stripped=stripped.get(entity.id, frozenset()),
                dropout=self.context_dropout if entity.id != source_id else 0.0,
            )
            unremovable += [entity.id] * unremoved
            relations.add_card(entity, len(slots), block, keys)
            slots += block
        for entity in command_zone_entities(record):
            block, keys, _ = self._entity_block(
                entity, record, stripped=frozenset(), dropout=self.context_dropout,
            )
            relations.add_card(entity, len(slots), block, keys)
            slots += block
        return EffectHeadInput(
            slots=tuple(slots), entity_slots=tuple(entity_slots),
            relations=relations.build(record), unremovable=tuple(unremovable),
        )

    def _entity_block(
        self, entity: EntityState, record: EffectRecord, *,
        stripped: frozenset[str], dropout: float,
    ) -> tuple[list[Slot], list[ProvenanceKey | None], bool]:
        """One entity's ``CARD`` slot, then an ``ABILITY`` slot per line it
        keeps, each charm root followed by its mode lines; positions restart.
        Beside the block, the key each slot reads (None for the card) and
        whether the strip left a stripped keyword's statement in a kept slot.
        ``stripped`` is gate 2's strip of this entity's keywords."""
        gone = frozenset(overlay_keyword(keyword) for keyword in stripped)
        keys: list[ProvenanceKey | None] = [None]
        block = [Slot(
            kind=SlotKind.CARD, position=0,
            features=card_features(entity, record, masked_keywords=stripped),
            entity_id=entity.id,
            subtypes=tuple(subtype_id(subtype) for subtype in entity.subtypes),
            chosen_types=chosen_subtypes(entity),
            real_face_subtypes=real_face_subtype_rows(entity),
            counters=counter_rows(entity),
        )]
        kept = [line for line in self._slot_lines(entity, record)
                if self._keeps(line, gone=gone, dropout=dropout)]
        for line in kept:
            for slot, key in self._line_slots(line, entity.id, len(block)):
                block.append(slot)
                keys.append(key)
        return block, keys, any(line.keywords & gone for line in kept)

    def _line_slots(
        self, line: SlotLine, entity_id: str, first: int,
    ) -> Iterator[tuple[Slot, ProvenanceKey]]:
        """A line's ``ABILITY`` slot, then one per charm mode, each with its key."""
        modes = self.lines.options_for(line.key)
        options = ((line.key, False), *((mode, True) for mode in modes))
        for position, (mode_key, option) in enumerate(options, first):
            row, vector = _split_e(self._zero_if_none(self.lines.e_for(mode_key)))
            yield Slot(
                kind=SlotKind.ABILITY, position=position,
                features=ability_features(
                    line, chosen=option and mode_key.option in line.chosen),
                e_row=row, e_vector=vector,
                entity_id=entity_id, option=option,
            ), mode_key

    def _zero_if_none(self, e: tuple[float, ...] | int | None) -> tuple[float, ...] | int:
        # Explicitly against None: row 0 is a valid matrix row and ``or``
        # would read it as a miss.
        return self._zero if e is None else e

    def _slot_lines(self, entity: EntityState, record: EffectRecord) -> tuple[SlotLine, ...]:
        """:func:`slot_lines`, a dungeon's room lines active for the room its
        player is in only (``global.dungeons[].room``, root spec § 9.3)."""
        lines = tuple(
            replace(line, keywords=line.keywords or overlay_keyword_set(
                self.lines.keyword_of(line.key)))
            for line in slot_lines(entity, self.lines.board_keys))
        room = dungeon_room(record, entity.id)
        if room is None:
            return lines
        return tuple(self._in_room(line, name_key(room)) for line in lines)

    def _in_room(self, line: SlotLine, room: str) -> SlotLine:
        named = self.lines.room_of(line.key)
        return line if named is None else replace(line, active=name_key(named) == room)

    def _keeps(self, line: SlotLine, *, gone: frozenset[str], dropout: float) -> bool:
        """Whether one of an entity's lines gets a slot."""
        if not self.lines.has_line(line.key):
            return False
        if line.keywords and line.keywords <= gone:
            # Dropped rather than zeroed: a zero token still says an ability
            # is here.
            return False
        return not (dropout > 0.0 and self.rng.random() < dropout)

    def _act_vector(self, record: EffectRecord):
        """The acting line's ``e``: the first of :func:`acting_keys` that
        resolves, so ``[ACT]`` reads exactly the line rarity and gate 1 key on.
        A merged line's keys all name the same row."""
        if not record.has_act_line:
            return self._zero
        vector = first_text(acting_keys(record), self.lines.e_for)
        return self._zero if vector is None else vector
