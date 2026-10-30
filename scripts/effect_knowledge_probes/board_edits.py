"""Edits to one record's board, each returning a new record (FR-084).

Every grid and sweep edits a real record to ask one question of the model: the
same board with one number, one keyword, one controller changed. An edit never
touches the record it starts from, and :class:`BoardEdit` chains them, so a
grid cell reads as the list of what it changed::

    BoardEdit(record).set_power("A", 3).set_toughness("B", 4).record

An input the record states in two places is edited in both by the caller that
knows the second place (``sweeps.set_untapped_lands``, the combat grid's damage
assignment); the edits here touch one field each.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class BoardEdit:
    """A record whose board is being edited; ``record`` is the result so far."""

    record: object

    # ── the board as a whole ────────────────────────────────────────────

    def with_entities(self, entities: Iterable) -> BoardEdit:
        record = self.record
        return BoardEdit(replace(record, state=replace(
            record.state, entities=tuple(entities))))

    def edit_entity(self, entity_id: str, edit: Callable) -> BoardEdit:
        """One entity replaced by ``edit(entity)``; every other one kept."""
        return self.with_entities(
            edit(e) if e.id == entity_id else e for e in self.record.state.entities
        )

    def set_step(self, step: str) -> BoardEdit:
        """The record as a snapshot of the given combat damage step."""
        state = self.record.state
        return BoardEdit(replace(self.record, state=replace(
            state, global_=replace(state.global_, combat_substep=step))))

    # ── one entity ──────────────────────────────────────────────────────

    def set_power(self, entity_id: str, power: int) -> BoardEdit:
        """Total power set; the base power absorbs the change, so boosts and
        counters keep their own channels."""
        from effects.domain.state_snapshot import PowerToughness

        def edit(entity):
            pt = entity.pt or PowerToughness(base=(0, 0))
            base_p = power - pt.boosts[0] - pt.counters[0]
            return replace(entity, pt=replace(pt, base=(base_p, pt.base[1])))

        return self.edit_entity(entity_id, edit)

    def set_toughness(self, entity_id: str, toughness: int) -> BoardEdit:
        """Total toughness set and marked damage cleared.

        The base toughness absorbs the change, so boosts and counters keep their
        own channels; damage is cleared so the toughness is what the ability
        faces.
        """
        return self.edit_entity(
            entity_id, lambda entity: entity.with_total_toughness(toughness))

    def set_marked_damage(self, entity_id: str, damage: int) -> BoardEdit:
        return self.edit_entity(entity_id, lambda e: replace(e, damage=damage))

    def clear_damage(self, *entity_ids: str) -> BoardEdit:
        edited = self
        for entity_id in entity_ids:
            edited = edited.set_marked_damage(entity_id, 0)
        return edited

    def grant_keyword(self, entity_id: str, keyword: str) -> BoardEdit:
        """``keyword`` granted until end of turn, the overlay channel the model reads."""
        def edit(entity):
            granted = entity.granted_temporary
            if keyword in granted.keywords:
                return entity
            return replace(entity, granted_temporary=replace(
                granted, keywords=(*granted.keywords, keyword)))

        return self.edit_entity(entity_id, edit)

    def set_controller(self, entity_id: str, player_id: str) -> BoardEdit:
        return self.edit_entity(entity_id, lambda e: replace(e, controller=player_id))

    def set_colours(self, entity_id: str, colours: tuple[str, ...]) -> BoardEdit:
        return self.edit_entity(entity_id, lambda e: replace(e, colors=tuple(colours)))

    def add_type(self, entity_id: str, card_type: str) -> BoardEdit:
        def edit(entity):
            if card_type in entity.types:
                return entity
            return replace(entity, types=(*entity.types, card_type))

        return self.edit_entity(entity_id, edit)

    def remove_type(self, entity_id: str, card_type: str) -> BoardEdit:
        return self.edit_entity(entity_id, lambda e: replace(
            e, types=tuple(t for t in e.types if t != card_type)))

    def add_supertype(self, entity_id: str, supertype: str) -> BoardEdit:
        return self.edit_entity(entity_id, lambda e: replace(
            e, supertypes=tuple(sorted({*e.supertypes, supertype}))))
