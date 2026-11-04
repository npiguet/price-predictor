"""Edits to one record's board, each returning a new record (FR-084).

Every grid and sweep edits a real record to ask one question of the model: the
same board with one number, one keyword, one controller changed. An edit never
touches the record it starts from, and :class:`BoardEdit` chains them, so a
grid cell reads as the list of what it changed::

    BoardEdit(record).set_power("A", 3).set_toughness("B", 4).record

An input the record states in two places is edited in both by the caller that
knows the second place (``sweeps.set_untapped_lands``, the combat grid's damage
assignment); the edits here touch one field each, but for a granted keyword,
which a gen-3 entity states as a list entry and a line, both edited here.

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
        """``keyword`` granted until end of turn, as a resolved effect grants it
        (root spec § 9.6): in the keyword list, which sets its kept flag where
        the keyword keeps one, and as a granted keyword line with no key, which
        the model reads through the keyword's reminder text. A keyword without
        a kept flag (fear, intimidate, unblockable) reaches the model through
        that line alone. A gen-1 or gen-2 entity has no lines, so its list is
        all an edit can change.

        ``keyword`` is in the package's spelling (``first_strike``,
        ``protection_from_red``); an entity that has it already is left alone.
        """
        from effects.domain.keyword_names import normalize_keyword
        from effects.domain.state_snapshot import CardLine, LineKind

        def edit(entity):
            granted = entity.granted_temporary
            if keyword not in granted.keywords:
                entity = replace(entity, granted_temporary=replace(
                    granted, keywords=(*granted.keywords, keyword)))
            fragment = keyword_fragment(keyword)
            if entity.lines is None or any(
                    line.in_force and (line.fragment == fragment
                                       or line.keyword == normalize_keyword(keyword))
                    for line in entity.lines):
                return entity
            line = CardLine(keys=(), kind=LineKind.KEYWORD.value, granted=True,
                            fragment=fragment)
            return replace(entity, lines=(*entity.lines, line))

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


#: Grants no Forge keyword spells: "can't be blocked" is a static in Forge, so
#: its line reads the text a converted card prints for that static.
_FRAGMENTS = {"unblockable": "CARDNAME can't be blocked."}


def keyword_fragment(keyword: str) -> str:
    """The fragment Forge writes on a keyword line for ``keyword``, given in
    the package's spelling: ``first_strike`` is ``First strike``,
    ``protection_from_red`` is ``Protection from red``. The keyword table
    matches a fragment's name case-insensitively, so the capital alone
    matters."""
    if keyword in _FRAGMENTS:
        return _FRAGMENTS[keyword]
    words = keyword.replace("_", " ")
    return words[:1].upper() + words[1:]
