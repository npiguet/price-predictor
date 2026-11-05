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
        return self.edit_global(combat_substep=step)

    def edit_global(self, **changes) -> BoardEdit:
        """The turn structure with ``changes`` (``active``, ``phase``, ...)."""
        state = self.record.state
        return BoardEdit(replace(self.record, state=replace(
            state, global_=replace(state.global_, **changes))))

    def set_active_player(self, player_id: str) -> BoardEdit:
        """Whose turn it is, and so who holds priority first in it."""
        return self.edit_global(active=player_id)

    def set_phase(self, phase: str) -> BoardEdit:
        """The phase or step, spelled as a record spells it (``main1``,
        ``combat_begin``); leaving combat's damage steps clears the substep."""
        substep = self.record.state.global_.combat_substep
        if phase not in ("combat_first_strike_damage", "combat_damage"):
            substep = None
        return self.edit_global(phase=phase, combat_substep=substep)

    def remove_entity(self, entity_id: str) -> BoardEdit:
        """One object gone from the snapshot, with every link naming it: what
        a destroyed permanent leaves of the board it was on."""
        from effects.domain.state_snapshot import LineRef

        def names(end) -> bool:
            return (end.entity if isinstance(end, LineRef) else end) == entity_id

        record = self.record
        links = record.state.links
        if links is not None:
            links = tuple(link for link in links
                          if not names(link.from_) and not names(link.to))
        return BoardEdit(replace(record, state=replace(
            record.state, links=links,
            entities=tuple(e for e in record.state.entities if e.id != entity_id))))

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
        ``protection_from_red``); an entity that has it already is left alone
        (``EntityState.with_granted_keyword``, which protection copies share).
        """
        fragment = keyword_fragment(keyword)
        return self.edit_entity(
            entity_id, lambda entity: entity.with_granted_keyword(keyword, fragment))

    def grant_protection(self, entity_id: str, quality) -> BoardEdit:
        """Protection from ``quality`` (``rule_outcomes.ProtectionQuality``)
        granted until end of turn, spelled as Forge writes its keyword line:
        ``Protection from red``, ``Protection:Goblin``,
        ``Protection:Card.MultiColor:multicolored``."""
        return self.edit_entity(entity_id, lambda entity: entity.with_granted_keyword(
            quality.keyword, quality.fragment))

    def set_line_active(self, entity_id: str, key, active: bool) -> BoardEdit:
        """Forge's answer to whether the static ``key`` on ``entity_id`` holds."""

        def edit(entity):
            if entity.lines is None:
                return entity
            return replace(entity, lines=tuple(
                replace(line, active=active) if key in line.keys and line.active is not None
                else line for line in entity.lines))

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
