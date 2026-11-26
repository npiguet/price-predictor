"""The relations between a surface's slots (root spec § 10).

Every link between two slots is a relation, a ``(from slot, to slot, type)``
triple acting only inside the trunk's attention.
"""

from __future__ import annotations

from functools import lru_cache

from effects.domain.card_features import (
    face_state_of,
)
from effects.domain.input_layout import (
    ACT_SLOT,
    Slot,
)
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    EffectRecord,
)
from effects.domain.relation_types import Direction, relation_type
from effects.domain.state_snapshot import (
    EntityState,
    FaceState,
    LineRef,
)

# ── relations ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=None)
def _relation_pair(family: str, target_index: int | None) -> tuple[int, int]:
    """A family's forward and reverse type rows (one target index's for ``targets``)."""
    return (relation_type(family, Direction.FORWARD, target_index),
            relation_type(family, Direction.REVERSE, target_index))


def name_key(name: str) -> str:
    return name.strip().lower()


class SlotRelations:
    """One surface's relations (root spec § 10.3), gathered while its slots
    are laid out and built once they are fixed.

    Every link is two triples, the family's forward type from the end the
    table names and its reverse type from the other, each triple once. The
    reader builds four families itself: ``line_of`` (each ability slot to its
    card), ``line_of_the_same_card`` (each pair of one card's ability slots),
    ``mode_of`` (a charm's mode slots to their root) and ``has_the_same_name``
    (each pair of cards whose names match but a last-known copy and its own
    card, and a card that chose a name to each card of that name). The rest
    are the record's ``links``. ``[ACT]`` reads ``targets`` to the cards and
    players the acting ability targets itself and ``source_of`` from its
    source, as the target and source flags do (root spec § 10.4): no
    ``targets`` to a permanent whose ability on the stack is the target
    (:meth:`StateSnapshot.targets_card`).

    A link end that is a line reads that line's slot; where the line has no
    slot (context dropout, gate 2's strip, a key with no rendered line) it
    reads the line's card, as the collector writes a link whose line Forge
    does not record. A link to an object the surface has no slot for is
    dropped, and so is a link both of whose ends read one slot.
    """

    def __init__(self) -> None:
        #: A player's or card's id to its slot.
        self.subjects: dict[str, int] = {}
        self._lines: dict[tuple[str, ProvenanceKey], int] = {}
        self._named: dict[str, list[int]] = {}
        self._chosen_names: list[tuple[int, tuple[str, ...]]] = []
        #: A last-known copy's slot to its card's id (``L<id>`` to ``E<id>``).
        self._copies: dict[int, str] = {}
        self._triples: dict[tuple[int, int, int], None] = {}

    def link(self, a: int | None, b: int | None, family: str,
             target_index: int | None = None) -> None:
        if a is None or b is None or a == b:
            return
        forward, reverse = _relation_pair(family, target_index)
        self._triples[(a, b, forward)] = None
        self._triples[(b, a, reverse)] = None

    def _pairs(self, rows: list[int], family: str,
               apart: frozenset[tuple[int, int]] = frozenset()) -> None:
        """Every pair of ``rows`` but those ``apart`` names."""
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if (a, b) not in apart:
                    self.link(a, b, family)

    def add_card(self, entity: EntityState, start: int, block: list[Slot],
                 keys: list[ProvenanceKey | None]) -> None:
        """One card's block, laid out from slot ``start``: its own lines'
        relations now, the rest noted for :meth:`build`."""
        self.subjects.setdefault(entity.id, start)
        lines: list[int] = []
        root = start
        for offset in range(1, len(block)):
            at = start + offset
            self._lines.setdefault((entity.id, keys[offset]), at)
            self.link(at, start, "line_of")
            if block[offset].option:
                self.link(at, root, "mode_of")
            else:
                root = at
            lines.append(at)
        self._pairs(lines, "line_of_the_same_card")
        self._note_names(entity, start)

    def _note_names(self, entity: EntityState, start: int) -> None:
        """What :meth:`_name_relations` reads of the card at slot ``start``:
        its name, the names it chose, and whether it is a last-known copy."""
        # A face-down card's name would be its real face's: it has none here.
        if entity.name and face_state_of(entity) != FaceState.FACE_DOWN:
            self._named.setdefault(name_key(entity.name), []).append(start)
        if entity.chosen is not None and entity.chosen.names:
            self._chosen_names.append((start, entity.chosen.names))
        if entity.is_last_known and entity.id.startswith("L"):
            self._copies[start] = "E" + entity.id[1:]

    def _end(self, end) -> int | None:
        if isinstance(end, LineRef):
            at = self._lines.get((end.entity, end.key))
            return at if at is not None else self.subjects.get(end.entity)
        return self.subjects.get(end)

    def build(self, record: EffectRecord) -> tuple[tuple[int, int, int], ...]:
        self._name_relations()
        for found in record.state.links:
            index = (found.index or 0) if found.type == "targets" else None
            self.link(self._end(found.from_), self._end(found.to), found.type, index)
        if record.has_act_line:
            self._act_relations(record.state)
        return tuple(self._triples)

    def _copy_pairs(self) -> frozenset[tuple[int, int]]:
        """Each last-known copy's slot with its card's, both ways: a copy is
        its card as it last existed, not another card of its name."""
        pairs = set()
        for copy, card_id in self._copies.items():
            card = self.subjects.get(card_id)
            pairs |= {(copy, card), (card, copy)}
        return frozenset(pairs)

    def _name_relations(self) -> None:
        apart = self._copy_pairs()
        for rows in self._named.values():
            self._pairs(rows, "has_the_same_name", apart)
        for chooser, names in self._chosen_names:
            for name in names:
                for at in self._named.get(name_key(name), ()):
                    self.link(chooser, at, "has_the_same_name")

    def _act_relations(self, state) -> None:
        """``[ACT]``'s ``targets`` to what it targets itself, as cast, each at
        its position, and ``source_of`` from its source."""
        refs = state.refs
        for index, target in enumerate(refs.as_cast):
            if state.targets_card(target):
                self.link(ACT_SLOT, self.subjects.get(target), "targets", index)
        if refs.source is not None:
            self.link(self.subjects.get(refs.source), ACT_SLOT, "source_of")
