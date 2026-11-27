"""Gate 2's perturbation must actually remove the keyword (FR-119 – FR-122).

It reaches the model through two channels — an ability token where the keyword
is printed or attachment-granted, an overlay bit where it is granted until end
of turn — and a strip that cleared only one of them would be measuring a board
that still has the keyword.
"""

from __future__ import annotations

from dataclasses import replace

from effects.application.surface_batching import SurfaceBatcher
from effects.domain.card_features import card_segments
from effects.domain.input_layout import SlotKind
from effects.domain.kept_keyword_flags import KEPT_KEYWORD_FLAGS
from effects.domain.records import EffectRecord
from effects.domain.state_snapshot import (
    CardLine,
    CombatStatus,
)
from effects.domain.surface_builder import NO_STRIP
from tests.unit.effects.application.gate_two_support import (
    _EQUIPPED,
    _FIRST_STRIKE,
    _OTHER,
    _PUMP,
    _batcher,
    _combat_record,
    _creature,
    _NoEncoder,
    _printed_board,
)
from tests.unit.effects.batch_builders import slots_of_kind
from tests.unit.effects.record_builders import make_entity, make_record


def _overlay_board(game: str = "g1") -> EffectRecord:
    """The attacker was granted first strike until end of turn."""
    attacker = _creature(
        "A", 2, 2, keywords=("first_strike",),
        combat=CombatStatus(attacking="P1", blocked_by=("B0",), became_blocked=True),
    )
    blocker = _creature(
        "B0", 2, 2, controller="P1", printed=(_OTHER,),
        combat=CombatStatus(blocking=("A",)),
    )
    return _combat_record(attacker, blocker, game=game, substep="first_strike")


def _rows(batcher: SurfaceBatcher, records) -> dict[str, int]:
    texts = batcher.batch_texts(records)
    return {text: row for row, text in enumerate(sorted(texts))}


# ── the perturbation ─────────────────────────────────────────────────────


class TestStrippingTheAbilityChannel:
    def test_the_keywords_own_ability_slot_is_dropped(self):
        record = _printed_board()
        batcher = _batcher()
        rows = _rows(batcher, [record])
        base = batcher.surface_builder(rows).build(record)
        stripped = batcher.surface_builder(rows).build(
            record, {"A": frozenset({"first_strike"})},
        )
        assert _ability_count(base, "A") == 2
        assert _ability_count(stripped, "A") == 1

    def test_the_entitys_other_lines_survive(self):
        """Only the keyword line goes; the activated ability is untouched."""
        record = _printed_board()
        batcher = _batcher()
        rows = _rows(batcher, [record])
        stripped = batcher.surface_builder(rows).build(
            record, {"A": frozenset({"first_strike"})},
        )
        kept = [
            slot.e_row for slot in slots_of_kind(stripped, SlotKind.ABILITY)
            if slot.entity_id == "A"
        ]
        assert kept == [rows["W: this creature gets +1/+0"]]

    def test_another_entitys_lines_are_intact(self):
        record = _printed_board()
        batcher = _batcher()
        rows = _rows(batcher, [record])
        stripped = batcher.surface_builder(rows).build(
            record, {"A": frozenset({"first_strike"})},
        )
        assert _ability_count(stripped, "B0") == 1

    def test_stripping_a_keyword_the_entity_does_not_carry_changes_nothing(self):
        record = _printed_board()
        batcher = _batcher()
        rows = _rows(batcher, [record])
        base = batcher.surface_builder(rows).build(record)
        stripped = batcher.surface_builder(rows).build(
            record, {"A": frozenset({"lifelink"})},
        )
        assert base.slots == stripped.slots

    def test_the_slot_is_dropped_rather_than_zeroed(self):
        """A zero token would still say "this creature has an ability here"."""
        record = _printed_board()
        batcher = _batcher()
        rows = _rows(batcher, [record])
        stripped = batcher.surface_builder(rows).build(
            record, {"A": frozenset({"first_strike"})},
        )
        assert all(
            slot.e_vector != (0.0,) * 4
            for slot in slots_of_kind(stripped, SlotKind.ABILITY)
        )


class TestStrippingTheOverlayChannel:
    def _overlay_bit(self, surface, entity_id: str) -> float:
        slot = next(
            s for s in slots_of_kind(surface, SlotKind.CARD) if s.entity_id == entity_id
        )
        return _overlay_column(slot.features, "first_strike")

    def test_the_granted_keyword_bit_is_cleared(self):
        record = _overlay_board()
        batcher = _batcher()
        rows = _rows(batcher, [record])
        base = batcher.surface_builder(rows).build(record)
        stripped = batcher.surface_builder(rows).build(
            record, {"A": frozenset({"first_strike"})},
        )
        assert self._overlay_bit(base, "A") == 1.0
        assert self._overlay_bit(stripped, "A") == 0.0


def _gen3_board() -> EffectRecord:
    """A gen-3 attacker stating first strike twice, printed and granted by an
    Equipment's static, with its keyword list and flag; the pump line beside
    them. The blocker prints nothing relevant."""
    attacker = replace(_creature(
        "A", 2, 2, keywords=("first_strike",),
        combat=CombatStatus(attacking="P1", blocked_by=("B0",), became_blocked=True),
    ), lines=(
        CardLine(keys=(_FIRST_STRIKE,), kind="keyword", granted=False,
                 fragment="First Strike"),
        CardLine(keys=(_PUMP,), kind="activated", granted=False),
        CardLine(keys=(_EQUIPPED,), kind="keyword", granted=True, active=True,
                 fragment="First Strike"),
    ))
    blocker = replace(_creature(
        "B0", 2, 2, controller="P1", combat=CombatStatus(blocking=("A",)),
    ), lines=(CardLine(keys=(_OTHER,), kind="keyword", granted=False, fragment="Trample"),))
    return _combat_record(attacker, blocker, substep="first_strike")


class TestStrippingAGen3Carrier:
    """Research M5: the strip edits the kept flag and every line of the
    keyword, printed or granted, so the twin is left with no line of it."""

    def _twin(self, record):
        batcher = _batcher()
        _batch, surfaces, _encoded = batcher.build(
            [record, record], _NoEncoder(),
            strip_per_record=[NO_STRIP, {"A": frozenset({"first_strike"})}],
        )
        return batcher, surfaces

    def test_a_stripped_keyword_leaves_no_line(self):
        record = _gen3_board()
        batcher, (base, stripped) = self._twin(record)
        rows = _rows(batcher, [record])
        assert _ability_count(base, "A") == 3
        kept = [slot.e_row for slot in slots_of_kind(stripped, SlotKind.ABILITY)
                if slot.entity_id == "A"]
        assert kept == [rows["W: this creature gets +1/+0"]]

    def test_its_flag_is_cleared_and_the_other_entity_is_intact(self):
        record = _gen3_board()
        _batcher_, (base, stripped) = self._twin(record)

        def flag(surface):
            slot = next(s for s in slots_of_kind(surface, SlotKind.CARD) if s.entity_id == "A")
            return _overlay_column(slot.features, "first_strike")

        assert (flag(base), flag(stripped)) == (1.0, 0.0)
        assert _ability_count(stripped, "B0") == _ability_count(base, "B0") == 1


class TestBatchThreading:
    def test_the_unperturbed_record_in_the_same_batch_is_unchanged(self):
        record = _printed_board()
        batcher = _batcher()
        _batch, surfaces, _encoded = batcher.build(
            [record, record], _NoEncoder(),
            strip_per_record=[NO_STRIP, {"A": frozenset({"first_strike"})}],
        )
        assert _ability_count(surfaces[0], "A") == 2
        assert _ability_count(surfaces[1], "A") == 1

    def test_no_strip_list_builds_exactly_as_before(self):
        record = _printed_board()
        batcher = _batcher()
        _batch, surfaces, _encoded = batcher.build([record], _NoEncoder())
        assert _ability_count(surfaces[0], "A") == 2


def _ability_count(surface, entity_id: str) -> int:
    return sum(
        1 for slot in slots_of_kind(surface, SlotKind.ABILITY)
        if slot.entity_id == entity_id
    )


def _overlay_column(features: tuple[float, ...], keyword: str) -> float:
    """Read one kept keyword's flag out of a ``[CARD]`` slot's features; every
    part of the card block has a fixed width, so any entity gives its place."""
    flags = card_segments(make_entity("E0", "probe"), make_record())["keyword_flags"]
    return features[flags][KEPT_KEYWORD_FLAGS.index(keyword)]
