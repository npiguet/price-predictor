"""Root spec § 9.2: the flags every ``[ABILITY]`` slot carries, granted,
by an effect, cast with, chosen, and what a keyword counter adds."""

from __future__ import annotations

from effects.domain.card_features import card_segments, counter_rows
from effects.domain.input_layout import SlotKind
from effects.domain.provenance import ProvenanceKey
from effects.domain.state_snapshot import GrantedTemporary
from tests.unit.effects.builders import slots_of_kind
from tests.unit.effects.domain.effect_head_input_support import (
    _ANTHEM,
    _CHOICE,
    _WARD,
    BuildSettings,
    _build,
    _entity,
    _flags,
    _line,
    _record,
    _snapshot,
)

_KICKER = ProvenanceKey("cardsfolder/k/kicker.txt", 0, "keyword", 0)
_SLIVER = ProvenanceKey("cardsfolder/f/fungus_sliver.txt", 0, "trigger", 0)


class TestLineFlags:
    """Root spec § 9.2: every ability slot carries its line's flags."""

    def _abilities(self, entity, settings: BuildSettings = BuildSettings()):
        return slots_of_kind(
            _build(_record(state=_snapshot(entities=(entity,))), settings), SlotKind.ABILITY)

    def test_every_ability_slot_has_the_flags_width(self):
        from effects.domain.slot_lines import ABILITY_FEATURE_WIDTH

        entity = _entity("E1", lines=(_line(_ANTHEM), _line(_WARD, active=False)))
        assert {len(slot.features) for slot in self._abilities(entity)} == {
            ABILITY_FEATURE_WIDTH}

    def test_a_face_not_up_is_slots_flagged_inactive(self):
        entity = _entity("E1", lines=(
            _line(_ANTHEM, active=True), _line(_WARD, kind="trigger", active=False)))
        assert [_flags(slot)["active"] for slot in self._abilities(entity)] == [1.0, 0.0]

    def _grant_flags(self, *lines) -> list[tuple[float, float]]:
        abilities = self._abilities(_entity("E1", lines=lines))
        return [(_flags(slot)["granted"], _flags(slot)["by_effect"]) for slot in abilities]

    def test_a_grant_by_a_static_is_granted_but_not_by_an_effect(self):
        assert self._grant_flags(
            _line(_ANTHEM, kind="activated", granted=True, grant="static")) == [(1.0, 0.0)]

    def test_a_trigger_a_static_grants_is_not_by_an_effect(self):
        """Finding 1.7: a second Fungus Sliver's trigger keys to its own
        trigger line, and is no effect's for that."""
        assert self._grant_flags(
            _line(_SLIVER, kind="trigger", granted=True, grant="static")) == [(1.0, 0.0)]

    def test_a_copied_trigger_line_reads_not_granted(self):
        """A copy prints what it copies: its trigger line is printed."""
        assert self._grant_flags(_line(_SLIVER, kind="trigger")) == [(0.0, 0.0)]

    def test_a_trait_a_card_grants_itself_merges_with_its_granting_line_printed(self):
        """A Class level's trigger is listed beside its ``Class:N`` line under
        the same key, both printed: one slot, not granted."""
        assert self._grant_flags(_line(_KICKER, kind="keyword"),
                                 _line(_KICKER, kind="trigger")) == [(0.0, 0.0)]

    def test_a_keyword_counter_line_is_granted_not_by_an_effect(self):
        assert self._grant_flags(_line(
            None, kind="keyword", granted=True, grant="counter", fragment="Trample",
        )) == [(1.0, 0.0)]

    def test_an_until_end_of_turn_granted_ability_is_by_an_effect(self):
        assert self._grant_flags(
            _line(_ANTHEM, kind="activated", granted=True, grant="effect")) == [(1.0, 1.0)]

    def test_a_static_an_effect_granted_is_by_an_effect(self):
        """The grant decides, not the key's trait kind: an animation's static."""
        assert self._grant_flags(
            _line(_WARD, kind="static", granted=True, grant="effect")) == [(1.0, 1.0)]

    def test_a_granted_line_written_before_grant_existed_is_not_by_an_effect(self):
        """A granted line with no ``grant`` (a record collected before the
        member) says nothing about why: nothing is guessed from its key."""
        assert self._grant_flags(
            _line(_SLIVER, kind="trigger", granted=True),
            _line(None, kind="keyword", granted=True, fragment="Flying"),
        ) == [(1.0, 0.0), (1.0, 0.0)]

    def test_a_keyword_an_effect_granted_reads_through_its_reminder_line(self):
        from effects.domain.slot_lines import keyword_line_fragment

        entity = _entity("E1", lines=(
            _line(None, kind="keyword", granted=True, grant="effect", fragment="Flying"),))
        keys = []
        abilities = self._abilities(
            entity, BuildSettings(e_for=lambda key: keys.append(key) or None))
        flags = _flags(abilities[0])
        assert (flags["granted"], flags["by_effect"]) == (1.0, 1.0)
        assert keyword_line_fragment(keys[-1]) == "Flying"

    def test_a_keyword_an_edit_grants_is_by_an_effect(self):
        """``with_granted_keyword`` grants as a resolved effect does."""
        entity = _entity("E1", lines=(_line(_ANTHEM),)).with_granted_keyword("flying", "Flying")
        flags = _flags(self._abilities(entity)[1])
        assert (flags["granted"], flags["by_effect"]) == (1.0, 1.0)

    def test_a_line_with_no_key_and_no_keyword_has_no_slot(self):
        entity = _entity("E1", lines=(_line(None, kind="trigger"), _line(_ANTHEM)))
        assert len(self._abilities(entity)) == 1

    def test_cast_with_carries_the_multikicker_count(self):
        entity = _entity("E1", lines=(_line(_KICKER, kind="keyword", cast_with=3),))
        flags = _flags(self._abilities(entity)[0])
        assert flags["cast_with"] == 1.0 and flags["count"] == 3.0

    def test_the_chosen_flag_sits_on_the_chosen_mode(self):
        modes = tuple(ProvenanceKey(_CHOICE.script_file, 0, "spell", 0, n) for n in range(3))
        entity = _entity("E1", lines=(_line(_CHOICE, kind="spell", chosen=(0, 2)),))
        abilities = self._abilities(
            entity, BuildSettings(options_for=lambda key: modes if key == _CHOICE else ()))
        assert [_flags(slot)["chosen"] for slot in abilities] == [0.0, 1.0, 0.0, 1.0]

    def test_a_keyword_counter_is_a_granted_keyword_line_not_a_counter(self):
        from effects.domain.input_vocabularies import counter_id
        from effects.domain.slot_lines import keyword_line_fragment, slot_lines

        entity = _entity("E1", counters={"FLYING": 1, "P1P1": 2})
        lines = slot_lines(entity)
        assert [keyword_line_fragment(line.key) for line in lines] == ["FLYING"]
        assert lines[0].granted and lines[0].grant == "counter" and not lines[0].by_effect
        assert counter_rows(entity) == ((counter_id("P1P1"), 2.0),)

    def test_a_keyword_counter_adds_no_line_where_the_keyword_has_one(self):
        from effects.domain.slot_lines import slot_lines

        entity = _entity("E1", counters={"FLYING": 1}, lines=(
            _line(None, kind="keyword", granted=True, fragment="Flying"),))
        assert len(slot_lines(entity)) == 1

    def test_stripping_a_keyword_drops_its_granted_line_and_its_flag(self):
        entity = _entity(
            "E1", granted_temporary=GrantedTemporary(keywords=("first_strike",)),
            lines=(_line(None, kind="keyword", granted=True, fragment="First Strike"),
                   _line(_ANTHEM)),
        )
        record = _record(state=_snapshot(entities=(entity,)))
        base = _build(record)
        stripped = _build(record, BuildSettings(stripped={"E1": frozenset({"first_strike"})}))
        assert len(slots_of_kind(base, SlotKind.ABILITY)) == 2
        assert len(slots_of_kind(stripped, SlotKind.ABILITY)) == 1
        flags = card_segments(entity, record)["keyword_flags"]
        assert sum(slots_of_kind(stripped, SlotKind.CARD)[0].features[flags]) == 0.0
        assert sum(slots_of_kind(base, SlotKind.CARD)[0].features[flags]) == 1.0
