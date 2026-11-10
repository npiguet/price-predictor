"""``rule_catalogue``: the catalogue's enums, value objects and derived output list (T009, T010)."""

from __future__ import annotations

import pytest

from effects.domain.effect_head_layout import VERDICT_BITS, HeadField
from effects.domain.records import RecordKind
from effects.domain.rule_catalogue import (
    FORK_OPERATIONS,
    POLARITIES,
    RECORD_KINDS,
    Departure,
    DepartureResolution,
    DepartureStatus,
    DetectorDeclaration,
    ForkOperation,
    MapVerdict,
    RuleMapEntry,
    head_outputs,
    is_model_output,
)

#: The hand-kept list ``scripts/rules_catalogue.py`` carried before the derived
#: one replaced it; every name in it must still be nameable.
_FORMER_HAND_LIST = {
    "gate", "zone_outcome", "tap_state", "damage_taken", "power_delta", "toughness_delta",
    "pt_duration", "types_gained", "types_lost", "colors_gained", "colors_lost",
    "type_color_duration", "keywords_gained", "keywords_lost", "control_change",
    "attached_to_change", "life_delta", "cards_drawn", "cards_discarded", "cards_milled",
    "library_events", "poison_delta", "target_legal", "attacker_legal", "blocker_legal",
    "min_blockers", "can_play", "affordable", "has_legal_target", "mana_paid", "trigger_fired",
    "created_objects",
}


def test_the_derived_output_list_keeps_every_name_of_the_hand_list():
    assert _FORMER_HAND_LIST <= head_outputs()


def test_the_derived_output_list_follows_the_layout():
    outputs = head_outputs()
    assert {f.value for f in HeadField} - {"counters_delta_other"} <= outputs
    assert set(VERDICT_BITS) <= outputs


@pytest.mark.parametrize("name, known", [
    ("zone_outcome", True),
    ("counters_delta_p1p1", True),
    ("mana_delta_R", True),
    ("new:timing", False),
    ("record:events", False),
    ("not_an_output", False),
])
def test_is_model_output(name, known):
    assert is_model_output(name) is known


def test_record_kinds_are_the_readers_own():
    assert RECORD_KINDS == {str(k) for k in RecordKind}


def test_enum_spellings():
    assert POLARITIES == {"fires", "does_not_fire", "other"}
    assert {str(v) for v in MapVerdict} == {"trained", "recorded", "missing", "planned"}
    assert FORK_OPERATIONS == {
        "pt", "keyword", "colour_type", "counters", "tap", "life", "replace_card",
        "remove_permanent", "add_permanent", "move_card"}


def test_a_detector_declaration_reads_from_json():
    declaration = DetectorDeclaration.from_json({
        "family": "trigger_mode", "record_kind": "trigger", "outcome": "fires on death",
        "near_miss": "a creature leaving otherwise", "params": {"mode": "ChangesZone"},
    })
    assert declaration.params == {"mode": "ChangesZone"}
    assert declaration.needs == ()


def test_a_departure_blocks_rule_tests_only_while_confirmed_and_unpatched():
    assert Departure(DepartureStatus.CONFIRMED, DepartureResolution.UNPATCHED).blocks_rule_tests
    assert not Departure(DepartureStatus.CONFIRMED, DepartureResolution.ACCEPTED).blocks_rule_tests
    assert not Departure(DepartureStatus.CONFIRMED, DepartureResolution.PATCHED).blocks_rule_tests
    assert not Departure(DepartureStatus.SUSPECTED).blocks_rule_tests
    parsed = Departure.from_json({"status": "confirmed", "resolution": "accepted"})
    assert parsed.resolution is DepartureResolution.ACCEPTED


def test_a_map_entry_round_trips_to_json():
    entry = RuleMapEntry("704.5f", MapVerdict.TRAINED, ("zone_outcome",))
    assert entry.to_json() == {"rule": "704.5f", "verdict": "trained", "outputs": ["zone_outcome"]}


def test_fork_operation_order_is_the_contract_order():
    assert [str(o) for o in ForkOperation] == [
        "pt", "keyword", "colour_type", "counters", "tap", "life", "replace_card",
        "remove_permanent", "add_permanent", "move_card"]


def test_a_script_edit_takes_its_lines_out_and_appends_its_own():
    from effects.domain.rule_catalogue import ScriptEdit

    edit = ScriptEdit("Bears", remove=("PT:2/2",), add=("PT:3/3", "K:Flying"))
    assert edit.applied(["Name:Bears", " PT:2/2", "Types:Creature Bear"]) == [
        "Name:Bears", "Types:Creature Bear", "PT:3/3", "K:Flying"]


def test_a_script_edit_refuses_a_line_its_script_lacks():
    from effects.domain.rule_catalogue import ScriptEdit

    with pytest.raises(ValueError, match="no script line"):
        ScriptEdit("Bears", remove=("PT:9/9",)).applied(["Name:Bears", "PT:2/2"])
