"""``rules_catalogue.py validate``'s detector, operation and departure checks (T020, T021).

Each failing case is a real scenario from ``rules/scenarios/`` edited by the
one fact its check refuses.
"""

from __future__ import annotations

import copy
import json

import pytest
import rules_catalogue as rc

_UNIT = "704"


@pytest.fixture(scope="module")
def families():
    return rc.detector_families()


@pytest.fixture
def declared():
    """704.3-1 as a fully declared gen-3 scenario: the edits start from it."""
    doc = json.loads((rc.RULES / "scenarios" / f"{_UNIT}.json").read_text(encoding="utf-8"))
    scenario = copy.deepcopy(next(s for s in doc["scenarios"] if s["id"] == "704.3-1"))
    scenario["detector"] = {
        "family": "sba_zone_move", "record_kind": "resolution",
        "outcome": "A creature with lethal damage marked dies in the state-based check.",
        "near_miss": "The creature has less damage than its toughness and stays.",
        "params": {"check": "lethal_damage"}, "needs": [],
    }
    for variant in scenario["variants"][1:]:
        variant["operation"] = {"unclassifiable": "removes marked damage"}
    return scenario


def _errors(scenario, families):
    return rc.validate_gen3_fields(scenario, families)


def test_a_fully_declared_scenario_passes(declared, families):
    assert _errors(declared, families) == []


def test_a_none_detector_with_its_reason_passes(declared, families):
    declared["detector"] = {"none": "the layer order shows in no record field"}
    assert _errors(declared, families) == []


@pytest.mark.parametrize("edit, message", [
    (lambda s: s.pop("detector"), "no detector"),
    (lambda s: s.update(detector={"none": ""}), "carries its reason"),
    (lambda s: s["detector"].update(family="no_such_family"), "unknown detector family"),
    (lambda s: s["detector"]["params"].update(toughness=3), "declares no parameter"),
    (lambda s: s["detector"].update(record_kind="trigger"), "is not the scenario's"),
    (lambda s: s["detector"].update(outcome=""), "no outcome"),
    (lambda s: s["variants"][0].update(delta="B has one more Goblin."), "base and its delta"),
    (lambda s: s["variants"][0].update(operation="pt"), "base variant carries an operation"),
    (lambda s: s["variants"][1].pop("operation"), "no operation"),
    (lambda s: s["variants"][1].update(operation="flip"), "unknown operation"),
    (lambda s: s["variants"][1].update(operation={"unclassifiable": " "}), "operation"),
    (lambda s: s.update(departure={"status": "maybe"}), "departure status"),
    (lambda s: s.update(departure={"status": "suspected", "resolution": "accepted"}),
     "only once confirmed"),
    (lambda s: s.update(departure={"status": "confirmed", "resolution": "ignored"}),
     "departure resolution"),
], ids=lambda value: value if isinstance(value, str) else "")
def test_each_check_refuses_its_one_edit(declared, families, edit, message):
    edit(declared)
    errors = _errors(declared, families)
    assert errors and any(message in error for error in errors), errors


@pytest.mark.parametrize("operation", ["remove_permanent", "add_permanent", "move_card"])
def test_a_variant_classified_into_a_board_edit_of_t031a_passes(declared, families, operation):
    declared["variants"][1]["operation"] = operation
    assert _errors(declared, families) == []


def test_a_parameter_naming_a_card_is_refused(declared, families):
    declared["detector"] = {
        "family": "trigger_mode", "record_kind": "resolution", "outcome": "o", "near_miss": "n",
        "params": {"mode": "ChangesZone", "selector": "Card.namedGoblin King"},
    }
    assert any("names a card" in error for error in _errors(declared, families))


def test_a_confirmed_accepted_departure_passes(declared, families):
    declared["departure"] = {"status": "confirmed", "resolution": "accepted",
                             "forge_behaviour": "Forge keeps it.", "note": ""}
    assert _errors(declared, families) == []


def test_the_registry_the_validator_reads_is_the_package_s(families):
    assert families["sba_zone_move"] == frozenset({"check"})


def test_the_real_tree_passes_every_check_but_the_gen3_fields():
    """The detector and operation passes (T030, T031) fill in what the gen-3
    checks ask; every earlier check holds over the real tree already."""
    rules = rc.load_rules()
    all_ids = {r["id"] for r in rules}
    for path in sorted((rc.RULES / "scenarios").glob("*.json")):
        unit = path.stem
        errors, _ = rc.validate_scenarios(
            rc.load_unit("scenarios", unit), rc.load_unit("triage", unit), all_ids,
            rc._AnyName())
        assert errors == [], (unit, errors[:3])
