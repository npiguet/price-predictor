"""``rules_catalogue.py validate``'s detector, operation and departure checks (T020, T021).

Each failing case is a real scenario from ``rules/scenarios/`` edited by the
one fact its check refuses.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

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


# --- directed-rules (T095a): the run-time lists of FR-014b and FR-038a ------------------------


def _scenario(sid, rules, cards, variants, *, kind="resolution", detector=None):
    """A scenario shaped like ``rules/scenarios/*.json``'s: a base variant, then
    ``variants`` as ``(label, polarity, operation)``."""
    return {
        "id": sid, "rules": rules, "title": sid, "record_kind": kind,
        "variants": [{"label": "base", "polarity": "fires", "delta": "", "expected": "e"}]
        + [{"label": label, "polarity": polarity, "delta": "d", "expected": "e",
            "operation": operation} for label, polarity, operation in variants],
        "cards": cards,
        "detector": detector or {"family": "sba_zone_move", "record_kind": kind,
                                 "outcome": "o", "near_miss": "n",
                                 "params": {"check": "toughness_zero"}, "needs": []},
    }


_UNCLASSIFIABLE = {"unclassifiable": "choice: a target"}


def _sides(fires, does_not_fire):
    out = {}
    if fires:
        out["fires"] = {"forge_ai": fires - fires // 3, "random_seat": fires // 3}
    if does_not_fire:
        out["does_not_fire"] = {"forge_ai": does_not_fire}
    return out


@pytest.fixture
def catalogue(tmp_path):
    """A rules dir, a converted tree and gen-2 counts covering each way a rule
    gets in or stays out."""
    rules = tmp_path / "rules"
    (rules / "scenarios").mkdir(parents=True)
    units = {
        "121": [_scenario("121.4-1", ["121.4"], ["Mind Rot", "Swamp"],
                          [("empty", "does_not_fire", "move_card"),
                           ("hand", "fires", "add_permanent")])],
        "400": [_scenario("400.6-1", ["400.6"], ["Rest in Peace"],
                          [("no_replacement", "does_not_fire", "remove_permanent")])],
        "500": [_scenario("500.1-1", ["500.1"], ["Common Card"],
                          [("off", "does_not_fire", "tap")]),
                _scenario("500.2-1", ["500.2"], ["Choice Card"],
                          [("other_target", "does_not_fire", _UNCLASSIFIABLE)]),
                _scenario("500.3-1", ["500.3"], ["Unseen Card"],
                          [("off", "does_not_fire", "life")]),
                _scenario("500.4-1", ["500.4"], ["Gen3 Card"],
                          [("off", "does_not_fire", "life")],
                          detector={"family": "sba_zone_move", "record_kind": "resolution",
                                    "outcome": "o", "near_miss": "n",
                                    "params": {"check": "toughness_zero"},
                                    "needs": ["failed_checks"]}),
                _scenario("500.5-1", ["500.5"], ["Untap Card"],
                          [("off", "does_not_fire", "tap")], kind="turn_action",
                          detector={"family": "outcome_event", "record_kind": "turn_action",
                                    "outcome": "o", "near_miss": "n",
                                    "params": {"event": "untapped"}, "needs": []}),
                _scenario("500.6-1", ["500.6"], ["Uncountable Card"],
                          [("off", "does_not_fire", "life")])],
        "702.9": [_scenario("702.9a-1", ["702.9a"], ["Cut"],
                            [("ribbons", "does_not_fire", "move_card")])],
        "702.10": [_scenario("702.10b-1", ["702.10b", "702.10c"], ["Ribbons", "Forest"],
                             [("no_haste", "does_not_fire", "keyword")])],
        "704": [_scenario("704.5f-1", ["704.5f"], ["Disfigure", "Drudge Skeletons"],
                          [("counters", "does_not_fire", "counters")])],
    }
    for unit, scenarios in units.items():
        (rules / "scenarios" / f"{unit}.json").write_text(
            json.dumps({"unit": unit, "verified": True, "scenarios": scenarios}),
            encoding="utf-8")
    counts = {
        "records": 1000,
        "rules": {
            "121.4": _sides(0, 94402),       # rare: the fires side is lacking
            "400.6": _sides(23335, 0),       # one-sided: no does_not_fire record
            "500.1": _sides(5000, 4000),     # common and two-sided: out
            "500.2": _sides(3, 0),           # rare, but nothing forkable: out
            "702.9a": _sides(99, 7),         # one short of common
            "702.10b": _sides(100, 0),       # common, one-sided
            "702.10c": _sides(100, 1),       # common and two-sided: out
            "704.5f": _sides(6814, 15221),   # always, though common
        },
        "not_countable": ["500.6"],
        "none": {},
        "review": [],
    }
    counts_path = tmp_path / "rule-counts-gen2.json"
    counts_path.write_text(json.dumps(counts), encoding="utf-8")
    cards = tmp_path / "cardsfolder" / "c"
    cards.mkdir(parents=True)
    (cards / "cut_ribbons.txt").write_text(
        "layout: split\nname: cut\ntypes: sorcery\n\nALTERNATE\n\nname: ribbons\n"
        "types: sorcery\n", encoding="utf-8")
    for name in ("mind rot", "swamp", "rest in peace", "unseen card", "forest", "disfigure",
                 "drudge skeletons"):
        (cards / f"{name.replace(' ', '_')}.txt").write_text(f"name: {name}\n",
                                                              encoding="utf-8")
    return argparse.Namespace(rules_dir=rules, counts=counts_path,
                              cards_folder=tmp_path / "cardsfolder")


def _run(args, capsys):
    rc.cmd_directed_rules(args)
    listed = (args.rules_dir / "directed-fork-rules.txt").read_text(encoding="utf-8")
    cards = (args.rules_dir / "coverage-wording-cards.txt").read_text(encoding="utf-8")
    return listed.splitlines(), cards.splitlines(), capsys.readouterr().out


def test_the_rare_and_one_sided_forkable_rules_and_704_5f_are_listed_by_unit(catalogue, capsys):
    listed, _, _ = _run(catalogue, capsys)
    # 500.3 has no record but a detector gen-2 records show: none of either side.
    assert listed == ["121.4", "400.6", "500.3", "702.9a", "702.10b", "704.5f"]


def test_the_wording_cards_are_those_rules_scenarios_cards_as_converted_names(
        catalogue, capsys):
    _, cards, out = _run(catalogue, capsys)
    # Ribbons is Cut's back face; 704.5f's cards are left out, since its counts
    # do not enable it; a 702.10b scenario's card is in though 702.10c is common.
    assert cards == ["cut", "forest", "mind rot", "rest in peace", "swamp", "unseen card"]
    assert "no converted card" not in out


def test_a_card_no_converted_file_carries_is_kept_and_named(catalogue, capsys):
    (catalogue.cards_folder / "c" / "forest.txt").unlink()
    _, cards, out = _run(catalogue, capsys)
    assert "forest" in cards
    assert "no converted card under" in out and out.rstrip().endswith("forest")


def test_each_rule_prints_its_counts_and_variants_lacking_polarity_first(catalogue, capsys):
    _, _, out = _run(catalogue, capsys)
    block = out[out.index("121.4"):out.index("400.6")].splitlines()
    assert block[0].split() == ["121.4", "fires", "0", "does_not_fire", "94402", "rare"]
    assert block[1].split() == ["*", "121.4-1", "hand", "(fires):", "add_permanent"]
    assert block[2].split() == ["121.4-1", "empty", "(does_not_fire):", "move_card"]
    assert "704.5f       fires      6814 does_not_fire     15221  always" in out
    assert out.splitlines()[-1].startswith(
        "6 rules (3 rare, 3 one-sided, 1 always), 6 wording cards")


def test_the_subcommand_is_wired(catalogue, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "rules_catalogue.py", "directed-rules", "--counts", str(catalogue.counts),
        "--cards-folder", str(catalogue.cards_folder), "--rules-dir", str(catalogue.rules_dir)])
    rc.main()
    assert (catalogue.rules_dir / "directed-fork-rules.txt").read_text(
        encoding="utf-8").splitlines()[0] == "121.4"


# --- gen3-fields (T108): newly learnable rules and the turn_action verdict -------------------


def _registry():
    """A gen3-fields.json holding one input, one output and one dropped name."""
    entry = {"slot": "card", "shape": "s", "source": "snapshot", "what": "w"}
    return {
        "about": "a", "conventions": {"text": "t"},
        "inputs": [{"id": "in.subtypes", **entry, "names": ["snapshot:subtypes"]}],
        "outputs": [{"id": "out.game_lost", **entry, "names": ["new:game_result"]}],
        "extensions": [], "planned": [], "fixes": [],
        "covered": {"names": ["record:event:untapped"], "reason": "r"},
        "dropped": [{"reason": "subgames", "names": ["new:subgame_boundary"]}],
        "overrides": [],
    }


def _heads_row(rule, output="trained", missing=(), outputs=("zone_outcome",)):
    return {"id": rule, "output": output, "outputs": list(outputs),
            "input": "incomplete" if missing else "complete", "inputs_missing": list(missing),
            "note": ""}


_TURN_ACTION = {"family": "outcome_event", "record_kind": "turn_action", "outcome": "o",
                "near_miss": "n", "params": {"event": "untapped"}, "needs": []}


@pytest.fixture
def mapped(tmp_path):
    """A rules dir as a map rerun leaves it: heads lines, scenarios, the registry."""
    rules = tmp_path / "rules"
    (rules / "heads").mkdir(parents=True)
    (rules / "scenarios").mkdir()
    (rules / rc.GEN3_FIELDS_NAME).write_text(json.dumps(_registry(), indent=2) + "\n",
                                             encoding="utf-8")
    scenarios = [
        _scenario("500.1-1", ["500.1"], ["A"], [("off", "does_not_fire", "tap")]),
        _scenario("500.5-1", ["500.5"], ["B"], [("off", "does_not_fire", "tap")],
                  kind="turn_action", detector=_TURN_ACTION),
        _scenario("500.6-1", ["500.6"], ["C"], [("off", "does_not_fire", "tap")],
                  kind="turn_action", detector=_TURN_ACTION),
        _scenario("500.6-2", ["500.6"], ["C"], [("off", "does_not_fire", "tap")],
                  detector={"none": "no record shows it"}),
        _scenario("500.7-1", ["500.7"], ["D"], [("off", "does_not_fire", "tap")],
                  kind="turn_action", detector=_TURN_ACTION),
        _scenario("500.7-2", ["500.7"], ["D"], [("off", "does_not_fire", "tap")]),
    ]
    (rules / "scenarios" / "500.json").write_text(
        json.dumps({"unit": "500", "verified": True, "scenarios": scenarios}), encoding="utf-8")
    _write_heads(rules, [
        _heads_row("500.1"),
        _heads_row("500.2", missing=["snapshot:subtypes"]),
        _heads_row("500.3", missing=["new:subgame_boundary"]),
        _heads_row("500.5", output="recorded", outputs=["record:event:untapped"]),
        _heads_row("500.6", output="recorded", outputs=["record:event:untapped"]),
        _heads_row("500.7", output="missing", outputs=["new:game_result"]),
    ])
    return argparse.Namespace(rules_dir=rules)


def _write_heads(rules, rows):
    (rules / "heads" / "500.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _edit_heads(rules, rule, **changes):
    rows = rc.read_jsonl(rules / "heads" / "500.jsonl")
    for row in rows:
        if row["id"] == rule:
            row.update(changes)
    _write_heads(rules, rows)


def _gen3_fields(args, capsys):
    rc.cmd_gen3_fields(args)
    return capsys.readouterr().out


def _registry_on_disk(args):
    return json.loads((args.rules_dir / rc.GEN3_FIELDS_NAME).read_text(encoding="utf-8"))


def _heads_on_disk(args):
    return {r["id"]: r for r in rc.read_jsonl(args.rules_dir / "heads" / "500.jsonl")}


def test_the_first_run_records_the_learnable_rules_as_the_baseline(mapped, capsys):
    out = _gen3_fields(mapped, capsys)
    assert "no previous learnable list: recording 1 rules as the baseline" in out
    assert "NEWLY LEARNABLE" not in out
    assert _registry_on_disk(mapped)["learnable"] == ["500.1"]


def test_a_rerun_prints_the_rules_the_change_made_learnable_and_records_them(mapped, capsys):
    _gen3_fields(mapped, capsys)
    _edit_heads(mapped.rules_dir, "500.2", input="complete", inputs_missing=[])
    _edit_heads(mapped.rules_dir, "500.7", output="trained", outputs=["zone_outcome"])
    out = _gen3_fields(mapped, capsys)
    assert [line for line in out.splitlines() if "LEARNABLE" in line] == [
        "NEWLY LEARNABLE 500.2", "NEWLY LEARNABLE 500.7"]
    assert "2 rules newly learnable, 0 no longer, against the previous gen3-fields.json" in out
    assert _registry_on_disk(mapped)["learnable"] == ["500.1", "500.2", "500.7"]


def test_a_rule_that_stops_being_learnable_is_named(mapped, capsys):
    _gen3_fields(mapped, capsys)
    _edit_heads(mapped.rules_dir, "500.1", output="recorded")
    out = _gen3_fields(mapped, capsys)
    assert "NO LONGER LEARNABLE 500.1" in out
    assert _registry_on_disk(mapped)["learnable"] == []


def test_recording_the_list_leaves_the_rest_of_the_registry_as_written(mapped, capsys):
    path = mapped.rules_dir / rc.GEN3_FIELDS_NAME
    before = path.read_text(encoding="utf-8")
    _gen3_fields(mapped, capsys)
    _edit_heads(mapped.rules_dir, "500.2", input="complete", inputs_missing=[])
    _gen3_fields(mapped, capsys)
    after = path.read_text(encoding="utf-8")
    assert after == before.rstrip()[:-1].rstrip() + ',\n  "learnable": ["500.1", "500.2"]\n}\n'


def test_learnable_after_gen3_is_recomputed_from_the_heads_line(mapped, capsys):
    _gen3_fields(mapped, capsys)
    heads = _heads_on_disk(mapped)
    assert heads["500.2"]["learnable_after_gen3"] is True
    assert heads["500.2"]["gen3_inputs"] == ["in.subtypes"]
    assert heads["500.3"]["learnable_after_gen3"] is False
    assert heads["500.3"]["gen3_dropped"] == ["new:subgame_boundary"]
    _edit_heads(mapped.rules_dir, "500.3", inputs_missing=["snapshot:subtypes"])
    _gen3_fields(mapped, capsys)
    assert _heads_on_disk(mapped)["500.3"]["learnable_after_gen3"] is True


def test_the_turn_action_only_rules_ignore_a_none_detector_and_not_a_declared_one(mapped):
    by_rule = rc.load_scenarios_by_rule(mapped.rules_dir)
    # 500.6's other scenario declares none; 500.7's other reads resolution records.
    assert rc.turn_action_rules(by_rule) == {"500.5", "500.6"}


@pytest.mark.parametrize("rule", ["500.5", "500.6"])
@pytest.mark.parametrize("verdict", ["trained", "missing", "planned"])
def test_a_turn_action_only_rule_not_marked_recorded_fails_and_writes_nothing(
        mapped, capsys, rule, verdict):
    _edit_heads(mapped.rules_dir, rule, output=verdict)
    heads_before = (mapped.rules_dir / "heads" / "500.jsonl").read_text(encoding="utf-8")
    registry_before = (mapped.rules_dir / rc.GEN3_FIELDS_NAME).read_text(encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        rc.cmd_gen3_fields(mapped)
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert f"VERDICT {rule}: shown by turn_action records only, verdict {verdict!r}" in out
    assert (mapped.rules_dir / "heads" / "500.jsonl").read_text(encoding="utf-8") == heads_before
    assert (mapped.rules_dir / rc.GEN3_FIELDS_NAME).read_text(
        encoding="utf-8") == registry_before
    assert not (mapped.rules_dir / "GEN3-FIELDS.md").exists()


def test_the_gen3_fields_subcommand_is_wired(mapped, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "rules_catalogue.py", "gen3-fields", "--rules-dir", str(mapped.rules_dir)])
    rc.main()
    assert _registry_on_disk(mapped)["learnable"] == ["500.1"]
    assert (mapped.rules_dir / "GEN3-FIELDS.md").exists()
