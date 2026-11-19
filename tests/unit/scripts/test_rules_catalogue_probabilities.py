"""``rules_catalogue.py directed-rules``' probabilities file (DF01): each listed rule's
own directed-fork probability, from the counts the list was decided from."""

from __future__ import annotations

import json
import sys

import pytest
import rules_catalogue as rc


def _row(counts, sides=("fires", "does_not_fire")):
    return {"counts": counts, "lacking": set(), "variants": [], "by_counts": True,
            "sides": frozenset(sides)}


def test_each_rule_gets_the_probability_its_two_sides_counts_give():
    chosen = {
        "121.2": _row((0, 94402)),      # a side at zero
        "702.9a": _row((40, 7000)),     # 60 of 100 missing
        "400.6": _row((300, 200)),      # both reached
        "704.5f": _row((99, 500)),      # nearly there: the floor
    }
    assert rc.directed_probabilities(chosen, 100) == pytest.approx(
        {"121.2": 1.0, "702.9a": 0.6, "400.6": 0.0, "704.5f": 0.05}
    )


def test_a_rule_the_counts_cannot_speak_for_is_forked_always():
    assert rc.directed_probabilities({"500.6": _row(None)}, 100) == {"500.6": 1.0}


def test_a_rule_is_judged_on_the_sides_its_registered_variants_make():
    """704.5b registers only a fires variant: once fires is full it is at 0, though
    its does_not_fire side stays empty for ever."""
    chosen = {
        "704.5b": _row((100, 0), sides=("fires",)),
        "702.118b": _row((0, 30), sides=("does_not_fire",)),
        "104.4a": _row((5, 0), sides=("fires",)),
    }
    assert rc.directed_probabilities(chosen, 100) == pytest.approx(
        {"704.5b": 0.0, "702.118b": 0.7, "104.4a": 0.95})


def test_a_rule_whose_registered_variants_are_all_other_is_left_to_the_run_default():
    """111.10p and 205.1a register only an `other` variant: no side to fill, so the
    file names no probability and the run's --directed-fork-probability applies, as
    `collect-variants` does."""
    assert rc.directed_probabilities({"205.1a": _row((0, 0), sides=())}, 100) == {}


def _real_chosen():
    counts = json.loads((rc.RULES / rc.DIRECTED_COUNTS).read_text(encoding="utf-8"))
    return counts, rc.directed_rules(counts, rc.load_scenarios_by_rule(rc.RULES),
                                     rc.registered_sides(rc.RULES))


def test_the_chosen_rules_are_the_registered_listed_rules():
    _, chosen = _real_chosen()
    listed = (rc.RULES / "directed-fork-rules.txt").read_text(encoding="utf-8").split()
    assert list(chosen) == listed


def test_the_probabilities_of_the_real_list_are_accepted_by_match_outcomes(tmp_path):
    from sealed.infrastructure.directed_fork_probabilities_file import (
        read_directed_fork_probabilities,
    )

    _, chosen = _real_chosen()
    listed = list(chosen)
    path = tmp_path / "p.json"
    path.write_text(json.dumps(rc.directed_probabilities(chosen, 100)), encoding="utf-8")
    read_directed_fork_probabilities(path, listed)


def test_704_5b_reaches_zero_once_its_fires_side_is_full():
    counts, _ = _real_chosen()
    counts["rules"].setdefault("704.5b", {}).setdefault("fires", {})["directed"] = 1000
    chosen = rc.directed_rules(counts, rc.load_scenarios_by_rule(rc.RULES),
                               rc.registered_sides(rc.RULES))
    assert rc.directed_probabilities(chosen, 100)["704.5b"] == 0.0


def test_the_subcommand_writes_the_file_when_asked(tmp_path, monkeypatch, capsys):
    rules = tmp_path / "rules"
    (rules / "scenarios").mkdir(parents=True)
    scenario = {
        "id": "121.2-1", "rules": ["121.2"], "title": "t", "record_kind": "resolution",
        "variants": [{"label": "base", "polarity": "fires", "delta": "", "expected": "e"},
                     {"label": "empty", "polarity": "does_not_fire", "delta": "d",
                      "expected": "e", "operation": "move_card"}],
        "cards": [],
        "detector": {"family": "sba_zone_move", "record_kind": "resolution", "outcome": "o",
                     "near_miss": "n", "params": {"check": "toughness_zero"}, "needs": []},
    }
    (rules / "scenarios" / "121.json").write_text(
        json.dumps({"unit": "121", "verified": True, "scenarios": [scenario]}), encoding="utf-8")
    counts = tmp_path / "counts.json"
    counts.write_text(json.dumps({
        "records": 1, "rules": {"121.2": {"does_not_fire": {"forge_ai": 40}}},
        "not_countable": [], "none": {}, "review": [],
    }), encoding="utf-8")
    (rules / "directed-fork-variants.tsv").write_text(
        "rule\tvariant\n121.2\tempty\n", encoding="utf-8")
    cards = tmp_path / "cards"
    cards.mkdir()
    out = tmp_path / "probabilities.json"
    monkeypatch.setattr(rc, "NOT_DIRECTED", {})
    monkeypatch.setattr(sys, "argv", [
        "rules_catalogue.py", "directed-rules", "--counts", str(counts),
        "--cards-folder", str(cards), "--rules-dir", str(rules),
        "--probabilities-output", str(out), "--count-per-side", "100"])
    rc.main()
    assert json.loads(out.read_text(encoding="utf-8")) == {"121.2": pytest.approx(0.6)}
    assert (rules / "directed-fork-rules.txt").read_text(encoding="utf-8") == "121.2\n"
