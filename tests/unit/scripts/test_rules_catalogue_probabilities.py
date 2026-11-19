"""``rules_catalogue.py directed-rules``' probabilities file (DF01): each listed rule's
own directed-fork probability, from the counts the list was decided from."""

from __future__ import annotations

import json
import sys

import pytest
import rules_catalogue as rc


def _row(counts):
    return {"counts": counts, "lacking": set(), "variants": [], "by_counts": True}


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
    cards = tmp_path / "cards"
    cards.mkdir()
    out = tmp_path / "probabilities.json"
    monkeypatch.setattr(rc, "NOT_DIRECTED", {})
    monkeypatch.setattr(sys, "argv", [
        "rules_catalogue.py", "directed-rules", "--counts", str(counts),
        "--cards-folder", str(cards), "--rules-dir", str(rules),
        "--probabilities-output", str(out), "--count-per-side", "100"])
    rc.main()
    assert json.loads(out.read_text(encoding="utf-8")) == {"121.2": 1.0}
    assert (rules / "directed-fork-rules.txt").read_text(encoding="utf-8") == "121.2\n"
