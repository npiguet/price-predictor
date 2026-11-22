"""``rules_catalogue.py directed-rules``' probabilities file (DF01): each listed rule's
records per registered side, from the counts the list was decided from, which
`match-outcomes` computes each rule's directed-fork probability from (DR1)."""

from __future__ import annotations

import json
import sys

import rules_catalogue as rc
from rules_catalogue_parts import directed_forks as forks
from rules_catalogue_parts import locations, reading


def _row(counts, sides=("fires", "does_not_fire")):
    return {"counts": counts, "lacking": set(), "variants": [], "by_counts": True,
            "sides": frozenset(sides)}


def test_each_rule_gets_the_records_of_its_registered_sides():
    chosen = {
        "121.2": _row((0, 94402)),
        "702.9a": _row((40, 7000)),
    }
    assert forks.directed_side_records(chosen) == {
        "121.2": {"fires": 0, "does_not_fire": 94402},
        "702.9a": {"fires": 40, "does_not_fire": 7000},
    }


def test_a_rule_the_counts_cannot_speak_for_starts_with_no_record_on_its_sides():
    assert forks.directed_side_records({"500.6": _row(None, sides=("fires",))}) == {
        "500.6": {"fires": 0}}


def test_a_rule_is_judged_on_the_sides_its_registered_variants_make():
    """704.5b registers only a fires variant: its does_not_fire side, empty for
    ever, is not written, so it reaches 0 once fires is full."""
    chosen = {
        "704.5b": _row((100, 0), sides=("fires",)),
        "702.118b": _row((0, 30), sides=("does_not_fire",)),
    }
    assert forks.directed_side_records(chosen) == {
        "704.5b": {"fires": 100}, "702.118b": {"does_not_fire": 30}}


def test_a_rule_whose_registered_variants_are_all_other_is_left_to_the_run_default():
    """111.10p and 205.1a register only an `other` variant: no side to fill, so the
    file names no rule and the run's --directed-fork-probability applies, as
    `collect-variants` does."""
    assert forks.directed_side_records({"205.1a": _row((0, 0), sides=())}) == {}


def test_the_document_carries_the_count_per_side():
    document = forks.probabilities_document({"704.5b": _row((5, 0), sides=("fires",))}, 100)
    assert document == {"count_per_side": 100, "rules": {"704.5b": {"fires": 5}}}


def _real_chosen():
    counts = json.loads((locations.RULES / forks.DIRECTED_COUNTS).read_text(encoding="utf-8"))
    return counts, forks.directed_rules(counts, reading.load_scenarios_by_rule(locations.RULES),
                                     forks.registered_sides(locations.RULES))


def test_the_chosen_rules_are_the_registered_listed_rules():
    _, chosen = _real_chosen()
    listed = (locations.RULES / "directed-fork-rules.txt").read_text(encoding="utf-8").split()
    assert list(chosen) == listed


def _real_needs(tmp_path, chosen):
    from sealed.infrastructure.directed_fork_probabilities_file import read_directed_fork_needs

    path = tmp_path / "p.json"
    path.write_text(json.dumps(forks.probabilities_document(chosen, 100)), encoding="utf-8")
    return read_directed_fork_needs(path, list(chosen))


def test_the_real_list_s_file_is_accepted_by_match_outcomes_at_its_counts_probabilities(
    tmp_path,
):
    """Every listed rule's registered variants make one side (two would be refused)."""
    from price_predictor.domain.directed_fork_probabilities import directed_fork_probability

    _, chosen = _real_chosen()
    starting = _real_needs(tmp_path, chosen).probabilities()
    assert starting == {
        rule: directed_fork_probability(
            [dict(zip(("fires", "does_not_fire"), row["counts"], strict=True))[side]
             for side in sorted(row["sides"])], 100)
        for rule, row in chosen.items() if row["sides"]
    }
    assert (starting["118.12b"], starting["704.5f"]) == (1.0, 0.0)


def test_a_rule_at_1_filled_by_this_run_s_kept_forks_is_written_at_0_for_the_next_spawn(
    tmp_path,
):
    """DR1: 118.12b (fires 0 in the counts) once the workers' status lines report
    100 kept forks."""
    from price_predictor.domain.worker_status import SlotStatuses, directed_fork_tally_document
    from tests.unit.worker_status_line import directed_counts, status_line

    _, chosen = _real_chosen()
    slot = SlotStatuses()
    slot.add_line(status_line("a", directed={"118.12b": directed_counts(
        matched=130, drawn=110, budget_blocked=10, taken=100, kept=100)}))
    tally = directed_fork_tally_document([slot])
    assert _real_needs(tmp_path, chosen).probabilities(tally)["118.12b"] == 0.0


def test_701_24e_reaches_zero_once_its_fires_side_is_full(tmp_path):
    counts, _ = _real_chosen()
    counts["rules"].setdefault("701.24e", {}).setdefault("fires", {})["directed"] = 1000
    chosen = forks.directed_rules(counts, reading.load_scenarios_by_rule(locations.RULES),
                               forks.registered_sides(locations.RULES))
    assert _real_needs(tmp_path, chosen).probabilities()["701.24e"] == 0.0


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
    monkeypatch.setattr(forks, "NOT_DIRECTED", {})
    monkeypatch.setattr(sys, "argv", [
        "rules_catalogue.py", "directed-rules", "--counts", str(counts),
        "--cards-folder", str(cards), "--rules-dir", str(rules),
        "--probabilities-output", str(out), "--count-per-side", "100"])
    rc.main()
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "count_per_side": 100, "rules": {"121.2": {"does_not_fire": 40}}}
    assert (rules / "directed-fork-rules.txt").read_text(encoding="utf-8") == "121.2\n"
