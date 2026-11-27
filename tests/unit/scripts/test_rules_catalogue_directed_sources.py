"""Which sources of the counts ``rules_catalogue.py directed-rules`` decides
the list from: a fork's record re-records a moment of its live game, so
a rule's side is lacking when only forks show it."""

from __future__ import annotations

from rules_catalogue_parts import directed_forks as forks
from rules_catalogue_parts import locations, reading

from effects.domain.rule_catalogue import Polarity

RULE = "121.2"
SCENARIOS = [{
    "id": "121.2-1", "rules": [RULE], "cards": ["Mind Rot"],
    "variants": [
        {"label": "base", "polarity": "fires"},
        {"label": "empty", "polarity": "does_not_fire", "operation": "move_card"},
        {"label": "hand", "polarity": "fires", "operation": "add_permanent"},
    ],
}]
#: The registry's sides of the rule, as ``registered_sides`` reads them.
SIDES = {RULE: {"empty": Polarity.DOES_NOT_FIRE, "hand": Polarity.FIRES}}


def _counts(sides: dict) -> dict:
    return {"rules": {RULE: sides}, "not_countable": [], "none": {}}


def _listed(sides: dict) -> dict:
    return forks.directed_rules(_counts(sides), {RULE: SCENARIOS}, SIDES)


def test_a_side_only_forks_show_is_lacking_and_the_rule_is_listed():
    chosen = _listed({"fires": {"forge_ai": 5000}, "does_not_fire": {"fork": 300}})
    assert chosen[RULE]["lacking"] == {"does_not_fire"}
    assert chosen[RULE]["counts"] == (5000, 0)


def test_the_same_side_shown_by_natural_records_is_not_listed():
    assert RULE not in _listed({"fires": {"forge_ai": 5000}, "does_not_fire": {"forge_ai": 300}})


def test_a_combat_re_recorded_by_its_probes_counts_once():
    """A rule rare on its natural records stays rare however many probes
    re-ran its combats."""
    chosen = _listed({"fires": {"forge_ai": 60, "fork": 120}, "does_not_fire": {"forge_ai": 9}})
    assert chosen[RULE]["counts"] == (60, 9)
    assert chosen[RULE]["lacking"] == {"fires"}


# --- F15: the directed forks' own records (``aimed``) decide nothing --------------------------


def test_a_side_only_the_directed_forks_show_is_lacking_and_the_rule_is_listed():
    """118.12b in the fourth 2026-11-21 review pilot: every fires record was
    one of the directed forks the pilot ran for it, so its natural records
    show no fires side."""
    chosen = _listed({"fires": {"aimed": 629}, "does_not_fire": {"forge_ai": 4000}})
    assert chosen[RULE]["lacking"] == {"fires"}
    assert chosen[RULE]["natural"] == (0, 4000)


def test_the_same_side_shown_by_forge_ai_records_is_not_listed():
    assert RULE not in _listed({"fires": {"forge_ai": 629}, "does_not_fire": {"forge_ai": 4000}})


def test_the_probabilities_still_count_the_directed_forks_records():
    """What a side holds toward ``--count-per-side`` is every record but a
    fork's: the list is decided on natural records alone."""
    chosen = _listed({"fires": {"aimed": 629, "fork": 50}, "does_not_fire": {"forge_ai": 4000}})
    assert forks.directed_side_records(chosen) == {RULE: {"fires": 629, "does_not_fire": 4000}}


def test_the_committed_list_comes_out_unchanged_on_the_committed_counts():
    import json

    counts = json.loads((locations.RULES / forks.DIRECTED_COUNTS).read_text(encoding="utf-8"))
    chosen = forks.directed_rules(counts, reading.load_scenarios_by_rule(locations.RULES),
                               forks.registered_sides(locations.RULES))
    listed = (locations.RULES / "directed-fork-rules.txt").read_text(encoding="utf-8").split()
    assert list(chosen) == listed
    assert all(row["natural"] == row["counts"] for row in chosen.values()), (
        "the committed gen-2 counts hold natural records only")
