"""Which sources of the counts ``rules_catalogue.py directed-rules`` decides
the list from (F01): a fork's record re-records a moment of its live game, so
a rule's side is lacking when only forks show it."""

from __future__ import annotations

import rules_catalogue as rc

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
    return rc.directed_rules(_counts(sides), {RULE: SCENARIOS}, SIDES)


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
