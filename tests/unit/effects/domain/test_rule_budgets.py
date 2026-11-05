"""Rule budgets (FR-054, FR-054a, FR-055; research R2): story 6's acceptance
scenarios 1, 2 and 7, and the caps around them."""

from __future__ import annotations

import random

from effects.domain.rule_budgets import BudgetedRecord, RuleRow, plan_rule_budgets
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import RuleSource

FIRES, NOT, OTHER = Polarity.FIRES, Polarity.DOES_NOT_FIRE, Polarity.OTHER
KLASS = "resolution-effect"
AI = RuleSource.FORGE_AI


def make(n, *tags, start=0, natural=1, text=None, source=AI, klass=KLASS, game=None):
    """``n`` records tagged ``tags``, hashes ``start…start+n-1``."""
    return [
        BudgetedRecord(
            game=game or f"g{start + i}", record_hash=start + i, klass=klass,
            text=text if text is not None else f"text-{start + i}",
            natural=natural, source=source, tags=tuple(tags),
        )
        for i in range(n)
    ]


def forty_rules(target_rule_records):
    """Scenario 1's class: 39 rules on one record each beside the rule under test."""
    filler = [
        record for k in range(39)
        for record in make(1, (f"rule-{k:02}", FIRES), start=10_000_000 + k)
    ]
    return target_rule_records + filler


def plan(records, *, target=400_000, budgeted=None, reuse_cap=4, text_cap=0):
    if budgeted is None:
        budgeted = {rule for record in records for rule, _ in record.tags}
    return plan_rule_budgets(
        records, class_targets={KLASS: target}, budgeted=budgeted,
        reuse_cap=reuse_cap, text_cap=text_cap,
    )


def written(budgets, rule, polarity, klass=KLASS) -> int:
    return budgets.rows[klass][rule][str(polarity)].written_total


def test_scenario_1_a_short_rule_repeats_to_the_reuse_cap_and_reports_its_shortfall():
    records = forty_rules(make(1000, ("x", FIRES)) + make(1000, ("x", NOT), start=1000))
    budgets = plan(records)
    assert budgets.shares[KLASS] == 10_000
    fires = budgets.rows[KLASS]["x"]["fires"]
    nots = budgets.rows[KLASS]["x"]["does_not_fire"]
    assert (fires.share, nots.share) == (5_000, 5_000)
    assert fires.written_total + nots.written_total == 8_000
    assert fires.shortfall + nots.shortfall == 2_000
    assert "x" not in budgets.one_sided.get(KLASS, ())


def test_scenario_2_a_copy_goes_to_the_rule_further_below_its_share():
    # Rule a is 4,000 below its half, rule b 1,000 below.
    records = (
        make(1, ("a", FIRES), ("b", FIRES), start=0, natural=0)
        + make(1000, ("a", FIRES), start=100)
        + make(4000, ("b", FIRES), start=10_000)
    )
    budgets = plan_rule_budgets(
        records, class_targets={KLASS: 20_000}, budgeted={"a", "b"},
        reuse_cap=1, text_cap=0,
    )
    assert budgets.added("g0", 0) == ("a",)


def test_scenario_7_one_polarity_fills_and_the_other_stays_short_and_is_named():
    records = forty_rules(make(1600, ("x", FIRES)) + make(400, ("x", NOT), start=1600))
    budgets = plan(records)
    fires = budgets.rows[KLASS]["x"]["fires"]
    nots = budgets.rows[KLASS]["x"]["does_not_fire"]
    assert (fires.share, nots.share) == (5_000, 5_000)
    assert (fires.written_total, nots.written_total) == (5_000, 1_600)
    assert fires.shortfall + nots.shortfall == 3_400
    assert "x" in budgets.one_sided[KLASS]


def test_a_short_side_is_never_given_to_the_other():
    records = make(10, ("x", FIRES)) + make(1, ("x", NOT), start=10)
    budgets = plan(records, target=1_000)
    assert written(budgets, "x", NOT) == 4
    assert written(budgets, "x", FIRES) == 40


def test_a_recorded_rule_gets_no_share_and_no_copy():
    records = make(3, ("t", FIRES)) + make(3, ("r", FIRES), start=3)
    budgets = plan(records, target=100, budgeted={"t"})
    assert budgets.shares[KLASS] == 100
    row = budgets.rows[KLASS]["r"]["fires"]
    assert row.share is None and row.shortfall == 0 and row.written_total == 3
    assert all("r" not in drawn for game in budgets.copies.values()
               for drawn in game.values())
    assert budgets.unbudgeted == {"r": {"fires": {"forge_ai": 3}}}


def test_other_counts_toward_the_rule_and_draws_no_half():
    records = make(2, ("x", OTHER)) + make(2, ("x", FIRES), start=2)
    budgets = plan(records, target=40)
    other = budgets.rows[KLASS]["x"]["other"]
    assert other.share is None and other.written_total == 2 and not other.added
    for i in (0, 1):
        assert budgets.added(f"g{i}", i) == ()


def test_a_multi_tagged_copy_counts_toward_each_rule():
    records = make(1, ("a", FIRES), ("b", NOT), natural=0)
    budgets = plan(records, target=100, reuse_cap=3)
    assert len(budgets.added("g0", 0)) == 3
    assert written(budgets, "a", FIRES) == 3
    assert written(budgets, "b", NOT) == 3


def test_a_record_never_passes_the_reuse_cap():
    records = make(5, ("x", FIRES), natural=2) + make(5, ("x", NOT), start=5, natural=4)
    budgets = plan(records, target=1_000, reuse_cap=4)
    for record in records:
        assert record.natural + len(budgets.added(record.game, record.record_hash)) <= 4
    assert budgets.rows[KLASS]["x"]["fires"].added == {"forge_ai": 10}
    assert budgets.rows[KLASS]["x"]["does_not_fire"].added == {}


def test_repeats_spread_over_a_rules_records():
    records = make(10, ("x", FIRES), natural=0)
    budgets = plan(records, target=30, reuse_cap=4)
    counts = sorted(len(budgets.added(r.game, r.record_hash)) for r in records)
    assert sum(counts) == 15 and counts[-1] - counts[0] <= 1


def test_the_text_cap_bounds_a_texts_total_in_the_class():
    records = make(10, ("x", FIRES), natural=1, text="same")
    budgets = plan(records, target=1_000, text_cap=25)
    assert written(budgets, "x", FIRES) == 25


def test_rows_split_available_and_written_by_source():
    records = (make(2, ("x", FIRES)) + make(1, ("x", FIRES), start=2, source=RuleSource.AIMED)
               + make(3, ("x", NOT), start=3, source=RuleSource.RANDOM_SEAT))
    row = plan(records, target=6).rows[KLASS]["x"]["fires"]
    assert row.available == {"forge_ai": 2, "aimed": 1}
    assert row.written == {"forge_ai": 2, "aimed": 1}
    assert row.shortfall == 0


def test_classes_are_budgeted_apart():
    records = make(4, ("x", FIRES)) + make(4, ("x", FIRES), start=4, klass="combat")
    budgets = plan_rule_budgets(
        records, class_targets={KLASS: 20, "combat": 0}, budgeted={"x"},
        reuse_cap=4, text_cap=0,
    )
    assert written(budgets, "x", FIRES) == 10
    assert written(budgets, "x", FIRES, klass="combat") == 4


def test_the_plan_is_a_function_of_the_records_not_their_order():
    records = (make(30, ("a", FIRES), ("b", NOT), natural=0)
               + make(20, ("b", FIRES), start=30) + make(5, ("a", NOT), start=50))
    first = plan(records, target=200)
    shuffled = list(records)
    random.Random(7).shuffle(shuffled)
    assert plan(shuffled, target=200) == first


def test_a_row_round_trips_through_its_dict():
    row = RuleRow(available={"forge_ai": 3}, share=5, written={"forge_ai": 4},
                  added={"forge_ai": 1}, shortfall=1)
    assert RuleRow.from_dict(row.as_dict()) == row
    other = RuleRow(available={}, share=None, written={}, added={}, shortfall=0)
    assert RuleRow.from_dict(other.as_dict()) == other
