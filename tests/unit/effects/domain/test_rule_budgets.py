"""Rule budgets (FR-054, FR-054a, FR-055; research R2): story 6's acceptance
scenarios 1, 2 and 7, and the caps around them."""

from __future__ import annotations

import random
from dataclasses import dataclass

from effects.domain.rule_budgets import BudgetedRecord, BudgetInputs, RuleRow, plan_rule_budgets
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import RuleSource

FIRES, NOT, OTHER = Polarity.FIRES, Polarity.DOES_NOT_FIRE, Polarity.OTHER
KLASS = "resolution-effect"
AI = RuleSource.FORGE_AI


@dataclass(frozen=True, slots=True)
class Shape:
    """What ``make`` gives every record it builds beside its tags."""

    start: int = 0
    natural: int = 1
    text: str | None = None
    source: RuleSource = AI
    klass: str = KLASS


def make(n, *tags, shape=Shape()):
    """``n`` records tagged ``tags``, hashes ``shape.start…shape.start+n-1``."""
    return [
        BudgetedRecord(
            game=f"g{shape.start + i}", record_hash=shape.start + i, klass=shape.klass,
            text=shape.text if shape.text is not None else f"text-{shape.start + i}",
            natural=shape.natural, source=shape.source, tags=tuple(tags),
        )
        for i in range(n)
    ]


def forty_rules(target_rule_records):
    """Scenario 1's class: 39 rules on one record each beside the rule under test."""
    filler = [
        record for k in range(39)
        for record in make(1, (f"rule-{k:02}", FIRES), shape=Shape(start=10_000_000 + k))
    ]
    return target_rule_records + filler


def plan(records, *, target=400_000, budgeted=None, reuse_cap=4, text_cap=0):
    if budgeted is None:
        budgeted = {rule for record in records for rule, _ in record.tags}
    return plan_rule_budgets(
        records, BudgetInputs(
            class_targets={KLASS: target}, budgeted=budgeted,
            reuse_cap=reuse_cap, text_cap=text_cap,
        ),
    )


def _added(budgets, game, record_hash) -> tuple[str, ...]:
    """The drawn rule of each added copy of one record; empty for none."""
    return budgets.copies.get(game, {}).get(record_hash, ())


def written(budgets, rule, polarity, klass=KLASS) -> int:
    return budgets.rows[klass][rule][str(polarity)].written_total


def test_scenario_1_a_short_rule_repeats_to_the_reuse_cap_and_reports_its_shortfall():
    records = forty_rules(
        make(1000, ("x", FIRES)) + make(1000, ("x", NOT), shape=Shape(start=1000)))
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
        make(1, ("a", FIRES), ("b", FIRES), shape=Shape(start=0, natural=0))
        + make(1000, ("a", FIRES), shape=Shape(start=100))
        + make(4000, ("b", FIRES), shape=Shape(start=10_000))
    )
    budgets = plan_rule_budgets(
        records, BudgetInputs(
            class_targets={KLASS: 20_000}, budgeted={"a", "b"},
            reuse_cap=1, text_cap=0,
        ),
    )
    assert _added(budgets, "g0", 0) == ("a",)


def test_scenario_7_one_polarity_fills_and_the_other_stays_short_and_is_named():
    records = forty_rules(make(1600, ("x", FIRES)) + make(400, ("x", NOT), shape=Shape(start=1600)))
    budgets = plan(records)
    fires = budgets.rows[KLASS]["x"]["fires"]
    nots = budgets.rows[KLASS]["x"]["does_not_fire"]
    assert (fires.share, nots.share) == (5_000, 5_000)
    assert (fires.written_total, nots.written_total) == (5_000, 1_600)
    assert fires.shortfall + nots.shortfall == 3_400
    assert "x" in budgets.one_sided[KLASS]


def test_a_short_side_is_never_given_to_the_other():
    records = make(10, ("x", FIRES)) + make(1, ("x", NOT), shape=Shape(start=10))
    budgets = plan(records, target=1_000)
    assert written(budgets, "x", NOT) == 4
    assert written(budgets, "x", FIRES) == 40


def test_a_recorded_rule_gets_no_share_and_no_copy():
    records = make(3, ("t", FIRES)) + make(3, ("r", FIRES), shape=Shape(start=3))
    budgets = plan(records, target=100, budgeted={"t"})
    assert budgets.shares[KLASS] == 100
    row = budgets.rows[KLASS]["r"]["fires"]
    assert row.share is None and row.shortfall == 0 and row.written_total == 3
    assert all("r" not in drawn for game in budgets.copies.values()
               for drawn in game.values())
    assert budgets.unbudgeted == {"r": {"fires": {"forge_ai": 3}}}


def test_other_counts_toward_the_rule_and_draws_no_half():
    records = make(2, ("x", OTHER)) + make(2, ("x", FIRES), shape=Shape(start=2))
    budgets = plan(records, target=40)
    other = budgets.rows[KLASS]["x"]["other"]
    assert other.share is None and other.written_total == 2 and not other.added
    for i in (0, 1):
        assert _added(budgets, f"g{i}", i) == ()


def test_a_multi_tagged_copy_counts_toward_each_rule():
    records = make(1, ("a", FIRES), ("b", NOT), shape=Shape(natural=0))
    budgets = plan(records, target=100, reuse_cap=3)
    assert len(_added(budgets, "g0", 0)) == 3
    assert written(budgets, "a", FIRES) == 3
    assert written(budgets, "b", NOT) == 3


def test_a_protection_copy_counts_toward_its_drawn_rule_alone():
    """Finding 14.2: a record offering 702.16b and 702.16j enters the budgets
    as one protection record tagged with both; 702.16b's half is already
    filled, so its one copy is drawn for 702.16j and adds to 702.16j only."""
    filled = make(2, ("702.16b", FIRES), shape=Shape(natural=1, start=10))
    offering = [BudgetedRecord(
        game="g0", record_hash=0, klass=KLASS, text="text-0", natural=0, source=AI,
        tags=(("702.16b", FIRES), ("702.16j", FIRES)), protection=True,
    )]
    budgets = plan(offering + filled, target=8, reuse_cap=1)
    assert _added(budgets, "g0", 0) == ("702.16j",)
    rows = budgets.rows[KLASS]
    assert rows["702.16j"]["fires"].added == {"forge_ai": 1}
    assert rows["702.16j"]["fires"].written == {"forge_ai": 1}
    assert rows["702.16b"]["fires"].added == {}
    assert rows["702.16b"]["fires"].written == {"forge_ai": 2}


def test_a_row_says_how_much_of_its_added_was_drawn_for_its_rule():
    """Finding 14.2: a natural record's copy counts toward every rule the record
    is tagged with, so 702.16e's ``added`` holds the two copies drawn for
    702.16b on a record tagged with both, beside the one drawn for 702.16e on
    another record. ``drawn`` is the part whose ``drawn_rule`` is the row's
    rule: the copies written for it."""
    other = make(1, ("702.16e", FIRES), shape=Shape(natural=1, start=10))
    both = make(1, ("702.16b", FIRES), ("702.16e", FIRES), shape=Shape(natural=0))
    budgets = plan(both + other, target=12, reuse_cap=2)
    assert _added(budgets, "g0", 0) == ("702.16b", "702.16b")
    assert _added(budgets, "g10", 10) == ("702.16e",)
    rows = budgets.rows[KLASS]
    assert (rows["702.16b"]["fires"].added, rows["702.16b"]["fires"].drawn) == (
        {"forge_ai": 2}, {"forge_ai": 2})
    assert (rows["702.16e"]["fires"].added, rows["702.16e"]["fires"].drawn) == (
        {"forge_ai": 3}, {"forge_ai": 1})
    assert RuleRow.from_dict(rows["702.16e"]["fires"].as_dict()) == rows["702.16e"]["fires"]


def test_a_record_never_passes_the_reuse_cap():
    records = (make(5, ("x", FIRES), shape=Shape(natural=2))
               + make(5, ("x", NOT), shape=Shape(start=5, natural=4)))
    budgets = plan(records, target=1_000, reuse_cap=4)
    for record in records:
        assert record.natural + len(_added(budgets, record.game, record.record_hash)) <= 4
    assert budgets.rows[KLASS]["x"]["fires"].added == {"forge_ai": 10}
    assert budgets.rows[KLASS]["x"]["does_not_fire"].added == {}


def test_repeats_spread_over_a_rules_records():
    records = make(10, ("x", FIRES), shape=Shape(natural=0))
    budgets = plan(records, target=30, reuse_cap=4)
    counts = sorted(len(_added(budgets, r.game, r.record_hash)) for r in records)
    assert sum(counts) == 15 and counts[-1] - counts[0] <= 1


def test_the_text_cap_bounds_a_texts_total_in_the_class():
    records = make(10, ("x", FIRES), shape=Shape(natural=1, text="same"))
    budgets = plan(records, target=1_000, text_cap=25)
    assert written(budgets, "x", FIRES) == 25


def test_the_text_cap_counts_the_texts_untagged_records_too():
    """Ten tagged records of a text whose class also writes fifteen untagged
    natural copies of it: the text is already at the cap of 25, so no copy is
    added, though its tagged records alone are well under it."""
    records = make(10, ("x", FIRES), shape=Shape(natural=1, text="same"))
    budgets = plan_rule_budgets(
        records, BudgetInputs(
            class_targets={KLASS: 1_000}, budgeted={"x"}, reuse_cap=4, text_cap=25,
            text_natural={KLASS: {"same": 25}},
        ),
    )
    assert written(budgets, "x", FIRES) == 10
    budgets = plan_rule_budgets(
        records, BudgetInputs(
            class_targets={KLASS: 1_000}, budgeted={"x"}, reuse_cap=4, text_cap=25,
            text_natural={KLASS: {"same": 20}},
        ),
    )
    assert written(budgets, "x", FIRES) == 15


def test_rows_split_available_and_written_by_source():
    aimed = Shape(start=2, source=RuleSource.AIMED)
    records = (make(2, ("x", FIRES)) + make(1, ("x", FIRES), shape=aimed)
               + make(3, ("x", NOT), shape=Shape(start=3, source=RuleSource.RANDOM_SEAT)))
    row = plan(records, target=6).rows[KLASS]["x"]["fires"]
    assert row.available == {"forge_ai": 2, "aimed": 1}
    assert row.written == {"forge_ai": 2, "aimed": 1}
    assert row.shortfall == 0


def test_a_fork_record_is_written_and_left_out_of_available():
    """F01: a probe re-records its live combat, so the records available to
    a rule count the combat once; its copies are written all the same."""
    records = (make(2, ("x", FIRES), shape=Shape(natural=1))
               + make(3, ("x", FIRES), shape=Shape(start=2, natural=1, source=RuleSource.FORK)))
    row = plan(records, target=6).rows[KLASS]["x"]["fires"]
    assert row.available == {"forge_ai": 2}
    assert row.written == {"forge_ai": 2, "fork": 3}


def test_a_side_only_forks_show_keeps_its_row():
    """Deathtouch's near miss is a probe that strips it: the side has no
    record available and its probes' copies written."""
    records = (make(2, ("x", FIRES), shape=Shape(natural=1))
               + make(3, ("x", NOT), shape=Shape(start=2, natural=1, source=RuleSource.FORK)))
    row = plan(records, target=6).rows[KLASS]["x"]["does_not_fire"]
    assert row.available == {}
    assert row.written == {"fork": 3}


def test_an_unbudgeted_rule_s_available_records_leave_out_forks():
    records = (make(2, ("y", NOT), shape=Shape(natural=1))
               + make(4, ("y", NOT), shape=Shape(start=2, natural=1, source=RuleSource.FORK)))
    budgets = plan(records, budgeted=set())
    assert budgets.unbudgeted == {"y": {"does_not_fire": {"forge_ai": 2}}}


def test_classes_are_budgeted_apart():
    records = make(4, ("x", FIRES)) + make(4, ("x", FIRES), shape=Shape(start=4, klass="combat"))
    budgets = plan_rule_budgets(
        records, BudgetInputs(
            class_targets={KLASS: 20, "combat": 0}, budgeted={"x"},
            reuse_cap=4, text_cap=0,
        ),
    )
    assert written(budgets, "x", FIRES) == 10
    assert written(budgets, "x", FIRES, klass="combat") == 4


def test_the_plan_is_a_function_of_the_records_not_their_order():
    records = (make(30, ("a", FIRES), ("b", NOT), shape=Shape(natural=0))
               + make(20, ("b", FIRES), shape=Shape(start=30))
               + make(5, ("a", NOT), shape=Shape(start=50)))
    first = plan(records, target=200)
    shuffled = list(records)
    random.Random(7).shuffle(shuffled)
    assert plan(shuffled, target=200) == first


def test_a_row_round_trips_through_its_dict():
    row = RuleRow(available={"forge_ai": 3}, share=5, written={"forge_ai": 4},
                  added={"forge_ai": 1}, drawn={"forge_ai": 1}, shortfall=1)
    assert RuleRow.from_dict(row.as_dict()) == row
    other = RuleRow(available={}, share=None, written={}, added={}, drawn={}, shortfall=0)
    assert RuleRow.from_dict(other.as_dict()) == other
