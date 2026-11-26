"""Rule-test and AUC-probe standard errors cluster by game (finding 12.7,
research E4: "every one clusters by game").

``n₊`` and ``n₋`` of the Hanley–McNeil bound count the distinct games among an
item's positive rows and among its negative rows, so 40 records from 4 games
have the standard error of 4 clusters. The records are the real gen-3
fixture's, their games edited.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.application.freeze_selection_set import (
    FreezePorts,
    ProbeItem,
    RuleTestRules,
    freeze_selection_set,
)
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import RuleTag
from effects.domain.rule_output_carriage import carried_outputs
from effects.domain.score_scales import auc_standard_error
from effects.domain.selection_set import error_of, item_key
from tests.unit.effects.application.test_freeze_selection_set import (
    GD,
    LENIENT,
    _Grid,
    _Losses,
    _manifest,
)

FIRES, NOT = Polarity.FIRES, Polarity.DOES_NOT_FIRE
RULE = "121.2"
SELECTION = ("s1", "s2", "s3", "s4")
REPORTING = ("r1", "r2", "r3", "r4")
HALVES = {GD: {"selection": list(SELECTION), "reporting": list(REPORTING)}}


@pytest.fixture(scope="module")
def records():
    """80 rule-test-eligible fixture records: per half 4 games of 10, the first
    two games' records positive, the last two's negative."""
    from tests.unit.effects.builders import curated_fixture_records

    eligible = [r for r in curated_fixture_records()
                if not (r.fork or r.interventional or r.synthetic or r.aimed_at is not None
                        or r.directed is not None)
                and carried_outputs(r, [r.actor_player], ("gate",))]
    assert len(eligible) >= 80
    games = SELECTION + REPORTING
    return [replace(record, game_id=games[i // 10]) for i, record in enumerate(eligible[:80])]


def _positive(record) -> bool:
    return record.game_id[1] in "12"


class _GameProbes:
    """One binary probe type, positive on the records of each half's first two games."""

    def items(self, records):
        for record in records:
            yield ProbeItem("dies", "binary", record.record_id, "E1", float(_positive(record)))


def _freeze(records):
    tags = {record.record_id: [RuleTag(
        RULE, FIRES if _positive(record) else NOT, observed=_positive(record), condition=(),
        outcome=(), scenario=f"{RULE}-1")] for record in records}
    rules = RuleTestRules(tag=lambda record: tags.get(record.record_id, []),
                          declared=frozenset({RULE}), trained=frozenset({RULE}))
    by_game: dict[str, list] = {}
    for record in records:
        by_game.setdefault(record.game_id, []).append(record)
    return freeze_selection_set(
        _manifest(HALVES), {GD: lambda: [list(rows) for rows in by_game.values()]},
        ports=FreezePorts(rules=rules, probes=_GameProbes(), grids=[_Grid()],
                          deviance=_Losses()),
        thresholds=LENIENT)


def test_forty_rule_test_records_from_four_games_have_the_error_of_four_clusters(records):
    frozen = _freeze(records)
    for half in ("selection", "reporting"):
        sides = frozen.rule_tests[GD][half][RULE]
        assert (len(sides["fires"]), len(sides["does_not_fire"])) == (20, 20)
        error = error_of(frozen.standard_errors[item_key(GD, half, "rule_tests", RULE)])
        assert error == pytest.approx(auc_standard_error(2, 2))


def test_a_probe_type_s_error_counts_the_distinct_games_of_each_label(records):
    frozen = _freeze(records)
    for half in ("selection", "reporting"):
        assert len(frozen.probes[GD][half]["dies"]) == 40
        error = error_of(frozen.standard_errors[item_key(GD, half, "probes", "dies")])
        assert error == pytest.approx(auc_standard_error(2, 2))


def test_the_clustered_error_counts_each_game_once_per_side():
    from effects.domain.score_scales import clustered_auc_standard_error

    assert clustered_auc_standard_error(["a"] * 20 + ["b"] * 20, ["c"] * 20 + ["d"] * 20) == \
        pytest.approx(auc_standard_error(2, 2))
    assert clustered_auc_standard_error(["a"], []) == float("inf")
