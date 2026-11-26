"""The control score's grids: each record's reading against that record's
answer, and the damage, combat and evasion grids weighing one third each
(findings 12.5, 12.6; root spec § 14.1).

The scorer runs over the gen-3 fixture records and a selection set frozen
over them by hand; the readings are set by hand in place of the model's, so
a model that knows each record's answer and one that knows only each cell's
base rate can be scored side by side.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from effects.application.control_items import component_score
from effects.application.control_score import ControlScorer
from effects.domain.selection_set import GRIDS, FrozenGrid, GridCell, TaggedRecord
from tests.unit.effects.application.test_control_score import (
    RULE_OUTPUTS,
    STRATA,
    _Editor,
    _selection,
)
from tests.unit.effects.builders import fixture_run

#: The second cell's answers, in the grid's record order: they differ by record.
HIGH = (1.0, 0.0, 1.0, 0.0)


@pytest.fixture(scope="module")
def run(corpus, records, tmp_path_factory):
    return fixture_run(corpus, records, tmp_path_factory.mktemp("control-grids"))


@pytest.fixture(scope="module")
def prepared(run, records):
    """A grid of four records whose ``high`` cell answers differ, prepared
    for its grids alone."""
    selection = _selection(records)
    games = set(selection.halves["card-disjoint"]["selection"])
    chosen = [r for r in records if r.state.entities and r.game_id in games][:4]
    grid = FrozenGrid((GridCell("low", (0.0,) * 4), GridCell("high", HIGH)),
                      tuple(TaggedRecord(r.record_id, r.game_id, (r.state.entities[0].id,))
                            for r in chosen))
    selection = replace(selection, grids={s: {"selection": {"damage": grid}} for s in STRATA})
    editor = _Editor()
    scorer = ControlScorer.prepare(
        selection, dict.fromkeys(STRATA, lambda: [records]), preparer=run.preparer,
        fields=list(run.fields.values()), rule_outputs=RULE_OUTPUTS, mask="hard",
        stratum_weight=0.5, deviance_floor=0.0, editor=editor, chunk_records=8,
        components=(GRIDS,))
    return scorer, grid, editor.asked[:len(chosen)]


def _scored(scorer, readings) -> float:
    """The damage grid's item when the model reads ``readings`` (``record_id
    -> per-cell reading``): each record's cells sit at the positions the
    editor's order gives them, two a record."""
    scorer._values = lambda cache, forward: readings
    score = scorer.epoch(None, None, ())
    return score.strata["card-disjoint"].items[GRIDS]["damage"]


def test_a_base_rate_model_scores_below_one_that_knows_each_record(prepared):
    """Finding 12.6: the cell means of the two models are the same, so a
    cell-mean score cannot tell them apart. Against each record's answer the
    base rate errs by 0.5 on every ``high`` read: MAE 0.25 over B = 0.375."""
    scorer, grid, asked = prepared
    answers = grid.record_answers()
    knows = np.array([value for rid in asked for value in answers[rid]])
    base_rate = np.array([0.0, float(np.mean(HIGH))] * len(asked))
    assert _scored(scorer, knows) == pytest.approx(1.0)
    assert _scored(scorer, base_rate) == pytest.approx(1 - 0.25 / 0.375)


GRID_ITEMS = {"damage": 1.0, "combat/none/blocker": 0.0,
              "combat-several/attacker-first-strike/attacker": 0.0, "evasion": 0.0}


def test_the_three_grids_weigh_equally_whatever_their_variants():
    """Finding 12.5: two combat variants and one damage grid, beside the
    evasion grid; the damage grid weighs one third of the component."""
    assert component_score(GRID_ITEMS, stratum="s", component=GRIDS) == pytest.approx(1 / 3)


def test_a_grid_with_no_scored_variant_is_left_out_not_counted_as_zero():
    without_evasion = {k: v for k, v in GRID_ITEMS.items() if k != "evasion"}
    assert component_score(without_evasion, stratum="s", component=GRIDS) == pytest.approx(0.5)
    unscored = {**GRID_ITEMS, "evasion": math.nan}
    assert component_score(unscored, stratum="s", component=GRIDS) == pytest.approx(0.5)


def test_each_variant_is_clipped_before_its_grid_averages_them():
    items = {"combat/none/blocker": -3.0, "combat/none/attacker": 1.0, "damage": 0.5}
    assert component_score(items, stratum="s", component=GRIDS) == pytest.approx(0.25)
