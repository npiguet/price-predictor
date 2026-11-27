"""The freeze checks the grid component's precision with the weights the score
gives it: the mean of the damage, combat and evasion grids' means (CPD1 I-2)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.application import freeze_selection_set as freeze_module
from effects.domain.score_scales import grouped_standard_error
from tests.unit.effects.application.test_freeze_selection_set import (
    LENIENT,
    SelectionSetRefused,
    _balanced_records,
    _freeze,
    _Grid,
    _two_halves,
)

ITEM_ERROR = 0.1
TARGET = replace(LENIENT, max_item_error=ITEM_ERROR, max_component_error=0.05)


def _grid_group(name: str) -> str:
    return "combat" if name.startswith("combat/") else name


def test_one_item_a_group_and_a_group_of_eighteen_variants_weigh_as_the_score_does() -> None:
    errors = {"damage": ITEM_ERROR} | {f"combat/v{i}": ITEM_ERROR for i in range(18)}
    assert grouped_standard_error(errors, _grid_group) == pytest.approx(0.0514, abs=1e-4)


def test_all_three_grids_weigh_a_third_each() -> None:
    errors = {"damage": ITEM_ERROR, "evasion": ITEM_ERROR} | {
        f"combat/v{i}": ITEM_ERROR for i in range(18)}
    assert grouped_standard_error(errors, _grid_group) == pytest.approx(0.0478, abs=1e-4)


def test_no_item_is_an_infinite_error() -> None:
    assert grouped_standard_error({}, _grid_group) == float("inf")


def _grid_misses(monkeypatch, names: list[str]) -> list[str]:
    """The details of the grids component's misses in one freeze: the other
    components miss the tight target on the toy records and are not read."""
    monkeypatch.setattr(freeze_module, "grid_standard_error", lambda answers: ITEM_ERROR)
    halves = _two_halves(2)
    records, tags = _balanced_records(halves)
    with pytest.raises(SelectionSetRefused) as refused:
        _freeze(halves, records, tags, thresholds=TARGET,
                grids=[_Grid(name=name) for name in names])
    return [m.detail for m in refused.value.misses if m.component == "grids"]


def test_a_freeze_without_the_evasion_grid_misses_the_component_target(monkeypatch) -> None:
    """Damage and 18 combat variants at 0.1 each: an equal mean of 19 items is
    0.023, the grouped mean the score reads is 0.051."""
    details = _grid_misses(monkeypatch, ["damage"] + [f"combat/v{i}" for i in range(18)])
    assert len(details) == 2
    assert all("standard error 0.0514 over 19 item(s)" in detail for detail in details)


def test_a_freeze_of_all_three_grids_meets_it(monkeypatch) -> None:
    assert _grid_misses(monkeypatch, ["damage", "evasion"]
                        + [f"combat/v{i}" for i in range(18)]) == []
