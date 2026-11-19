"""The steered round's exit code when it plays nothing (contracts/cli.md,
``collect-variants``: 0 when every unit was already done, 1 when units remain
that no deck could hold)."""

from __future__ import annotations

from unittest.mock import MagicMock

from effects.application import steered_variants
from effects.application.collect_variants import CollectVariantsConfig, VariantRound
from effects.application.steered_variants import SteerOutcome, collect_steered
from tests.unit.effects.application.test_steered_variants import _steer, world  # noqa: F401


def test_short_pairs_with_no_variant_to_make_is_not_an_outcome_of_nothing_short(
    world, monkeypatch,  # noqa: F811
):
    world[0].variants = lambda pairs, counts, total: []
    outcome, played, _ = _steer(world, monkeypatch, MagicMock())
    assert outcome is SteerOutcome.NO_VARIANT and played == []


def test_no_short_pair_left_is_nothing_short(world, monkeypatch):  # noqa: F811
    world[1]["rules"]["702.2b"]["does_not_fire"]["forge_ai"] = 500
    outcome, _, _ = _steer(world, monkeypatch, MagicMock())
    assert outcome is SteerOutcome.NOTHING_SHORT


def _exit_of(outcome, monkeypatch, tmp_path, world_):  # noqa: F811
    import json

    _, counts, rules = world_
    counts_path = tmp_path / "counts.json"
    counts_path.write_text(json.dumps(counts), encoding="utf-8")
    monkeypatch.setattr(steered_variants, "unstage_variants", MagicMock())
    monkeypatch.setattr(steered_variants, "steer", lambda *args: outcome)
    monkeypatch.setattr(steered_variants, "finish", lambda *a, **k: 0)
    config = CollectVariantsConfig(
        round=VariantRound.STEERED, rule_counts=counts_path, rules_dir=rules,
        forge_cards_path=tmp_path / "cardsfolder", effect_records=tmp_path / "records",
        variant_scripts=tmp_path / "variant-scripts", cards_folders=(tmp_path / "none",),
    )
    return collect_steered(config, [])


def test_a_round_with_short_pairs_and_no_variant_exits_one(
    world, monkeypatch, tmp_path, fake_supervisor,  # noqa: F811
):
    assert _exit_of(SteerOutcome.NO_VARIANT, monkeypatch, tmp_path, world) == 1


def test_a_round_with_nothing_short_exits_zero(
    world, monkeypatch, tmp_path, fake_supervisor,  # noqa: F811
):
    assert _exit_of(SteerOutcome.NOTHING_SHORT, monkeypatch, tmp_path, world) == 0
