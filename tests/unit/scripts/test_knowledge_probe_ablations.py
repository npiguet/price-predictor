"""The two ablations on a tiny random gen-3 model."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from effect_knowledge_probes import (
    ablation,
    board,
)

from tests.unit.scripts.knowledge_probe_support import E_DIM


def _cache_items(probe_model, records):
    """Line items with ``e`` drawn at random: the cache statistics' input."""
    rng = np.random.default_rng(0)
    texts = list(probe_model.batcher.batch_texts(records))
    return [SimpleNamespace(text=text, api=f"api{i % 3}", e=rng.standard_normal(E_DIM))
            for i, text in enumerate(texts)]


class TestAblationsOnGen3:
    def test_ability_slot_noise_runs_over_gen3_records(self, records, probe_model):
        stats = ablation.cache_statistics(_cache_items(probe_model, records))
        result = ablation.run_ablation(probe_model, records[:24], stats, batch_size=8)
        assert result
        for body in result.values():
            assert np.isfinite(body["base_loss"])
            for replacement in ablation.REPLACEMENTS:
                assert set(body[replacement]) == set(ablation.SCOPES)
                assert all(np.isfinite(v) for v in body[replacement].values())
        # The card blocks' ability slots carry e on gen-3, so their noise moves the loss.
        assert any(body["noise"]["cards"] != 0.0 for body in result.values())

    def test_a_gen3_permanent_lists_its_own_unlisted_lines(self, records):
        from effects.domain.state_snapshot import CardLine

        entity = next(e for r in records for e in r.state.entities if e.lines)
        lines = (CardLine(keys=(), kind="static", granted=False),
                 CardLine(keys=(), kind="trigger", granted=True),
                 CardLine(keys=(), kind="replacement", granted=False, active=False),
                 CardLine(keys=(), kind="keyword", granted=False, fragment="Flying"))
        assert board.unlisted_kinds(replace(entity, lines=lines)) == {"static"}
        assert board.unlisted_kinds(replace(entity, lines=())) == frozenset()

    def test_the_board_abilities_ablation_runs_over_gen3_subsets(
        self, records, sidecars, probe_model,
    ):
        chosen = {"card-disjoint": board.collect(records, sidecars, per_subset=6)}
        assert any(chosen["card-disjoint"].values())
        stats = ablation.cache_statistics(_cache_items(probe_model, records))
        result = board.run(probe_model, chosen, stats)
        measured = [name for name in board.SUBSETS if result[name].get("ablation")]
        assert measured
        assert "# Board abilities" in board.table(result)
