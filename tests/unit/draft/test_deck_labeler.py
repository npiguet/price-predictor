"""Batched pod labeling matches per-seat labeling (perf path for generate-draft-data)."""

from __future__ import annotations

import numpy as np
import torch

from draft.application.generate_draft_data import _label_pools, _PickerLabeler
from sealed.domain.picker_model import PickerConfig, PickerModel
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer

_DIM = 64  # >= FEATURE_COUNT (32) and divisible by the head counts below


class _FakeLocator:
    """Embeddings by name (IS_LAND=0 -> all spells); no card text."""

    def __init__(self) -> None:
        self._tbl: dict[str, np.ndarray] = {}

    def load_embedding(self, name: str) -> np.ndarray:
        if name not in self._tbl:
            v = np.random.randn(_DIM).astype(np.float32)
            v[-32] = 0.0  # IS_LAND flag off
            self._tbl[name] = v
        return self._tbl[name].copy()

    def load_text(self, name: str):
        return None


def _picker_labeler() -> _PickerLabeler:
    torch.manual_seed(0)
    np.random.seed(0)
    scorer = SetTransformerScorer(
        ScorerConfig(d_model=_DIM, n_heads=4, n_seeds=4, d_ff=128, mlp_hidden=32, dropout=0.0),
    ).eval()
    picker = PickerModel(
        PickerConfig(embedding_dim=_DIM, n_heads=8, n_layers=2, dropout=0.0),
    ).eval()
    lab = _PickerLabeler.__new__(_PickerLabeler)
    lab._torch = torch
    lab._model = picker
    lab._scorer = scorer
    lab._locator = _FakeLocator()
    lab._device = torch.device("cpu")
    return lab


def test_batched_picker_matches_per_pool() -> None:
    lab = _picker_labeler()
    pools = [
        [f"p0_{i}" for i in range(45)],   # normal draft pool
        [f"p1_{i}" for i in range(30)],   # shorter
        ["only", "two"],                  # < 23 -> failed build
        [f"p3_{i}" for i in range(60)],   # longer (padding stress)
    ]
    single = [lab.build_and_score(p) for p in pools]
    batched = lab.build_and_score_many(pools)

    assert len(batched) == len(pools)
    for (s_deck, s_score), (b_deck, b_score) in zip(single, batched):
        assert s_deck == b_deck  # identical card selection
        if s_score is None:
            assert b_score is None
        else:
            assert b_score is not None and abs(s_score - b_score) < 1e-4


def test_too_small_pool_fails_in_batch() -> None:
    lab = _picker_labeler()
    results = lab.build_and_score_many([["a", "b"], [f"c{i}" for i in range(45)]])
    assert results[0] == ([], None)        # < 23 embeddable cards
    assert results[1][0] and results[1][1] is not None


class _RecordingLabeler:
    """Implements both methods; records which one _label_pools calls."""

    def __init__(self) -> None:
        self.many_calls = 0

    def build_and_score(self, pool):
        return (["x"], 1.0)

    def build_and_score_many(self, pools):
        self.many_calls += 1
        return [(["x"], float(len(p))) for p in pools]


class _SingleOnlyLabeler:
    def build_and_score(self, pool):
        return (list(pool), float(len(pool)))


def test_label_pools_prefers_batch_method() -> None:
    lab = _RecordingLabeler()
    out = _label_pools(lab, [["a"], ["b", "c"]])
    assert lab.many_calls == 1
    assert out == [(["x"], 1.0), (["x"], 2.0)]


def test_label_pools_falls_back_to_per_pool() -> None:
    out = _label_pools(_SingleOnlyLabeler(), [["a"], ["b", "c"]])
    assert out == [(["a"], 1.0), (["b", "c"], 2.0)]
