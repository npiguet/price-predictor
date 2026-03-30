"""T022 — Unit tests for SampleStage1UseCase."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.optim as optim

from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.application.train_stage1 import TrainingState
from sealed.application.sample_stage1 import SampleStage1UseCase
from sealed.infrastructure.pool_loader import card_npz_path
from sealed.infrastructure.pool_model_store import PoolModelStore


MINI = PoolTransformerConfig(
    n_slots=4,
    d_model=12,
    n_layers=1,
    n_heads=2,
    card_embed_dim=8,
    ff_dim=16,
    dropout=0.0,
)

EMBED_DIM = 8
BASIC_LANDS = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]
BOOSTER_CARDS = ["Card1", "Card2", "Card3", "Card4"]


def _write_npz(path: Path, embed_dim: int = EMBED_DIM) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embedding=np.ones(embed_dim, dtype=np.float32))


def _setup_env(tmp_path: Path, n_pools: int = 3):
    cards_path = tmp_path / "cards"
    for name in BOOSTER_CARDS + BASIC_LANDS:
        _write_npz(card_npz_path(cards_path, name))

    pools_path = tmp_path / "pools"
    pools_path.mkdir(parents=True)
    lines = [";".join(BOOSTER_CARDS)] * n_pools
    (pools_path / "pools.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    state = TrainingState()

    model_path = tmp_path / "models" / "latest.pt"
    model_path.parent.mkdir(parents=True)
    store = PoolModelStore()
    store.save(model_path, model, optimizer, state)

    return pools_path, cards_path, model_path


# ── missing checkpoint ────────────────────────────────────────────────────────

def test_missing_checkpoint_raises_file_not_found(tmp_path):
    _, cards_path, _ = _setup_env(tmp_path)
    pools_path = tmp_path / "pools"
    model_path = tmp_path / "nonexistent.pt"

    use_case = SampleStage1UseCase()
    with pytest.raises(FileNotFoundError):
        with patch("sealed.application.sample_stage1.PoolTransformerConfig", return_value=MINI):
            use_case.execute(pools_path, cards_path, model_path, n_samples=1)


# ── n_samples controls output ─────────────────────────────────────────────────

def test_n_samples_controls_count(tmp_path, capsys):
    pools_path, cards_path, model_path = _setup_env(tmp_path, n_pools=10)

    use_case = SampleStage1UseCase()
    with patch("sealed.application.sample_stage1.PoolTransformerConfig", return_value=MINI):
        use_case.execute(pools_path, cards_path, model_path, n_samples=3)

    captured = capsys.readouterr()
    # Each sample prints a "--- Sample N" header
    count = captured.out.count("--- Sample")
    assert count == 3


def test_n_samples_capped_at_pool_count(tmp_path, capsys):
    """n_samples > number of pools should not crash."""
    pools_path, cards_path, model_path = _setup_env(tmp_path, n_pools=2)

    use_case = SampleStage1UseCase()
    with patch("sealed.application.sample_stage1.PoolTransformerConfig", return_value=MINI):
        use_case.execute(pools_path, cards_path, model_path, n_samples=100)

    captured = capsys.readouterr()
    count = captured.out.count("--- Sample")
    assert count == 2  # only 2 pools available


# ── output format ─────────────────────────────────────────────────────────────

def test_output_format_contains_status(tmp_path, capsys):
    pools_path, cards_path, model_path = _setup_env(tmp_path, n_pools=2)

    use_case = SampleStage1UseCase()
    with patch("sealed.application.sample_stage1.PoolTransformerConfig", return_value=MINI):
        use_case.execute(pools_path, cards_path, model_path, n_samples=2)

    captured = capsys.readouterr()
    # Each sample header should contain SUCCESS or TERMINATED
    headers = [ln for ln in captured.out.splitlines() if ln.startswith("--- Sample")]
    assert len(headers) == 2
    for h in headers:
        assert "SUCCESS" in h or "TERMINATED" in h


def test_output_contains_card_names(tmp_path, capsys):
    pools_path, cards_path, model_path = _setup_env(tmp_path, n_pools=1)

    use_case = SampleStage1UseCase()
    with patch("sealed.application.sample_stage1.PoolTransformerConfig", return_value=MINI):
        use_case.execute(pools_path, cards_path, model_path, n_samples=1)

    captured = capsys.readouterr()
    # At least one booster card name should appear in the output
    has_card = any(card in captured.out for card in BOOSTER_CARDS)
    assert has_card
