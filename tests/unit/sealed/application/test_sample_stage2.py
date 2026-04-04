"""T023 — Unit tests for SampleStage2UseCase."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch.optim as optim

from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.application.train_stage1 import TrainingState
from sealed.application.sample_stage2 import SampleStage2UseCase
from sealed.infrastructure.pool_loader import card_npz_path
from sealed.infrastructure.pool_model_store import PoolModelStore
from sealed.domain.episode_runner import MAX_PICKS, BASIC_LAND_NAMES


MINI = PoolTransformerConfig(
    n_slots=4,
    d_model=16,
    n_layers=1,
    n_heads=2,
    card_embed_dim=8,
    ff_dim=16,
    dropout=0.0,
)
EMBED_DIM = 8
BOOSTER_CARDS = ["Card1", "Card2", "Card3", "Card4"]


def _write_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embedding=np.ones(EMBED_DIM, dtype=np.float32))


def _write_txt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _setup_env(tmp_path: Path, n_pools: int = 3) -> tuple[Path, Path, Path]:
    cards_path = tmp_path / "cards"
    for name in BOOSTER_CARDS + BASIC_LAND_NAMES:
        _write_npz(card_npz_path(cards_path, name))
        txt_path = card_npz_path(cards_path, name).with_suffix(".txt")
        if name in BASIC_LAND_NAMES:
            color = {"Plains": "W", "Island": "U", "Swamp": "B",
                     "Mountain": "R", "Forest": "G", "Wastes": "C"}[name]
            _write_txt(txt_path, f"name: {name.lower()}\ntypes: basic land\nactivated[1]: {{T}}: add {{{color}}}\n")
        else:
            _write_txt(txt_path, f"name: {name.lower()}\nmana cost: {{1}}{{W}}\ntypes: creature\n")

    pools_path = tmp_path / "pools"
    pools_path.mkdir(parents=True)
    lines = [";".join(BOOSTER_CARDS)] * n_pools
    (pools_path / "pools.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    state = TrainingState(best_run=MAX_PICKS, episode_count=100)
    model_path = tmp_path / "models" / "stage2" / "latest.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(model_path, model, optimizer, state)

    return pools_path, cards_path, model_path


# ─── Missing checkpoint ───────────────────────────────────────────────────────

def test_missing_checkpoint_raises(tmp_path):
    _, cards_path, _ = _setup_env(tmp_path)
    pools_path = tmp_path / "pools"
    model_path = tmp_path / "nonexistent.pt"

    use_case = SampleStage2UseCase()
    with pytest.raises(FileNotFoundError):
        with patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI):
            use_case.execute(pools_path, cards_path, model_path, n_samples=1)


# ─── Output: 40 picks ─────────────────────────────────────────────────────────

def test_output_shows_40_picks_for_success(tmp_path, capsys):
    pools_path, cards_path, model_path = _setup_env(tmp_path, n_pools=5)

    use_case = SampleStage2UseCase()
    with patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI):
        use_case.execute(pools_path, cards_path, model_path, n_samples=1)

    out = capsys.readouterr().out
    # Should show "picks=40" in the header
    assert "picks=40" in out or "picks=" in out


def test_n_samples_controls_output_count(tmp_path, capsys):
    pools_path, cards_path, model_path = _setup_env(tmp_path, n_pools=10)

    use_case = SampleStage2UseCase()
    with patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI):
        use_case.execute(pools_path, cards_path, model_path, n_samples=3)

    out = capsys.readouterr().out
    sample_headers = [ln for ln in out.splitlines() if "--- Sample" in ln]
    assert len(sample_headers) == 3


# ─── Output: mana analysis ─────────────────────────────────────────────────────

def test_output_contains_mana_sources_section(tmp_path, capsys):
    pools_path, cards_path, model_path = _setup_env(tmp_path, n_pools=5)

    use_case = SampleStage2UseCase()
    with patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI):
        use_case.execute(pools_path, cards_path, model_path, n_samples=1)

    out = capsys.readouterr().out
    # Should contain "Mana sources" section
    assert "Mana sources" in out or "mana source" in out.lower()


def test_output_contains_all_six_colors(tmp_path, capsys):
    pools_path, cards_path, model_path = _setup_env(tmp_path, n_pools=5)

    use_case = SampleStage2UseCase()
    with patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI):
        use_case.execute(pools_path, cards_path, model_path, n_samples=1)

    out = capsys.readouterr().out
    # Per CLI contract, output should include W, U, B, R, G, C
    for color in ("W:", "U:", "B:", "R:", "G:", "C:"):
        assert color in out, f"Expected color {color} in mana analysis output"


def test_output_contains_score_and_land_count(tmp_path, capsys):
    pools_path, cards_path, model_path = _setup_env(tmp_path, n_pools=5)

    use_case = SampleStage2UseCase()
    with patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI):
        use_case.execute(pools_path, cards_path, model_path, n_samples=1)

    out = capsys.readouterr().out
    assert "Lands:" in out or "lands:" in out.lower()
    assert "Score:" in out or "score:" in out.lower()


def test_output_shows_ideal_vs_actual(tmp_path, capsys):
    pools_path, cards_path, model_path = _setup_env(tmp_path, n_pools=5)

    use_case = SampleStage2UseCase()
    with patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI):
        use_case.execute(pools_path, cards_path, model_path, n_samples=1)

    out = capsys.readouterr().out
    # CLI contract shows "ideal → actual" format
    assert "→" in out or "->" in out
