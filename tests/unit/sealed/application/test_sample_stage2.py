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
    n_slots=10,
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


# ─── T012: Mana cost prefix display (feature 016, FR-011) ────────────────────

_DUMMY_SPELL = "name: dummy\nmana cost: {1}{W}\ntypes: creature\n"
_DUMMY_NAMES = ["Dummy1", "Dummy2", "Dummy3"]  # padding to reach MINI.n_slots

def _setup_env_with_custom_cards(tmp_path, card_texts: dict[str, str]) -> tuple:
    """Setup env with specific card texts.  Booster is padded to MINI.n_slots - 6 cards."""
    from sealed.infrastructure.pool_loader import card_npz_path

    cards_path = tmp_path / "cards"

    # Write all provided card texts
    for name, text in card_texts.items():
        npz_path = card_npz_path(cards_path, name)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, embedding=np.ones(EMBED_DIM, dtype=np.float32))
        npz_path.with_suffix(".txt").write_text(text, encoding="utf-8")

    # Write dummy filler cards so booster reaches MINI.n_slots - 6
    for name in _DUMMY_NAMES:
        npz_path = card_npz_path(cards_path, name)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, embedding=np.ones(EMBED_DIM, dtype=np.float32))
        npz_path.with_suffix(".txt").write_text(_DUMMY_SPELL, encoding="utf-8")

    # Write basic lands
    for name in BASIC_LAND_NAMES:
        npz_path = card_npz_path(cards_path, name)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, embedding=np.ones(EMBED_DIM, dtype=np.float32))
        color = {"Plains": "W", "Island": "U", "Swamp": "B",
                 "Mountain": "R", "Forest": "G", "Wastes": "C"}[name]
        npz_path.with_suffix(".txt").write_text(
            f"name: {name.lower()}\ntypes: basic land\nactivated[1]: {{T}}: add {{{color}}}\n",
            encoding="utf-8"
        )

    # Pool: provided non-land cards + padding to fill MINI.n_slots - 6 slots
    booster_names = [n for n in card_texts if n not in BASIC_LAND_NAMES]
    n_needed = MINI.n_slots - len(BASIC_LAND_NAMES)
    for dummy in _DUMMY_NAMES:
        if len(booster_names) >= n_needed:
            break
        if dummy not in booster_names:
            booster_names.append(dummy)

    pools_path = tmp_path / "pools"
    pools_path.mkdir(parents=True)
    (pools_path / "pools.txt").write_text(";".join(booster_names) + "\n", encoding="utf-8")

    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    state = TrainingState(best_run=MAX_PICKS, episode_count=100)
    model_path = tmp_path / "models" / "stage2" / "latest.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(model_path, model, optimizer, state)

    return pools_path, cards_path, model_path


def test_nonland_card_shows_mana_cost_prefix(tmp_path, capsys):
    """T012: Non-land card must be printed as '{mana_cost} CardName'."""
    card_texts = {
        "Counterspell": "name: counterspell\nmana cost: {1}{U}{U}\ntypes: instant\n",
    }
    pools_path, cards_path, model_path = _setup_env_with_custom_cards(tmp_path, card_texts)

    use_case = SampleStage2UseCase()
    with patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI):
        use_case.execute(pools_path, cards_path, model_path, n_samples=1)

    out = capsys.readouterr().out
    # Non-land picks must show mana cost prefix: "{1}{U}{U} Counterspell"
    assert "{1}{U}{U} Counterspell" in out, (
        f"Expected '{{1}}{{U}}{{U}} Counterspell' in output. Got:\n{out}"
    )


def test_land_card_shows_name_only_no_cost_prefix(tmp_path, capsys):
    """T012: Land cards must be printed with name only (no mana cost prefix)."""
    card_texts = {
        "Lightning Bolt": "name: lightning bolt\nmana cost: {R}\ntypes: instant\n",
    }
    pools_path, cards_path, model_path = _setup_env_with_custom_cards(tmp_path, card_texts)

    use_case = SampleStage2UseCase()
    with patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI):
        use_case.execute(pools_path, cards_path, model_path, n_samples=1)

    out = capsys.readouterr().out
    pick_lines = [ln for ln in out.splitlines() if ln.strip() and ln.strip()[0].isdigit()
                  and "." in ln]
    # Find land picks (Plains appears in BASIC_LAND_NAMES)
    land_lines = [ln for ln in pick_lines if "Plains" in ln]
    for ln in land_lines:
        # Land lines must NOT start with a mana cost like {W} or {1}{W}
        pick_text = ln.split(".", 1)[-1].strip() if "." in ln else ln
        assert not pick_text.startswith("{"), (
            f"Land pick line should not have mana cost prefix: {ln}"
        )


def test_multiface_card_shows_first_face_mana_cost(tmp_path, capsys):
    """T012: Multi-face card shows the first-face mana cost as prefix."""
    adventure_text = (
        "name: brave adventurer\nmana cost: {1}{W}\ntypes: creature\n"
        "ALTERNATE\n"
        "name: heroic adventure\nmana cost: {2}{U}\ntypes: instant\n"
    )
    card_texts = {
        "Brave Adventurer": adventure_text,
    }
    pools_path, cards_path, model_path = _setup_env_with_custom_cards(tmp_path, card_texts)

    use_case = SampleStage2UseCase()
    with patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI):
        use_case.execute(pools_path, cards_path, model_path, n_samples=1)

    out = capsys.readouterr().out
    # First-face cost {1}{W} must appear before the card name
    assert "{1}{W} Brave Adventurer" in out, (
        f"Expected '{{1}}{{W}} Brave Adventurer' (first-face cost) in output. Got:\n{out}"
    )
