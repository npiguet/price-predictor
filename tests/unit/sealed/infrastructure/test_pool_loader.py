"""T006 — Unit tests for PoolLoader."""
from __future__ import annotations

import numpy as np
import pytest

from sealed.infrastructure.pool_loader import PoolLoader, BASIC_LAND_NAMES, card_npz_path

EMBED_DIM = 512
_LOADER = PoolLoader()


def _write_npz(path, embed_dim=EMBED_DIM):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embedding=np.ones(embed_dim, dtype=np.float32))


def _write_card_npz(cards_path, card_name, embed_dim=EMBED_DIM):
    _write_npz(card_npz_path(cards_path, card_name), embed_dim)


# ── load_pools ────────────────────────────────────────────────────────────────

def test_load_pools_missing_file_raises_value_error(tmp_path):
    missing = tmp_path / "pools.txt"
    with pytest.raises(ValueError, match="pools.txt not found or empty"):
        _LOADER.load_pools(missing)


def test_load_pools_empty_file_raises_value_error(tmp_path):
    f = tmp_path / "pools.txt"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="pools.txt not found or empty"):
        _LOADER.load_pools(f)


def test_load_pools_whitespace_only_raises_value_error(tmp_path):
    f = tmp_path / "pools.txt"
    f.write_text("   \n\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="pools.txt not found or empty"):
        _LOADER.load_pools(f)


def test_load_pools_returns_list(tmp_path):
    f = tmp_path / "pools.txt"
    f.write_text("CardA;CardB;CardC\nCardD;CardE\n", encoding="utf-8")
    result = _LOADER.load_pools(f)
    assert result == ["CardA;CardB;CardC", "CardD;CardE"]


def test_load_pools_strips_blank_lines(tmp_path):
    f = tmp_path / "pools.txt"
    f.write_text("\nCardA;CardB\n\nCardC;CardD\n\n", encoding="utf-8")
    result = _LOADER.load_pools(f)
    assert len(result) == 2


# ── assemble_pool_tensor ──────────────────────────────────────────────────────

def test_assemble_pool_tensor_missing_npz_raises_fnf(tmp_path):
    # Create basic land files but not the booster card
    for land in BASIC_LAND_NAMES:
        _write_card_npz(tmp_path, land)
    with pytest.raises(FileNotFoundError) as exc_info:
        _LOADER.assemble_pool_tensor("MissingCard", tmp_path)
    assert "MissingCard" in str(exc_info.value)


def test_assemble_pool_tensor_shape_one_booster(tmp_path):
    """1 booster + 6 basic lands, 512-dim embeddings → [max(7,96), 516] tensor."""
    _write_card_npz(tmp_path, "CardA", EMBED_DIM)
    for land in BASIC_LAND_NAMES:
        _write_card_npz(tmp_path, land, EMBED_DIM)

    tensor = _LOADER.assemble_pool_tensor("CardA", tmp_path)
    # max(7, 96) = 96 rows; 512+4=516 cols
    assert tensor.shape == (96, EMBED_DIM + 4)


def test_assemble_pool_tensor_all_flags_zero(tmp_path):
    """Flags (last 4 columns) must all be 0 on the assembled tensor."""
    _write_card_npz(tmp_path, "CardA", EMBED_DIM)
    for land in BASIC_LAND_NAMES:
        _write_card_npz(tmp_path, land, EMBED_DIM)

    tensor = _LOADER.assemble_pool_tensor("CardA", tmp_path)
    flags = tensor[:, EMBED_DIM:]
    assert (flags == 0).all()


def test_assemble_pool_tensor_embeddings_loaded(tmp_path):
    """The booster card embedding should appear in the first row."""
    emb = np.arange(EMBED_DIM, dtype=np.float32)
    path = card_npz_path(tmp_path, "CardA")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embedding=emb)
    for land in BASIC_LAND_NAMES:
        _write_card_npz(tmp_path, land, EMBED_DIM)

    tensor = _LOADER.assemble_pool_tensor("CardA", tmp_path)
    np.testing.assert_array_equal(tensor[0, :EMBED_DIM].numpy(), emb)


def test_assemble_pool_tensor_missing_basic_land(tmp_path):
    """Missing basic land .npz should raise FileNotFoundError with the land name."""
    _write_card_npz(tmp_path, "CardA", EMBED_DIM)
    # Deliberately omit "Plains"
    for land in BASIC_LAND_NAMES[1:]:
        _write_card_npz(tmp_path, land, EMBED_DIM)
    with pytest.raises(FileNotFoundError) as exc_info:
        _LOADER.assemble_pool_tensor("CardA", tmp_path)
    assert "Plains" in str(exc_info.value)
