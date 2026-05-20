"""Unit tests for PickDecksUseCase."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from sealed.application.pick_decks import PickDecksConfig, PickDecksUseCase
from sealed.domain.card_embedding_layout import FEATURE_COUNT, IS_LAND

WIDTH = 40
_IS_LAND_OFFSET = WIDTH - FEATURE_COUNT + IS_LAND


class _StubPicker:
    """Picker whose logits decrease with pool position (index 0 ranks highest)."""

    def __call__(self, cards, mask):
        n = cards.shape[1]
        logits = torch.arange(n, 0, -1, dtype=torch.float32).unsqueeze(0)
        return logits, torch.zeros(1)


def _write_pools(path: Path, pools: list[tuple[str, list[str]]]) -> None:
    path.write_text(
        "\n".join(f"{set_code};" + "|".join(names) for set_code, names in pools) + "\n",
        encoding="utf-8",
    )


def _make_locator(land_names: set[str], spell_names: set[str]) -> MagicMock:
    locator = MagicMock()
    rng = np.random.default_rng(7)
    embeddings: dict[str, np.ndarray] = {}
    for n in land_names | spell_names:
        emb = rng.standard_normal(WIDTH).astype(np.float32)
        emb[-FEATURE_COUNT:] = 0.0
        emb[_IS_LAND_OFFSET] = 1.0 if n in land_names else 0.0
        embeddings[n] = emb

    def load_embedding(name: str):
        return embeddings.get(name)

    def load_text(name: str):
        if name not in embeddings:
            return None
        text = MagicMock()
        text.mana_cost_line.return_value = None
        return text

    locator.load_embedding.side_effect = load_embedding
    locator.load_text.side_effect = load_text
    return locator


def _patch(use_case, locator, model=None):
    if model is None:
        model = _StubPicker()
    use_case._load_picker = MagicMock(return_value=(model, WIDTH))
    return patch(
        "sealed.application.pick_decks.ConvertedCardLocator", return_value=locator,
    )


class TestPickDecks:
    def test_zero_lands_yields_40_cards(self, tmp_path):
        spells = [f"s{i}" for i in range(30)]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, [("MH3", spells)])
        locator = _make_locator(set(), set(spells))
        use_case = PickDecksUseCase()
        config = PickDecksConfig(
            pools_path=pools_path, label="picker", cards_path=tmp_path / "c",
            output=tmp_path / "decks.txt",
        )
        with _patch(use_case, locator):
            written = use_case.execute(config)
        assert written == 1
        label, set_code, names = config.output.read_text("utf-8").strip().split(";", 2)
        cards = names.split("|")
        assert len(cards) == 40
        # 23 spells + 17 basics (no nonbasic land in the pool).
        nonbasic = [c for c in cards if c in spells]
        assert len(nonbasic) == 23

    def test_picked_lands_still_total_40(self, tmp_path):
        # Lands ranked first (StubPicker logits decrease with position) so they
        # are taken before the 23-spell quota fills: 23 spells + 5 lands + 12
        # basics = 40.
        lands = [f"land{i}" for i in range(5)]
        spells = [f"s{i}" for i in range(23)]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, [("BLB", lands + spells)])
        locator = _make_locator(set(lands), set(spells))
        use_case = PickDecksUseCase()
        config = PickDecksConfig(
            pools_path=pools_path, label="picker", cards_path=tmp_path / "c",
            output=tmp_path / "decks.txt",
        )
        with _patch(use_case, locator):
            use_case.execute(config)
        _, _, names = config.output.read_text("utf-8").strip().split(";", 2)
        cards = names.split("|")
        assert len(cards) == 40
        picked_lands = [c for c in cards if c in lands]
        assert len(picked_lands) == 5

    def test_label_written_verbatim(self, tmp_path):
        spells = [f"s{i}" for i in range(30)]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, [("MH3", spells), ("BLB", spells)])
        locator = _make_locator(set(), set(spells))
        use_case = PickDecksUseCase()
        config = PickDecksConfig(
            pools_path=pools_path, label="picker-gen5", cards_path=tmp_path / "c",
            output=tmp_path / "decks.txt",
        )
        with _patch(use_case, locator):
            use_case.execute(config)
        for line in config.output.read_text("utf-8").strip().splitlines():
            assert line.split(";", 1)[0] == "picker-gen5"

    def test_pool_with_too_few_cards_skipped(self, tmp_path):
        small = [f"s{i}" for i in range(10)]
        big = [f"b{i}" for i in range(30)]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, [("MH3", small), ("BLB", big)])
        locator = _make_locator(set(), set(small) | set(big))
        use_case = PickDecksUseCase()
        config = PickDecksConfig(
            pools_path=pools_path, label="picker", cards_path=tmp_path / "c",
            output=tmp_path / "decks.txt",
        )
        with _patch(use_case, locator):
            written = use_case.execute(config)
        assert written == 1
        assert config.output.read_text("utf-8").strip().startswith("picker;BLB;")

    def test_resume_skips_existing_lines(self, tmp_path):
        all_cards = [f"c{i}" for i in range(90)]
        pools = [
            ("MH3", all_cards[:30]),
            ("BLB", all_cards[30:60]),
            ("RVR", all_cards[60:90]),
        ]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, pools)
        output = tmp_path / "decks.txt"
        output.write_text("picker;MH3;preexisting\n", encoding="utf-8")
        locator = _make_locator(set(), set(all_cards))
        use_case = PickDecksUseCase()
        config = PickDecksConfig(
            pools_path=pools_path, label="picker", cards_path=tmp_path / "c",
            output=output, resume=True,
        )
        with _patch(use_case, locator):
            written = use_case.execute(config)
        assert written == 2
        lines = output.read_text("utf-8").splitlines()
        assert len(lines) == 3
        assert lines[0] == "picker;MH3;preexisting"
        assert lines[1].startswith("picker;BLB;")
        assert lines[2].startswith("picker;RVR;")


class TestWidthMismatch:
    def test_picker_cache_width_mismatch_fails_fast(self, tmp_path):
        from sealed.domain.picker_model import PickerConfig, PickerModel
        from sealed.infrastructure.picker_store import PickerStore

        ckpt = tmp_path / "picker.pt"
        model = PickerModel(PickerConfig(embedding_dim=WIDTH, d_model=WIDTH, n_heads=2))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        PickerStore().save_checkpoint(
            model, opt, epoch=0, best_val_reward=0.0, config=model.config, path=ckpt,
        )

        spells = [f"s{i}" for i in range(30)]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, [("MH3", spells)])
        # Cache rows are 36-wide, but the checkpoint expects 40.
        locator = MagicMock()
        embeddings = {n: np.zeros(36, dtype=np.float32) for n in spells}
        locator.load_embedding.side_effect = lambda n: embeddings.get(n)
        use_case = PickDecksUseCase()
        config = PickDecksConfig(
            pools_path=pools_path, label="picker", cards_path=tmp_path / "c",
            picker_checkpoint=ckpt, output=tmp_path / "decks.txt",
        )
        with patch(
            "sealed.application.pick_decks.ConvertedCardLocator", return_value=locator,
        ), pytest.raises(ValueError, match="wide"):
            use_case.execute(config)
