"""Unit tests for BuildDecksUseCase."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from sealed.application.build_decks import BuildDecksConfig, BuildDecksUseCase
from sealed.domain.card_embedding_layout import FEATURE_COUNT, IS_LAND, total_dim
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer

D_MODEL = total_dim(256)
_IS_LAND_OFFSET = D_MODEL - FEATURE_COUNT + IS_LAND


def _make_model() -> SetTransformerScorer:
    model = SetTransformerScorer(ScorerConfig(
        n_layers=1, n_heads=4, n_seeds=4, d_ff=544, mlp_hidden=64,
    ))
    model.eval()
    return model


def _write_pools(path: Path, pools: list[tuple[str, list[str]]]) -> None:
    path.write_text(
        "\n".join(f"{set_code};" + "|".join(names) for set_code, names in pools)
        + "\n",
        encoding="utf-8",
    )


def _make_locator(known_cards: set[str]) -> MagicMock:
    """Locator that returns deterministic embeddings for known cards, None otherwise.

    `load_text` returns a stub with `mana_cost_line()` returning `None`,
    which `compute_basic_lands` accepts (it falls back to even WUBRG split).
    """
    locator = MagicMock()
    rng = np.random.default_rng(7)
    embeddings: dict[str, np.ndarray] = {}
    for n in known_cards:
        emb = rng.standard_normal(D_MODEL).astype(np.float32)
        emb[_IS_LAND_OFFSET] = 0.0  # synthetic spell — no land flag set
        embeddings[n] = emb

    def load_embedding(name: str):
        return embeddings.get(name)

    def load_text(name: str):
        if name not in known_cards:
            return None
        text = MagicMock()
        text.mana_cost_line.return_value = None
        return text

    locator.load_embedding.side_effect = load_embedding
    locator.load_text.side_effect = load_text
    return locator


def _patch_use_case(model, locator):
    """Patch the loader and locator constructor used by BuildDecksUseCase."""
    use_case = BuildDecksUseCase()
    use_case._load_model = MagicMock(return_value=model)
    return use_case, patch(
        "sealed.application.build_decks.ConvertedCardLocator",
        return_value=locator,
    )


class TestBuildDecksUseCase:
    def test_writes_one_line_per_pool_with_set_code_prefix(self, tmp_path):
        pool_names = [f"card_{i}" for i in range(30)]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, [("MH3", pool_names), ("BLB", pool_names)])

        model = _make_model()
        locator = _make_locator(set(pool_names))
        use_case, locator_patch = _patch_use_case(model, locator)

        config = BuildDecksConfig(
            pools_path=pools_path,
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=tmp_path / "decks.txt",
        )

        with locator_patch:
            written = use_case.execute(config)

        assert written == 2
        lines = config.output.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("MH3;")
        assert lines[1].startswith("BLB;")

    def test_each_deck_has_40_cards(self, tmp_path):
        pool_names = [f"card_{i}" for i in range(30)]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, [("MH3", pool_names)])

        model = _make_model()
        locator = _make_locator(set(pool_names))
        use_case, locator_patch = _patch_use_case(model, locator)

        config = BuildDecksConfig(
            pools_path=pools_path,
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=tmp_path / "decks.txt",
        )

        with locator_patch:
            use_case.execute(config)

        line = config.output.read_text(encoding="utf-8").strip()
        _, _, names_field = line.partition(";")
        assert len(names_field.split("|")) == 40

    def test_pool_with_too_few_embeddable_cards_is_skipped(self, tmp_path):
        small_pool = [f"card_{i}" for i in range(10)]  # < 23
        big_pool = [f"big_{i}" for i in range(30)]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, [("MH3", small_pool), ("BLB", big_pool)])

        model = _make_model()
        locator = _make_locator(set(small_pool) | set(big_pool))
        use_case, locator_patch = _patch_use_case(model, locator)

        config = BuildDecksConfig(
            pools_path=pools_path,
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=tmp_path / "decks.txt",
        )

        with locator_patch:
            written = use_case.execute(config)

        assert written == 1
        lines = config.output.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("BLB;")

    def test_pool_with_only_unknown_cards_is_skipped(self, tmp_path):
        """Cards not present in the locator are filtered out."""
        pool_names = [f"missing_{i}" for i in range(30)]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, [("MH3", pool_names)])

        model = _make_model()
        locator = _make_locator(set())  # nothing known
        use_case, locator_patch = _patch_use_case(model, locator)

        config = BuildDecksConfig(
            pools_path=pools_path,
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=tmp_path / "decks.txt",
        )

        with locator_patch:
            written = use_case.execute(config)

        assert written == 0
        assert config.output.read_text(encoding="utf-8") == ""

    def test_creates_output_parent_directory(self, tmp_path):
        pool_names = [f"card_{i}" for i in range(30)]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, [("MH3", pool_names)])

        model = _make_model()
        locator = _make_locator(set(pool_names))
        use_case, locator_patch = _patch_use_case(model, locator)

        config = BuildDecksConfig(
            pools_path=pools_path,
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=tmp_path / "nested" / "out" / "decks.txt",
        )

        with locator_patch:
            use_case.execute(config)

        assert config.output.exists()


class TestGeneratedDecksFixture:
    """Verify the shared fixture round-trips correctly through parsing."""

    def test_fixture_yields_three_decks(self, synthetic_generated_decks_file):
        from sealed.infrastructure.pool_file_reader import parse_pools

        rows = parse_pools(synthetic_generated_decks_file)
        assert len(rows) == 3
        for set_code, deck in rows:
            assert set_code in {"MH3", "BLB"}
            assert len(deck) == 40
