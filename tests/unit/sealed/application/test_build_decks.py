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
            label="gen-test",
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=tmp_path / "decks.txt",
        )

        with locator_patch:
            written = use_case.execute(config)

        assert written == 2
        lines = config.output.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("gen-test;MH3;")
        assert lines[1].startswith("gen-test;BLB;")

    def test_each_deck_has_40_cards(self, tmp_path):
        pool_names = [f"card_{i}" for i in range(30)]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, [("MH3", pool_names)])

        model = _make_model()
        locator = _make_locator(set(pool_names))
        use_case, locator_patch = _patch_use_case(model, locator)

        config = BuildDecksConfig(
            pools_path=pools_path,
            label="gen-test",
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=tmp_path / "decks.txt",
        )

        with locator_patch:
            use_case.execute(config)

        line = config.output.read_text(encoding="utf-8").strip()
        label, set_code, names_field = line.split(";", 2)
        assert label == "gen-test"
        assert set_code == "MH3"
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
            label="gen-test",
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=tmp_path / "decks.txt",
        )

        with locator_patch:
            written = use_case.execute(config)

        assert written == 1
        lines = config.output.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("gen-test;BLB;")

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
            label="gen-test",
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=tmp_path / "decks.txt",
        )

        with locator_patch:
            written = use_case.execute(config)

        assert written == 0
        assert config.output.read_text(encoding="utf-8") == ""

    def test_label_appears_as_first_column_for_every_deck(self, tmp_path):
        pool_names = [f"card_{i}" for i in range(30)]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, [("MH3", pool_names), ("BLB", pool_names)])

        model = _make_model()
        locator = _make_locator(set(pool_names))
        use_case, locator_patch = _patch_use_case(model, locator)

        config = BuildDecksConfig(
            pools_path=pools_path,
            label="gen-3-experimental",
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=tmp_path / "decks.txt",
        )

        with locator_patch:
            use_case.execute(config)

        for line in config.output.read_text(encoding="utf-8").strip().splitlines():
            assert line.split(";", 1)[0] == "gen-3-experimental"

    def test_creates_output_parent_directory(self, tmp_path):
        pool_names = [f"card_{i}" for i in range(30)]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, [("MH3", pool_names)])

        model = _make_model()
        locator = _make_locator(set(pool_names))
        use_case, locator_patch = _patch_use_case(model, locator)

        config = BuildDecksConfig(
            pools_path=pools_path,
            label="gen-test",
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=tmp_path / "nested" / "out" / "decks.txt",
        )

        with locator_patch:
            use_case.execute(config)

        assert config.output.exists()


class TestResume:
    """``--resume`` controls whether the output file is truncated (default)
    or appended to with the first ``N`` pools skipped (where ``N`` is the
    count of complete lines already in the output file)."""

    def _three_pool_setup(self, tmp_path: Path) -> tuple[Path, set[str]]:
        """Build a 3-pool input file with disjoint card lists."""
        all_cards = [f"card_{i}" for i in range(90)]
        pools = [
            ("MH3", all_cards[:30]),
            ("BLB", all_cards[30:60]),
            ("RVR", all_cards[60:90]),
        ]
        pools_path = tmp_path / "pools.txt"
        _write_pools(pools_path, pools)
        return pools_path, set(all_cards)

    def test_resume_with_one_existing_line_skips_first_pool(self, tmp_path):
        pools_path, known = self._three_pool_setup(tmp_path)
        output = tmp_path / "decks.txt"
        existing = "gen-test;MH3;preexisting_card_0|preexisting_card_1\n"
        output.write_text(existing, encoding="utf-8")

        model = _make_model()
        locator = _make_locator(known)
        use_case, locator_patch = _patch_use_case(model, locator)

        config = BuildDecksConfig(
            pools_path=pools_path,
            label="gen-test",
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=output,
            resume=True,
        )

        with locator_patch:
            written = use_case.execute(config)

        assert written == 2  # only the last 2 pools are processed
        lines = output.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert lines[0] == existing.rstrip("\n")  # pre-existing line untouched
        assert lines[1].startswith("gen-test;BLB;")
        assert lines[2].startswith("gen-test;RVR;")

    def test_resume_with_empty_output_starts_from_beginning(self, tmp_path):
        pools_path, known = self._three_pool_setup(tmp_path)
        output = tmp_path / "decks.txt"
        output.write_text("", encoding="utf-8")

        model = _make_model()
        locator = _make_locator(known)
        use_case, locator_patch = _patch_use_case(model, locator)

        config = BuildDecksConfig(
            pools_path=pools_path,
            label="gen-test",
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=output,
            resume=True,
        )

        with locator_patch:
            written = use_case.execute(config)

        assert written == 3
        lines = output.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("gen-test;MH3;")

    def test_resume_with_missing_output_starts_from_beginning(self, tmp_path):
        pools_path, known = self._three_pool_setup(tmp_path)
        output = tmp_path / "decks.txt"  # does not exist

        model = _make_model()
        locator = _make_locator(known)
        use_case, locator_patch = _patch_use_case(model, locator)

        config = BuildDecksConfig(
            pools_path=pools_path,
            label="gen-test",
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=output,
            resume=True,
        )

        with locator_patch:
            written = use_case.execute(config)

        assert written == 3
        assert output.exists()

    def test_resume_truncates_partial_last_line(self, tmp_path):
        pools_path, known = self._three_pool_setup(tmp_path)
        output = tmp_path / "decks.txt"
        # One complete line + a partial line with no trailing newline
        # (simulating a process killed between two of the per-deck
        # out.write() calls).
        output.write_text(
            "gen-test;MH3;complete_line\n"
            "gen-test;BLB;partial_line_no_newline",
            encoding="utf-8",
        )

        model = _make_model()
        locator = _make_locator(known)
        use_case, locator_patch = _patch_use_case(model, locator)

        config = BuildDecksConfig(
            pools_path=pools_path,
            label="gen-test",
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=output,
            resume=True,
        )

        with locator_patch:
            use_case.execute(config)

        text = output.read_text(encoding="utf-8")
        # Skip count was 1 (only one complete line). The partial line is
        # truncated, then the last 2 pools are appended.
        assert "partial_line_no_newline" not in text
        lines = text.splitlines()
        assert len(lines) == 3
        assert lines[0] == "gen-test;MH3;complete_line"
        assert lines[1].startswith("gen-test;BLB;")
        assert lines[2].startswith("gen-test;RVR;")

    def test_default_truncates_existing_file(self, tmp_path):
        """Without ``--resume``, the output file is overwritten — regression
        guard for the default behavior."""
        pools_path, known = self._three_pool_setup(tmp_path)
        output = tmp_path / "decks.txt"
        output.write_text(
            "old-label;FOO;old_card_1|old_card_2\n", encoding="utf-8",
        )

        model = _make_model()
        locator = _make_locator(known)
        use_case, locator_patch = _patch_use_case(model, locator)

        config = BuildDecksConfig(
            pools_path=pools_path,
            label="gen-test",
            checkpoint=tmp_path / "fake.pt",
            cards_path=tmp_path / "cards",
            output=output,
            # resume defaults to False
        )

        with locator_patch:
            written = use_case.execute(config)

        assert written == 3
        text = output.read_text(encoding="utf-8")
        assert "old-label" not in text
        assert "old_card_1" not in text
        assert text.count("\n") == 3


class TestGeneratedDecksFixture:
    """Verify the shared fixture round-trips correctly through parsing."""

    def test_fixture_yields_three_decks(self, synthetic_generated_decks_file):
        from sealed.infrastructure.pool_file_reader import parse_generated_decks

        decks = parse_generated_decks(synthetic_generated_decks_file)
        assert len(decks) == 3
        for deck in decks:
            assert deck.label == "gen-test"
            assert deck.set_code in {"MH3", "BLB"}
            assert len(deck.cards) == 40
