"""The checks that read the cache rather than the model.

Also covers T089: ``--vocab-path`` and ``--keyword-definitions`` default to what
the checkpoint recorded. That has to be tested **by execution** — a default
resolved from a loaded checkpoint is invisible to a parser-only contract test,
which would pass while the command read the wrong vocabulary.
"""

from __future__ import annotations

import numpy as np
import pytest

from effects.application.evaluate_effect_model import CheckStatus
from effects.application.geometry_checks import (
    CachedVectors,
    check_decodability,
    check_nearest_neighbours,
    check_umap,
    check_variant_geometry,
    check_ward,
    load_cache,
    nearest_neighbours,
    write_scorer_smoke_cache,
)
from effects.domain.ability_cache_layout import ARRAY_KEY
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.domain.ward_twins import BARE_KEYWORDS, WARD_TEXT, WARD_TWINS
from effects.infrastructure.effect_model_store import (
    SplitProvenance,
    content_hash,
    resolve_inference_paths,
)
from effects.infrastructure.sidecar_io import sidecar_path_for, write_sidecar


def _write_source(
    cards_root, abilities_root, card: str, stem: str, texts: list[str],
    vectors: list[list[float]], *, variant: str = "full",
):
    """One converted source plus its cache rows, row-aligned."""
    script_file = f"cardsfolder/{stem}.txt"
    write_sidecar(
        ProvenanceSidecar(
            card=card,
            script_file=script_file,
            lines=tuple(
                SidecarLine(
                    line_index=i, line_kind="static",
                    provenance=(ProvenanceKey(script_file, 0, "static", i),),
                    script_text=text,
                )
                for i, text in enumerate(texts)
            ),
        ),
        sidecar_path_for(cards_root / f"{stem}.txt"),
    )
    suffix = ".npz" if variant == "full" else f".{variant}.npz"
    target = abilities_root / "cardsfolder" / f"{stem}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, **{ARRAY_KEY: np.array(vectors, dtype=np.float32)},
    )


@pytest.fixture
def cache(tmp_path):
    cards = tmp_path / "cardsfolder"
    (cards / "s").mkdir(parents=True)
    abilities = tmp_path / "abilities"
    _write_source(
        cards, abilities, "Serra Angel", "s/serra_angel",
        ["flying", "vigilance"], [[1.0, 0.0], [0.0, 1.0]],
    )
    _write_source(
        cards, abilities, "Shock", "s/shock",
        ["deal 2 damage to any target"], [[0.7, 0.7]],
    )
    return cards, abilities


class TestLoadingTheCache:
    def test_rows_are_keyed_by_the_text_they_encode(self, cache):
        cards, abilities = cache
        loaded = load_cache(abilities, cards)
        assert set(loaded.texts()) == {
            "flying", "vigilance", "deal 2 damage to any target",
        }

    def test_a_row_carries_the_vector_its_sidecar_line_points_at(self, cache):
        cards, abilities = cache
        loaded = load_cache(abilities, cards)
        np.testing.assert_allclose(loaded.by_text["flying"], [1.0, 0.0])
        np.testing.assert_allclose(loaded.by_text["vigilance"], [0.0, 1.0])

    def test_a_reprinted_text_is_one_point_in_the_space(self, tmp_path):
        """Counting it twice would make the space look more populated."""
        cards = tmp_path / "cardsfolder"
        (cards / "a").mkdir(parents=True)
        abilities = tmp_path / "abilities"
        for stem in ("a/one", "a/two"):
            _write_source(cards, abilities, stem, stem, ["flying"], [[1.0, 0.0]])
        assert len(load_cache(abilities, cards)) == 1

    def test_pooled_card_vectors_are_mean_then_max(self, cache):
        cards, abilities = cache
        loaded = load_cache(abilities, cards)
        np.testing.assert_allclose(
            loaded.by_card["Serra Angel"], [0.5, 0.5, 1.0, 1.0],
        )

    def test_a_misaligned_cache_is_skipped_with_a_warning(self, tmp_path, caplog):
        cards = tmp_path / "cardsfolder"
        (cards / "s").mkdir(parents=True)
        abilities = tmp_path / "abilities"
        _write_source(
            cards, abilities, "Broken", "s/broken", ["a", "b"], [[1.0, 0.0]],
        )
        with caplog.at_level("WARNING"):
            assert len(load_cache(abilities, cards)) == 0
        assert "cache rows" in caplog.text

    def test_an_absent_cache_loads_empty_rather_than_raising(self, tmp_path):
        assert len(load_cache(tmp_path / "nothing", tmp_path)) == 0

    def test_a_variant_cache_loads_separately(self, cache):
        cards, abilities = cache
        _write_source(
            cards, abilities, "Serra Angel", "s/serra_angel",
            ["flying", "vigilance"], [[9.0, 9.0], [9.0, 9.0]],
            variant="no-state",
        )
        assert load_cache(abilities, cards, variant="no-state").by_text[
            "flying"
        ].tolist() == [9.0, 9.0]
        np.testing.assert_allclose(load_cache(abilities, cards).by_text["flying"],
                                   [1.0, 0.0])


class TestNearestNeighbours:
    def test_the_nearest_neighbour_is_the_most_similar_text(self):
        cached = CachedVectors(by_text={
            "a": np.array([1.0, 0.0]),
            "close": np.array([0.99, 0.14]),
            "far": np.array([0.0, 1.0]),
        })
        assert nearest_neighbours(cached, "a", k=1)[0][0] == "close"

    def test_the_query_is_not_its_own_neighbour(self):
        cached = CachedVectors(by_text={
            "a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0]),
        })
        assert [t for t, _ in nearest_neighbours(cached, "a")] == ["b"]

    def test_an_unknown_query_has_no_neighbours(self):
        cached = CachedVectors(by_text={"a": np.array([1.0, 0.0])})
        assert nearest_neighbours(cached, "never encoded") == []

    def test_the_check_skips_on_an_empty_cache(self):
        result = check_nearest_neighbours(CachedVectors(), ("draw a card",))
        assert result.status is CheckStatus.SKIPPED

    def test_the_check_reports_and_blocks_nothing(self, cache):
        cards, abilities = cache
        result = check_nearest_neighbours(
            load_cache(abilities, cards), ("flying",),
        )
        assert result.status is CheckStatus.REPORTED
        assert not result.blocks


class TestUmap:
    def test_too_few_vectors_skips(self, cache):
        cards, abilities = cache
        assert check_umap(load_cache(abilities, cards)).status is (
            CheckStatus.SKIPPED
        )

    def test_a_populated_cache_projects(self):
        rng = np.random.default_rng(0)
        cached = CachedVectors(by_text={
            f"text {i}": rng.normal(size=8) for i in range(30)
        })
        result = check_umap(cached)
        assert result.status in (CheckStatus.REPORTED, CheckStatus.SKIPPED)


class TestWardCanary:
    def test_it_skips_when_ward_is_not_in_the_cache(self, cache):
        cards, abilities = cache
        assert check_ward(load_cache(abilities, cards)).status is (
            CheckStatus.SKIPPED
        )

    def test_it_matches_ward_by_a_normalized_prefix(self):
        """Converted text differs from the checked-in wording in whitespace."""
        rng = np.random.default_rng(1)
        by_text = {
            "  WARD {2}  (Whenever this permanent becomes the target of a "
            "spell or ability an opponent controls, counter it unless that "
            "player pays {2}.)": np.array([1.0, 0.0, 0.0]),
        }
        for twin in WARD_TWINS:
            by_text[twin] = np.array([0.99, 0.14, 0.0])
        for keyword in BARE_KEYWORDS:
            by_text[keyword] = rng.normal(size=3)
        result = check_ward(CachedVectors(by_text=by_text))
        assert result.status is CheckStatus.REPORTED
        assert "functional twin" in result.detail

    def test_the_ward_text_and_its_twins_are_checked_in(self):
        assert WARD_TEXT
        assert len(WARD_TWINS) >= 3
        assert len(BARE_KEYWORDS) >= 10
        assert "ward" not in " ".join(WARD_TWINS).lower()


class TestDecodability:
    def test_it_skips_on_an_empty_cache(self, tmp_path):
        result = check_decodability(CachedVectors(), {}, tmp_path / "wr.txt")
        assert result.status is CheckStatus.SKIPPED

    def test_it_skips_without_a_win_rate_table(self, cache):
        cards, abilities = cache
        result = check_decodability(
            load_cache(abilities, cards), {}, cards / "absent.txt",
        )
        assert result.status is CheckStatus.SKIPPED
        assert "win-rate table" in result.detail

    def test_it_skips_when_too_few_cards_overlap(self, cache, tmp_path):
        cards, abilities = cache
        win_rates = tmp_path / "cards-win-rates.txt"
        win_rates.write_text("card_name;a\n", encoding="utf-8")
        result = check_decodability(
            load_cache(abilities, cards), {}, win_rates,
        )
        assert result.status is CheckStatus.SKIPPED
        assert "too few" in result.detail


class TestVariantComparison:
    def test_it_skips_without_the_variant_cache(self, cache):
        cards, abilities = cache
        result = check_variant_geometry(
            load_cache(abilities, cards), CachedVectors(), "no-state",
        )
        assert result.status is CheckStatus.SKIPPED
        assert "--variant no-state" in result.detail

    def test_it_compares_the_two_geometries(self):
        rng = np.random.default_rng(2)
        full = CachedVectors(by_text={
            f"t{i}": rng.normal(size=8) for i in range(50)
        })
        collapsed = CachedVectors(by_text={
            f"t{i}": np.ones(8) + 0.001 * rng.normal(size=8) for i in range(50)
        })
        result = check_variant_geometry(full, collapsed, "no-state")
        assert result.status is CheckStatus.REPORTED
        assert result.values["full_cosine"] < result.values["no-state_cosine"]


class TestScorerSmokeCache:
    def test_it_concatenates_rather_than_replacing(self, tmp_path):
        """The test asks "does this add anything", not "is this better than
        nothing"."""
        from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

        cached = CachedVectors(by_card={"Shock": np.array([1.0, 2.0], np.float32)})
        sealed = {"Shock": np.array([9.0, 9.0, 9.0], np.float32)}
        scratch = tmp_path / "scratch"
        locator = ConvertedCardLocator(tmp_path / "cardsfolder")
        assert write_scorer_smoke_cache(cached, sealed, scratch, locator) == 1
        written = next(scratch.rglob("*.npz"))
        with np.load(written) as data:
            combined = data["embedding"]
        np.testing.assert_allclose(combined, [9.0, 9.0, 9.0, 1.0, 2.0])

    def test_it_writes_only_into_the_scratch_tree(self, tmp_path):
        """Never output/cardsfolder/: the sealed pipeline reads that tree."""
        from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

        real = tmp_path / "cardsfolder"
        real.mkdir()
        scratch = tmp_path / "scratch"
        write_scorer_smoke_cache(
            CachedVectors(by_card={"Shock": np.array([1.0], np.float32)}),
            {"Shock": np.array([9.0], np.float32)},
            scratch, ConvertedCardLocator(real),
        )
        assert list(real.rglob("*.npz")) == []
        assert list(scratch.rglob("*.npz"))

    def test_a_card_the_sealed_cache_lacks_is_skipped(self, tmp_path):
        from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

        assert write_scorer_smoke_cache(
            CachedVectors(by_card={"Shock": np.array([1.0], np.float32)}),
            {}, tmp_path / "scratch", ConvertedCardLocator(tmp_path / "cards"),
        ) == 0


class TestInferencePathsResolveFromTheCheckpoint:
    """T089: FR-097 tested by execution, not by parser inspection."""

    def _provenance(self, tmp_path):
        vocab = tmp_path / "recorded-vocab.txt"
        vocab.write_text("[PAD]\n[UNK]\n", encoding="utf-8")
        keywords = tmp_path / "recorded-kw.json"
        keywords.write_text("{}", encoding="utf-8")
        return SplitProvenance(
            vocab_path=str(vocab), keyword_definitions_path=str(keywords),
            vocab_hash=content_hash(vocab),
            keyword_definitions_hash=content_hash(keywords),
        ), vocab, keywords

    def test_absent_flags_resolve_to_the_recorded_paths(self, tmp_path):
        provenance, vocab, keywords = self._provenance(tmp_path)
        resolved = resolve_inference_paths(
            provenance, vocab_path=None, keyword_path=None,
        )
        assert resolved == (vocab, keywords)

    def test_the_resolved_paths_pass_the_hash_check(self, tmp_path):
        provenance, _vocab, _keywords = self._provenance(tmp_path)
        vocab_path, keyword_path = resolve_inference_paths(
            provenance, vocab_path=None, keyword_path=None,
        )
        provenance.verify_hashes(
            vocab_path=vocab_path, keyword_path=keyword_path,
        )

    def test_an_explicit_flag_overrides_the_recorded_path(self, tmp_path):
        provenance, _vocab, keywords = self._provenance(tmp_path)
        chosen = tmp_path / "chosen.txt"
        chosen.write_text("[PAD]\n[UNK]\n", encoding="utf-8")
        vocab_path, keyword_path = resolve_inference_paths(
            provenance, vocab_path=chosen, keyword_path=None,
        )
        assert vocab_path == chosen
        assert keyword_path == keywords

    def test_an_override_to_a_different_file_still_fails_the_hash_check(
        self, tmp_path,
    ):
        provenance, _vocab, _keywords = self._provenance(tmp_path)
        other = tmp_path / "other.txt"
        other.write_text("[PAD]\n[UNK]\ndifferent\n", encoding="utf-8")
        from effects.infrastructure.effect_model_store import HashMismatchError

        vocab_path, keyword_path = resolve_inference_paths(
            provenance, vocab_path=other, keyword_path=None,
        )
        with pytest.raises(HashMismatchError):
            provenance.verify_hashes(
                vocab_path=vocab_path, keyword_path=keyword_path,
            )
