"""The checks that read the cache rather than the model.

Also covers T089: ``--vocab-path`` and ``--keyword-definitions`` default to what
the checkpoint recorded. That has to be tested **by execution** — a default
resolved from a loaded checkpoint is invisible to a parser-only contract test,
which would pass while the command read the wrong vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from effects.application.evaluate_effect_model import (
    NEIGHBOUR_QUERIES,
    CheckStatus,
    EvaluateEffectModelConfig,
    load_shipping_cache,
)
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
from effects.domain.line_query import LineQuery, normalize_prose
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.domain.ward_twins import BARE_KEYWORDS, WARD, WARD_TWINS
from effects.infrastructure.effect_model_store import (
    SplitProvenance,
    content_hash,
    resolve_inference_paths,
)
from effects.infrastructure.sidecar_io import sidecar_path_for, write_sidecar


def _write_source(
    cards_root, abilities_root, card: str, stem: str, texts: list[str],
    vectors: list[list[float]], *, variant: str = "full",
    prose: list[str] | None = None, tree: str = "cardsfolder",
):
    """One converted source plus its cache rows, row-aligned.

    ``texts`` are the lines' script; ``prose``, when given, is written into the
    converted ``.txt`` beside the sidecar, one ``static:`` line per ability, so
    the two surfaces can differ the way they do on a real card.
    """
    script_file = f"{tree}/{stem}.txt"
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
    if prose is not None:
        (cards_root / f"{stem}.txt").write_text(
            "\n".join(f"static: {line}" for line in prose) + "\n",
            encoding="utf-8",
        )
    suffix = ".npz" if variant == "full" else f".{variant}.npz"
    target = abilities_root / tree / f"{stem}{suffix}"
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
        ["Flying", "Vigilance"], [[1.0, 0.0], [0.0, 1.0]],
        prose=["flying", "vigilance"],
    )
    _write_source(
        cards, abilities, "Shock", "s/shock",
        ["DealDamage | NumDmg$ 2"], [[0.7, 0.7]],
        prose=["CARDNAME deals 2 damage to any target."],
    )
    return cards, abilities


def _two_cards_one_prose(tmp_path):
    """Two cards printing one prose line that compiles to two scripts."""
    cards = tmp_path / "cardsfolder"
    (cards / "a").mkdir(parents=True)
    abilities = tmp_path / "abilities"
    prose = [TAX_PROSE]
    _write_source(
        cards, abilities, "one", "a/one", ["RaiseCost | Amount$ 2"],
        [[1.0, 0.0]], prose=prose,
    )
    _write_source(
        cards, abilities, "two", "a/two", ["RaiseCost | Amount$ 2 | Type$ Spell"],
        [[0.0, 1.0]], prose=prose,
    )
    return cards, abilities


TAX_PROSE = "spells your opponents cast that target CARDNAME cost {2} more to cast."


class TestLoadingTheCache:
    def test_rows_are_keyed_by_the_text_they_encode(self, cache):
        cards, abilities = cache
        loaded = load_cache(abilities, cards, surface="script")
        assert set(loaded.texts()) == {
            "Flying", "Vigilance", "DealDamage | NumDmg$ 2",
        }

    def test_the_prose_surface_keys_on_prose(self, cache):
        cards, abilities = cache
        loaded = load_cache(abilities, cards, surface="prose")
        assert "CARDNAME deals 2 damage to any target." in loaded.texts()

    def test_the_surface_is_never_defaulted(self, cache):
        """A default is how the evaluator came to key a script model on prose."""
        cards, abilities = cache
        with pytest.raises(TypeError):
            load_cache(abilities, cards)

    def test_one_prose_on_two_scripts_is_two_points_on_the_script_surface(
        self, tmp_path,
    ):
        """The model read two different texts; merging them hides one."""
        cards, abilities = _two_cards_one_prose(tmp_path)
        assert len(load_cache(abilities, cards, surface="script")) == 2
        assert len(load_cache(abilities, cards, surface="prose")) == 1

    def test_a_row_carries_the_vector_its_sidecar_line_points_at(self, cache):
        cards, abilities = cache
        loaded = load_cache(abilities, cards, surface="script")
        np.testing.assert_allclose(loaded.by_text["Flying"], [1.0, 0.0])
        np.testing.assert_allclose(loaded.by_text["Vigilance"], [0.0, 1.0])

    def test_a_reprinted_text_is_one_point_in_the_space(self, tmp_path):
        """Counting it twice would make the space look more populated."""
        cards = tmp_path / "cardsfolder"
        (cards / "a").mkdir(parents=True)
        abilities = tmp_path / "abilities"
        for stem in ("a/one", "a/two"):
            _write_source(cards, abilities, stem, stem, ["flying"], [[1.0, 0.0]])
        assert len(load_cache(abilities, cards, surface="script")) == 1

    def test_a_reprint_is_one_point_even_when_its_vectors_differ_in_float_noise(
        self, tmp_path,
    ):
        """One text encoded in two batches differs in the fifth decimal.

        Deduplicating by the vector's bytes keeps every such copy, so gate 3
        would read half again as many points as there are texts.
        """
        cards = tmp_path / "cardsfolder"
        (cards / "a").mkdir(parents=True)
        abilities = tmp_path / "abilities"
        _write_source(cards, abilities, "one", "a/one", ["Flying"], [[1.0, 0.0]])
        _write_source(
            cards, abilities, "two", "a/two", ["Flying"], [[1.0 + 1.5e-5, 0.0]],
        )
        assert len(load_cache(abilities, cards, surface="script")) == 1

    def test_pooled_card_vectors_are_mean_then_max(self, cache):
        cards, abilities = cache
        loaded = load_cache(abilities, cards, surface="script")
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
            assert len(load_cache(abilities, cards, surface="script")) == 0
        assert "cache rows" in caplog.text

    def test_an_absent_cache_loads_empty_rather_than_raising(self, tmp_path):
        assert len(
            load_cache(tmp_path / "nothing", tmp_path, surface="script"),
        ) == 0

    def test_a_variant_cache_loads_separately(self, cache):
        cards, abilities = cache
        _write_source(
            cards, abilities, "Serra Angel", "s/serra_angel",
            ["Flying", "Vigilance"], [[9.0, 9.0], [9.0, 9.0]],
            variant="no-state", prose=["flying", "vigilance"],
        )
        variant = load_cache(
            abilities, cards, variant="no-state", surface="script",
        )
        assert variant.by_text["Flying"].tolist() == [9.0, 9.0]
        np.testing.assert_allclose(
            load_cache(abilities, cards, surface="script").by_text["Flying"],
            [1.0, 0.0],
        )


class TestTheShippingCache:
    """Gate 3's population: one vector per unique text the shipping cache serves."""

    def _config(self, tmp_path):
        cards = tmp_path / "cardsfolder"
        tokens = tmp_path / "tokenscripts"
        variants = tmp_path / "variant-scripts"
        for folder in (cards / "s", tokens, variants):
            folder.mkdir(parents=True)
        abilities = tmp_path / "abilities"
        _write_source(
            cards, abilities, "Serra Angel", "s/serra_angel", ["Flying"],
            [[1.0, 0.0]], prose=["flying"],
        )
        _write_source(
            tokens, abilities, "Food", "c_a_food_sac", ["Sacrifice | GainLife$ 3"],
            [[0.0, 1.0]], prose=["{2}, {T}, sacrifice CARDNAME: you gain 3 life."],
            tree="tokenscripts",
        )
        _write_source(
            variants, abilities, "Shock (variant)", "shock_variant",
            ["DealDamage | NumDmg$ 5"], [[0.5, 0.5]], tree="variant-scripts",
        )
        return EvaluateEffectModelConfig(
            cards_folders=(cards, tokens), variant_scripts=variants,
            abilities_root=abilities,
        )

    def test_it_spans_the_card_and_token_trees(self, tmp_path):
        loaded = load_shipping_cache(self._config(tmp_path), surface="script")
        assert set(loaded.texts()) == {"Flying", "Sacrifice | GainLife$ 3"}

    def test_synthetic_variant_scripts_are_not_in_it(self, tmp_path):
        """A perturbed script is collection input, not a line the cache serves."""
        loaded = load_shipping_cache(self._config(tmp_path), surface="script")
        assert "DealDamage | NumDmg$ 5" not in loaded.texts()

    def test_a_token_is_not_a_card_for_the_pooled_vectors(self, tmp_path):
        loaded = load_shipping_cache(self._config(tmp_path), surface="script")
        assert set(loaded.by_card) == {"Serra Angel"}


class TestResolvingAQuery:
    def test_a_query_resolves_by_prose_to_the_script_the_model_read(self, cache):
        cards, abilities = cache
        loaded = load_cache(abilities, cards, surface="script")
        resolved = loaded.resolve(
            LineQuery("cardname deals 2 damage to any target."),
        )
        assert resolved.key == "DealDamage | NumDmg$ 2"

    def test_matching_ignores_case_and_whitespace_only(self, cache):
        cards, abilities = cache
        loaded = load_cache(abilities, cards, surface="script")
        assert loaded.resolve(LineQuery("  FLYING ")).key == "Flying"
        assert loaded.resolve(LineQuery("flying.")).key is None
        assert loaded.resolve(LineQuery("fly")).key is None

    def test_an_absent_line_says_so(self, cache):
        cards, abilities = cache
        resolved = load_cache(abilities, cards, surface="script").resolve(
            LineQuery("draw a card."),
        )
        assert resolved.key is None
        assert "no line" in resolved.problem

    def test_a_prose_on_several_scripts_is_ambiguous_without_a_card(
        self, tmp_path,
    ):
        cards, abilities = _two_cards_one_prose(tmp_path)
        loaded = load_cache(abilities, cards, surface="script")
        ambiguous = loaded.resolve(LineQuery(TAX_PROSE))
        assert ambiguous.key is None
        assert "2 scripts" in ambiguous.problem
        assert loaded.resolve(LineQuery(TAX_PROSE, card="Two")).key == (
            "RaiseCost | Amount$ 2 | Type$ Spell"
        )

    def test_a_card_that_does_not_print_the_line_does_not_resolve(self, cache):
        cards, abilities = cache
        resolved = load_cache(abilities, cards, surface="script").resolve(
            LineQuery("flying", card="shock"),
        )
        assert resolved.key is None
        assert "shock" in resolved.problem

    def test_a_line_is_displayed_in_prose(self, cache):
        cards, abilities = cache
        loaded = load_cache(abilities, cards, surface="script")
        assert loaded.label("DealDamage | NumDmg$ 2") == (
            "CARDNAME deals 2 damage to any target."
        )


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
        result = check_nearest_neighbours(
            CachedVectors(), (LineQuery("draw a card."),),
        )
        assert result.status is CheckStatus.SKIPPED

    def test_the_check_reports_and_blocks_nothing(self, cache):
        cards, abilities = cache
        result = check_nearest_neighbours(
            load_cache(abilities, cards, surface="script"),
            (LineQuery("flying"),),
        )
        assert result.status is CheckStatus.REPORTED
        assert not result.blocks

    def test_neighbours_are_listed_in_prose_for_the_reader(self, cache):
        cards, abilities = cache
        result = check_nearest_neighbours(
            load_cache(abilities, cards, surface="script"),
            (LineQuery("flying"),),
        )
        assert "CARDNAME deals 2 damage to any target." in result.detail
        assert "DealDamage" not in result.detail

    def test_a_query_that_resolves_to_nothing_is_named(self, cache):
        """A silently dropped query shrinks the comparison without saying so."""
        cards, abilities = cache
        result = check_nearest_neighbours(
            load_cache(abilities, cards, surface="script"),
            (LineQuery("flying"), LineQuery("draw a card.")),
        )
        assert "draw a card." in result.detail
        assert "unresolved" in result.detail
        assert result.values["unresolved_queries"] == 1

    def test_the_checked_in_queries_use_the_converters_spelling(self):
        for query in NEIGHBOUR_QUERIES:
            assert query.prose == query.prose.strip()
            assert query.prose.endswith(".")
            assert "this creature" not in query.prose


class TestUmap:
    def test_too_few_vectors_skips(self, cache):
        cards, abilities = cache
        assert check_umap(
            load_cache(abilities, cards, surface="script"),
        ).status is CheckStatus.SKIPPED

    def test_a_populated_cache_projects(self):
        rng = np.random.default_rng(0)
        cached = CachedVectors(by_text={
            f"text {i}": rng.normal(size=8) for i in range(30)
        })
        result = check_umap(cached)
        assert result.status in (CheckStatus.REPORTED, CheckStatus.SKIPPED)


def _ward_cache(tmp_path, *, missing: tuple[LineQuery, ...] = ()):
    """Ward, its twins and the bare keywords, as the converter spells them."""
    cards = tmp_path / "cardsfolder"
    (cards / "w").mkdir(parents=True)
    abilities = tmp_path / "abilities"
    rng = np.random.default_rng(1)
    _write_source(
        cards, abilities, "Aboleth Spawn", "w/aboleth_spawn", ["Ward:2"],
        [[1.0, 0.0, 0.0]], prose=[WARD.prose],
    )
    for number, twin in enumerate(WARD_TWINS):
        if twin in missing:
            continue
        _write_source(
            cards, abilities, twin.card or f"twin {number}", f"w/twin_{number}",
            [f"Trigger | Twin$ {number}"], [[0.99, 0.14, 0.0]],
            prose=[twin.prose],
        )
    for number, keyword in enumerate(BARE_KEYWORDS):
        _write_source(
            cards, abilities, f"bare {number}", f"w/bare_{number}",
            [keyword.prose.title()], [rng.normal(size=3).tolist()],
            prose=[keyword.prose],
        )
    return load_cache(abilities, cards, surface="script")


class TestWardCanary:
    def test_it_skips_when_ward_is_not_in_the_cache(self, cache):
        cards, abilities = cache
        result = check_ward(load_cache(abilities, cards, surface="script"))
        assert result.status is CheckStatus.SKIPPED
        assert "ward {2}" in result.detail

    def test_it_resolves_every_line_through_its_prose(self, tmp_path):
        result = check_ward(_ward_cache(tmp_path))
        assert result.status is CheckStatus.REPORTED
        assert "every functional twin" in result.detail
        assert result.values["twins_compared"] == len(WARD_TWINS)
        assert result.values["bare_keywords_compared"] == len(BARE_KEYWORDS)

    def test_a_twin_that_resolves_to_nothing_is_named(self, tmp_path):
        """Otherwise the canary compares against fewer twins than it lists."""
        absent = WARD_TWINS[-1]
        result = check_ward(_ward_cache(tmp_path, missing=(absent,)))
        assert result.values["twins_compared"] == len(WARD_TWINS) - 1
        assert "unresolved" in result.detail
        assert absent.prose in result.detail

    def test_the_ward_text_and_its_twins_are_checked_in(self):
        assert WARD.prose == "ward {2}"
        assert len(WARD_TWINS) >= 3
        assert len(BARE_KEYWORDS) >= 10
        assert "ward" not in " ".join(t.prose for t in WARD_TWINS).lower()

    def test_every_line_is_spelled_the_way_the_converter_renders_it(self):
        """Lowercase but ``CARDNAME``, no reminder text, a period on a sentence."""
        for twin in WARD_TWINS:
            bare = twin.prose.replace("CARDNAME", "")
            assert bare == bare.lower()
            assert "this creature" not in twin.prose
            assert twin.prose.endswith(".")
            assert twin.card
        for line in (WARD, *BARE_KEYWORDS):
            assert "(" not in line.prose
            assert not line.prose.endswith(".")

    @pytest.mark.skipif(
        not Path("output/cardsfolder").is_dir(),
        reason="needs the converted corpus",
    )
    def test_every_pinned_line_is_one_its_card_prints(self):
        """Twins and neighbour queries alike: a pin to a card that does not
        print the line resolves to nothing on the real corpus."""
        from effects.infrastructure.sidecar_io import (
            converted_text_path,
            prose_for,
            prose_lines,
            read_sidecar,
        )
        from sealed.infrastructure.converted_card_locator import (
            ConvertedCardLocator,
        )

        locator = ConvertedCardLocator(Path("output/cardsfolder"))
        for query in (*WARD_TWINS, *NEIGHBOUR_QUERIES):
            path = locator.expected_path(query.card, ".provenance.json")
            sidecar = read_sidecar(path)
            rendered = prose_lines(converted_text_path(path))
            printed = {
                normalize_prose(prose_for(line, rendered) or "")
                for line in sidecar.lines
            }
            assert normalize_prose(query.prose) in printed, query.card


class TestDecodability:
    def test_it_skips_on_an_empty_cache(self, tmp_path):
        result = check_decodability(CachedVectors(), {}, tmp_path / "wr.txt")
        assert result.status is CheckStatus.SKIPPED

    def test_it_skips_without_a_win_rate_table(self, cache):
        cards, abilities = cache
        result = check_decodability(
            load_cache(abilities, cards, surface="script"), {}, cards / "absent.txt",
        )
        assert result.status is CheckStatus.SKIPPED
        assert "win-rate table" in result.detail

    def test_it_skips_when_too_few_cards_overlap(self, cache, tmp_path):
        cards, abilities = cache
        win_rates = tmp_path / "cards-win-rates.txt"
        win_rates.write_text("card_name;a\n", encoding="utf-8")
        result = check_decodability(
            load_cache(abilities, cards, surface="script"), {}, win_rates,
        )
        assert result.status is CheckStatus.SKIPPED
        assert "too few" in result.detail


class TestVariantComparison:
    def test_it_skips_without_the_variant_cache(self, cache):
        cards, abilities = cache
        result = check_variant_geometry(
            load_cache(abilities, cards, surface="script"), CachedVectors(), "no-state",
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
