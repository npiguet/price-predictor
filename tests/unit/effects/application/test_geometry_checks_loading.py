"""Loading the ability cache, and the evaluation files' place beside the checkpoint."""


from __future__ import annotations

import numpy as np
import pytest

from effects.application.evaluation_context import (
    EvaluateEffectModelConfig,
)
from effects.domain.baselines import Baseline
from tests.unit.effects.application.geometry_checks_support import (
    SourceRows,
    _two_cards_one_prose,
    _write_source,
)
from tests.unit.effects.builders import read_cache


class TestLoadingTheCache:
    def test_rows_are_keyed_by_the_text_they_encode(self, cache):
        cards, abilities = cache
        loaded = read_cache(abilities, cards, surface="script")
        assert set(loaded.texts) == {
            "Flying", "Vigilance", "DealDamage | NumDmg$ 2",
        }

    def test_the_prose_surface_keys_on_prose(self, cache):
        cards, abilities = cache
        loaded = read_cache(abilities, cards, surface="prose")
        assert "CARDNAME deals 2 damage to any target." in loaded.texts

    def test_the_surface_is_never_defaulted(self, cache):
        """A default is how the evaluator came to key a script model on prose."""
        from effects.infrastructure.ability_cache_store import SidecarIndex

        cards, _ = cache
        with pytest.raises(TypeError):
            SidecarIndex(cards)

    def test_one_prose_on_two_scripts_is_two_points_on_the_script_surface(
        self, tmp_path,
    ):
        """The model read two different texts; merging them hides one."""
        cards, abilities = _two_cards_one_prose(tmp_path)
        assert len(read_cache(abilities, cards, surface="script")) == 2
        assert len(read_cache(abilities, cards, surface="prose")) == 1

    def test_a_row_carries_the_vector_its_sidecar_line_points_at(self, cache):
        cards, abilities = cache
        loaded = read_cache(abilities, cards, surface="script")
        np.testing.assert_allclose(loaded.by_text["Flying"], [1.0, 0.0])
        np.testing.assert_allclose(loaded.by_text["Vigilance"], [0.0, 1.0])

    def test_a_reprinted_text_is_one_point_in_the_space(self, tmp_path):
        """Counting it twice would make the space look more populated."""
        cards = tmp_path / "cardsfolder"
        (cards / "a").mkdir(parents=True)
        abilities = tmp_path / "abilities"
        for stem in ("a/one", "a/two"):
            _write_source(cards, abilities, SourceRows(stem, stem, ["flying"], [[1.0, 0.0]]))
        assert len(read_cache(abilities, cards, surface="script")) == 1

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
        _write_source(cards, abilities, SourceRows("one", "a/one", ["Flying"], [[1.0, 0.0]]))
        _write_source(
            cards, abilities,
            SourceRows("two", "a/two", ["Flying"], [[1.0 + 1.5e-5, 0.0]]),
        )
        assert len(read_cache(abilities, cards, surface="script")) == 1

    def test_pooled_card_vectors_are_mean_then_max(self, cache):
        cards, abilities = cache
        loaded = read_cache(abilities, cards, surface="script")
        np.testing.assert_allclose(
            loaded.by_card["Serra Angel"], [0.5, 0.5, 1.0, 1.0],
        )

    def test_a_misaligned_cache_is_skipped_with_a_warning(self, tmp_path, caplog):
        cards = tmp_path / "cardsfolder"
        (cards / "s").mkdir(parents=True)
        abilities = tmp_path / "abilities"
        _write_source(cards, abilities, SourceRows("Broken", "s/broken", ["a", "b"], [[1.0, 0.0]]))
        with caplog.at_level("WARNING"):
            assert len(read_cache(abilities, cards, surface="script")) == 0
        assert "cache rows" in caplog.text

    def test_an_absent_cache_loads_empty_rather_than_raising(self, tmp_path):
        assert len(
            read_cache(tmp_path / "nothing", tmp_path, surface="script"),
        ) == 0

    def test_a_variant_cache_loads_separately(self, cache):
        cards, abilities = cache
        _write_source(
            cards, abilities,
            SourceRows(
                "Serra Angel",
                "s/serra_angel",
                ["Flying", "Vigilance"],
                [[9.0, 9.0], [9.0, 9.0]],
                variant=Baseline.NO_STATE,
                prose=["flying", "vigilance"],
            ),
        )
        variant = read_cache(
            abilities, cards, variant=Baseline.NO_STATE, surface="script",
        )
        assert variant.by_text["Flying"].tolist() == [9.0, 9.0]
        np.testing.assert_allclose(
            read_cache(abilities, cards, surface="script").by_text["Flying"],
            [1.0, 0.0],
        )

    def test_every_variant_joins_the_sidecars_read_once(self, cache):
        """The evaluator reads the trees once for its three variants."""
        from effects.application.geometry_checks import cached_vectors
        from effects.infrastructure.ability_cache_store import SidecarIndex
        from effects.infrastructure.sidecar_io import sidecar_paths

        cards, abilities = cache
        _write_source(
            cards, abilities,
            SourceRows(
                "Serra Angel",
                "s/serra_angel",
                ["Flying", "Vigilance"],
                [[9.0, 9.0], [9.0, 9.0]],
                variant=Baseline.NO_STATE,
                prose=["flying", "vigilance"],
            ),
        )
        expected = {
            variant: read_cache(abilities, cards, surface="script", variant=variant)
            for variant in ("full", "no-state")
        }
        index = SidecarIndex(cards, surface="script")
        for path in sidecar_paths(cards):
            path.unlink()
        for variant, loaded in expected.items():
            joined = cached_vectors(index, abilities, variant=variant)
            assert joined.texts == loaded.texts
            np.testing.assert_array_equal(joined.matrix, loaded.matrix)
            assert joined.by_prose == loaded.by_prose
            assert set(joined.by_card) == set(loaded.by_card)


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
            cards, abilities,
            SourceRows("Serra Angel", "s/serra_angel", ["Flying"], [[1.0, 0.0]], prose=["flying"]),
        )
        _write_source(
            tokens, abilities,
            SourceRows(
                "Food",
                "c_a_food_sac",
                ["Sacrifice | GainLife$ 3"],
                [[0.0, 1.0]],
                prose=["{2}, {T}, sacrifice CARDNAME: you gain 3 life."],
                tree="tokenscripts",
            ),
        )
        _write_source(
            variants, abilities,
            SourceRows(
                "Shock (variant)",
                "shock_variant",
                ["DealDamage | NumDmg$ 5"],
                [[0.5, 0.5]],
                tree="variant-scripts",
            ),
        )
        return EvaluateEffectModelConfig(
            cards_folders=(cards, tokens), variant_scripts=variants,
            abilities_root=abilities,
        )

    def _load(self, tmp_path):
        config = self._config(tmp_path)
        return read_cache(
            config.abilities_root, config.cards_folders, surface="script",
        )

    def test_it_spans_the_card_and_token_trees(self, tmp_path):
        loaded = self._load(tmp_path)
        assert set(loaded.texts) == {"Flying", "Sacrifice | GainLife$ 3"}

    def test_synthetic_variant_scripts_are_not_in_it(self, tmp_path):
        """A perturbed script is collection input, not a line the cache serves."""
        loaded = self._load(tmp_path)
        assert "DealDamage | NumDmg$ 5" not in loaded.texts

    def test_a_token_is_not_a_card_for_the_pooled_vectors(self, tmp_path):
        loaded = self._load(tmp_path)
        assert set(loaded.by_card) == {"Serra Angel"}


def test_the_evaluation_files_go_beside_the_checkpoint_named_for_it(tmp_path):
    from effects.application.evaluation_context import EvaluateEffectModelConfig

    config = EvaluateEffectModelConfig(checkpoint=tmp_path / "arm" / "latest.pt")
    assert config.output_dir == tmp_path / "arm" / "latest.evaluation"
