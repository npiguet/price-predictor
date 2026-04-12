"""Tests for feature engineering: Card → numeric feature vector."""

import numpy as np
import pytest

from price_predictor.application.feature_engineering import FeatureEngineering
from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import ManaCost, PrintingData, RECOGNIZED_FORMATS


@pytest.fixture
def cards_for_fitting() -> list[Card]:
    """A diverse set of cards to fit the feature engineering on."""
    return [
        Card(
            name="Grizzly Bears",
            types=["Creature"],
            subtypes=["Bear"],
            mana_cost=ManaCost.parse("1 G"),
            power="2",
            toughness="2",
            keywords=[],
            oracle_text="",
        ),
        Card(
            name="Lightning Bolt",
            types=["Instant"],
            mana_cost=ManaCost.parse("R"),
            oracle_text="Lightning Bolt deals 3 damage to any target.",
            ability_count=1,
        ),
        Card(
            name="Serra Angel",
            types=["Creature"],
            subtypes=["Angel"],
            mana_cost=ManaCost.parse("3 W W"),
            power="4",
            toughness="4",
            keywords=["Flying", "Vigilance"],
            oracle_text="Flying, vigilance",
        ),
        Card(
            name="Jace, the Mind Sculptor",
            types=["Planeswalker"],
            supertypes=["Legendary"],
            subtypes=["Jace"],
            mana_cost=ManaCost.parse("2 U U"),
            loyalty="3",
            oracle_text="Look at the top card. Draw three cards.",
            ability_count=4,
        ),
        Card(
            name="Island",
            types=["Land"],
            supertypes=["Basic"],
            subtypes=["Island"],
            mana_cost=None,
            oracle_text="{T}: Add {U}.",
            ability_count=1,
        ),
        Card(
            name="Sol Ring",
            types=["Artifact"],
            mana_cost=ManaCost.parse("1"),
            oracle_text="{T}: Add {C}{C}.",
            ability_count=1,
        ),
    ]


@pytest.fixture
def fitted_fe(cards_for_fitting: list[Card]) -> FeatureEngineering:
    fe = FeatureEngineering(random_seed=42)
    fe.fit(cards_for_fitting)
    return fe


class TestFeatureEngineeringFit:
    def test_fit_returns_self(self, cards_for_fitting: list[Card]) -> None:
        fe = FeatureEngineering()
        result = fe.fit(cards_for_fitting)
        assert result is fe

    def test_fit_learns_keywords(self, fitted_fe: FeatureEngineering) -> None:
        assert "Flying" in fitted_fe.top_keywords_
        assert "Vigilance" in fitted_fe.top_keywords_

    def test_fit_learns_tfidf(self, fitted_fe: FeatureEngineering) -> None:
        assert fitted_fe.tfidf_.vocabulary_ is not None
        assert len(fitted_fe.tfidf_.vocabulary_) > 0

    def test_transform_before_fit_raises(self) -> None:
        from sklearn.exceptions import NotFittedError
        fe = FeatureEngineering()
        card = Card(name="Test", types=["Creature"], power="1", toughness="1")
        with pytest.raises(NotFittedError):
            fe.transform([card])


class TestFeatureEngineeringTransform:
    def test_output_shape(
        self, fitted_fe: FeatureEngineering, cards_for_fitting: list[Card]
    ) -> None:
        result = fitted_fe.transform(cards_for_fitting)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(cards_for_fitting)
        assert result.shape[1] > 0

    def test_all_cards_same_feature_count(
        self, fitted_fe: FeatureEngineering, cards_for_fitting: list[Card]
    ) -> None:
        result = fitted_fe.transform(cards_for_fitting)
        assert result.shape[1] == fitted_fe.get_feature_count()

    def test_creature_has_mana_value(self, fitted_fe: FeatureEngineering) -> None:
        card = Card(
            name="Test Creature",
            types=["Creature"],
            mana_cost=ManaCost.parse("2 W W"),
            power="3",
            toughness="3",
        )
        result = fitted_fe.transform([card])
        # Feature 0: has_mana_cost = 1.0
        assert result[0, 0] == 1.0
        # Feature 1: total_mana_value = 4 (2 + W + W)
        assert result[0, 1] == 4.0

    def test_color_encoding(self, fitted_fe: FeatureEngineering) -> None:
        card = Card(
            name="Multicolor",
            types=["Creature"],
            mana_cost=ManaCost.parse("1 U R"),
            power="2",
            toughness="2",
        )
        result = fitted_fe.transform([card])
        # Features [2..6] are W, U, B, R, G
        assert result[0, 2] == 0.0  # W
        assert result[0, 3] == 1.0  # U
        assert result[0, 4] == 0.0  # B
        assert result[0, 5] == 1.0  # R
        assert result[0, 6] == 0.0  # G
        # Feature [7] is color_count
        assert result[0, 7] == 2.0

    def test_land_no_mana_cost(self, fitted_fe: FeatureEngineering) -> None:
        card = Card(
            name="Island",
            types=["Land"],
            supertypes=["Basic"],
            subtypes=["Island"],
            mana_cost=None,
        )
        result = fitted_fe.transform([card])
        # All mana features (0..12) should be 0 (has_mana_cost=0, rest=0)
        for i in range(13):
            assert result[0, i] == 0.0

    def test_type_encoding(self, fitted_fe: FeatureEngineering) -> None:
        card = Card(
            name="Test",
            types=["Creature", "Enchantment"],
            mana_cost=ManaCost.parse("2 G"),
            power="1",
            toughness="1",
        )
        result = fitted_fe.transform([card])
        # Types start at index 13: Creature, Instant, Sorcery, Enchantment, ...
        assert result[0, 13] == 1.0   # Creature
        assert result[0, 14] == 0.0   # Instant
        assert result[0, 15] == 0.0   # Sorcery
        assert result[0, 16] == 1.0   # Enchantment

    def test_supertype_encoding(self, fitted_fe: FeatureEngineering) -> None:
        card = Card(
            name="Test Legendary",
            types=["Creature"],
            supertypes=["Legendary"],
            mana_cost=ManaCost.parse("3"),
            power="2",
            toughness="2",
        )
        result = fitted_fe.transform([card])
        # Supertypes start at index 28 (13 mana + 15 types), order:
        # Legendary, Basic, Snow, World, Ongoing, Host
        assert result[0, 28] == 1.0  # Legendary
        assert result[0, 29] == 0.0  # Basic
        assert result[0, 30] == 0.0  # Snow

    def test_colorless_mana_encoding(self, fitted_fe: FeatureEngineering) -> None:
        card = Card(
            name="Matter Reshaper",
            types=["Creature"],
            subtypes=["Eldrazi"],
            mana_cost=ManaCost.parse("2 C"),
            power="3",
            toughness="2",
        )
        result = fitted_fe.transform([card])
        # Feature 0: has_mana_cost = 1.0
        assert result[0, 0] == 1.0
        # Feature 1: total_mana_value = 3
        assert result[0, 1] == 3.0
        # Feature 8: generic_mana = 2
        assert result[0, 8] == 2.0
        # Feature 9: colorless_mana = 1
        assert result[0, 9] == 1.0
        # Feature 7: color_count = 0
        assert result[0, 7] == 0.0

    def test_has_mana_cost_distinguishes_zero_cost_from_no_cost(
        self, fitted_fe: FeatureEngineering
    ) -> None:
        """has_mana_cost (feature 0) is 1.0 for {0}-cost cards, 0.0 for lands."""
        zero_cost = Card(
            name="Black Lotus",
            types=["Artifact"],
            mana_cost=ManaCost.parse("0"),
        )
        no_cost = Card(
            name="Island",
            types=["Land"],
            supertypes=["Basic"],
            mana_cost=None,
        )
        result_zero = fitted_fe.transform([zero_cost])
        result_none = fitted_fe.transform([no_cost])
        assert result_zero[0, 0] == 1.0  # has mana cost
        assert result_none[0, 0] == 0.0  # no mana cost

    def test_star_power_indicator(self, fitted_fe: FeatureEngineering) -> None:
        card = Card(
            name="Star Power",
            types=["Creature"],
            mana_cost=ManaCost.parse("3"),
            power="*",
            toughness="*",
        )
        fitted_fe.transform([card])
        # Verifies no error is raised when transforming star P/T

    def test_deterministic_output(self, fitted_fe: FeatureEngineering) -> None:
        card = Card(
            name="Bolt",
            types=["Instant"],
            mana_cost=ManaCost.parse("R"),
            oracle_text="Deal 3 damage.",
            ability_count=1,
        )
        result1 = fitted_fe.transform([card])
        result2 = fitted_fe.transform([card])
        np.testing.assert_array_equal(result1, result2)

    def test_partial_attributes(self, fitted_fe: FeatureEngineering) -> None:
        """Card with minimal attributes still produces valid features."""
        card = Card(name="Minimal", types=["Creature"])
        result = fitted_fe.transform([card])
        assert result.shape == (1, fitted_fe.get_feature_count())
        assert not np.any(np.isnan(result))

    def test_scheme_type_encoding(self, fitted_fe: FeatureEngineering) -> None:
        card = Card(
            name="Your Puny Minds Cannot Fathom",
            types=["Scheme"],
            mana_cost=None,
            oracle_text="Draw four cards.",
        )
        result = fitted_fe.transform([card])
        # Types at index 13+: Creature(13)..Battle(20), Scheme(21)
        assert result[0, 21] == 1.0  # Scheme
        # All other types should be 0
        for i in range(13, 21):
            assert result[0, i] == 0.0

    def test_layout_encoding(self, fitted_fe: FeatureEngineering) -> None:
        card = Card(
            name="Split Card",
            types=["Instant"],
            mana_cost=ManaCost.parse("1 R"),
            layout="split",
        )
        result = fitted_fe.transform([card])
        # Layout one-hot is near the end of the dense features
        # Check that it produces valid output without errors
        assert result.shape[1] == fitted_fe.get_feature_count()


class TestPrintingDataFeatures:
    """Tests for the 19 printing-data features appended to the dense vector."""

    PRINTING_FEATURE_COUNT = 19

    def test_card_with_printing_data_produces_19_additional_features(
        self, fitted_fe: FeatureEngineering
    ) -> None:
        """A Card with printing_data populated produces 19 additional dense features
        at the END of the feature vector, with correct values."""
        pd = PrintingData(
            is_reserved=True,
            rarity="mythic",
            printings_count=3,
            release_year=2018,
            legalities=["commander", "legacy"],
            is_abu=True,
        )
        card = Card(
            name="Test Reserved",
            types=["Creature"],
            mana_cost=ManaCost.parse("3 W W"),
            power="4",
            toughness="4",
            printing_data=pd,
        )
        result = fitted_fe.transform([card])
        total_features = fitted_fe.get_feature_count()
        assert result.shape == (1, total_features)

        tfidf_count = len(fitted_fe.tfidf_.vocabulary_)
        dense_count = total_features - tfidf_count
        pd_start = dense_count - self.PRINTING_FEATURE_COUNT

        row = result[0]

        # is_reserved = 1.0
        assert row[pd_start] == 1.0
        # is_abu = 1.0
        assert row[pd_start + 1] == 1.0

        # rarity one-hot: common=0, uncommon=0, rare=0, mythic=1
        assert row[pd_start + 2] == 0.0  # common
        assert row[pd_start + 3] == 0.0  # uncommon
        assert row[pd_start + 4] == 0.0  # rare
        assert row[pd_start + 5] == 1.0  # mythic

        # printings_count = 3.0
        assert row[pd_start + 6] == 3.0

        # release_year normalized = (2018 - 1992) / 34 ≈ 0.7647
        assert row[pd_start + 7] == pytest.approx((2018 - 1992) / 34.0)

        # legalities_count = 2.0
        assert row[pd_start + 8] == 2.0

        # format multi-hot (10 positions matching RECOGNIZED_FORMATS order)
        fmt_start = pd_start + 9
        for i, fmt in enumerate(RECOGNIZED_FORMATS):
            expected = 1.0 if fmt in ("commander", "legacy") else 0.0
            assert row[fmt_start + i] == expected, (
                f"Format {fmt} at position {i}: expected {expected}, got {row[fmt_start + i]}"
            )

    def test_card_without_printing_data_produces_19_zeros(
        self, fitted_fe: FeatureEngineering
    ) -> None:
        """A Card with printing_data=None produces 19 zeros at the end of the dense block."""
        card = Card(
            name="No Printing Data",
            types=["Creature"],
            mana_cost=ManaCost.parse("2 G"),
            power="2",
            toughness="2",
            printing_data=None,
        )
        result = fitted_fe.transform([card])
        total_features = fitted_fe.get_feature_count()
        tfidf_count = len(fitted_fe.tfidf_.vocabulary_)
        dense_count = total_features - tfidf_count
        pd_start = dense_count - self.PRINTING_FEATURE_COUNT

        row = result[0]
        for i in range(self.PRINTING_FEATURE_COUNT):
            assert row[pd_start + i] == 0.0, (
                f"Printing data feature at offset {i}: expected 0.0, got {row[pd_start + i]}"
            )

    def test_release_year_feature_present(
        self, fitted_fe: FeatureEngineering
    ) -> None:
        """Two cards differing only by release_year produce different feature vectors,
        proving the release_year feature is wired into the dense block."""
        early = Card(
            name="Early", types=["Creature"], mana_cost=ManaCost.parse("2 G"),
            power="2", toughness="2",
            printing_data=PrintingData(release_year=1995),
        )
        recent = Card(
            name="Recent", types=["Creature"], mana_cost=ManaCost.parse("2 G"),
            power="2", toughness="2",
            printing_data=PrintingData(release_year=2024),
        )
        early_vec = fitted_fe.transform([early])[0]
        recent_vec = fitted_fe.transform([recent])[0]
        assert not np.array_equal(early_vec, recent_vec)
