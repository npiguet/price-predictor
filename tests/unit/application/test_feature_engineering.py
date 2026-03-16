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
        assert "Flying" in fitted_fe._top_keywords
        assert "Vigilance" in fitted_fe._top_keywords

    def test_fit_learns_tfidf(self, fitted_fe: FeatureEngineering) -> None:
        assert fitted_fe._tfidf.vocabulary_ is not None
        assert len(fitted_fe._tfidf.vocabulary_) > 0

    def test_transform_before_fit_raises(self) -> None:
        fe = FeatureEngineering()
        card = Card(name="Test", types=["Creature"], power="1", toughness="1")
        with pytest.raises(RuntimeError, match="must be fitted"):
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
        # First feature is mana value
        assert result[0, 0] == 4.0  # 2 + W + W = 4

    def test_color_encoding(self, fitted_fe: FeatureEngineering) -> None:
        card = Card(
            name="Multicolor",
            types=["Creature"],
            mana_cost=ManaCost.parse("1 U R"),
            power="2",
            toughness="2",
        )
        result = fitted_fe.transform([card])
        # Features [1..5] are W, U, B, R, G
        assert result[0, 1] == 0.0  # W
        assert result[0, 2] == 1.0  # U
        assert result[0, 3] == 0.0  # B
        assert result[0, 4] == 1.0  # R
        assert result[0, 5] == 0.0  # G
        # Feature [6] is color_count
        assert result[0, 6] == 2.0

    def test_land_no_mana_cost(self, fitted_fe: FeatureEngineering) -> None:
        card = Card(
            name="Island",
            types=["Land"],
            supertypes=["Basic"],
            subtypes=["Island"],
            mana_cost=None,
        )
        result = fitted_fe.transform([card])
        # All mana features (0..11) should be 0
        for i in range(12):
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
        # Types start at index 12: Creature, Instant, Sorcery, Enchantment, ...
        assert result[0, 12] == 1.0   # Creature
        assert result[0, 13] == 0.0   # Instant
        assert result[0, 14] == 0.0   # Sorcery
        assert result[0, 15] == 1.0   # Enchantment

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
        # Supertypes at index 25: Legendary, Basic, Snow, World, Ongoing, Host
        assert result[0, 25] == 1.0  # Legendary
        assert result[0, 26] == 0.0  # Basic
        assert result[0, 27] == 0.0  # Snow

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
        # Feature 0: total_mana_value = 3
        assert result[0, 0] == 3.0
        # Feature 7: generic_mana = 2
        assert result[0, 7] == 2.0
        # Feature 8: colorless_mana = 1
        assert result[0, 8] == 1.0
        # Feature 6: color_count = 0
        assert result[0, 6] == 0.0

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
        # Types at index 12+: Creature(12)..Battle(19), Scheme(20)
        assert result[0, 20] == 1.0  # Scheme
        # All other types should be 0
        for i in range(12, 20):
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
    """Tests for the 17 printing-data features appended to the dense vector."""

    def test_card_with_printing_data_produces_17_additional_features(
        self, fitted_fe: FeatureEngineering
    ) -> None:
        """A Card with printing_data populated produces 17 additional dense features
        at the END of the feature vector, with correct values."""
        pd = PrintingData(
            is_reserved=True,
            rarity="mythic",
            printings_count=3,
            set_code="uma",
            legalities=["commander", "legacy"],
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

        # The 17 printing data features are at the end of the dense block,
        # just before the TF-IDF features.
        # The fitted TF-IDF vocab may be smaller than 500 (depends on training data).
        tfidf_count = len(fitted_fe._tfidf.vocabulary_)
        dense_count = total_features - tfidf_count
        # Printing data starts at dense_count - 17
        pd_start = dense_count - 17

        row = result[0]

        # is_reserved = 1.0
        assert row[pd_start] == 1.0

        # rarity one-hot: common=0, uncommon=0, rare=0, mythic=1
        assert row[pd_start + 1] == 0.0  # common
        assert row[pd_start + 2] == 0.0  # uncommon
        assert row[pd_start + 3] == 0.0  # rare
        assert row[pd_start + 4] == 1.0  # mythic

        # printings_count = 3.0
        assert row[pd_start + 5] == 3.0

        # legalities_count = 2.0
        assert row[pd_start + 6] == 2.0

        # format multi-hot (10 positions matching RECOGNIZED_FORMATS order)
        # RECOGNIZED_FORMATS = ("standard", "pioneer", "modern", "brawl", "legacy",
        #                       "vintage", "pauper", "commander", "penny", "oathbreaker")
        fmt_start = pd_start + 7
        for i, fmt in enumerate(RECOGNIZED_FORMATS):
            expected = 1.0 if fmt in ("commander", "legacy") else 0.0
            assert row[fmt_start + i] == expected, (
                f"Format {fmt} at position {i}: expected {expected}, got {row[fmt_start + i]}"
            )

    def test_card_without_printing_data_produces_17_zeros(
        self, fitted_fe: FeatureEngineering
    ) -> None:
        """A Card with printing_data=None produces 17 zeros at the end of the dense block."""
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
        tfidf_count = len(fitted_fe._tfidf.vocabulary_)
        dense_count = total_features - tfidf_count
        pd_start = dense_count - 17

        row = result[0]
        for i in range(17):
            assert row[pd_start + i] == 0.0, (
                f"Printing data feature at offset {i}: expected 0.0, got {row[pd_start + i]}"
            )

    def test_feature_count_increased_by_17(
        self, fitted_fe: FeatureEngineering
    ) -> None:
        """Dense feature count is 93 (was 76 before printing data).
        The 17 extra features come from printing data.
        With a full 500-word TF-IDF vocabulary: 93 + 500 = 593.
        With the test fixture's small vocabulary, we verify
        dense = 93 and that it is exactly 17 more than the old 76."""
        tfidf_count = len(fitted_fe._tfidf.vocabulary_)
        total = fitted_fe.get_feature_count()
        dense_count = total - tfidf_count
        # Dense features should be 93 (76 old + 17 printing data)
        assert dense_count == 93
        # The increase from the old dense count (76) is exactly 17
        old_dense_count = 76
        assert dense_count - old_dense_count == 17
