"""Tests for domain entities."""

import pytest

from price_predictor.domain.entities import (
    Card,
    EvaluationMetrics,
    PriceEstimate,
    TrainedModel,
    TrainingExample,
    TransformerConfig,
)
from price_predictor.domain.value_objects import ManaCost

# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------


class TestCard:
    """Tests for the Card frozen dataclass."""

    def test_construction_with_all_fields(self):
        mana = ManaCost.parse("2 W W")
        card = Card(
            name="Serra Angel",
            types=["Creature"],
            supertypes=["Legendary"],
            subtypes=["Angel"],
            mana_cost=mana,
            oracle_text="Flying, vigilance",
            keywords=["flying", "vigilance"],
            power="4",
            toughness="4",
            loyalty=None,
            layout="normal",
            ability_count=2,
        )
        assert card.name == "Serra Angel"
        assert card.types == ["Creature"]
        assert card.supertypes == ["Legendary"]
        assert card.subtypes == ["Angel"]
        assert card.mana_cost is not None
        assert card.mana_cost.total_mana_value == 4.0
        assert card.oracle_text == "Flying, vigilance"
        assert card.keywords == ["flying", "vigilance"]
        assert card.power == "4"
        assert card.toughness == "4"
        assert card.loyalty is None
        assert card.layout == "normal"
        assert card.ability_count == 2

    def test_construction_with_minimal_fields(self):
        card = Card(name="Lightning Bolt", types=["Instant"])
        assert card.name == "Lightning Bolt"
        assert card.types == ["Instant"]
        assert card.supertypes == []
        assert card.subtypes == []
        assert card.mana_cost is None
        assert card.oracle_text is None
        assert card.keywords == []
        assert card.power is None
        assert card.toughness is None
        assert card.loyalty is None
        assert card.layout == "normal"
        assert card.ability_count == 0

    def test_null_optional_fields(self):
        card = Card(
            name="Ancestral Recall",
            types=["Instant"],
            mana_cost=None,
            oracle_text=None,
            power=None,
            toughness=None,
        )
        assert card.mana_cost is None
        assert card.oracle_text is None
        assert card.power is None
        assert card.toughness is None

    def test_star_power_toughness_is_valid(self):
        card = Card(
            name="Tarmogoyf",
            types=["Creature"],
            power="*",
            toughness="1+*",
        )
        assert card.power == "*"
        assert card.toughness == "1+*"

    def test_x_power_is_valid(self):
        card = Card(
            name="Stonecoil Serpent",
            types=["Creature"],
            power="X",
            toughness="X",
        )
        assert card.power == "X"
        assert card.toughness == "X"

    def test_invalid_power_raises_value_error(self):
        with pytest.raises(ValueError, match="power"):
            Card(name="Bad Card", types=["Creature"], power="abc")

    def test_invalid_toughness_raises_value_error(self):
        with pytest.raises(ValueError, match="toughness"):
            Card(name="Bad Card", types=["Creature"], toughness="abc")

    def test_empty_name_raises_value_error(self):
        with pytest.raises(ValueError, match="name"):
            Card(name="", types=["Creature"])

    def test_empty_types_raises_value_error(self):
        with pytest.raises(ValueError, match="type"):
            Card(name="Nameless", types=[])

    def test_invalid_layout_raises_value_error(self):
        with pytest.raises(ValueError, match="layout"):
            Card(name="Bad Layout", types=["Instant"], layout="transform")

    def test_valid_layouts(self):
        for layout in ("normal", "doublefaced", "split", "adventure", "modal", "flip"):
            card = Card(name="Test", types=["Instant"], layout=layout)
            assert card.layout == layout

    def test_frozen(self):
        card = Card(name="Bolt", types=["Instant"])
        with pytest.raises(AttributeError):
            card.name = "Changed"


# ---------------------------------------------------------------------------
# PriceEstimate
# ---------------------------------------------------------------------------


class TestPriceEstimate:
    """Tests for the PriceEstimate frozen dataclass."""

    def test_construction(self):
        estimate = PriceEstimate(predicted_price_eur=1.50, model_version="v1.0")
        assert estimate.predicted_price_eur == 1.50
        assert estimate.model_version == "v1.0"

    def test_zero_price_is_valid(self):
        estimate = PriceEstimate(predicted_price_eur=0.0, model_version="v1.0")
        assert estimate.predicted_price_eur == 0.0

    def test_negative_price_raises_value_error(self):
        with pytest.raises(ValueError, match="predicted_price_eur"):
            PriceEstimate(predicted_price_eur=-0.01, model_version="v1.0")

    def test_empty_model_version_raises_value_error(self):
        with pytest.raises(ValueError, match="model_version"):
            PriceEstimate(predicted_price_eur=1.0, model_version="")

    def test_frozen(self):
        estimate = PriceEstimate(predicted_price_eur=1.0, model_version="v1")
        with pytest.raises(AttributeError):
            estimate.predicted_price_eur = 2.0


# ---------------------------------------------------------------------------
# TrainingExample
# ---------------------------------------------------------------------------


class TestTrainingExample:
    """Tests for the TrainingExample frozen dataclass."""

    def test_valid_training_example(self):
        card = Card(name="Bolt", types=["Instant"])
        example = TrainingExample(card=card, actual_price_eur=0.25)
        assert example.card.name == "Bolt"
        assert example.actual_price_eur == 0.25

    def test_zero_price_raises_value_error(self):
        card = Card(name="Bolt", types=["Instant"])
        with pytest.raises(ValueError, match="actual_price_eur"):
            TrainingExample(card=card, actual_price_eur=0.0)

    def test_negative_price_raises_value_error(self):
        card = Card(name="Bolt", types=["Instant"])
        with pytest.raises(ValueError, match="actual_price_eur"):
            TrainingExample(card=card, actual_price_eur=-5.0)

    def test_frozen(self):
        card = Card(name="Bolt", types=["Instant"])
        example = TrainingExample(card=card, actual_price_eur=1.0)
        with pytest.raises(AttributeError):
            example.actual_price_eur = 2.0


# ---------------------------------------------------------------------------
# TrainedModel
# ---------------------------------------------------------------------------


class TestTrainedModel:
    """Tests for the TrainedModel frozen dataclass."""

    def test_construction_without_metrics(self):
        model = TrainedModel(
            model_version="v1.0",
            training_date="2026-03-01",
            card_count=1000,
            price_range_min_eur=0.02,
            price_range_max_eur=500.0,
        )
        assert model.model_version == "v1.0"
        assert model.training_date == "2026-03-01"
        assert model.card_count == 1000
        assert model.price_range_min_eur == 0.02
        assert model.price_range_max_eur == 500.0
        assert model.metrics is None

    def test_construction_with_metrics(self):
        metrics = EvaluationMetrics(
            mean_absolute_error_eur=0.50,
            median_percentage_error=12.5,
            top_20_overlap=0.85,
            sample_count=200,
        )
        model = TrainedModel(
            model_version="v2.0",
            training_date="2026-03-01",
            card_count=5000,
            price_range_min_eur=0.01,
            price_range_max_eur=1000.0,
            metrics=metrics,
        )
        assert model.metrics is not None
        assert model.metrics.mean_absolute_error_eur == 0.50
        assert model.metrics.sample_count == 200

    def test_frozen(self):
        model = TrainedModel(
            model_version="v1",
            training_date="2026-03-01",
            card_count=100,
            price_range_min_eur=0.01,
            price_range_max_eur=100.0,
        )
        with pytest.raises(AttributeError):
            model.model_version = "v2"


# ---------------------------------------------------------------------------
# EvaluationMetrics
# ---------------------------------------------------------------------------


class TestEvaluationMetrics:
    """Tests for the EvaluationMetrics frozen dataclass."""

    def test_construction(self):
        metrics = EvaluationMetrics(
            mean_absolute_error_eur=1.23,
            median_percentage_error=15.0,
            top_20_overlap=0.75,
            sample_count=500,
        )
        assert metrics.mean_absolute_error_eur == 1.23
        assert metrics.median_percentage_error == 15.0
        assert metrics.top_20_overlap == 0.75
        assert metrics.sample_count == 500

    def test_frozen(self):
        metrics = EvaluationMetrics(
            mean_absolute_error_eur=1.0,
            median_percentage_error=10.0,
            top_20_overlap=0.5,
            sample_count=100,
        )
        with pytest.raises(AttributeError):
            metrics.sample_count = 200


# ---------------------------------------------------------------------------
# TransformerConfig
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    defaults = dict(
        d_model=128,
        n_layers=4,
        n_heads=4,
        ff_dim=512,
        max_seq_len=256,
        vocab_size=30522,
        dropout=0.1,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestTransformerConfig:
    """Tests for the TransformerConfig frozen dataclass."""

    def test_construction_with_valid_defaults(self):
        cfg = _make_config()
        assert cfg.d_model == 128
        assert cfg.n_layers == 4
        assert cfg.n_heads == 4
        assert cfg.ff_dim == 512
        assert cfg.max_seq_len == 256
        assert cfg.vocab_size == 30522
        assert cfg.dropout == 0.1

    def test_frozen(self):
        cfg = _make_config()
        with pytest.raises(AttributeError):
            cfg.d_model = 256

    def test_zero_d_model_raises(self):
        with pytest.raises(ValueError, match="d_model"):
            _make_config(d_model=0)

    def test_zero_n_layers_raises(self):
        with pytest.raises(ValueError, match="n_layers"):
            _make_config(n_layers=0)

    def test_zero_n_heads_raises(self):
        with pytest.raises(ValueError, match="n_heads"):
            _make_config(n_heads=0)

    def test_zero_ff_dim_raises(self):
        with pytest.raises(ValueError, match="ff_dim"):
            _make_config(ff_dim=0)

    def test_zero_max_seq_len_raises(self):
        with pytest.raises(ValueError, match="max_seq_len"):
            _make_config(max_seq_len=0)

    def test_zero_vocab_size_raises(self):
        with pytest.raises(ValueError, match="vocab_size"):
            _make_config(vocab_size=0)

    def test_negative_d_model_raises(self):
        with pytest.raises(ValueError, match="d_model"):
            _make_config(d_model=-1)

    def test_d_model_not_divisible_by_n_heads_raises(self):
        with pytest.raises(ValueError, match="d_model"):
            _make_config(d_model=128, n_heads=3)

    def test_dropout_negative_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            _make_config(dropout=-0.1)

    def test_dropout_one_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            _make_config(dropout=1.0)

    def test_dropout_above_one_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            _make_config(dropout=1.5)

    def test_dropout_zero_is_valid(self):
        cfg = _make_config(dropout=0.0)
        assert cfg.dropout == 0.0
