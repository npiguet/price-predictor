"""Domain entities for the card price predictor."""

from __future__ import annotations

from dataclasses import dataclass, field

from price_predictor.domain.card_taxonomy import VALID_LAYOUTS
from price_predictor.domain.power_toughness import validate_pt_chars
from price_predictor.domain.value_objects import ManaCost, PrintingData


@dataclass(frozen=True)
class Card:
    """A Magic: The Gathering card described by its game attributes."""

    name: str
    types: list[str]
    supertypes: list[str] = field(default_factory=list)
    subtypes: list[str] = field(default_factory=list)
    mana_cost: ManaCost | None = None
    oracle_text: str | None = None
    keywords: list[str] = field(default_factory=list)
    power: str | None = None
    toughness: str | None = None
    loyalty: str | None = None
    layout: str = "normal"
    ability_count: int = 0
    printing_data: PrintingData | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Card name must not be empty")
        if not self.types:
            raise ValueError("Card must have at least one type")
        validate_pt_chars(self.power, "power")
        validate_pt_chars(self.toughness, "toughness")
        if self.layout not in VALID_LAYOUTS:
            raise ValueError(
                f"layout must be one of {sorted(VALID_LAYOUTS)}, got '{self.layout}'"
            )

    def is_land(self) -> bool:
        return any(t.lower() == "land" for t in self.types)

    def has_devoid(self) -> bool:
        return "devoid" in (self.oracle_text or "").lower()

    def is_colorless(self) -> bool:
        if self.has_devoid() or self.mana_cost is None:
            return True
        return self.mana_cost.color_count == 0


@dataclass(frozen=True)
class PriceEstimate:
    """The system's predicted EUR market price for a card."""

    predicted_price_eur: float
    model_version: str

    def __post_init__(self) -> None:
        if self.predicted_price_eur < 0:
            raise ValueError("predicted_price_eur must be >= 0")
        if not self.model_version:
            raise ValueError("model_version must not be empty")


@dataclass(frozen=True)
class TrainingExample:
    """A Card paired with its known EUR market price."""

    card: Card
    actual_price_eur: float

    def __post_init__(self) -> None:
        if self.actual_price_eur <= 0:
            raise ValueError("actual_price_eur must be > 0")


@dataclass(frozen=True)
class EvaluationMetrics:
    """Accuracy metrics from model evaluation."""

    mean_absolute_error_eur: float
    median_percentage_error: float
    median_abs_error_log: float
    top_20_overlap: float
    sample_count: int


@dataclass(frozen=True)
class TrainedModel:
    """Metadata about a trained prediction model."""

    model_version: str
    training_date: str
    card_count: int
    price_range_min_eur: float
    price_range_max_eur: float
    metrics: EvaluationMetrics | None = None


@dataclass(frozen=True)
class TransformerConfig:
    """Immutable configuration for reconstructing a transformer model architecture.

    log_offset controls the price transform used during training and inference:
        target  = log(price + log_offset)
        inverse = exp(prediction) - log_offset

    A smaller offset (e.g. 0.5) gives more gradient signal to mid/high-price
    cards; the default of 2.0 preserves backward compatibility.
    """

    d_model: int
    n_layers: int
    n_heads: int
    ff_dim: int
    max_seq_len: int
    vocab_size: int
    dropout: float
    regression_hidden_dim: int = 64
    log_offset: float = 2.0
    meta_dim: int = 15  # must match metadata_encoder.encode_metadata output length

    def __post_init__(self) -> None:
        for name in (
            "d_model", "n_layers", "n_heads", "ff_dim",
            "max_seq_len", "vocab_size", "regression_hidden_dim", "meta_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.log_offset <= 0.0:
            raise ValueError(f"log_offset must be > 0, got {self.log_offset}")

    @property
    def pooled_dim(self) -> int:
        """Width of the pooled card vector — ``cat([max_pool, mean_pool])`` over
        the encoder's token outputs (mirrors ``SealedEncoderConfig.pooled_dim``)."""
        return 2 * self.d_model
