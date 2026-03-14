"""Domain entities for the card price predictor."""

from __future__ import annotations

from dataclasses import dataclass, field

from price_predictor.domain.value_objects import ManaCost

VALID_LAYOUTS = frozenset(
    {"normal", "doublefaced", "split", "adventure", "modal", "flip"}
)

VALID_PT_PATTERN_CHARS = frozenset("0123456789*X+-")


def _validate_pt(value: str | None, field_name: str) -> None:
    if value is None:
        return
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty if provided")
    if not all(c in VALID_PT_PATTERN_CHARS for c in stripped):
        raise ValueError(
            f"{field_name} must be a number, '*', or 'X', got '{value}'"
        )


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

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Card name must not be empty")
        if not self.types:
            raise ValueError("Card must have at least one type")
        _validate_pt(self.power, "power")
        _validate_pt(self.toughness, "toughness")
        if self.layout not in VALID_LAYOUTS:
            raise ValueError(
                f"layout must be one of {sorted(VALID_LAYOUTS)}, got '{self.layout}'"
            )


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
    """Immutable configuration for reconstructing a transformer model architecture."""

    d_model: int
    n_layers: int
    n_heads: int
    ff_dim: int
    max_seq_len: int
    vocab_size: int
    dropout: float

    def __post_init__(self) -> None:
        for name in ("d_model", "n_layers", "n_heads", "ff_dim", "max_seq_len", "vocab_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
