"""Feature engineering: transform Card entities into numeric feature vectors."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

from price_predictor.domain.card_taxonomy import (
    CARD_TYPES,
    LAYOUTS,
    RARITIES,
    SUPERTYPES,
)
from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import RECOGNIZED_FORMATS

_TOP_KEYWORD_LIMIT = 30


def _parse_pt(value: str | None) -> tuple[float, bool]:
    """Parse power/toughness string to (numeric_value, is_star).
    Returns (nan, False) if value is None.
    Returns (nan, True) if value is '*' or 'X'.
    """
    if value is None:
        return float("nan"), False
    stripped = value.strip()
    if stripped in ("*", "X"):
        return float("nan"), True
    try:
        return float(stripped), False
    except ValueError:
        return float("nan"), False


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Transforms Card entities into numeric feature vectors for ML models."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def fit(self, X: list[Card], y: object = None) -> "FeatureEngineering":
        """Learn TF-IDF vocabulary and top-30 keywords from training cards."""
        keyword_counts: dict[str, int] = {}
        for card in X:
            for kw in card.keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        sorted_kw = sorted(keyword_counts.items(), key=lambda x: -x[1])
        self.top_keywords_ = [kw for kw, _ in sorted_kw[:_TOP_KEYWORD_LIMIT]]

        self.tfidf_ = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            lowercase=True,
        )
        texts = [card.oracle_text or "" for card in X]
        self.tfidf_.fit(texts)

        # Probe one card to discover the dense feature width — guarantees
        # get_feature_count never drifts from the actual transform output.
        probe = X[0] if X else Card(name="probe", types=["Creature"])
        self.n_dense_features_ = sum(
            len(group(probe)) for group in self._feature_groups()
        )
        return self

    def transform(self, X: list[Card]) -> np.ndarray:
        """Transform cards into a numeric feature matrix."""
        check_is_fitted(self, ["top_keywords_", "tfidf_"])

        dense = np.array(
            [self._dense_features(card) for card in X],
            dtype=np.float64,
        )
        texts = [card.oracle_text or "" for card in X]
        tfidf_matrix = self.tfidf_.transform(texts).toarray()
        return np.hstack([dense, tfidf_matrix])

    def get_feature_count(self) -> int:
        """Return the total number of features produced by transform."""
        check_is_fitted(self, ["tfidf_"])
        return self.n_dense_features_ + len(self.tfidf_.vocabulary_)

    def _feature_groups(self) -> tuple[Callable[[Card], list[float]], ...]:
        return (
            self._mana_features,
            self._type_features,
            self._supertype_features,
            self._keyword_features,
            self._pt_features,
            self._layout_features,
            self._printing_features,
        )

    def _dense_features(self, card: Card) -> list[float]:
        result: list[float] = []
        for group in self._feature_groups():
            result.extend(group(card))
        return result

    def _mana_features(self, card: Card) -> list[float]:
        mc = card.mana_cost
        if mc is None:
            return [0.0] * 13
        return [
            1.0,  # has_mana_cost
            mc.total_mana_value,
            float(mc.w), float(mc.u), float(mc.b), float(mc.r), float(mc.g),
            float(mc.color_count),
            float(mc.generic_mana),
            float(mc.colorless_mana),
            float(mc.has_x),
            float(mc.has_hybrid),
            float(mc.has_phyrexian),
        ]

    def _type_features(self, card: Card) -> list[float]:
        return [1.0 if ct in card.types else 0.0 for ct in CARD_TYPES]

    def _supertype_features(self, card: Card) -> list[float]:
        features = [1.0 if st in card.supertypes else 0.0 for st in SUPERTYPES]
        features.append(float(len(card.subtypes)))
        return features

    def _keyword_features(self, card: Card) -> list[float]:
        card_keywords_set = set(card.keywords)
        features = [1.0 if kw in card_keywords_set else 0.0 for kw in self.top_keywords_]
        # Pad to fixed width when training data was too small to learn full top-K.
        features.extend([0.0] * (_TOP_KEYWORD_LIMIT - len(self.top_keywords_)))
        features.append(float(len(card.keywords)))
        features.append(float(len(card.oracle_text)) if card.oracle_text else 0.0)
        return features

    def _pt_features(self, card: Card) -> list[float]:
        power_val, power_star = _parse_pt(card.power)
        toughness_val, toughness_star = _parse_pt(card.toughness)
        loyalty = 0.0
        if card.loyalty is not None:
            try:
                loyalty = float(card.loyalty)
            except ValueError:
                loyalty = 0.0
        return [
            power_val if not np.isnan(power_val) else 0.0,
            toughness_val if not np.isnan(toughness_val) else 0.0,
            float(power_star),
            float(toughness_star),
            loyalty,
            float(card.ability_count),
        ]

    def _layout_features(self, card: Card) -> list[float]:
        return [1.0 if card.layout == layout else 0.0 for layout in LAYOUTS]

    def _printing_features(self, card: Card) -> list[float]:
        pd = card.printing_data
        if pd is None:
            return [0.0] * 19
        features = [
            1.0 if pd.is_reserved else 0.0,
            1.0 if pd.is_abu else 0.0,
        ]
        effective_rarity = pd.rarity if pd.rarity in RARITIES else "rare"
        features.extend(1.0 if effective_rarity == r else 0.0 for r in RARITIES)
        features.append(float(pd.printings_count))
        features.append(pd.normalized_release_year())
        legalities_set = set(pd.legalities)
        features.append(float(len(legalities_set)))
        features.extend(1.0 if fmt in legalities_set else 0.0 for fmt in RECOGNIZED_FORMATS)
        return features
