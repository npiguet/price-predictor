"""Feature engineering: transform Card entities into numeric feature vectors."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from price_predictor.domain.entities import Card

CARD_TYPES = [
    "Creature", "Instant", "Sorcery", "Enchantment",
    "Artifact", "Planeswalker", "Land", "Battle",
    "Scheme", "Plane", "Conspiracy", "Vanguard", "Phenomenon",
]
SUPERTYPES = ["Legendary", "Basic", "Snow", "World", "Ongoing", "Host"]
LAYOUTS = ["normal", "doublefaced", "split", "adventure", "modal", "flip"]


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


class FeatureEngineering:
    """Transforms Card entities into numeric feature vectors for ML models."""

    def __init__(self, random_seed: int = 42):
        self._random_seed = random_seed
        self._top_keywords: list[str] = []
        self._tfidf = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            lowercase=True,
        )
        self._is_fitted = False

    def fit(self, cards: list[Card]) -> "FeatureEngineering":
        """Learn TF-IDF vocabulary and top-30 keywords from training cards."""
        # Learn top keywords
        keyword_counts: dict[str, int] = {}
        for card in cards:
            for kw in card.keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        sorted_kw = sorted(keyword_counts.items(), key=lambda x: -x[1])
        self._top_keywords = [kw for kw, _ in sorted_kw[:30]]

        # Fit TF-IDF on oracle texts
        texts = [card.oracle_text or "" for card in cards]
        self._tfidf.fit(texts)

        self._is_fitted = True
        return self

    def transform(self, cards: list[Card]) -> np.ndarray:
        """Transform cards into a numeric feature matrix."""
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineering must be fitted before transform")

        rows = []
        for card in cards:
            row = self._transform_single(card)
            rows.append(row)

        # Combine dense features
        dense = np.array([r[0] for r in rows], dtype=np.float64)

        # TF-IDF sparse → dense
        texts = [card.oracle_text or "" for card in cards]
        tfidf_matrix = self._tfidf.transform(texts).toarray()

        return np.hstack([dense, tfidf_matrix])

    def _transform_single(self, card: Card) -> tuple[list[float], None]:
        """Transform a single card into a dense feature list."""
        features: list[float] = []

        # Mana cost features
        mc = card.mana_cost
        if mc is not None:
            features.append(mc.total_mana_value)
            features.extend([float(mc.w), float(mc.u), float(mc.b), float(mc.r), float(mc.g)])
            features.append(float(mc.color_count))
            features.append(float(mc.generic_mana))
            features.append(float(mc.colorless_mana))
            features.append(float(mc.has_x))
            features.append(float(mc.has_hybrid))
            features.append(float(mc.has_phyrexian))
        else:
            features.extend([0.0] * 12)  # All mana features = 0

        # Card type multi-hot
        for ct in CARD_TYPES:
            features.append(1.0 if ct in card.types else 0.0)

        # Supertype multi-hot
        for st in SUPERTYPES:
            features.append(1.0 if st in card.supertypes else 0.0)

        # Subtypes count
        features.append(float(len(card.subtypes)))

        # Keywords multi-hot (top 30) — always 30 features
        card_keywords_set = set(card.keywords)
        for kw in self._top_keywords:
            features.append(1.0 if kw in card_keywords_set else 0.0)
        # Pad to 30 if fewer keywords were learned from training data
        features.extend([0.0] * (30 - len(self._top_keywords)))

        # Keyword count
        features.append(float(len(card.keywords)))

        # Oracle text length
        features.append(float(len(card.oracle_text)) if card.oracle_text else 0.0)

        # Power, toughness
        power_val, power_star = _parse_pt(card.power)
        toughness_val, toughness_star = _parse_pt(card.toughness)
        features.append(power_val if not np.isnan(power_val) else 0.0)
        features.append(toughness_val if not np.isnan(toughness_val) else 0.0)
        features.append(float(power_star))
        features.append(float(toughness_star))

        # Loyalty
        if card.loyalty is not None:
            try:
                features.append(float(card.loyalty))
            except ValueError:
                features.append(0.0)
        else:
            features.append(0.0)

        # Ability count
        features.append(float(card.ability_count))

        # Layout one-hot
        for layout in LAYOUTS:
            features.append(1.0 if card.layout == layout else 0.0)

        return features, None

    def get_feature_count(self) -> int:
        """Return the total number of features produced by transform."""
        # Dense features: 12 (mana) + 13 (types) + 6 (supertypes) + 1 (subtypes count)
        #   + 30 (keywords) + 1 (kw count) + 1 (text len)
        #   + 2 (power/toughness) + 2 (star indicators) + 1 (loyalty) + 1 (ability count)
        #   + 6 (layout)
        dense = 12 + 13 + 6 + 1 + 30 + 1 + 1 + 2 + 2 + 1 + 1 + 6
        if self._is_fitted and hasattr(self._tfidf, "vocabulary_"):
            tfidf = len(self._tfidf.vocabulary_)
        else:
            tfidf = self._tfidf.max_features or 500
        return dense + tfidf
