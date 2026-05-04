"""Unit tests for the stratified card-level train/val split (FR-018)."""

from __future__ import annotations

from sealed.application.train_encoder import CardLabel, _split_cards


def _label_map(values: dict[str, float]) -> dict[str, CardLabel]:
    return {
        name: CardLabel(
            card_name=name,
            wins_when_played=int(v * 100),
            wins_when_in_deck=100,
            shrunk_label=v,
        )
        for name, v in values.items()
    }


class TestSplitCards:
    def test_sets_are_disjoint(self):
        labels = _label_map({f"card_{i}": (i / 100.0) for i in range(50)})
        train, val = _split_cards(labels, val_fraction=0.2, seed=42)
        assert set(train).isdisjoint(set(val))
        assert set(train) | set(val) == set(labels.keys())

    def test_val_fraction_approximate(self):
        labels = _label_map({f"card_{i}": (i / 100.0) for i in range(50)})
        _, val = _split_cards(labels, val_fraction=0.2, seed=42)
        # 20% of 50 = 10; allow one off (rounding per quartile).
        assert 8 <= len(val) <= 12

    def test_deterministic(self):
        labels = _label_map({f"card_{i}": (i / 100.0) for i in range(50)})
        a = _split_cards(labels, val_fraction=0.2, seed=42)
        b = _split_cards(labels, val_fraction=0.2, seed=42)
        assert a == b

    def test_handles_degenerate_distribution(self):
        # Every card has the same label — only one quantile bin.
        labels = _label_map({f"card_{i}": 0.5 for i in range(20)})
        train, val = _split_cards(labels, val_fraction=0.2, seed=42)
        assert set(train).isdisjoint(set(val))
        assert len(val) == 4  # 20% of 20

    def test_val_covers_label_range(self):
        # Stratification by quartile means val should contain low + high cards.
        labels = _label_map({f"card_{i}": (i / 100.0) for i in range(100)})
        _, val = _split_cards(labels, val_fraction=0.2, seed=42)
        val_values = sorted(labels[n].shrunk_label for n in val)
        assert val_values[0] < 0.30, "val must include low-quartile cards"
        assert val_values[-1] > 0.70, "val must include high-quartile cards"
