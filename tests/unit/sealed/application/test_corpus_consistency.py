"""Unit tests for the corpus consistency check (FR-023d)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sealed.application.train_encoder import (
    CardLabel,
    CorpusInconsistencyError,
    _check_corpus_consistency,
)
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator


def _label(name: str) -> CardLabel:
    return CardLabel(card_name=name, wins_when_played=1, wins_when_in_deck=1, shrunk_label=1.0)


def _seed_card(folder: Path, card_name: str) -> None:
    """Create a converted card .txt at the expected on-disk location."""
    sanitized = card_name.lower().replace(" ", "_").replace(",", "")
    letter = sanitized[0]
    (folder / letter).mkdir(parents=True, exist_ok=True)
    (folder / letter / f"{sanitized}.txt").write_text(
        f"name: {card_name}\ntype: instant\n", encoding="utf-8",
    )


class TestCorpusConsistency:
    def test_passes_when_all_cards_present(self, tmp_path: Path):
        for name in ("Lightning Bolt", "Grizzly Bears"):
            _seed_card(tmp_path, name)
        labels = {n: _label(n) for n in ("Lightning Bolt", "Grizzly Bears")}
        _check_corpus_consistency(labels, ConvertedCardLocator(tmp_path))

    def test_raises_with_missing_card_names(self, tmp_path: Path):
        _seed_card(tmp_path, "Lightning Bolt")
        labels = {
            n: _label(n)
            for n in ("Lightning Bolt", "Counterspell", "Llanowar Elves")
        }
        with pytest.raises(CorpusInconsistencyError) as ex:
            _check_corpus_consistency(labels, ConvertedCardLocator(tmp_path))
        msg = str(ex.value)
        assert "2 card(s)" in msg
        assert "Counterspell" in msg
        assert "Llanowar Elves" in msg
        assert "python -m price_predictor convert" in msg

    def test_caps_displayed_card_names_at_20(self, tmp_path: Path):
        labels = {f"Card {i}": _label(f"Card {i}") for i in range(30)}
        with pytest.raises(CorpusInconsistencyError) as ex:
            _check_corpus_consistency(labels, ConvertedCardLocator(tmp_path))
        msg = str(ex.value)
        assert "and 10 more" in msg
