"""Unit tests for EncodeCardsUseCase."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from sealed.application.encode_cards import (
    EncodeCardsConfig,
    EncodeCardsResult,
    EncodeCardsUseCase,
)


def _execute(use_case, cards_dir, encoder, store, *, clean=False):
    return use_case.execute(
        EncodeCardsConfig(cards_path=cards_dir, clean=clean),
        encoder,
        store,
    )


def _make_fixture_dir(tmp_path: Path, cards: dict[str, str]) -> Path:
    """Write card .txt files and return the directory."""
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    for name, text in cards.items():
        (cards_dir / name).write_text(text, encoding="utf-8")
    return cards_dir


def _mock_encoder(embedding: np.ndarray | None = None):
    enc = MagicMock()
    enc.encode.return_value = embedding if embedding is not None else np.zeros(16, dtype=np.float32)
    return enc


def _mock_store():
    store = MagicMock()
    store.save = MagicMock()
    return store


class TestEncodeCardsSkipLogic:
    def test_skips_card_with_existing_npz(self, tmp_path):
        cards_dir = _make_fixture_dir(tmp_path, {"card_a.txt": "type: Creature"})
        npz_path = cards_dir / "card_a.npz"
        npz_path.touch()  # simulate already-encoded

        use_case = EncodeCardsUseCase()
        encoder = _mock_encoder()
        store = _mock_store()
        result = _execute(use_case, cards_dir, encoder, store)

        encoder.encode.assert_not_called()
        store.save.assert_not_called()
        assert result.skipped == 1
        assert result.processed == 0

    def test_encodes_card_without_npz(self, tmp_path):
        cards_dir = _make_fixture_dir(tmp_path, {"card_a.txt": "type: Creature"})

        use_case = EncodeCardsUseCase()
        encoder = _mock_encoder()
        store = _mock_store()
        result = _execute(use_case, cards_dir, encoder, store)

        encoder.encode.assert_called_once()
        store.save.assert_called_once()
        assert result.processed == 1
        assert result.skipped == 0

    def test_partial_folder_processes_only_missing(self, tmp_path):
        cards_dir = _make_fixture_dir(tmp_path, {
            "card_a.txt": "type: Creature",
            "card_b.txt": "type: Instant",
            "card_c.txt": "type: Land",
        })
        (cards_dir / "card_b.npz").touch()  # already encoded

        use_case = EncodeCardsUseCase()
        encoder = _mock_encoder()
        store = _mock_store()
        result = _execute(use_case, cards_dir, encoder, store)

        assert result.processed == 2
        assert result.skipped == 1


class TestEncodeCardsErrorHandling:
    def test_error_in_one_card_does_not_stop_others(self, tmp_path):
        cards_dir = _make_fixture_dir(tmp_path, {
            "card_a.txt": "type: Creature",
            "card_b.txt": "type: Instant",
        })

        use_case = EncodeCardsUseCase()
        encoder = _mock_encoder()
        # Make the first call raise, second succeed
        encoder.encode.side_effect = [RuntimeError("bad card"), np.zeros(16, dtype=np.float32)]
        store = _mock_store()
        result = _execute(use_case, cards_dir, encoder, store)

        assert len(result.errors) == 1
        assert result.processed == 1

    def test_error_messages_accumulate(self, tmp_path):
        cards_dir = _make_fixture_dir(tmp_path, {
            "card_a.txt": "x",
            "card_b.txt": "x",
        })

        use_case = EncodeCardsUseCase()
        encoder = _mock_encoder()
        encoder.encode.side_effect = RuntimeError("boom")
        store = _mock_store()
        result = _execute(use_case, cards_dir, encoder, store)

        assert len(result.errors) == 2


class TestEncodeCardsResult:
    def test_result_dataclass_fields(self):
        r = EncodeCardsResult(processed=10, skipped=5, errors=["err1"])
        assert r.processed == 10
        assert r.skipped == 5
        assert r.errors == ["err1"]

    def test_empty_errors_by_default(self):
        r = EncodeCardsResult(processed=0, skipped=0, errors=[])
        assert r.errors == []


class TestEncodeCards544DimSkipLogic:
    """Verify that 544-dim encoding correctly handles skip logic."""

    def test_fresh_encode_produces_file(self, tmp_path):
        """With no existing .npz, encode produces a file."""
        cards_dir = _make_fixture_dir(tmp_path, {"card_a.txt": "type: Creature"})
        embedding_544 = np.zeros(544, dtype=np.float32)

        use_case = EncodeCardsUseCase()
        encoder = _mock_encoder(embedding_544)
        store = _mock_store()
        result = _execute(use_case, cards_dir, encoder, store)

        assert result.processed == 1
        saved_embedding = store.save.call_args[0][1]
        assert saved_embedding.shape == (544,)

    def test_existing_544_dim_file_skipped(self, tmp_path):
        """Cards with existing .npz (any dimension) are skipped."""
        cards_dir = _make_fixture_dir(tmp_path, {"card_a.txt": "type: Creature"})
        npz_path = cards_dir / "card_a.npz"
        np.savez_compressed(npz_path, embedding=np.zeros(544, dtype=np.float32))

        use_case = EncodeCardsUseCase()
        encoder = _mock_encoder()
        store = _mock_store()
        result = _execute(use_case, cards_dir, encoder, store)

        encoder.encode.assert_not_called()
        assert result.skipped == 1

    def test_existing_512_dim_file_skipped_not_upgraded(self, tmp_path):
        """Existing 512-dim files are NOT auto-upgraded; they are skipped."""
        cards_dir = _make_fixture_dir(tmp_path, {"card_a.txt": "type: Creature"})
        npz_path = cards_dir / "card_a.npz"
        np.savez_compressed(npz_path, embedding=np.zeros(512, dtype=np.float32))

        use_case = EncodeCardsUseCase()
        encoder = _mock_encoder()
        store = _mock_store()
        result = _execute(use_case, cards_dir, encoder, store)

        encoder.encode.assert_not_called()
        assert result.skipped == 1

    def test_clean_deletes_and_re_encodes(self, tmp_path):
        """clean=True deletes all .npz files before encoding so cards are re-encoded."""
        cards_dir = _make_fixture_dir(tmp_path, {"card_a.txt": "type: Creature"})
        npz_path = cards_dir / "card_a.npz"
        np.savez_compressed(npz_path, embedding=np.zeros(512, dtype=np.float32))

        use_case = EncodeCardsUseCase()
        encoder = _mock_encoder(np.zeros(544, dtype=np.float32))
        store = _mock_store()
        result = _execute(use_case, cards_dir, encoder, store, clean=True)

        assert result.processed == 1
        assert result.skipped == 0


class TestEncodeCardsSaveArgs:
    def test_save_called_with_npz_path(self, tmp_path):
        cards_dir = _make_fixture_dir(tmp_path, {"mycard.txt": "type: Creature"})

        use_case = EncodeCardsUseCase()
        encoder = _mock_encoder()
        store = _mock_store()
        _execute(use_case, cards_dir, encoder, store)

        saved_path = store.save.call_args[0][0]
        assert saved_path.suffix == ".npz"
        assert saved_path.stem == "mycard"
