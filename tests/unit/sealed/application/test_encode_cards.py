"""Unit tests for EncodeCardsUseCase."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from sealed.application.encode_cards import (
    EncodeCardsConfig,
    EncodeCardsResult,
    EncodeCardsUseCase,
)


def _execute(cards_dir, encoder, store, *, clean=False):
    return EncodeCardsUseCase().execute(
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


def _encode_over_existing(
    tmp_path: Path, *, existing_dim: int, embedding: np.ndarray | None = None, clean: bool = False,
):
    """Encode one card whose ``.npz`` already holds a zero vector of ``existing_dim``."""
    cards_dir = _make_fixture_dir(tmp_path, {"card_a.txt": "type: Creature"})
    npz_path = cards_dir / "card_a.npz"
    np.savez_compressed(npz_path, embedding=np.zeros(existing_dim, dtype=np.float32))
    encoder = _mock_encoder(embedding)
    result = _execute(cards_dir, encoder, _mock_store(), clean=clean)
    return encoder, result


class TestEncodeCardsSkipLogic:
    def test_skips_card_with_existing_npz(self, tmp_path):
        cards_dir = _make_fixture_dir(tmp_path, {"card_a.txt": "type: Creature"})
        npz_path = cards_dir / "card_a.npz"
        npz_path.touch()  # simulate already-encoded

        encoder = _mock_encoder()
        store = _mock_store()
        result = _execute(cards_dir, encoder, store)

        encoder.encode.assert_not_called()
        store.save.assert_not_called()
        assert result.skipped == 1
        assert result.processed == 0

    def test_encodes_card_without_npz(self, tmp_path):
        cards_dir = _make_fixture_dir(tmp_path, {"card_a.txt": "type: Creature"})

        encoder = _mock_encoder()
        store = _mock_store()
        result = _execute(cards_dir, encoder, store)

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

        encoder = _mock_encoder()
        store = _mock_store()
        result = _execute(cards_dir, encoder, store)

        assert result.processed == 2
        assert result.skipped == 1


class TestEncodeCardsErrorHandling:
    def test_error_in_one_card_does_not_stop_others(self, tmp_path):
        cards_dir = _make_fixture_dir(tmp_path, {
            "card_a.txt": "type: Creature",
            "card_b.txt": "type: Instant",
        })

        encoder = _mock_encoder()
        # Make the first call raise, second succeed
        encoder.encode.side_effect = [RuntimeError("bad card"), np.zeros(16, dtype=np.float32)]
        store = _mock_store()
        result = _execute(cards_dir, encoder, store)

        assert len(result.errors) == 1
        assert result.processed == 1

    def test_error_messages_accumulate(self, tmp_path):
        cards_dir = _make_fixture_dir(tmp_path, {
            "card_a.txt": "x",
            "card_b.txt": "x",
        })

        encoder = _mock_encoder()
        encoder.encode.side_effect = RuntimeError("boom")
        store = _mock_store()
        result = _execute(cards_dir, encoder, store)

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

        encoder = _mock_encoder(embedding_544)
        store = _mock_store()
        result = _execute(cards_dir, encoder, store)

        assert result.processed == 1
        saved_embedding = store.save.call_args[0][1]
        assert saved_embedding.shape == (544,)

    def test_existing_544_dim_file_skipped(self, tmp_path):
        """Cards with existing .npz (any dimension) are skipped."""
        encoder, result = _encode_over_existing(tmp_path, existing_dim=544)

        encoder.encode.assert_not_called()
        assert result.skipped == 1

    def test_existing_512_dim_file_skipped_not_upgraded(self, tmp_path):
        """Existing 512-dim files are NOT auto-upgraded; they are skipped."""
        encoder, result = _encode_over_existing(tmp_path, existing_dim=512)

        encoder.encode.assert_not_called()
        assert result.skipped == 1

    def test_clean_deletes_and_re_encodes(self, tmp_path):
        """clean=True deletes all .npz files before encoding so cards are re-encoded."""
        encoder, result = _encode_over_existing(
            tmp_path, existing_dim=512, embedding=np.zeros(544, dtype=np.float32), clean=True,
        )

        assert result.processed == 1
        assert result.skipped == 0


class TestEncodeCardsSaveArgs:
    def test_save_called_with_npz_path(self, tmp_path):
        cards_dir = _make_fixture_dir(tmp_path, {"mycard.txt": "type: Creature"})

        encoder = _mock_encoder()
        store = _mock_store()
        _execute(cards_dir, encoder, store)

        saved_path = store.save.call_args[0][0]
        assert saved_path.suffix == ".npz"
        assert saved_path.stem == "mycard"


class TestEmbeddingsApartFromTheCards:
    """``embeddings_path``: the ``.npz`` files go to a tree of their own, laid
    out like the cards, and ``--clean`` empties only that tree."""

    def _trees(self, tmp_path):
        cards = tmp_path / "converted" / "cardsfolder"
        (cards / "a").mkdir(parents=True)
        (cards / "a" / "card_a.txt").write_text("type: Creature", encoding="utf-8")
        (cards / "b").mkdir()
        (cards / "b" / "card_b.txt").write_text("type: Instant", encoding="utf-8")
        return cards, tmp_path / "cache" / "run"

    def _run(self, cards, embeddings, *, clean=False):
        from sealed.infrastructure.embedding_store import EmbeddingStore

        return EncodeCardsUseCase().execute(
            EncodeCardsConfig(cards_path=cards, clean=clean, embeddings_path=embeddings),
            _mock_encoder(np.ones(4, dtype=np.float32)), EmbeddingStore(),
        )

    def test_the_embeddings_mirror_the_cards_tree_elsewhere(self, tmp_path):
        cards, embeddings = self._trees(tmp_path)
        result = self._run(cards, embeddings)
        assert result.processed == 2
        assert sorted(p.relative_to(embeddings).as_posix()
                      for p in embeddings.rglob("*.npz")) == ["a/card_a.npz", "b/card_b.npz"]
        assert not list(cards.rglob("*.npz"))

    def test_a_card_already_encoded_there_is_skipped(self, tmp_path):
        cards, embeddings = self._trees(tmp_path)
        self._run(cards, embeddings)
        assert self._run(cards, embeddings).skipped == 2

    def test_an_embedding_beside_the_cards_does_not_count_as_encoded(self, tmp_path):
        cards, embeddings = self._trees(tmp_path)
        np.savez_compressed(cards / "a" / "card_a.npz", embedding=np.zeros(4, dtype=np.float32))
        assert self._run(cards, embeddings).processed == 2

    def test_clean_deletes_only_the_embeddings_tree(self, tmp_path):
        cards, embeddings = self._trees(tmp_path)
        beside = cards / "a" / "card_a.npz"
        np.savez_compressed(beside, embedding=np.zeros(4, dtype=np.float32))
        self._run(cards, embeddings)
        assert self._run(cards, embeddings, clean=True).processed == 2
        assert beside.exists()

    def test_without_one_the_embeddings_go_beside_the_cards(self, tmp_path):
        cards, _ = self._trees(tmp_path)
        self._run(cards, None)
        assert sorted(p.relative_to(cards).as_posix()
                      for p in cards.rglob("*.npz")) == ["a/card_a.npz", "b/card_b.npz"]


class TestEncodeCardsScorerCheckpointHappyPath:
    """T035 (US2): with a Phase B scorer checkpoint, encode-cards constructs a
    fresh CardPriceTransformerModel from `encoder_config`, loads weights from
    `encoder_state_dict`, and re-encodes every `.txt` (including never-seen
    cards) into a sibling `.npz` (FR-014, FR-015, US2 acceptance scenario 3)."""

    def test_uses_encoder_config_to_construct_model(self, tmp_path):
        """When --scorer-checkpoint is set, the encoder's TransformerConfig is
        rebuilt from the checkpoint's `encoder_config` (no dependency on the
        price-predictor `latest.pt` being present/unchanged)."""
        from argparse import Namespace
        from dataclasses import asdict

        import torch

        from price_predictor.domain.entities import TransformerConfig
        from price_predictor.infrastructure.tokenizer_store import save_vocabulary
        from price_predictor.infrastructure.transformer_model import (
            CardPriceTransformerModel,
        )
        from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
        from sealed.infrastructure.cli_encode_cards import run_encode_cards
        from sealed.infrastructure.scorer_store import ScorerStore

        encoder_cfg = TransformerConfig(
            d_model=16, n_layers=1, n_heads=2, ff_dim=32,
            max_seq_len=8, vocab_size=32, dropout=0.0,
        )
        encoder = CardPriceTransformerModel(encoder_cfg)

        scorer_cfg = ScorerConfig()
        scorer = SetTransformerScorer(scorer_cfg)
        opt = torch.optim.AdamW(scorer.parameters(), lr=1e-3)
        ckpt = tmp_path / "phase_b.pt"
        ScorerStore().save_checkpoint(
            scorer, opt, epoch=0, best_val_accuracy=0.5,
            config=scorer_cfg, path=ckpt,
            encoder_state_dict=encoder.state_dict(),
            encoder_config=asdict(encoder_cfg),
            train_config={},
        )
        vocab_path = tmp_path / "vocab.txt"
        save_vocabulary({"[PAD]": 0, "[UNK]": 1, "type": 2, "creature": 3}, vocab_path)

        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        letter = cards_dir / "n"
        letter.mkdir()
        # Card unseen by training (US2 scenario 3 — re-encoded too).
        (letter / "never_seen.txt").write_text(
            "name: never_seen\ntype: creature\n", encoding="utf-8",
        )

        ns = Namespace(
            encoder_checkpoint=None,
            scorer_checkpoint=str(ckpt),
            vocab_path=str(vocab_path),
            cards_path=str(cards_dir),
            embeddings_path=None,
            clean=True,
        )
        rc = run_encode_cards(ns)
        assert rc == 0
        npz = letter / "never_seen.npz"
        assert npz.exists()
        import numpy as np
        with np.load(npz) as f:
            arr = f["embedding"]
        # Shape = 2 * d_model + FEATURE_COUNT (32) = 64
        assert arr.shape == (2 * encoder_cfg.d_model + 32,)
