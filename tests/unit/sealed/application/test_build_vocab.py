"""Unit tests for the sealed build-vocab wrapper (FR-007/FR-008/FR-009)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sealed.application.build_vocab import (
    BuildVocabConfig,
    EmptyCardsFolderError,
)
from sealed.application.build_vocab import (
    run as run_build_vocab,
)


def _config(tmp_path: Path, **fields) -> BuildVocabConfig:
    """A config whose printings file is absent: the default would parse the
    real 600 MB ``resources/AllPrintings.json``, about 5 GB in memory."""
    return BuildVocabConfig(printings_path=tmp_path / "no-printings.json", **fields)


def _seed_corpus(folder: Path, lines: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(lines):
        (folder / f"card_{i}.txt").write_text(line, encoding="utf-8")


_TWO_CARDS = [
    "name: lightning bolt\nmana cost: {R}\ntypes: instant\n"
    "spell[1]: cardname deals 3 damage to any target.\n",
    "name: grizzly bears\nmana cost: {1}{G}\ntypes: creature\n",
]


def _build(tmp_path: Path, corpus: list[str], target: Path, target_size: int) -> list[str]:
    """Seed ``corpus``, build the vocabulary into ``target``, return its tokens."""
    cards = tmp_path / "cardsfolder"
    _seed_corpus(cards, corpus)
    run_build_vocab(_config(
        tmp_path, cards_folder=cards, vocab_path=target, target_size=target_size,
    ))
    return target.read_text(encoding="utf-8").splitlines()


class TestBuildVocab:
    def test_writes_vocab_to_sealed_path(self, tmp_path: Path):
        target = tmp_path / "models" / "sealed" / "encoder" / "vocab.txt"
        lines = _build(tmp_path, _TWO_CARDS, target, 5000)
        assert lines[:3] == ["[PAD]", "[UNK]", "cardname"]

    def test_emitted_file_contains_mask_token(self, tmp_path: Path):
        """FR-009a: the sealed build-vocab wrapper emits a vocab file
        whose tokens include ``[MASK]`` (the reserved MLM token used by
        ``train-encoder``).
        """
        tokens = _build(tmp_path, _TWO_CARDS, tmp_path / "vocab.txt", 5000)
        assert "[MASK]" in tokens

    def test_does_not_modify_price_side_vocab(self, tmp_path: Path):
        cards = tmp_path / "cardsfolder"
        _seed_corpus(cards, ["name: lightning bolt\ntypes: instant\n"])
        price_path = tmp_path / "price-vocab.txt"
        sentinel = "DO_NOT_TOUCH_THIS\n"
        price_path.write_text(sentinel, encoding="utf-8")

        sealed_path = tmp_path / "sealed-vocab.txt"
        run_build_vocab(_config(
            tmp_path, cards_folder=cards, vocab_path=sealed_path, target_size=5000,
        ))
        # Price-side file untouched (FR-008).
        assert price_path.read_text(encoding="utf-8") == sentinel

    def test_empty_folder_raises(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(EmptyCardsFolderError):
            run_build_vocab(_config(
                tmp_path,
                cards_folder=empty,
                vocab_path=tmp_path / "vocab.txt",
                target_size=5000,
            ))

    def test_target_size_below_seeds_raises(self, tmp_path: Path):
        cards = tmp_path / "cardsfolder"
        _seed_corpus(cards, ["name: lightning bolt\ntypes: instant\n"])
        with pytest.raises(ValueError, match="--target-size"):
            run_build_vocab(_config(
                tmp_path,
                cards_folder=cards,
                vocab_path=tmp_path / "vocab.txt",
                target_size=2,
            ))

    def test_truncation_preserves_specials(self, tmp_path: Path):
        corpus = [
            "name: lightning bolt\nmana cost: {R}\ntypes: instant\n"
            "spell[1]: cardname deals damage to any target.\n",
            "name: grizzly bears\nmana cost: {1}{G}\ntypes: creature\n"
            "p/t: 2/2\n",
        ]
        lines = _build(tmp_path, corpus, tmp_path / "vocab.txt", 100)
        assert lines[:3] == ["[PAD]", "[UNK]", "cardname"]
        assert len(lines) <= 100
