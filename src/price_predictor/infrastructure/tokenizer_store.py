"""Infrastructure: load and save MtgTokenizer vocabulary files."""

from __future__ import annotations

from pathlib import Path

from price_predictor.domain.tokenizer import MtgTokenizer


def save_vocabulary(vocab: dict[str, int], path: Path) -> None:
    """Write vocab.txt: one token per line, line number = token ID.

    Tokens are sorted by their ID value before writing. The file is
    UTF-8 encoded with no trailing whitespace or blank lines.
    """
    tokens_by_id = sorted(vocab.items(), key=lambda x: x[1])
    lines = [token for token, _ in tokens_by_id]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_vocabulary(vocab_path: Path) -> dict[str, int]:
    """Load a vocab.txt file into a ``{token: id}`` dict.

    Validates that ``[PAD]`` is at line 0 and ``[UNK]`` is at line 1.
    Returns the dict so callers (e.g. the sealed train-encoder) can
    resolve the IDs of reserved tokens such as ``[MASK]`` without
    constructing a tokenizer.
    """
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    lines = vocab_path.read_text(encoding="utf-8").splitlines()

    if not lines or lines[0] != MtgTokenizer.PAD:
        got = lines[0] if lines else "empty file"
        raise ValueError(
            f"vocab.txt must have '[PAD]' at line 0, got: {got!r}"
        )
    if len(lines) < 2 or lines[1] != MtgTokenizer.UNK:
        got = lines[1] if len(lines) > 1 else "missing"
        raise ValueError(
            f"vocab.txt must have '[UNK]' at line 1, got: {got!r}"
        )

    return {token: i for i, token in enumerate(lines)}


def load_tokenizer(vocab_path: Path) -> MtgTokenizer:
    """Load an MtgTokenizer from a vocab.txt file."""
    return MtgTokenizer(load_vocabulary(vocab_path))
