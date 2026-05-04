"""``build-vocab``: thin sealed-side wrapper around the price-predictor
``build_vocabulary`` utility.

Per FR-008, the sealed vocabulary file lives at
``models/sealed/encoder/vocab.txt`` and is independent from
``models/price-predictor/transformer/vocab.txt``: writing one MUST NOT
modify the other. Per FR-009 / Decision D-1, ``--target-size`` is a
post-hoc truncation of the corpus-frequency vocab; seeded special and
domain tokens are always preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from price_predictor.application.build_vocabulary import build_vocabulary
from price_predictor.infrastructure.tokenizer_store import save_vocabulary


@dataclass
class BuildVocabConfig:
    cards_folder: Path = field(
        default_factory=lambda: Path("output/cardsfolder/"),
    )
    vocab_path: Path = field(
        default_factory=lambda: Path("models/sealed/encoder/vocab.txt"),
    )
    target_size: int = 5000
    printings_path: Path = field(
        default_factory=lambda: Path("resources/AllPrintings.json"),
    )


class EmptyCardsFolderError(RuntimeError):
    """Raised when the cards-folder has no ``*.txt`` files. Maps to exit code 1."""


def _truncate_to_target_size(
    vocab: dict[str, int], domain_token_count: int, target_size: int,
) -> dict[str, int]:
    """Keep the seeded specials + domain tokens, then drop the long tail of
    corpus-frequency tokens beyond ``target_size``.

    The price-side ``build_vocabulary`` orders entries (special, domain,
    set-code fragments, frequency-sorted corpus tokens, leftover mana
    symbols) so truncating from the tail preserves the seeded slots.
    """
    if target_size < domain_token_count:
        raise ValueError(
            f"--target-size ({target_size}) is smaller than the seeded "
            f"token count ({domain_token_count}); seeded tokens are always "
            "preserved per Decision D-1."
        )
    if len(vocab) <= target_size:
        return vocab
    items = sorted(vocab.items(), key=lambda kv: kv[1])[:target_size]
    return {tok: i for i, (tok, _) in enumerate(items)}


def run(config: BuildVocabConfig) -> int:
    """Execute the build-vocab pipeline. Returns the resulting vocab size."""
    cards_folder = Path(config.cards_folder)
    if not cards_folder.is_dir() or not any(cards_folder.rglob("*.txt")):
        raise EmptyCardsFolderError(
            f"Cards folder is empty: {cards_folder}. "
            "Run python -m price_predictor convert first."
        )

    printings_path = (
        config.printings_path if config.printings_path.exists() else None
    )

    result = build_vocabulary(
        cards_path=cards_folder,
        freq_threshold=2,
        printings_path=printings_path,
    )

    truncated = _truncate_to_target_size(
        result.vocab, result.domain_token_count, config.target_size,
    )

    config.vocab_path.parent.mkdir(parents=True, exist_ok=True)
    save_vocabulary(truncated, config.vocab_path)

    coverage = result.coverage_pct if len(truncated) == len(result.vocab) else None
    coverage_str = f" (corpus coverage: {coverage}%)" if coverage is not None else ""
    print(f"Wrote {len(truncated)} tokens to {config.vocab_path}{coverage_str}")
    return len(truncated)
