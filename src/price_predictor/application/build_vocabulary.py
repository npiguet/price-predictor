"""Build MTG domain vocabulary from a converted card corpus.

Corpus results (full ./output/ at freq_threshold=5, 2026-03-17):
  vocab_size=2451, coverage_pct=99.5%, unk_pct=0.5%
  SC-001 PASS (2451 < 10000), SC-002 PASS (99.5% >= 95%)

name: lines are stripped before frequency counting (mirroring MtgTokenizer)
so card-name proper nouns never enter the vocab. This pruned ~3,200 slots
vs earlier builds that included name-line tokens.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from price_predictor.domain.tokenizer import MtgTokenizer

# Multi-word keywords derived from forge.game.keyword.Keyword enum.
# Entries whose display name contains a space, normalized to:
# lowercase, apostrophes removed, spaces replaced with underscores.
MULTI_WORD_KEYWORDS: tuple[str, ...] = (
    "aura_swap",
    "bands_with_other",
    "battle_cry",
    "choose_a_background",
    "cumulative_upkeep",
    "doctors_companion",
    "double_agenda",
    "double_strike",
    "double_team",
    "first_strike",
    "for_mirrodin",
    "hidden_agenda",
    "job_select",
    "level_up",
    "living_metal",
    "living_weapon",
    "more_than_meets_the_eye",
    "partner_with",
    "read_ahead",
    "space_sculptor",
    "split_second",
    "start_your_engines",
    "starting_intensity",
    "umbra_armor",
)

# Fixed domain terms seeded into every vocabulary regardless of corpus frequency.
_GAME_ZONES: tuple[str, ...] = (
    "battlefield",
    "exile",
    "graveyard",
    "hand",
    "library",
    "stack",
    "command",
    "zone",
)

_COLOR_NAMES: tuple[str, ...] = (
    "black",
    "blue",
    "colorless",
    "green",
    "red",
    "white",
)

# Printing-data field names and values fed to the model as side-channel
# metadata at training time. These never appear in raw converted card texts,
# so they would always be UNK without explicit seeding — causing rarity,
# reserved list, and legality information to be completely invisible.
_PRINTING_DATA_TERMS: tuple[str, ...] = (
    # Boolean values (used by reserved:)
    "false",
    "true",
    # Rarity values
    "common",
    "mythic",
    "rare",
    "uncommon",
    # Enrichment field names
    "abu",
    "legalities",
    "printings",
    "rarity",
    "reserved",
    # Format names (used in legalities: line)
    "brawl",
    "commander",
    "legacy",
    "modern",
    "oathbreaker",
    "pauper",
    "penny",
    "pioneer",
    "standard",
    "vintage",
    # Mana cost sentinel: cards with no mana cost line get "mana cost: none"
    # inserted by MtgTokenizer so lands are distinguishable from {0}-cost cards.
    "none",
)

_MANA_SYMBOL_PATTERN = re.compile(r"\{[^}]+\}")
_LETTER_FRAGMENT_PATTERN = re.compile(r"[a-z]+")


def _extract_set_code_tokens(printings_path: Path) -> list[str]:
    """Extract alphabetic token fragments from all set codes in AllPrintings.json.

    Set codes tokenize as letter sequences: "2XM" → ["xm"], "ELD" → ["eld"],
    "M21" → ["m"]. Digit sequences are already well-covered by mana costs in
    the corpus and are skipped here.

    Returns a sorted, deduplicated list of letter-sequence tokens.
    """
    with open(printings_path, encoding="utf-8") as f:
        data = json.load(f)
    tokens: set[str] = set()
    for code in data.get("data", {}).keys():
        for fragment in _LETTER_FRAGMENT_PATTERN.findall(code.lower()):
            tokens.add(fragment)
    return sorted(tokens)


@dataclass(frozen=True)
class VocabBuildResult:
    """Result returned by build_vocabulary()."""

    vocab: dict[str, int]
    vocab_size: int
    domain_token_count: int
    freq_threshold_token_count: int
    coverage_pct: float
    unk_pct: float


def _bootstrap_tokenizer() -> MtgTokenizer:
    """Build a tokenizer suitable for vocabulary scanning.

    The token IDs are irrelevant here — only the tokenize() output is used.
    Multi-word keywords must be present in the vocab so that MtgTokenizer's
    longest-first replacement step recognizes them.
    """
    bootstrap_vocab: dict[str, int] = {
        MtgTokenizer.PAD: MtgTokenizer.PAD_ID,
        MtgTokenizer.UNK: MtgTokenizer.UNK_ID,
    }
    for kw in MULTI_WORD_KEYWORDS:
        bootstrap_vocab[kw] = len(bootstrap_vocab)
    return MtgTokenizer(bootstrap_vocab)


def _add_token(vocab: dict[str, int], token: str) -> None:
    if token not in vocab:
        vocab[token] = len(vocab)


def _seed_special_tokens(vocab: dict[str, int]) -> None:
    """Seed [PAD], [UNK], and the cardname placeholder."""
    _add_token(vocab, "[PAD]")
    _add_token(vocab, "[UNK]")
    _add_token(vocab, "cardname")


def _seed_domain_terms(vocab: dict[str, int]) -> None:
    """Seed game zones, color names, multi-word keywords, and printing-data terms."""
    for zone in sorted(_GAME_ZONES):
        _add_token(vocab, zone)
    for color in sorted(_COLOR_NAMES):
        _add_token(vocab, color)
    for kw in sorted(MULTI_WORD_KEYWORDS):
        _add_token(vocab, kw)
    for term in sorted(_PRINTING_DATA_TERMS):
        _add_token(vocab, term)


def _seed_set_codes(vocab: dict[str, int], printings_path: Path | None) -> None:
    """Seed alphabetic fragments of every MTGJSON set code, when available."""
    if printings_path is None:
        return
    for token in _extract_set_code_tokens(printings_path):
        _add_token(vocab, token)


def _scan_corpus_frequencies(
    cards_path: Path, tokenizer: MtgTokenizer
) -> tuple[Counter[str], int]:
    """Walk the corpus and return (token_counts, total_occurrences)."""
    token_counts: Counter[str] = Counter()
    total_token_occurrences = 0
    for txt_file in cards_path.rglob("*.txt"):
        try:
            text = txt_file.read_text(encoding="utf-8")
        except OSError:
            continue
        tokens = tokenizer.tokenize(text)
        token_counts.update(tokens)
        total_token_occurrences += len(tokens)
    return token_counts, total_token_occurrences


def _add_freq_tokens(
    vocab: dict[str, int], counts: Counter[str], freq_threshold: int
) -> None:
    """Add corpus tokens meeting freq_threshold, sorted desc by freq then alpha."""
    freq_tokens = [
        (token, count)
        for token, count in counts.items()
        if count >= freq_threshold
        and token not in vocab
        and not _MANA_SYMBOL_PATTERN.fullmatch(token)
    ]
    freq_tokens.sort(key=lambda x: (-x[1], x[0]))
    for token, _ in freq_tokens:
        _add_token(vocab, token)


def _add_remaining_mana_symbols(vocab: dict[str, int], counts: Counter[str]) -> None:
    """Add every mana symbol seen in the corpus that isn't already in the vocab."""
    for token in counts:
        if _MANA_SYMBOL_PATTERN.fullmatch(token) and token not in vocab:
            _add_token(vocab, token)


def _compute_coverage(
    counts: Counter[str], vocab: dict[str, int], total_occurrences: int
) -> tuple[float, float]:
    """Return (coverage_pct, unk_pct) of corpus tokens captured by the vocab."""
    if total_occurrences == 0:
        return 100.0, 0.0
    covered = sum(count for token, count in counts.items() if token in vocab)
    coverage_pct = round(covered / total_occurrences * 100.0, 1)
    return coverage_pct, round(100.0 - coverage_pct, 1)


def build_vocabulary(
    cards_path: Path,
    freq_threshold: int = 5,
    printings_path: Path | None = None,
) -> VocabBuildResult:
    """Build a compact MTG domain vocabulary from the converted card corpus.

    Token ID assignment order:
      0: [PAD]
      1: [UNK]
      2: cardname
      3+: fixed domain terms (zones, colors, multi-word keywords,
          printing-data terms) — alphabetical within each group
      then: set-code letter fragments from AllPrintings (if printings_path given)
      then: corpus-frequency tokens (desc freq, alpha tie-break)
      then: any remaining mana symbols not yet included

    Args:
        cards_path: Directory containing converted card .txt files.
        freq_threshold: Minimum corpus frequency for a word to be included.
        printings_path: Optional path to AllPrintings.json. When supplied,
            alphabetic fragments of every set code are seeded into the fixed
            domain section so set codes in enriched training texts are never UNK.

    Returns:
        VocabBuildResult with vocab dict and coverage statistics.
    """
    vocab: dict[str, int] = {}

    _seed_special_tokens(vocab)
    _seed_domain_terms(vocab)
    _seed_set_codes(vocab, printings_path)
    domain_token_count = len(vocab)

    counts, total_occurrences = _scan_corpus_frequencies(
        cards_path, _bootstrap_tokenizer(),
    )

    _add_freq_tokens(vocab, counts, freq_threshold)
    freq_threshold_token_count = len(vocab) - domain_token_count

    _add_remaining_mana_symbols(vocab, counts)

    coverage_pct, unk_pct = _compute_coverage(counts, vocab, total_occurrences)

    return VocabBuildResult(
        vocab=vocab,
        vocab_size=len(vocab),
        domain_token_count=domain_token_count,
        freq_threshold_token_count=freq_threshold_token_count,
        coverage_pct=coverage_pct,
        unk_pct=unk_pct,
    )
