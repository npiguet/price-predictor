"""Build MTG domain vocabulary from a converted card corpus.

Corpus results (full ./output/ at freq_threshold=5, 2026-03-17, pre-enrichment-fix):
  vocab_size=5064, coverage_pct=98.4%, unk_pct=1.6%
  SC-001 PASS (5064 < 10000), SC-002 PASS (98.4% >= 95%)

Note: vocab was rebuilt after adding printing-data fixed-domain terms (rarity,
format names, field names) which were missing from the raw card corpus but are
present in enriched training texts.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

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

# Printing-data field names and values added by card_enrichment.py at training
# time. These never appear in raw converted card texts, so they would always be
# UNK without explicit seeding — causing rarity, reserved list, and legality
# information to be completely invisible to the model.
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


def _selective_normalize(text: str) -> str:
    """Lowercase non-brace-enclosed text; keep mana symbols as-is."""
    parts = re.split(r"(\{[^}]+\})", text)
    return "".join(p if p.startswith("{") else p.lower() for p in parts)


def _replace_multi_word_keywords(text: str) -> str:
    """Replace multi-word keyword display forms with underscore form.

    Sorted longest-first to handle overlaps (e.g. "double strike" before "strike").
    """
    sorted_kws = sorted(MULTI_WORD_KEYWORDS, key=len, reverse=True)
    for kw in sorted_kws:
        display = kw.replace("_", " ")
        text = text.replace(display, kw)
    return text


def _tokenize_text(text: str) -> list[str]:
    """Full tokenization pipeline: normalize → replace keywords → split."""
    text = _selective_normalize(text)
    text = _replace_multi_word_keywords(text)
    return re.findall(r"[a-z_]+|\{[^}]+\}|\d+|[^\s\w]", text)


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

    def _add(token: str) -> None:
        if token not in vocab:
            vocab[token] = len(vocab)

    # 1. Special tokens
    _add("[PAD]")
    _add("[UNK]")

    # 2. Structural placeholder
    _add("cardname")

    # 3. Fixed domain terms — alphabetical within each group
    for zone in sorted(_GAME_ZONES):
        _add(zone)

    for color in sorted(_COLOR_NAMES):
        _add(color)

    for kw in sorted(MULTI_WORD_KEYWORDS):
        _add(kw)

    for term in sorted(_PRINTING_DATA_TERMS):
        _add(term)

    # 4. Set-code letter fragments from AllPrintings (optional)
    if printings_path is not None:
        for token in _extract_set_code_tokens(printings_path):
            _add(token)

    domain_token_count = len(vocab)

    # 5. Scan card corpus
    txt_files = list(cards_path.rglob("*.txt"))

    token_counts: Counter[str] = Counter()
    total_token_occurrences = 0

    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding="utf-8")
        except OSError:
            continue
        tokens = _tokenize_text(text)
        token_counts.update(tokens)
        total_token_occurrences += len(tokens)

    # 6. Add corpus tokens meeting freq_threshold (sorted by desc freq, alpha tie-break)
    freq_tokens = [
        (token, count)
        for token, count in token_counts.items()
        if count >= freq_threshold and token not in vocab
        and not _MANA_SYMBOL_PATTERN.fullmatch(token)
    ]
    freq_tokens.sort(key=lambda x: (-x[1], x[0]))

    for token, _ in freq_tokens:
        _add(token)

    freq_threshold_token_count = len(vocab) - domain_token_count

    # 6. Add all mana symbols not yet in vocab (regardless of frequency)
    for token in token_counts:
        if _MANA_SYMBOL_PATTERN.fullmatch(token) and token not in vocab:
            _add(token)

    # 7. Compute coverage statistics
    if total_token_occurrences > 0:
        covered = sum(
            count for token, count in token_counts.items() if token in vocab
        )
        coverage_pct = round(covered / total_token_occurrences * 100.0, 1)
    else:
        coverage_pct = 100.0

    unk_pct = round(100.0 - coverage_pct, 1)

    return VocabBuildResult(
        vocab=vocab,
        vocab_size=len(vocab),
        domain_token_count=domain_token_count,
        freq_threshold_token_count=freq_threshold_token_count,
        coverage_pct=coverage_pct,
        unk_pct=unk_pct,
    )
