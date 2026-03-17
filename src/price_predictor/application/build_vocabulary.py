"""Build MTG domain vocabulary from a converted card corpus.

Corpus results (full ./output/ at freq_threshold=5, 2026-03-17):
  vocab_size=5064, coverage_pct=98.4%, unk_pct=1.6%
  SC-001 PASS (5064 < 10000), SC-002 PASS (98.4% >= 95%)
"""

from __future__ import annotations

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

_MANA_SYMBOL_PATTERN = re.compile(r"\{[^}]+\}")


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
) -> VocabBuildResult:
    """Build a compact MTG domain vocabulary from the converted card corpus.

    Token ID assignment order:
      0: [PAD]
      1: [UNK]
      2: cardname
      3+: fixed domain terms (zones, colors, multi-word keywords) — alphabetical
      then: corpus-frequency tokens (desc freq, alpha tie-break)
      then: any remaining mana symbols not yet included

    Args:
        cards_path: Directory containing converted card .txt files.
        freq_threshold: Minimum corpus frequency for a word to be included.

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

    domain_token_count = len(vocab)

    # 4. Scan corpus
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

    # 5. Add corpus tokens meeting freq_threshold (sorted by desc freq, alpha tie-break)
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
