"""``train-encoder``: train a sealed encoder + nine regression heads + an
MLM head on per-card winnability and an auxiliary mask-prediction
objective.

The training loop reads ``cards-played.txt`` twice (primary + per-color
slices), computes nine raw and shrunk per-card labels per FR-011, splits
cards into train/val with FR-018 stratification, trains the dual-pool
encoder + regression heads + MLM head jointly under FR-017's weighted
MSE + cross-entropy objective, and saves only encoder weights to
``models/sealed/encoder/``.

Per FR-013, aggregation is inline: there is no separate
``aggregate-labels`` subcommand. The human-readable label snapshot lives
at ``output/sealed/cards-win-rates.txt``.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.infrastructure.tokenizer_store import (
    load_tokenizer,
    load_vocabulary,
)
from sealed.domain.encoder_model import (
    COLOR_ORDER,
    SealedEncoderConfig,
    SealedEncoderModel,
)
from sealed.infrastructure.cards_played_reader import (
    CardsPlayedRow,
    iter_rows,
)
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.encoder_store import SealedEncoderStore

# ── Hardcoded constants (FR-022) ────────────────────────────────────────
D_MODEL: int = 256
FF_DIM: int = 1024
VAL_FRACTION: float = 0.2
RANDOM_SEED: int = 42
_MAX_SEQ_LEN_MULTIPLE: int = 8
_MIN_MAX_SEQ_LEN: int = 64
_WIN_RATES_FILENAME: str = "cards-win-rates.txt"
_MASK_TOKEN: str = "[MASK]"
_PAD_TOKEN: str = "[PAD]"
_UNK_TOKEN: str = "[UNK]"
_CARDNAME_TOKEN: str = "cardname"
_MASKED_SPECIAL_TOKENS: tuple[str, ...] = (
    _PAD_TOKEN, _UNK_TOKEN, _CARDNAME_TOKEN, _MASK_TOKEN,
)

# Head ordering — tensor index → head name. Matches the column order in
# ``cards-win-rates.txt`` and the FR-017 / FR-017a head list.
_SIGNED_HEAD_NAMES: tuple[str, ...] = (
    "score_play", "score_draw", "played_rate", "cast_lift",
)
_COLOR_LIFT_HEAD_NAMES: tuple[str, ...] = tuple(
    f"color_lift_{c}" for c in COLOR_ORDER
)
_ALL_HEAD_NAMES: tuple[str, ...] = _SIGNED_HEAD_NAMES + _COLOR_LIFT_HEAD_NAMES
N_HEADS: int = len(_ALL_HEAD_NAMES)  # 9
_COLOR_LIFT_OFFSET: int = len(_SIGNED_HEAD_NAMES)  # 4

# Per FR-017, the four signed heads sum unweighted into L_reg; the five
# color_lift heads sum with a 1/5 prefactor. _HEAD_LOSS_WEIGHT[h] gives
# the prefactor used when combining head h's weighted-MSE term into L_reg.
_HEAD_LOSS_WEIGHT: tuple[float, ...] = (
    1.0, 1.0, 1.0, 1.0,           # score_play / score_draw / played_rate / cast_lift
    0.2, 0.2, 0.2, 0.2, 0.2,      # color_lift_W..G
)


def _log(message: str) -> None:
    """Print a timestamped progress line to stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# ── Per-card counters and labels (replaces v1 CardLabel) ────────────────


def _empty_color_dict() -> dict[str, int]:
    return {color: 0 for color in COLOR_ORDER}


@dataclass
class CardCounters:
    """All raw counters maintained per card across the two aggregation passes.

    Pass 1 fills the four primary counters and the four ``@play`` subset
    counters; pass 2 fills the four per-color counter dicts (one entry
    per WUBRG letter). The ``@draw`` counterparts are derivable by
    subtraction; the cast-lift "not played" counterparts are derivable
    from ``wins_when_in_deck - wins_when_played`` (and the loss analog).
    """

    wins_when_played: int = 0
    losses_when_played: int = 0
    wins_when_in_deck: int = 0
    losses_when_in_deck: int = 0
    wins_when_played_at_play: int = 0
    losses_when_played_at_play: int = 0
    wins_when_in_deck_at_play: int = 0
    losses_when_in_deck_at_play: int = 0
    wins_when_played_with: dict[str, int] = field(default_factory=_empty_color_dict)
    losses_when_played_with: dict[str, int] = field(default_factory=_empty_color_dict)
    wins_when_in_deck_with: dict[str, int] = field(default_factory=_empty_color_dict)
    losses_when_in_deck_with: dict[str, int] = field(default_factory=_empty_color_dict)


@dataclass(frozen=True)
class CardLabels:
    """Nine raw + nine shrunk labels per card.

    ``None`` cells signal that the head's slice denominator was zero
    (FR-012). Such cells are written as the empty string in
    ``cards-win-rates.txt`` and contribute zero loss at training time
    via the per-card head_mask. ``raw_color_lift`` and
    ``shrunk_color_lift`` are dicts keyed by WUBRG letter so the table
    layout stays explicit; mapping to tensor positions happens in
    ``_WinnabilityDataset``.
    """

    card_name: str
    counters: CardCounters
    raw_score_play: float | None
    shrunk_score_play: float | None
    raw_score_draw: float | None
    shrunk_score_draw: float | None
    raw_played_rate: float | None
    shrunk_played_rate: float | None
    raw_cast_lift: float | None
    shrunk_cast_lift: float | None
    raw_color_lift: dict[str, float | None]
    shrunk_color_lift: dict[str, float | None]


CardLabelMap = dict[str, CardLabels]


# ── Color extraction helper (FR-010b, Decision D-16) ────────────────────

_MANA_SYMBOL_PATTERN = re.compile(r"\{[^}]+\}")
_COLOR_LETTERS: frozenset[str] = frozenset(COLOR_ORDER)


def _colors_from_mana_cost(line: str | None) -> set[str]:
    """Return the WUBRG colors present in a converted ``mana cost:`` line.

    Hybrid (``{W/U}``), Phyrexian (``{W/P}``), and mono-hybrid (``{2/W}``)
    symbols all contribute their color letter. Generic (``{2}``),
    colorless (``{C}``), and `X` (``{X}``) contribute nothing. Cards with
    no ``mana cost:`` line (``line is None``) contribute no colors.
    """
    if not line:
        return set()
    colors: set[str] = set()
    for symbol in _MANA_SYMBOL_PATTERN.findall(line):
        for ch in symbol.upper():
            if ch in _COLOR_LETTERS:
                colors.add(ch)
    return colors


# ── Aggregation pass 1 (primary + @play counters) ──────────────────────


def _record_pass_one(
    counters: dict[str, CardCounters],
    deck_played: list[str],
    deck_not_played: list[str],
    *,
    won: bool,
    at_play: bool,
) -> None:
    """Fold one side's deck into the per-card counter dict for pass 1.

    ``deck_played`` lists distinct card names that entered the
    battlefield/stack; ``deck_not_played`` lists distinct names that
    stayed in the deck. The two sets are disjoint per FR-004.
    """
    played_set = set(deck_played)
    in_deck_set = played_set | set(deck_not_played)
    for name in in_deck_set:
        c = counters.setdefault(name, CardCounters())
        if won:
            c.wins_when_in_deck += 1
            if name in played_set:
                c.wins_when_played += 1
            if at_play:
                c.wins_when_in_deck_at_play += 1
                if name in played_set:
                    c.wins_when_played_at_play += 1
        else:
            c.losses_when_in_deck += 1
            if name in played_set:
                c.losses_when_played += 1
            if at_play:
                c.losses_when_in_deck_at_play += 1
                if name in played_set:
                    c.losses_when_played_at_play += 1


def _aggregate_pass_one(
    rows: Iterable[CardsPlayedRow],
) -> dict[str, CardCounters]:
    """First streaming pass over ``cards-played.txt``.

    Fills the four primary and four ``@play`` counters on every card
    observed in either side's deck (FR-010a). The per-color counters in
    each ``CardCounters`` stay zero — pass 2 fills them after the corpus
    consistency check.
    """
    counters: dict[str, CardCounters] = {}
    for row in rows:
        a_won = row.winner == "A"
        a_at_play = row.starter == "A"
        _record_pass_one(
            counters,
            row.cards_played_a,
            row.cards_not_played_a,
            won=a_won,
            at_play=a_at_play,
        )
        _record_pass_one(
            counters,
            row.cards_played_b,
            row.cards_not_played_b,
            won=not a_won,
            at_play=not a_at_play,
        )
    return counters


# ── Corpus consistency check (FR-023d) ──────────────────────────────────

_MISSING_CARD_DISPLAY_CAP: int = 20


def _drop_missing_cards(
    counters: dict[str, CardCounters], locator: ConvertedCardLocator,
) -> int:
    """Run after pass 1, before pass 2.

    Cards referenced by ``cards-played.txt`` that have no converted
    ``.txt`` under the cards folder cannot be tokenized, so they're
    dropped from ``counters`` (and therefore from the label map, the
    train/val split, and the dataset). Missing cards are *reported* via
    a log warning naming up to ``_MISSING_CARD_DISPLAY_CAP`` of them
    plus the total — but they do not block the run. Returns the number
    of cards dropped.

    Cards with zero observations in both sides' decks won't be in
    ``counters`` at all (pass 1 only inserts on first sighting), so this
    only ever drops cards the model would otherwise have trained on.
    """
    missing = sorted(
        name for name in counters if locator.text_path(name) is None
    )
    if not missing:
        return 0
    for name in missing:
        del counters[name]
    head = missing[:_MISSING_CARD_DISPLAY_CAP]
    suffix = (
        f" (and {len(missing) - _MISSING_CARD_DISPLAY_CAP} more)"
        if len(missing) > _MISSING_CARD_DISPLAY_CAP else ""
    )
    _log(
        f"WARNING: {len(missing)} card(s) referenced by cards-played.txt "
        f"have no converted .txt under the cards folder; dropping them from "
        f"training. First {len(head)}: " + ", ".join(head) + suffix
        + ". Run python -m price_predictor convert to add them."
    )
    return len(missing)


# ── Aggregation pass 2 (per-color slices) ──────────────────────────────


def _record_pass_two_side(
    counters: dict[str, CardCounters],
    deck_played: list[str],
    deck_not_played: list[str],
    deck_colors: set[str],
    *,
    won: bool,
) -> None:
    """Fold one side's deck into the per-color counters for pass 2."""
    played_set = set(deck_played)
    in_deck_set = played_set | set(deck_not_played)
    for name in in_deck_set:
        c = counters.get(name)
        if c is None:
            continue
        for color in deck_colors:
            if won:
                c.wins_when_in_deck_with[color] += 1
                if name in played_set:
                    c.wins_when_played_with[color] += 1
            else:
                c.losses_when_in_deck_with[color] += 1
                if name in played_set:
                    c.losses_when_played_with[color] += 1


def _aggregate_pass_two(
    rows: Iterable[CardsPlayedRow],
    counters: dict[str, CardCounters],
    locator: ConvertedCardLocator,
) -> None:
    """Second streaming pass over ``cards-played.txt`` (FR-010b).

    For each card observed in pass 1, lazily resolve its color set from
    the converted card text's ``mana cost:`` line and cache the result.
    Each game's deck-color set is the union over its contents; per-color
    counters increment for every (card, color) pair where the card was
    in a deck running that color.
    """
    color_cache: dict[str, set[str]] = {}

    def colors_of(name: str) -> set[str]:
        cached = color_cache.get(name)
        if cached is not None:
            return cached
        converted = locator.load_text(name)
        cost_line = converted.mana_cost_line() if converted is not None else None
        result = _colors_from_mana_cost(cost_line)
        color_cache[name] = result
        return result

    for row in rows:
        a_won = row.winner == "A"
        deck_a = set(row.cards_played_a) | set(row.cards_not_played_a)
        deck_b = set(row.cards_played_b) | set(row.cards_not_played_b)
        colors_a: set[str] = set()
        for name in deck_a:
            colors_a |= colors_of(name)
        colors_b: set[str] = set()
        for name in deck_b:
            colors_b |= colors_of(name)
        _record_pass_two_side(
            counters, row.cards_played_a, row.cards_not_played_a,
            colors_a, won=a_won,
        )
        _record_pass_two_side(
            counters, row.cards_played_b, row.cards_not_played_b,
            colors_b, won=not a_won,
        )


# ── Label arithmetic (FR-011 / FR-012) ──────────────────────────────────


def _safe_div(num: float, denom: float) -> float | None:
    return num / denom if denom > 0 else None


def _build_label_map(
    counters: dict[str, CardCounters], shrinkage_k: float,
) -> CardLabelMap:
    """Compute 9 raw + 9 shrunk labels per card (FR-011, FR-012).

    Cards with zero total in-deck observations are excluded entirely.
    Cells whose slice denominator is zero are stored as ``None`` in
    both raw and shrunk columns; downstream code (the writer and the
    dataset) handles the empty cell as "no signal" and zeroes its
    loss contribution.
    """
    k = float(shrinkage_k)
    half_k = k / 2.0
    labels: CardLabelMap = {}
    for name, c in counters.items():
        in_deck_total = c.wins_when_in_deck + c.losses_when_in_deck
        if in_deck_total == 0:
            continue

        # ── score_play ─────────────────────────────────────────
        in_deck_play = c.wins_when_in_deck_at_play + c.losses_when_in_deck_at_play
        score_play_num = c.wins_when_played_at_play - c.losses_when_played_at_play
        raw_score_play = _safe_div(score_play_num, in_deck_play)
        shrunk_score_play = (
            None if in_deck_play == 0
            else score_play_num / (in_deck_play + k)
        )

        # ── score_draw (= primary - @play) ────────────────────
        wins_in_deck_draw = c.wins_when_in_deck - c.wins_when_in_deck_at_play
        losses_in_deck_draw = c.losses_when_in_deck - c.losses_when_in_deck_at_play
        wins_played_draw = c.wins_when_played - c.wins_when_played_at_play
        losses_played_draw = c.losses_when_played - c.losses_when_played_at_play
        in_deck_draw = wins_in_deck_draw + losses_in_deck_draw
        score_draw_num = wins_played_draw - losses_played_draw
        raw_score_draw = _safe_div(score_draw_num, in_deck_draw)
        shrunk_score_draw = (
            None if in_deck_draw == 0
            else score_draw_num / (in_deck_draw + k)
        )

        # ── played_rate ────────────────────────────────────────
        played_total = c.wins_when_played + c.losses_when_played
        raw_played_rate = played_total / in_deck_total
        shrunk_played_rate = (played_total + half_k) / (in_deck_total + k)

        # ── cast_lift ──────────────────────────────────────────
        wins_not_played = c.wins_when_in_deck - c.wins_when_played
        losses_not_played = c.losses_when_in_deck - c.losses_when_played
        not_played_total = wins_not_played + losses_not_played
        m = min(played_total, not_played_total)
        if m == 0:
            raw_cast_lift: float | None = None
            shrunk_cast_lift: float | None = None
        else:
            played_rate = c.wins_when_played / played_total
            not_played_rate = wins_not_played / not_played_total
            raw_cast_lift = played_rate - not_played_rate
            shrunk_played = (
                (c.wins_when_played + half_k) / (played_total + k)
            )
            shrunk_not_played = (
                (wins_not_played + half_k) / (not_played_total + k)
            )
            shrunk_cast_lift = shrunk_played - shrunk_not_played

        # ── color_lift_X ───────────────────────────────────────
        # Overall (raw and shrunk) signed score for the card.
        overall_num = c.wins_when_played - c.losses_when_played
        raw_overall = overall_num / in_deck_total
        shrunk_overall = overall_num / (in_deck_total + k)

        raw_color_lift: dict[str, float | None] = {}
        shrunk_color_lift: dict[str, float | None] = {}
        for color in COLOR_ORDER:
            in_deck_color = (
                c.wins_when_in_deck_with[color]
                + c.losses_when_in_deck_with[color]
            )
            played_with_color_num = (
                c.wins_when_played_with[color]
                - c.losses_when_played_with[color]
            )
            if in_deck_color == 0:
                raw_color_lift[color] = None
                shrunk_color_lift[color] = None
                continue
            raw_color_lift[color] = (
                played_with_color_num / in_deck_color - raw_overall
            )
            shrunk_color_lift[color] = (
                played_with_color_num / (in_deck_color + k) - shrunk_overall
            )

        labels[name] = CardLabels(
            card_name=name,
            counters=c,
            raw_score_play=raw_score_play,
            shrunk_score_play=shrunk_score_play,
            raw_score_draw=raw_score_draw,
            shrunk_score_draw=shrunk_score_draw,
            raw_played_rate=raw_played_rate,
            shrunk_played_rate=shrunk_played_rate,
            raw_cast_lift=raw_cast_lift,
            shrunk_cast_lift=shrunk_cast_lift,
            raw_color_lift=raw_color_lift,
            shrunk_color_lift=shrunk_color_lift,
        )
    return labels


# ── cards-win-rates.txt writer (FR-013a, Decision D-15) ────────────────


def _format_cell(value: float | None) -> str:
    return "" if value is None else f"{value:.5f}"


_WIN_RATES_HEADER_FIELDS: tuple[str, ...] = (
    "card_name",
    "wins_when_played", "wins_when_in_deck",
    "losses_when_played", "losses_when_in_deck",
    "raw_score_play", "shrunk_score_play",
    "raw_score_draw", "shrunk_score_draw",
    "raw_played_rate", "shrunk_played_rate",
    "raw_cast_lift", "shrunk_cast_lift",
    "raw_color_lift_W", "shrunk_color_lift_W",
    "raw_color_lift_U", "shrunk_color_lift_U",
    "raw_color_lift_B", "shrunk_color_lift_B",
    "raw_color_lift_R", "shrunk_color_lift_R",
    "raw_color_lift_G", "shrunk_color_lift_G",
)
_WIN_RATES_HEADER: str = ";".join(_WIN_RATES_HEADER_FIELDS)


def _sort_key(lbl: CardLabels) -> tuple[int, float, str]:
    """Sort by ``shrunk_score_play`` descending, empty cells at the end.

    Returned tuple is consumed with ``reverse=False``: the leading
    integer puts non-empty values first; the negated float reverses
    descending order; the trailing card name breaks ties stably.
    """
    if lbl.shrunk_score_play is None:
        return (1, 0.0, lbl.card_name)
    return (0, -lbl.shrunk_score_play, lbl.card_name)


def _write_win_rates(labels: CardLabelMap, path: Path) -> None:
    """Write the per-card label snapshot in the FR-013a 24-column schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(labels.values(), key=_sort_key)
    lines = [_WIN_RATES_HEADER]
    for lbl in rows:
        c = lbl.counters
        fields_out = [
            lbl.card_name,
            str(c.wins_when_played), str(c.wins_when_in_deck),
            str(c.losses_when_played), str(c.losses_when_in_deck),
            _format_cell(lbl.raw_score_play),
            _format_cell(lbl.shrunk_score_play),
            _format_cell(lbl.raw_score_draw),
            _format_cell(lbl.shrunk_score_draw),
            _format_cell(lbl.raw_played_rate),
            _format_cell(lbl.shrunk_played_rate),
            _format_cell(lbl.raw_cast_lift),
            _format_cell(lbl.shrunk_cast_lift),
        ]
        for color in COLOR_ORDER:
            fields_out.append(_format_cell(lbl.raw_color_lift[color]))
            fields_out.append(_format_cell(lbl.shrunk_color_lift[color]))
        lines.append(";".join(fields_out))
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Train/val split (FR-018, Decision D-4) ──────────────────────────────


def _stratification_value(lbl: CardLabels) -> float | None:
    """Return the FR-018 fallback-chain stratification value for a card.

    Order: ``score_play`` → ``score_draw`` → ``cast_lift`` →
    ``color_lift_W`` → ``color_lift_U`` → ``color_lift_B`` →
    ``color_lift_R`` → ``color_lift_G``. Returns ``None`` when every
    signed cell is empty (the card is bound for the catch-all stratum).
    """
    if lbl.shrunk_score_play is not None:
        return lbl.shrunk_score_play
    if lbl.shrunk_score_draw is not None:
        return lbl.shrunk_score_draw
    if lbl.shrunk_cast_lift is not None:
        return lbl.shrunk_cast_lift
    for color in COLOR_ORDER:
        v = lbl.shrunk_color_lift[color]
        if v is not None:
            return v
    return None


def _split_cards(
    labels: CardLabelMap,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    """Card-level disjoint split, stratified by ``score_play`` quartile
    with the FR-018 fallback chain for cards whose ``score_play`` cell
    is empty. Cards with no non-empty signed-head cell go into a single
    catch-all stratum.
    """
    if not labels:
        return [], []
    names = sorted(labels.keys())
    # Use NaN as the sentinel for "card has no signed-cell signal" — a
    # plain ``or`` on the FR-018 fallback value misroutes 0.0 (neutral
    # signed score) into the catch-all stratum.
    strat_raw: list[float] = []
    for n in names:
        v = _stratification_value(labels[n])
        strat_raw.append(float("nan") if v is None else v)
    strat_values = np.array(strat_raw, dtype=np.float64)
    catch_all_mask = np.isnan(strat_values)

    # Quartile-stratify the cards whose signed-cell value is non-NaN.
    bin_assignments = np.full(len(names), -1, dtype=np.int64)
    if not catch_all_mask.all():
        finite = strat_values[~catch_all_mask]
        edges = np.unique(np.quantile(finite, [0.25, 0.5, 0.75]))
        bin_assignments[~catch_all_mask] = np.digitize(finite, edges)

    rng = np.random.default_rng(seed)
    train: list[str] = []
    val: list[str] = []
    # Iterate every distinct stratum (each non-negative bin id, plus -1
    # for the catch-all). The deterministic iteration order is the
    # sorted set of bin ids — same `seed` always shuffles the same buckets.
    unique_bins = sorted(set(int(b) for b in bin_assignments.tolist()))
    for b in unique_bins:
        bucket_idxs = np.where(bin_assignments == b)[0]
        rng.shuffle(bucket_idxs)
        n_val = int(round(len(bucket_idxs) * val_fraction))
        val_set = set(bucket_idxs[:n_val].tolist())
        for idx in bucket_idxs:
            (val if idx in val_set else train).append(names[idx])
    # Small-corpus safety: if per-stratum rounding zeroed out the val
    # set entirely, transfer the population's expected total val
    # quota (round(N * val_fraction), at least 1) from train to val.
    # Pick from the largest stratum first so the kept-back cards
    # remain proportional to stratum size.
    if not val and len(train) > 1:
        target = max(1, int(round(len(names) * val_fraction)))
        # Sort train by stratum size descending so we steal from
        # bigger buckets first; keep determinism via the same RNG state.
        bucket_sizes = {b: int((bin_assignments == b).sum()) for b in unique_bins}
        train_with_bucket = [
            (n, bucket_sizes[int(bin_assignments[names.index(n)])])
            for n in train
        ]
        train_with_bucket.sort(key=lambda x: (-x[1], x[0]))
        moved = [n for n, _ in train_with_bucket[:target]]
        val.extend(moved)
        train = [n for n in train if n not in set(moved)]
    train.sort()
    val.sort()
    return train, val


# ── Per-card label / weight / mask tensors ──────────────────────────────


def _label_value(lbl: CardLabels, head_idx: int) -> float | None:
    if head_idx == 0:
        return lbl.shrunk_score_play
    if head_idx == 1:
        return lbl.shrunk_score_draw
    if head_idx == 2:
        return lbl.shrunk_played_rate
    if head_idx == 3:
        return lbl.shrunk_cast_lift
    color = COLOR_ORDER[head_idx - _COLOR_LIFT_OFFSET]
    return lbl.shrunk_color_lift[color]


def _per_head_weight(lbl: CardLabels, head_idx: int, k: float) -> float:
    """FR-017a per-head sample weight for one card.

    Returns 0.0 when the head's slice denominator is zero (which is
    exactly the condition that makes the cell empty per FR-012).
    """
    c = lbl.counters
    if head_idx == 0:
        n = c.wins_when_in_deck_at_play + c.losses_when_in_deck_at_play
    elif head_idx == 1:
        n = (
            (c.wins_when_in_deck - c.wins_when_in_deck_at_play)
            + (c.losses_when_in_deck - c.losses_when_in_deck_at_play)
        )
    elif head_idx == 2:
        n = c.wins_when_in_deck + c.losses_when_in_deck
    elif head_idx == 3:
        played_total = c.wins_when_played + c.losses_when_played
        in_deck_total = c.wins_when_in_deck + c.losses_when_in_deck
        n = min(played_total, in_deck_total - played_total)
    else:
        color = COLOR_ORDER[head_idx - _COLOR_LIFT_OFFSET]
        n = c.wins_when_in_deck_with[color] + c.losses_when_in_deck_with[color]
    if n <= 0:
        return 0.0
    return n / (n + k)


# ── Dataset ─────────────────────────────────────────────────────────────


class _WinnabilityDataset(Dataset):
    """Per-card training tuple for the multi-head + MLM encoder.

    ``__getitem__`` returns ``(input_ids, attention_mask, labels[9],
    weights[9], head_mask[9])``. Labels carry the shrunk-form value at
    each head position, with 0.0 substituted at empty cells; weights
    follow FR-017a; head_mask is 1.0 where the cell is non-empty and
    0.0 elsewhere. The card text feeds the encoder with the ``name:``
    line stripped per FR-014a.
    """

    def __init__(
        self,
        card_names: Sequence[str],
        labels: CardLabelMap,
        tokenizer: MtgTokenizer,
        locator: ConvertedCardLocator,
        max_seq_len: int,
        shrinkage_k: float,
    ) -> None:
        self._names = list(card_names)
        self._labels = labels
        self._tokenizer = tokenizer
        self._locator = locator
        self._max_seq_len = max_seq_len
        self._shrinkage_k = shrinkage_k

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(
        self, idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        name = self._names[idx]
        text = self._read_card_text(name)
        ids, mask = self._tokenizer.encode(text, self._max_seq_len)
        lbl = self._labels[name]
        labels_v = torch.zeros(N_HEADS, dtype=torch.float32)
        weights_v = torch.zeros(N_HEADS, dtype=torch.float32)
        head_mask_v = torch.zeros(N_HEADS, dtype=torch.float32)
        for h in range(N_HEADS):
            v = _label_value(lbl, h)
            if v is None:
                continue
            labels_v[h] = float(v)
            weights_v[h] = _per_head_weight(lbl, h, self._shrinkage_k)
            head_mask_v[h] = 1.0
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            labels_v,
            weights_v,
            head_mask_v,
        )

    def _read_card_text(self, name: str) -> str:
        converted = self._locator.load_text(name)
        if converted is None:
            raise FileNotFoundError(
                f"No converted card text on disk for {name!r}",
            )
        return converted.without_name_line()


def _measure_max_seq_len(
    card_names: Iterable[str],
    locator: ConvertedCardLocator,
    tokenizer: MtgTokenizer,
) -> int:
    """Longest tokenized card length, rounded up to a multiple of 8 (FR-022)."""
    max_len = 1
    for name in card_names:
        converted = locator.load_text(name)
        if converted is None:
            continue
        tokens = tokenizer.tokenize(converted.without_name_line())
        if len(tokens) > max_len:
            max_len = len(tokens)
    rounded = math.ceil(max_len / _MAX_SEQ_LEN_MULTIPLE) * _MAX_SEQ_LEN_MULTIPLE
    return max(rounded, _MIN_MAX_SEQ_LEN)


# ── MLM mask draw (FR-014b, Decision D-10) ──────────────────────────────


def _draw_mlm_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_prob: float,
    mask_token_id: int,
    special_token_ids: Iterable[int],
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw an MLM mask and produce the masked-input tensor.

    Returns ``(masked_ids, mask_positions)`` where ``mask_positions`` is
    a ``(B, T)`` boolean tensor marking the positions selected for
    cross-entropy loss. Eligible positions are real (``attention_mask
    == 1``) AND carry a non-special token id; selected positions are
    overwritten with ``mask_token_id`` in ``masked_ids``. The
    mask is drawn fresh on every call so the same card sees different
    masks across epochs (Decision D-10).
    """
    eligible = attention_mask.bool()
    for special_id in special_token_ids:
        eligible = eligible & (input_ids != special_id)
    probs = torch.full_like(input_ids, fill_value=0, dtype=torch.float32)
    probs[eligible] = mask_prob
    mask_positions = torch.bernoulli(probs, generator=generator).bool()
    masked_ids = input_ids.clone()
    masked_ids[mask_positions] = mask_token_id
    return masked_ids, mask_positions


# ── Per-batch per-head sum-to-1 weighted MSE (FR-017, Decision D-11) ───


def _per_batch_weighted_mse(
    predictions: dict[str, torch.Tensor],
    labels: torch.Tensor,
    weights: torch.Tensor,
    head_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute ``L_reg`` for one batch.

    Each of the 9 heads gets a per-batch sum-to-1 weighted MSE term;
    the four signed-head terms sum unweighted, and the five color-lift
    terms sum with the FR-017 ``(1/5)`` prefactor. Heads whose total
    sample weight is zero in the batch contribute exactly zero (the
    short-circuit is on the head_mask sum, not on a denominator clamp,
    so a future numerator drift cannot leak through).
    """
    # Stack predictions into a (B, 9) tensor in the same head order.
    per_head_preds: list[torch.Tensor] = [
        predictions["score_play"],
        predictions["score_draw"],
        predictions["played_rate"],
        predictions["cast_lift"],
    ]
    color_lift = predictions["color_lift"]  # (B, 5)
    for k in range(len(COLOR_ORDER)):
        per_head_preds.append(color_lift[:, k])
    preds = torch.stack(per_head_preds, dim=1)  # (B, 9)

    sq_err = (preds - labels) ** 2  # (B, 9)
    weighted = sq_err * weights * head_mask  # (B, 9)
    weight_total = (weights * head_mask).sum(dim=0)  # (9,)
    head_weight_sum = head_mask.sum(dim=0)  # (9,)

    loss = preds.new_zeros(())
    for h in range(N_HEADS):
        if float(head_weight_sum[h].item()) == 0.0:
            # No card in the batch contributes to head h — exact zero.
            continue
        denom = weight_total[h]
        if float(denom.item()) == 0.0:
            # head_mask is non-empty but every contributing weight is zero
            # (FR-017a returns 0 for empty cells). Still exactly zero.
            continue
        loss = loss + _HEAD_LOSS_WEIGHT[h] * (weighted[:, h].sum() / denom)
    return loss


# ── Config ──────────────────────────────────────────────────────────────


@dataclass
class TrainEncoderConfig:
    cards_played_path: Path = field(
        default_factory=lambda: Path("output/sealed/cards-played.txt"),
    )
    cards_folder: Path = field(
        default_factory=lambda: Path("output/cardsfolder/"),
    )
    vocab_path: Path = field(
        default_factory=lambda: Path("models/sealed/encoder/vocab.txt"),
    )
    model_output_dir: Path = field(
        default_factory=lambda: Path("models/sealed/encoder/"),
    )
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-4
    patience: int = 20
    dropout: float = 0.1
    n_layers: int = 6
    n_heads: int = 4
    n_pool_queries: int = 4
    shrinkage_k: float = 20.0
    mlm_weight: float = 0.1
    mlm_mask_prob: float = 0.15

    # FR-022 hardcoded constants — kept on the dataclass for traceability.
    d_model: int = D_MODEL
    ff_dim: int = FF_DIM
    val_fraction: float = VAL_FRACTION
    random_seed: int = RANDOM_SEED


# ── Pre-flight ──────────────────────────────────────────────────────────


class _PreFlightError(RuntimeError):
    """Raised when a pre-flight check fails. Carries an exit code so the
    CLI handler can map directly to ``contracts/cli.md`` § Exit codes.
    """

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _run_preflight(config: TrainEncoderConfig) -> None:
    if not config.vocab_path.exists():
        raise _PreFlightError(
            f"Vocabulary file not found: {config.vocab_path}. "
            "Run python -m sealed build-vocab first.",
            exit_code=2,
        )
    vocab = load_vocabulary(config.vocab_path)
    if _MASK_TOKEN not in vocab:
        raise _PreFlightError(
            f"Vocabulary at {config.vocab_path} is missing the reserved "
            f"{_MASK_TOKEN!r} token required by the MLM auxiliary loss "
            "(FR-009a). Re-run python -m sealed build-vocab to refresh.",
            exit_code=2,
        )
    if (
        not config.cards_played_path.exists()
        or config.cards_played_path.stat().st_size == 0
    ):
        raise _PreFlightError(
            f"cards-played.txt missing or empty: {config.cards_played_path}. "
            "Run python -m sealed match-outcomes to collect data first.",
            exit_code=3,
        )
    if not config.cards_folder.is_dir() or not any(
        config.cards_folder.rglob("*.txt")
    ):
        raise _PreFlightError(
            f"Cards folder is empty or missing: {config.cards_folder}. "
            "Run python -m price_predictor convert to build the corpus.",
            exit_code=4,
        )
    # n_pool_queries / d_model relationship is validated again inside
    # SealedEncoderConfig.__post_init__, but we surface it as exit code 6
    # at the CLI handler level (the dataclass raises ValueError).
    if D_MODEL % config.n_pool_queries != 0:
        raise _PreFlightError(
            f"d_model ({D_MODEL}) must be divisible by n_pool_queries "
            f"({config.n_pool_queries})",
            exit_code=6,
        )


# ── Best-checkpoint helper ──────────────────────────────────────────────


@dataclass
class _BestCheckpoint:
    """Track best-by-val-loss state through training. Mirrors the shape of
    ``price_predictor.application.train_transformer._BestCheckpoint``.

    Follow-up: when a third training entry point arrives, extract this to
    a shared utility (research.md "Third-instance check").
    """

    best_val_loss: float = float("inf")
    best_epoch: int = -1
    best_state_dict: dict | None = None

    def update(self, model: SealedEncoderModel, epoch: int, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return True
        return False


# ── Training loop ──────────────────────────────────────────────────────


def _make_optimizer(
    model: SealedEncoderModel, lr: float, total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW + linear warmup over the first 5% of total scheduled steps.

    After the warmup window the learning rate stays at ``lr`` for the
    remainder of the run (FR-022 / Decision D-14).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup_steps = max(1, int(math.ceil(total_steps * 0.05)))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class _TrainContext:
    """Read-only context passed through the training and evaluation loops."""

    mlm_weight: float
    mlm_mask_prob: float
    mask_token_id: int
    special_token_ids: tuple[int, ...]


def _forward_batch(
    model: SealedEncoderModel,
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    head_mask: torch.Tensor,
    ctx: _TrainContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one batch end-to-end and return ``(L_reg, L_mlm, L_total)``."""
    masked_ids, mask_positions = _draw_mlm_mask(
        ids, attention_mask, ctx.mlm_mask_prob,
        ctx.mask_token_id, ctx.special_token_ids,
    )
    predictions = model(masked_ids, attention_mask)
    l_reg = _per_batch_weighted_mse(predictions, labels, weights, head_mask)
    mlm_logits = predictions["mlm_logits"]
    if mask_positions.any():
        flat_logits = mlm_logits[mask_positions]
        flat_targets = ids[mask_positions]
        l_mlm = F.cross_entropy(flat_logits, flat_targets)
    else:
        l_mlm = mlm_logits.new_zeros(())
    l_total = l_reg + ctx.mlm_weight * l_mlm
    return l_reg, l_mlm, l_total


def _train_epoch(
    model: SealedEncoderModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    ctx: _TrainContext,
) -> tuple[float, float, float]:
    """Train for one epoch. Returns ``(reg_avg, mlm_avg, total_avg)``."""
    model.train()
    sum_reg = 0.0
    sum_mlm = 0.0
    sum_total = 0.0
    n_batches = 0
    for ids, mask, labels, weights, head_mask in loader:
        ids = ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        head_mask = head_mask.to(device)
        optimizer.zero_grad()
        l_reg, l_mlm, l_total = _forward_batch(
            model, ids, mask, labels, weights, head_mask, ctx,
        )
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        sum_reg += float(l_reg.item())
        sum_mlm += float(l_mlm.item())
        sum_total += float(l_total.item())
        n_batches += 1
    n = max(n_batches, 1)
    return sum_reg / n, sum_mlm / n, sum_total / n


@torch.no_grad()
def _eval_loss(
    model: SealedEncoderModel,
    loader: DataLoader,
    device: torch.device,
    ctx: _TrainContext,
) -> tuple[float, float, float]:
    """Evaluate on the val set. Returns ``(reg_avg, mlm_avg, total_avg)``.

    The MLM mask is drawn at val time too (with the same
    ``--mlm-mask-prob``) so val numbers stay comparable across epochs
    (Decision D-12).
    """
    model.eval()
    sum_reg = 0.0
    sum_mlm = 0.0
    sum_total = 0.0
    n_batches = 0
    for ids, mask, labels, weights, head_mask in loader:
        ids = ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        head_mask = head_mask.to(device)
        l_reg, l_mlm, l_total = _forward_batch(
            model, ids, mask, labels, weights, head_mask, ctx,
        )
        sum_reg += float(l_reg.item())
        sum_mlm += float(l_mlm.item())
        sum_total += float(l_total.item())
        n_batches += 1
    n = max(n_batches, 1)
    return sum_reg / n, sum_mlm / n, sum_total / n


def _resolve_special_token_ids(
    vocab: dict[str, int],
) -> tuple[int, tuple[int, ...]]:
    """Return ``(mask_id, special_ids)`` for the MLM mask draw."""
    mask_id = vocab[_MASK_TOKEN]
    special_ids = tuple(
        vocab[t] for t in _MASKED_SPECIAL_TOKENS if t in vocab
    )
    return mask_id, special_ids


def run(config: TrainEncoderConfig) -> Path:
    """Execute the train-encoder pipeline.

    Returns the path of the saved ``latest.pt``. Raises ``_PreFlightError``
    on pre-flight failure (CLI handler maps to exit codes). Cards
    referenced by ``cards-played.txt`` that lack a converted ``.txt`` are
    reported via a log warning and dropped — they do not block the run.
    """
    _run_preflight(config)

    locator = ConvertedCardLocator(config.cards_folder)

    _log(f"Reading {config.cards_played_path} (pass 1)")
    counters = _aggregate_pass_one(iter_rows(config.cards_played_path))
    _log(f"Pass 1 collected counters for {len(counters)} card(s)")

    n_dropped = _drop_missing_cards(counters, locator)
    if n_dropped:
        _log(f"Continuing with {len(counters)} card(s) after dropping {n_dropped}")

    _log(f"Reading {config.cards_played_path} (pass 2: per-color slices)")
    _aggregate_pass_two(iter_rows(config.cards_played_path), counters, locator)

    labels = _build_label_map(counters, config.shrinkage_k)
    _log(
        f"Built label map: {len(labels)} card(s) with "
        f"wins_when_in_deck + losses_when_in_deck > 0",
    )

    win_rates_path = Path("output/sealed") / _WIN_RATES_FILENAME
    _write_win_rates(labels, win_rates_path)
    _log(f"Wrote per-card label snapshot to {win_rates_path}")

    train_names, val_names = _split_cards(
        labels, val_fraction=config.val_fraction, seed=config.random_seed,
    )
    if not train_names or not val_names:
        raise RuntimeError(
            f"Not enough cards for a train/val split: "
            f"train={len(train_names)}, val={len(val_names)}"
        )
    _log(f"Card-level split: {len(train_names)} train / {len(val_names)} val")

    tokenizer = load_tokenizer(config.vocab_path)
    vocab = load_vocabulary(config.vocab_path)
    mask_token_id, special_token_ids = _resolve_special_token_ids(vocab)
    max_seq_len = _measure_max_seq_len(labels.keys(), locator, tokenizer)
    _log(f"max_seq_len = {max_seq_len} (rounded to multiple of 8)")

    encoder_config = SealedEncoderConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        ff_dim=FF_DIM,
        max_seq_len=max_seq_len,
        dropout=config.dropout,
        n_pool_queries=config.n_pool_queries,
    )

    torch.manual_seed(config.random_seed)
    model = SealedEncoderModel(encoder_config)
    device = _select_device()
    model.to(device)
    _log(f"Sealed encoder on {device}")

    train_ds = _WinnabilityDataset(
        train_names, labels, tokenizer, locator, max_seq_len, config.shrinkage_k,
    )
    val_ds = _WinnabilityDataset(
        val_names, labels, tokenizer, locator, max_seq_len, config.shrinkage_k,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    total_steps = max(1, config.epochs * len(train_loader))
    optimizer, scheduler = _make_optimizer(model, config.lr, total_steps)

    ctx = _TrainContext(
        mlm_weight=config.mlm_weight,
        mlm_mask_prob=config.mlm_mask_prob,
        mask_token_id=mask_token_id,
        special_token_ids=special_token_ids,
    )

    best = _BestCheckpoint()
    epochs_since_best = 0
    for epoch in range(1, config.epochs + 1):
        train_reg, train_mlm, train_total = _train_epoch(
            model, train_loader, optimizer, scheduler, device, ctx,
        )
        val_reg, val_mlm, val_total = _eval_loss(model, val_loader, device, ctx)
        improved = best.update(model, epoch, val_total)
        marker = "*" if improved else " "
        _log(
            f"Epoch {epoch}/{config.epochs}  "
            f"train_loss={train_total:.4f} (reg={train_reg:.4f}, mlm={train_mlm:.4f})  "
            f"val_loss={val_total:.4f} (reg={val_reg:.4f}, mlm={val_mlm:.4f})  "
            f"best={marker}"
        )
        if improved:
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= config.patience:
                _log(
                    f"Early stop after {epochs_since_best} epochs "
                    f"without improvement.",
                )
                break

    if best.best_state_dict is not None:
        model.load_state_dict(best.best_state_dict)
        _log(
            f"Restored best epoch {best.best_epoch} "
            f"(val_loss={best.best_val_loss:.4f})",
        )

    store = SealedEncoderStore()
    saved = store.save_encoder(model, encoder_config, config.model_output_dir)
    latest = config.model_output_dir / "latest.pt"
    _log(
        f"Best epoch: {best.best_epoch}, val_loss: {best.best_val_loss:.4f}. "
        f"Saved {saved}",
    )
    return latest
