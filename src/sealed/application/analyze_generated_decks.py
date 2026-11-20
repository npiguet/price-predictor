"""Aggregate composition stats over generated 40-card decks.

The engine behind ``sealed analyze-generated-decks`` (and, via reuse,
``draft analyze-generated-decks``): given a list of ``GeneratedDeck`` objects it
reports color preference, color-count distribution, mana curve, type balance,
basic/nonbasic land split, pip distribution, and — when MTGJSON is available —
rarity distribution, with per-label breakdowns when more than one label is
present.

Decoupled from the data source: callers build the ``GeneratedDeck`` list (from
``LABEL;SET_CODE;…`` files, or from a ``drafts.jsonl`` corpus's per-seat decks)
and pass it to :func:`analyze_decks`. Moved here from the former
``scripts/analyze_generated_decks.py`` so both the sealed and draft CLIs share
one implementation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import PrintingData
from price_predictor.infrastructure.converted_card_parser import parse_converted_file
from price_predictor.infrastructure.mtgjson_loader import build_printing_data_map
from sealed.infrastructure.converted_card_locator import (
    BASIC_LAND_NAMES,
    ConvertedCardLocator,
)
from sealed.infrastructure.pool_file_reader import GeneratedDeck

ALLPRINTINGS_PATH = Path("resources/AllPrintings.json")
ALLPRICES_PATH = Path("resources/AllPricesToday.json")

WUBRG = ("W", "U", "B", "R", "G")
TYPE_LABELS = (
    "Creature", "Instant", "Sorcery", "Artifact",
    "Enchantment", "Planeswalker", "Land",
)
MV_BUCKETS = ("0", "1", "2", "3", "4", "5", "6", "7+")
STANDARD_RARITIES = ("common", "uncommon", "rare", "mythic")


@dataclass
class DeckStats:
    label: str
    set_code: str
    colors_present: set[str]
    nonland_count: int
    nonland_mv_sum: float
    mv_bucket_counts: Counter
    type_counts: Counter
    basic_land_count: int
    nonbasic_land_count: int
    pip_counts: Counter
    rarity_counts: Counter | None
    unresolved_count: int


def analyze_decks(
    decks: list[GeneratedDeck],
    cards_path: Path,
    *,
    no_rarity: bool,
    source_summary: list[tuple[str, int]],
) -> None:
    """Compute per-deck stats and print the global + per-label reports.

    ``source_summary`` is the ``(source, count)`` lines printed under
    ``=== Decks loaded ===`` (one per generated-decks file, or one for the
    drafts corpus). ``decks`` must be non-empty.
    """
    distinct_names = sorted({name for deck in decks for name in deck.cards})
    card_cache = _resolve_cards(distinct_names, ConvertedCardLocator(cards_path))
    unresolved_total = sum(
        1 for name, card in card_cache.items()
        if card is None and name.lower() not in BASIC_LAND_NAMES
    )
    metadata, rarity_skip_reason = _load_rarity_metadata(no_rarity)
    all_stats = [compute_deck_stats(deck, card_cache, metadata) for deck in decks]

    print("=== Decks loaded ===")
    for source, count in source_summary:
        print(f"  {source}: {count}")
    print(f"  total: {len(decks)}")
    print(f"  distinct nonbasic card names: {len(distinct_names)}")
    print(f"  unresolved card names (skipped from stats): {unresolved_total}")
    print()

    _print_report(all_stats, "Global", rarity_skip_reason)
    _print_per_label_reports(all_stats, rarity_skip_reason)


def _resolve_cards(
    names: list[str], locator: ConvertedCardLocator,
) -> dict[str, Card | None]:
    card_cache: dict[str, Card | None] = {}
    for name in names:
        path = locator.text_path(name)
        card_cache[name] = parse_converted_file(path) if path else None
    return card_cache


def _load_rarity_metadata(
    no_rarity: bool,
) -> tuple[dict[str, PrintingData] | None, str | None]:
    """The MTGJSON printing data, or ``None`` plus the reason it is skipped."""
    if no_rarity:
        return None, "--no-rarity flag set"
    if not (ALLPRINTINGS_PATH.exists() and ALLPRICES_PATH.exists()):
        return None, f"missing {ALLPRINTINGS_PATH} or {ALLPRICES_PATH}"
    metadata, _ = build_printing_data_map(ALLPRINTINGS_PATH, ALLPRICES_PATH)
    return metadata, None


def _print_per_label_reports(
    all_stats: list[DeckStats], rarity_skip_reason: str | None,
) -> None:
    labels = sorted({s.label for s in all_stats})
    if len(labels) <= 1:
        return
    for label in labels:
        subset = [s for s in all_stats if s.label == label]
        _print_report(
            subset,
            f"Per-label: {label} ({len(subset)} decks)",
            rarity_skip_reason,
        )


class _DeckTally:
    """The running counts of one deck while ``compute_deck_stats`` walks it."""

    def __init__(self, with_rarity: bool) -> None:
        self.colors_present: set[str] = set()
        self.nonland_count = 0
        self.nonland_mv_sum = 0.0
        self.mv_buckets: Counter = Counter()
        self.type_counts: Counter = Counter()
        self.basic = 0
        self.nonbasic = 0
        self.pip_counts: Counter = Counter()
        self.rarity_counts: Counter | None = Counter() if with_rarity else None
        self.unresolved = 0

    def count_rarity(self, name: str, metadata: dict[str, PrintingData]) -> None:
        pd = metadata.get(name)
        if pd is not None and self.rarity_counts is not None:
            self.rarity_counts[pd.rarity] += 1

    def count_basic(self) -> None:
        self.basic += 1
        self.type_counts["Land"] += 1

    def count_card(self, card: Card) -> None:
        for t in card.types:
            self.type_counts[t] += 1
        if card.is_land():
            self.nonbasic += 1
            return
        self.nonland_count += 1
        if card.mana_cost is not None:
            self._count_mana_cost(card.mana_cost)

    def _count_mana_cost(self, mc) -> None:
        self.nonland_mv_sum += mc.total_mana_value
        self.mv_buckets[_bucket_mv(mc.total_mana_value)] += 1
        for color, count in (
            ("W", mc.w), ("U", mc.u), ("B", mc.b), ("R", mc.r), ("G", mc.g),
        ):
            if count > 0:
                self.colors_present.add(color)
                self.pip_counts[color] += count

    def to_stats(self, deck: GeneratedDeck) -> DeckStats:
        return DeckStats(
            label=deck.label,
            set_code=deck.set_code,
            colors_present=self.colors_present,
            nonland_count=self.nonland_count,
            nonland_mv_sum=self.nonland_mv_sum,
            mv_bucket_counts=self.mv_buckets,
            type_counts=self.type_counts,
            basic_land_count=self.basic,
            nonbasic_land_count=self.nonbasic,
            pip_counts=self.pip_counts,
            rarity_counts=self.rarity_counts,
            unresolved_count=self.unresolved,
        )


def compute_deck_stats(
    deck: GeneratedDeck,
    card_cache: dict[str, Card | None],
    metadata: dict[str, PrintingData] | None,
) -> DeckStats:
    tally = _DeckTally(metadata is not None)
    for name in deck.cards:
        if metadata is not None:
            tally.count_rarity(name, metadata)
        if name.lower() in BASIC_LAND_NAMES:
            tally.count_basic()
            continue
        card = card_cache.get(name)
        if card is None:
            tally.unresolved += 1
            continue
        tally.count_card(card)
    return tally.to_stats(deck)


def _bucket_mv(mv: float) -> str:
    n = int(mv)
    if n >= 7:
        return "7+"
    return str(n)


def _sum_counters(counters: list[Counter]) -> Counter:
    totals: Counter = Counter()
    for counter in counters:
        for k, v in counter.items():
            totals[k] += v
    return totals


def _print_report(
    stats: list[DeckStats],
    title: str,
    rarity_skip_reason: str | None,
) -> None:
    n = len(stats)
    print(f"### {title} ###")
    print()
    _print_color_presence(stats, n)
    _print_color_count_distribution(stats, n)
    _print_pip_share_by_rank(stats)
    _print_mana_curve(stats)
    _print_type_balance(stats, n)
    _print_land_breakdown(stats, n)
    _print_pip_distribution(stats, n)
    _print_rarity_distribution(stats, n, rarity_skip_reason)


def _print_color_presence(stats: list[DeckStats], n: int) -> None:
    print("=== Color presence (% of decks) ===")
    print(
        "  (color present iff any nonland mana cost contains it; "
        "devoid/colorless cards contribute nothing)"
    )
    for color in WUBRG:
        c = sum(1 for s in stats if color in s.colors_present)
        _print_bar(color, c, n)
    print()


def _color_count_label(k: int) -> str:
    if k == 0:
        return "0 (colorless)"
    if k == 1:
        return "1 (mono)"
    return f"{k}-color"


def _print_color_count_distribution(stats: list[DeckStats], n: int) -> None:
    print("=== Color count distribution (% of decks) ===")
    bucket_counts: Counter = Counter(
        _color_count_label(len(s.colors_present)) for s in stats
    )
    for label in (
        "0 (colorless)", "1 (mono)", "2-color",
        "3-color", "4-color", "5-color",
    ):
        c = bucket_counts.get(label, 0)
        _print_bar(label, c, n)
    print()


def _pip_rank_groups(stats: list[DeckStats]) -> dict[int, list[list[float]]]:
    rank_groups: dict[int, list[list[float]]] = {c: [] for c in range(1, 6)}
    for s in stats:
        pips = sorted(
            (v for v in s.pip_counts.values() if v > 0), reverse=True
        )
        total = sum(pips)
        if not pips or total == 0:
            continue
        cc = len(pips)
        if cc in rank_groups:
            rank_groups[cc].append([p / total for p in pips])
    return rank_groups


def _pip_rank_cell(rank: int, cc: int, group: list[list[float]]) -> str:
    if rank < cc and group:
        avg = mean(p[rank] for p in group) * 100
        return f"{avg:6.1f}%".rjust(7)
    return "-".rjust(7)


def _print_pip_share_by_rank(stats: list[DeckStats]) -> None:
    print("=== Pip share by rank (avg %, grouped by deck color count) ===")
    print("  (within each deck, colors are sorted by pip count desc;")
    print("   cells show the avg share held by the Nth-ranked color)")
    rank_groups = _pip_rank_groups(stats)
    cols = list(range(1, 6))
    pad = " " * 12
    print(f"  {pad} | " + " | ".join(f"{c}-color".rjust(7) for c in cols))
    print(
        f"  {'decks'.ljust(12)} | "
        + " | ".join(f"{len(rank_groups[c])}".rjust(7) for c in cols)
    )
    ordinals = ("1st", "2nd", "3rd", "4th", "5th")
    for rank in range(5):
        row_label = f"{ordinals[rank]} color".ljust(12)
        cells = [_pip_rank_cell(rank, cc, rank_groups[cc]) for cc in cols]
        print(f"  {row_label} | " + " | ".join(cells))
    print()


def _print_mana_curve(stats: list[DeckStats]) -> None:
    total_nonland_mv = sum(s.nonland_mv_sum for s in stats)
    total_nonland = sum(s.nonland_count for s in stats)
    avg_mv = total_nonland_mv / total_nonland if total_nonland else 0.0
    print("=== Mana curve (nonland cards only) ===")
    print(f"  avg mana value: {avg_mv:.2f}")
    bucket_totals = _sum_counters([s.mv_bucket_counts for s in stats])
    for bucket in MV_BUCKETS:
        c = bucket_totals.get(bucket, 0)
        _print_bar(f"MV {bucket}", c, total_nonland)
    print()


def _print_type_balance(stats: list[DeckStats], n: int) -> None:
    print("=== Type balance (avg per deck) ===")
    print("  (multi-typed cards count under each type, so rows can sum to >40)")
    type_totals = _sum_counters([s.type_counts for s in stats])
    for t in TYPE_LABELS:
        avg = type_totals.get(t, 0) / n if n else 0.0
        print(f"  {t:<14}: {avg:5.2f}")
    print()


def _print_land_breakdown(stats: list[DeckStats], n: int) -> None:
    avg_basic = mean(s.basic_land_count for s in stats) if n else 0.0
    avg_nonbasic = mean(s.nonbasic_land_count for s in stats) if n else 0.0
    print("=== Land breakdown (avg per deck) ===")
    print(f"  basic    : {avg_basic:5.2f}")
    print(f"  nonbasic : {avg_nonbasic:5.2f}")
    print(f"  total    : {avg_basic + avg_nonbasic:5.2f}")
    print()


def _share_of(count: float, n: int, grand_total: float) -> tuple[float, float, str]:
    """Average per deck, percent of ``grand_total`` and the ``#`` bar."""
    avg = count / n if n else 0.0
    pct = 100 * count / grand_total if grand_total else 0.0
    return avg, pct, "#" * int(pct / 2)


def _print_pip_distribution(stats: list[DeckStats], n: int) -> None:
    print("=== Pip distribution (across nonland mana costs) ===")
    pip_totals = _sum_counters([s.pip_counts for s in stats])
    total_pips = sum(pip_totals.values())
    for color in WUBRG:
        avg, pct, bar = _share_of(pip_totals.get(color, 0), n, total_pips)
        print(f"  {color}: {avg:5.2f} avg/deck  ({pct:5.1f}%) {bar}")
    print()


def _print_rarity_distribution(
    stats: list[DeckStats], n: int, rarity_skip_reason: str | None,
) -> None:
    if rarity_skip_reason is not None:
        print(f"=== Rarity distribution: skipped ({rarity_skip_reason}) ===")
        print()
        return

    print("=== Rarity distribution (avg per deck) ===")
    rarity_totals = _sum_counters(
        [s.rarity_counts for s in stats if s.rarity_counts is not None]
    )
    total_rarity = sum(rarity_totals.values())
    for r in STANDARD_RARITIES:
        avg, pct, bar = _share_of(rarity_totals.get(r, 0), n, total_rarity)
        print(f"  {r:<8}: {avg:5.2f} avg/deck  ({pct:5.1f}%) {bar}")
    other_total = sum(
        v for k, v in rarity_totals.items() if k not in STANDARD_RARITIES
    )
    if other_total:
        avg, pct, bar = _share_of(other_total, n, total_rarity)
        print(f"  {'other':<8}: {avg:5.2f} avg/deck  ({pct:5.1f}%) {bar}")
    total_slots = n * 40
    print(
        f"  (mapped {total_rarity}/{total_slots} card slots to MTGJSON "
        f"rarity data)"
    )
    print()


def _print_bar(label: str, count: int, total: int) -> None:
    pct = 100 * count / total if total else 0.0
    bar = "#" * int(pct / 2)
    print(f"  {label:<16}: {count:>5}/{total:<5} ({pct:5.1f}%) {bar}")
