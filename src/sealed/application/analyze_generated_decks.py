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
from price_predictor.infrastructure.mtgjson_loader import build_metadata_map
from sealed.infrastructure.converted_card_locator import (
    BASIC_LAND_NAMES,
    ConvertedCardLocator,
)
from sealed.infrastructure.pool_file_reader import GeneratedDeck

DEFAULT_DECKS_PATH = Path("output/sealed/generated-decks.txt")
DEFAULT_CARDS_PATH = Path("output/cardsfolder")
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
    locator = ConvertedCardLocator(cards_path)
    distinct_names = sorted({name for deck in decks for name in deck.cards})
    card_cache: dict[str, Card | None] = {}
    for name in distinct_names:
        path = locator.text_path(name)
        card_cache[name] = parse_converted_file(path) if path else None

    unresolved_total = sum(
        1 for name, card in card_cache.items()
        if card is None and name.lower() not in BASIC_LAND_NAMES
    )

    metadata: dict[str, PrintingData] | None = None
    rarity_skip_reason: str | None = None
    if no_rarity:
        rarity_skip_reason = "--no-rarity flag set"
    elif not (ALLPRINTINGS_PATH.exists() and ALLPRICES_PATH.exists()):
        rarity_skip_reason = f"missing {ALLPRINTINGS_PATH} or {ALLPRICES_PATH}"
    else:
        metadata, _ = build_metadata_map(ALLPRINTINGS_PATH, ALLPRICES_PATH)

    all_stats = [compute_deck_stats(deck, card_cache, metadata) for deck in decks]

    print("=== Decks loaded ===")
    for source, count in source_summary:
        print(f"  {source}: {count}")
    print(f"  total: {len(decks)}")
    print(f"  distinct nonbasic card names: {len(distinct_names)}")
    print(f"  unresolved card names (skipped from stats): {unresolved_total}")
    print()

    _print_report(all_stats, "Global", rarity_skip_reason)

    labels = sorted({s.label for s in all_stats})
    if len(labels) > 1:
        for label in labels:
            subset = [s for s in all_stats if s.label == label]
            _print_report(
                subset,
                f"Per-label: {label} ({len(subset)} decks)",
                rarity_skip_reason,
            )


def compute_deck_stats(
    deck: GeneratedDeck,
    card_cache: dict[str, Card | None],
    metadata: dict[str, PrintingData] | None,
) -> DeckStats:
    colors_present: set[str] = set()
    nonland_count = 0
    nonland_mv_sum = 0.0
    mv_buckets: Counter = Counter()
    type_counts: Counter = Counter()
    basic = 0
    nonbasic = 0
    pip_counts: Counter = Counter()
    rarity_counts: Counter | None = Counter() if metadata is not None else None
    unresolved = 0

    for name in deck.cards:
        if metadata is not None and rarity_counts is not None:
            pd = metadata.get(name)
            if pd is not None:
                rarity_counts[pd.rarity] += 1

        if name.lower() in BASIC_LAND_NAMES:
            basic += 1
            type_counts["Land"] += 1
            continue

        card = card_cache.get(name)
        if card is None:
            unresolved += 1
            continue

        for t in card.types:
            type_counts[t] += 1

        if card.is_land():
            nonbasic += 1
            continue

        nonland_count += 1
        mc = card.mana_cost
        if mc is None:
            continue

        nonland_mv_sum += mc.total_mana_value
        mv_buckets[_bucket_mv(mc.total_mana_value)] += 1
        for color, count in (
            ("W", mc.w), ("U", mc.u), ("B", mc.b), ("R", mc.r), ("G", mc.g),
        ):
            if count > 0:
                colors_present.add(color)
                pip_counts[color] += count

    return DeckStats(
        label=deck.label,
        set_code=deck.set_code,
        colors_present=colors_present,
        nonland_count=nonland_count,
        nonland_mv_sum=nonland_mv_sum,
        mv_bucket_counts=mv_buckets,
        type_counts=type_counts,
        basic_land_count=basic,
        nonbasic_land_count=nonbasic,
        pip_counts=pip_counts,
        rarity_counts=rarity_counts,
        unresolved_count=unresolved,
    )


def _bucket_mv(mv: float) -> str:
    n = int(mv)
    if n >= 7:
        return "7+"
    return str(n)


def _print_report(
    stats: list[DeckStats],
    title: str,
    rarity_skip_reason: str | None,
) -> None:
    n = len(stats)
    print(f"### {title} ###")
    print()

    print("=== Color presence (% of decks) ===")
    print(
        "  (color present iff any nonland mana cost contains it; "
        "devoid/colorless cards contribute nothing)"
    )
    for color in WUBRG:
        c = sum(1 for s in stats if color in s.colors_present)
        _print_bar(color, c, n)
    print()

    print("=== Color count distribution (% of decks) ===")
    bucket_counts: Counter = Counter()
    for s in stats:
        k = len(s.colors_present)
        if k == 0:
            bucket_counts["0 (colorless)"] += 1
        elif k == 1:
            bucket_counts["1 (mono)"] += 1
        else:
            bucket_counts[f"{k}-color"] += 1
    for label in (
        "0 (colorless)", "1 (mono)", "2-color",
        "3-color", "4-color", "5-color",
    ):
        c = bucket_counts.get(label, 0)
        _print_bar(label, c, n)
    print()

    print("=== Pip share by rank (avg %, grouped by deck color count) ===")
    print("  (within each deck, colors are sorted by pip count desc;")
    print("   cells show the avg share held by the Nth-ranked color)")
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
        cells = []
        for cc in cols:
            if rank < cc and rank_groups[cc]:
                avg = mean(p[rank] for p in rank_groups[cc]) * 100
                cells.append(f"{avg:6.1f}%".rjust(7))
            else:
                cells.append("-".rjust(7))
        print(f"  {row_label} | " + " | ".join(cells))
    print()

    total_nonland_mv = sum(s.nonland_mv_sum for s in stats)
    total_nonland = sum(s.nonland_count for s in stats)
    avg_mv = total_nonland_mv / total_nonland if total_nonland else 0.0
    print("=== Mana curve (nonland cards only) ===")
    print(f"  avg mana value: {avg_mv:.2f}")
    bucket_totals: Counter = Counter()
    for s in stats:
        for k, v in s.mv_bucket_counts.items():
            bucket_totals[k] += v
    for bucket in MV_BUCKETS:
        c = bucket_totals.get(bucket, 0)
        _print_bar(f"MV {bucket}", c, total_nonland)
    print()

    print("=== Type balance (avg per deck) ===")
    print("  (multi-typed cards count under each type, so rows can sum to >40)")
    type_totals: Counter = Counter()
    for s in stats:
        for k, v in s.type_counts.items():
            type_totals[k] += v
    for t in TYPE_LABELS:
        avg = type_totals.get(t, 0) / n if n else 0.0
        print(f"  {t:<14}: {avg:5.2f}")
    print()

    avg_basic = mean(s.basic_land_count for s in stats) if n else 0.0
    avg_nonbasic = mean(s.nonbasic_land_count for s in stats) if n else 0.0
    print("=== Land breakdown (avg per deck) ===")
    print(f"  basic    : {avg_basic:5.2f}")
    print(f"  nonbasic : {avg_nonbasic:5.2f}")
    print(f"  total    : {avg_basic + avg_nonbasic:5.2f}")
    print()

    print("=== Pip distribution (across nonland mana costs) ===")
    pip_totals: Counter = Counter()
    for s in stats:
        for k, v in s.pip_counts.items():
            pip_totals[k] += v
    total_pips = sum(pip_totals.values())
    for color in WUBRG:
        avg = pip_totals.get(color, 0) / n if n else 0.0
        pct = 100 * pip_totals.get(color, 0) / total_pips if total_pips else 0.0
        bar = "#" * int(pct / 2)
        print(f"  {color}: {avg:5.2f} avg/deck  ({pct:5.1f}%) {bar}")
    print()

    if rarity_skip_reason is not None:
        print(f"=== Rarity distribution: skipped ({rarity_skip_reason}) ===")
        print()
        return

    print("=== Rarity distribution (avg per deck) ===")
    rarity_totals: Counter = Counter()
    for s in stats:
        if s.rarity_counts is not None:
            for k, v in s.rarity_counts.items():
                rarity_totals[k] += v
    total_rarity = sum(rarity_totals.values())
    for r in STANDARD_RARITIES:
        avg = rarity_totals.get(r, 0) / n if n else 0.0
        pct = 100 * rarity_totals.get(r, 0) / total_rarity if total_rarity else 0.0
        bar = "#" * int(pct / 2)
        print(f"  {r:<8}: {avg:5.2f} avg/deck  ({pct:5.1f}%) {bar}")
    other_total = sum(
        v for k, v in rarity_totals.items() if k not in STANDARD_RARITIES
    )
    if other_total:
        avg = other_total / n if n else 0.0
        pct = 100 * other_total / total_rarity if total_rarity else 0.0
        bar = "#" * int(pct / 2)
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
