"""Canonicalize ``output/sealed/cards-played.txt``.

Forge logs Card.getCurrentState().getName() during play — i.e. the
*display* name. Three classes of mismatch appear in the file:

  1. Flavor names (Universes Beyond reskins) — e.g. "Cloud Strife" is
     a flavorName of "Najeela, the Blade-Blossom".
  2. Card faces (split / DFC / adventure / room) — e.g. "Fire" is a
     faceName of "Fire // Ice".
  3. Forge-internal triggered-ability names — "...(N)'s Effect",
     "...(N)'s Boon", "...(N)'s Regeneration", "...Paradigm",
     "...Capstone", "Emblem - ...", plus mechanic placeholders
     "The Initiative" / "The Ring".

This script reads the file, rewrites flavor/face names to their
canonical MTG name via ``resources/AllPrintings.json``, drops junk
entries via regex+blacklist, enforces FR-004 (a canonical name in
both played and not_played columns goes to played), and emits the
cleaned dataset to ``output/sealed/cards-played.cleaned.txt`` along
with a diagnostic report at ``output-tmp/canonicalization_report.txt``.

Run ``--dry-run`` to produce only the report without writing the
cleaned file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sealed.infrastructure.cards_played_reader import iter_rows
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

DEFAULT_INPUT = Path("output/sealed/cards-played.txt")
DEFAULT_OUTPUT = Path("output/sealed/cards-played.cleaned.txt")
DEFAULT_REPORT = Path("output-tmp/canonicalization_report.txt")
DEFAULT_ALLPRINTINGS = Path("resources/AllPrintings.json")
DEFAULT_CARDS_FOLDER = Path("output/cardsfolder")


# Forge-internal triggered-ability names + emblem prefix.
_JUNK_PATTERN = re.compile(
    r"\(\d+\)'s (Effect|Boon|Regeneration|Trigger)$"
    r"|^Emblem [—-] "
    r"| Paradigm$"
    r"| Capstone$"
)
_JUNK_EXACT: frozenset[str] = frozenset({
    "The Initiative",
    "The Ring",
})


def _junk_reason(name: str, drop_tokens: frozenset[str]) -> str | None:
    """Return a short tag if ``name`` is junk; ``None`` otherwise."""
    if name in _JUNK_EXACT:
        return "exact"
    if name in drop_tokens:
        return "MTGJSON token"
    m = _JUNK_PATTERN.search(name)
    if m is None:
        return None
    # Map the matched alternation to a human-readable label.
    if name.startswith("Emblem "):
        return "Emblem prefix"
    if name.endswith(" Paradigm"):
        return "Paradigm suffix"
    if name.endswith(" Capstone"):
        return "Capstone suffix"
    # The remaining patterns are the parenthesized triggered abilities.
    return f"(N)'s {m.group(1)}"


def build_alias_map(
    printings_path: Path,
) -> tuple[dict[str, str], frozenset[str], list[tuple[str, list[str]]]]:
    """Build ``{alias: canonical_name}``, the token-name drop set, and conflicts.

    Walks every card in AllPrintings.json. For each card with a
    canonical ``name``:

      * ``name -> name``     (identity)
      * ``faceName -> name`` (split / DFC / adventure / room halves)
      * ``flavorName -> name`` (UB reskins)

    Token names (under ``set_info.tokens[]``) are collected separately
    as a drop set — these are game-state entities (emblems, dungeons,
    The Monarch, Treasure tokens, etc.) that Forge tracks like cards
    but are never deck contents.

    Conflicts (same alias mapping to multiple distinct canonical names
    across printings) are resolved by:
      1. identity match wins (``"Lightning Bolt"`` -> ``"Lightning Bolt"``,
         not the faceName form);
      2. otherwise pick lexicographically smallest as a deterministic
         fallback.
    """
    print(f"Loading {printings_path}...", file=sys.stderr, flush=True)
    with printings_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # alias -> set of canonical names
    multimap: dict[str, set[str]] = defaultdict(set)
    token_names: set[str] = set()
    sets_data = data.get("data", {})
    for set_info in sets_data.values():
        cards = (
            set_info if isinstance(set_info, list)
            else set_info.get("cards", [])
        )
        for c in cards:
            name = c.get("name")
            if not name:
                continue
            multimap[name].add(name)
            face = c.get("faceName")
            if face:
                multimap[face].add(name)
            flavor = c.get("flavorName")
            if flavor:
                multimap[flavor].add(name)
        if isinstance(set_info, dict):
            for t in set_info.get("tokens", []):
                tname = t.get("name")
                if tname:
                    token_names.add(tname)

    alias_map: dict[str, str] = {}
    conflicts: list[tuple[str, list[str]]] = []
    for alias, canonicals in multimap.items():
        if len(canonicals) == 1:
            alias_map[alias] = next(iter(canonicals))
            continue
        if alias in canonicals:
            alias_map[alias] = alias
            continue
        chosen = min(canonicals)
        alias_map[alias] = chosen
        conflicts.append((alias, sorted(canonicals)))
    # Tokens that share a name with a real card stay mapped (e.g.
    # "Goblin" is both a tribe-token name and not a real card — but
    # there is no real card "Goblin"; the token check below catches
    # those). When a real card *does* share a name, alias_map's
    # identity rule takes precedence and the name passes through.
    real_card_names = {n for ns in multimap.values() for n in ns}
    drop_tokens = frozenset(token_names - real_card_names)
    print(
        f"  {len(alias_map)} alias entries, "
        f"{len(drop_tokens)} token-only names to drop, "
        f"{len(conflicts)} conflicts",
        file=sys.stderr,
        flush=True,
    )
    return alias_map, drop_tokens, conflicts


def _rewrite_column(
    names: list[str],
    alias_map: dict[str, str],
    drop_tokens: frozenset[str],
    locator: ConvertedCardLocator,
    junk_counter: Counter[tuple[str, str]],
    rewrite_counter: Counter[tuple[str, str]],
    unmapped_counter: Counter[str],
) -> set[str]:
    """Rewrite one card-list column. Mutates the counters in place."""
    cleaned: set[str] = set()
    for raw in names:
        reason = _junk_reason(raw, drop_tokens)
        if reason is not None:
            junk_counter[(raw, reason)] += 1
            continue
        canonical = alias_map.get(raw)
        if canonical is None:
            # Fall back to Forge's converted-card corpus — handles real
            # cards that aren't in our AllPrintings snapshot yet (e.g.
            # Iron Man, Futurist Paragon from a newer set).
            if locator.text_path(raw) is not None:
                cleaned.add(raw)
                # No rewrite recorded — name is already canonical.
                continue
            unmapped_counter[raw] += 1
            continue
        if canonical != raw:
            rewrite_counter[(raw, canonical)] += 1
        cleaned.add(canonical)
    return cleaned


def _format_row(
    timestamp: str, run_id: str, set_code: str,
    method_a: str, method_b: str,
    cards_played_a: set[str], cards_played_b: set[str],
    cards_not_played_a: set[str], cards_not_played_b: set[str],
    winner: str, starter: str,
) -> str:
    return ";".join((
        timestamp, run_id, set_code, method_a, method_b,
        "|".join(sorted(cards_played_a)),
        "|".join(sorted(cards_played_b)),
        "|".join(sorted(cards_not_played_a)),
        "|".join(sorted(cards_not_played_b)),
        winner, starter,
    ))


def _process(
    input_path: Path,
    alias_map: dict[str, str],
    drop_tokens: frozenset[str],
    locator: ConvertedCardLocator,
    output_path: Path | None,
) -> dict:
    """Stream-process the input. Returns counts + counters for the report."""
    rows_in = 0
    rows_written = 0
    rows_dropped_empty = 0
    junk_counter: Counter[tuple[str, str]] = Counter()
    rewrite_counter: Counter[tuple[str, str]] = Counter()
    unmapped_counter: Counter[str] = Counter()

    out_handle = output_path.open("w", encoding="utf-8") if output_path else None
    try:
        for row in iter_rows(input_path):
            rows_in += 1
            played_a = _rewrite_column(
                row.cards_played_a, alias_map, drop_tokens, locator,
                junk_counter, rewrite_counter, unmapped_counter,
            )
            played_b = _rewrite_column(
                row.cards_played_b, alias_map, drop_tokens, locator,
                junk_counter, rewrite_counter, unmapped_counter,
            )
            not_played_a = _rewrite_column(
                row.cards_not_played_a, alias_map, drop_tokens, locator,
                junk_counter, rewrite_counter, unmapped_counter,
            )
            not_played_b = _rewrite_column(
                row.cards_not_played_b, alias_map, drop_tokens, locator,
                junk_counter, rewrite_counter, unmapped_counter,
            )
            # FR-004: a canonical that ended up in both played and
            # not_played (from two distinct faces of the same card)
            # belongs in played only.
            not_played_a -= played_a
            not_played_b -= played_b
            if not (played_a or played_b or not_played_a or not_played_b):
                rows_dropped_empty += 1
                continue
            if out_handle is not None:
                line = _format_row(
                    row.timestamp, row.run_id, row.set_code,
                    row.method_a, row.method_b,
                    played_a, played_b, not_played_a, not_played_b,
                    row.winner, row.starter,
                )
                out_handle.write(line + "\n")
            rows_written += 1
            if rows_in % 100_000 == 0:
                print(f"  ... {rows_in:,} rows processed", file=sys.stderr, flush=True)
    finally:
        if out_handle is not None:
            out_handle.close()

    return {
        "rows_in": rows_in,
        "rows_written": rows_written,
        "rows_dropped_empty": rows_dropped_empty,
        "junk_counter": junk_counter,
        "rewrite_counter": rewrite_counter,
        "unmapped_counter": unmapped_counter,
    }


def _write_report(report_path: Path, stats: dict) -> None:
    """Render the diagnostic report to ``report_path``."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rewrite_counter: Counter[tuple[str, str]] = stats["rewrite_counter"]
    junk_counter: Counter[tuple[str, str]] = stats["junk_counter"]
    unmapped_counter: Counter[str] = stats["unmapped_counter"]

    total_rewrites = sum(rewrite_counter.values())
    total_junk = sum(junk_counter.values())
    total_unmapped = sum(unmapped_counter.values())

    # Rewrites
    rewrites_by_total = Counter()
    for (src, dst), count in rewrite_counter.items():
        rewrites_by_total[(src, dst)] = count

    # Column widths for prettier output
    def _max_width(items: list[str]) -> int:
        return max((len(s) for s in items), default=0)

    rewrite_keys = sorted(
        rewrite_counter.keys(), key=lambda k: (-rewrite_counter[k], k[0]),
    )
    junk_keys = sorted(
        junk_counter.keys(), key=lambda k: (-junk_counter[k], k[0]),
    )
    unmapped_keys = sorted(
        unmapped_counter.keys(), key=lambda n: (-unmapped_counter[n], n),
    )

    src_width = _max_width([s for s, _ in rewrite_keys])
    dst_width = _max_width([d for _, d in rewrite_keys])
    junk_name_width = _max_width([n for n, _ in junk_keys])
    unmapped_width = _max_width(unmapped_keys)

    lines: list[str] = []
    lines.append("# cards-played.txt canonicalization report")
    lines.append("")
    lines.append(f"Input rows:           {stats['rows_in']:>10,}")
    lines.append(f"Output rows written:  {stats['rows_written']:>10,}")
    lines.append(f"Rows dropped (empty): {stats['rows_dropped_empty']:>10,}")
    lines.append(f"Names rewritten:      {total_rewrites:>10,}  "
                 f"({len(rewrite_keys):,} distinct)")
    lines.append(f"Names dropped (junk): {total_junk:>10,}  "
                 f"({len(junk_keys):,} distinct)")
    lines.append(f"Names dropped (unmapped): {total_unmapped:>10,}  "
                 f"({len(unmapped_keys):,} distinct)")
    lines.append("")

    lines.append("## Rewrites")
    if not rewrite_keys:
        lines.append("(none)")
    else:
        for src, dst in rewrite_keys:
            count = rewrite_counter[(src, dst)]
            lines.append(
                f"{src:<{src_width}}  ->  {dst:<{dst_width}}  {count:>8,}"
            )
    lines.append("")

    lines.append("## Dropped (junk pattern)")
    if not junk_keys:
        lines.append("(none)")
    else:
        for name, reason in junk_keys:
            count = junk_counter[(name, reason)]
            lines.append(
                f"{name:<{junk_name_width}}  {count:>8,}  {reason}"
            )
    lines.append("")

    lines.append("## Dropped (unmapped - no alias, no Forge script)")
    if not unmapped_keys:
        lines.append("(none)")
    else:
        for name in unmapped_keys:
            count = unmapped_counter[name]
            lines.append(f"{name:<{unmapped_width}}  {count:>8,}")
    lines.append("")

    # Sanity invariant: every input row was either written or dropped.
    sanity_ok = (
        stats["rows_written"] + stats["rows_dropped_empty"] == stats["rows_in"]
    )
    lines.append(
        f"# Sanity: rows_written ({stats['rows_written']:,}) + "
        f"rows_dropped_empty ({stats['rows_dropped_empty']:,}) "
        f"= rows_in ({stats['rows_in']:,}): "
        f"{'OK' if sanity_ok else 'MISMATCH'}"
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    # UTF-8 stdout so card names with macrons / accents round-trip
    # on Windows (default codec cp1252 chokes on them).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Canonicalize cards-played.txt against AllPrintings.json.",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Input cards-played.txt (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Cleaned-output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT,
        help=f"Diagnostic report path (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--allprintings", type=Path, default=DEFAULT_ALLPRINTINGS,
        help=f"AllPrintings.json (default: {DEFAULT_ALLPRINTINGS})",
    )
    parser.add_argument(
        "--cards-folder", type=Path, default=DEFAULT_CARDS_FOLDER,
        help=f"Converted card corpus (default: {DEFAULT_CARDS_FOLDER})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Emit the report but do not write the cleaned dataset.",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Error: {args.input} does not exist", file=sys.stderr)
        return 1
    if not args.allprintings.exists():
        print(f"Error: {args.allprintings} does not exist", file=sys.stderr)
        return 1

    alias_map, drop_tokens, conflicts = build_alias_map(args.allprintings)
    if conflicts:
        print(
            f"  warning: {len(conflicts)} alias conflicts "
            f"(same name maps to multiple canonical cards) — "
            f"picking lexicographically-smallest. "
            f"First 5: {conflicts[:5]}",
            file=sys.stderr,
        )

    locator = ConvertedCardLocator(args.cards_folder)
    output_path = None if args.dry_run else args.output
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {args.input}...", file=sys.stderr, flush=True)
    stats = _process(args.input, alias_map, drop_tokens, locator, output_path)

    _write_report(args.report, stats)

    print(
        f"\nRows in: {stats['rows_in']:,}  "
        f"written: {stats['rows_written']:,}  "
        f"dropped-empty: {stats['rows_dropped_empty']:,}",
        file=sys.stderr,
    )
    print(
        f"Names rewritten: {sum(stats['rewrite_counter'].values()):,}  "
        f"junk-dropped: {sum(stats['junk_counter'].values()):,}  "
        f"unmapped-dropped: {sum(stats['unmapped_counter'].values()):,}",
        file=sys.stderr,
    )
    print(f"Diagnostic report: {args.report}", file=sys.stderr)
    if args.dry_run:
        print(f"(dry-run: {args.output} NOT written)", file=sys.stderr)
    else:
        print(f"Cleaned dataset: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
