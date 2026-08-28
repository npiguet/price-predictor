"""Extract Forge-side card annotations that the sealed pipeline never sees.

Two annotation sources live in the Forge repo, both hand-maintained by Forge
contributors and both *upstream* of every deck in the training corpus:

1. ``AI:RemoveDeck:{All,Random,NonCommander}`` lines in the card scripts
   (``forge-gui/res/cardsfolder/**/*.txt``). Forge's own deck builders treat
   these as a soft/hard blacklist -- ``CardRanker.getScores`` subtracts 20.0
   from a card's pick score when ``getRemAIDecks()`` is set. Any card the
   Forge builders systematically refuse to play is therefore under-represented
   in the *played* half of the win-rate labels, which is how a human-authored
   blacklist can leak into a learned scorer ("blacklist inheritance").

2. The limited draft rankings under ``forge-gui/res/draft/rankings/*.rnk``,
   read by ``forge.gamemodes.limited.ReadDraftRankings`` (via ``DraftRankCache``
   / ``CardRanker.getRawScore``). Each file is one set's human pick order:

       //Rank|Name|Rarity|Set
       #1|Lavalanche|R|ARB
       #2|Behemoth Sledge|U|ARB

   Forge normalizes these to ``rank / max_rank_in_that_set``, i.e. a number in
   (0, 1] where **lower is better** (#1 is the best card in the set). This is
   distilled human limited taste; correlating it with the empirical win-rate
   labels tests how much "borrowed human taste" the labels encode.

Output: ``forge_hints.csv`` -- one row per referable card name with

    name                 Forge card name (primary face; split cards also get
                         an "A // B" row so pool/win-rate spellings join)
    ai_remove_deck       0/1
    ai_remove_deck_kind  All | Random | NonCommander | "" (which flavour)
    draft_rank           mean normalized rank over the sets the card is in
                         (lower = better), empty when unranked
    draft_rank_best      min normalized rank over those sets, empty when unranked
    draft_rank_n_sets    how many .rnk files list the card
    mv                   mana value parsed from the primary ``ManaCost:`` line
    is_creature          0/1 from the primary ``Types:`` line

``mv``/``is_creature`` are carried here (rather than re-derived from the
converted-card corpus) so downstream probes can bucket cards for matched
comparisons without a second pass over 30k files.

Read-only with respect to the Forge repo. CPU only, ~30s for the full scan.
"""

from __future__ import annotations

import argparse
import csv
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

FORGE = Path(__file__).resolve().parents[2].parent / "forge"  # sibling checkout
CARDSFOLDER = FORGE / "forge-gui" / "res" / "cardsfolder"
RANKINGS_DIR = FORGE / "forge-gui" / "res" / "draft" / "rankings"
SCRATCH = Path(__file__).resolve().parents[2] / "output" / "scorer-probes"
SCRATCH.mkdir(parents=True, exist_ok=True)
DEFAULT_OUT = SCRATCH / "forge_hints.csv"

REMOVE_KINDS = ("All", "Random", "NonCommander")


# ------------------------------------------------------------------ utilities

def strip_accents(s: str) -> str:
    """ASCII-fold, matching ``StringUtils.stripAccents`` in ReadDraftRankings."""
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def rank_key(name: str) -> str:
    """Lookup key Forge uses for the rankings map (see getRanking)."""
    return strip_accents(name).replace(" // ", " ").strip()


def parse_mana_cost(cost: str) -> float | None:
    """Mana value from a Forge ``ManaCost:`` string.

    Tokens are space-separated: bare digits are generic, ``X``/``Y``/``Z`` count
    0, hybrid-with-generic (``2/B``, ``2B``) counts its digit, everything else
    (colored, snow, colorless-C, Phyrexian, hybrid colored pairs) counts 1.
    Crude by design -- it only has to be good enough to bucket cards.
    """
    cost = cost.strip()
    if not cost or cost.lower() == "no cost":
        return None
    total = 0.0
    for tok in cost.split():
        if tok.isdigit():
            total += int(tok)
        elif tok.upper() in ("X", "Y", "Z"):
            continue
        elif tok[0].isdigit():          # "2/B" / "2B" hybrid
            total += int(tok[0])
        else:
            total += 1
    return total


# -------------------------------------------------------------- card scripts

def scan_card_scripts(limit: int | None = None) -> tuple[list[dict], Counter]:
    """Parse every card script; one record per script file."""
    stats: Counter = Counter()
    records: list[dict] = []
    paths = sorted(CARDSFOLDER.rglob("*.txt"))
    stats["files_found"] = len(paths)
    if limit:
        paths = paths[:limit]

    for path in paths:
        names: list[str] = []
        mana: str | None = None
        types: str | None = None
        kinds: list[str] = []
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n").rstrip("\r")
                    if line.startswith("Name:"):
                        names.append(line[5:].strip())
                    elif line.startswith("ManaCost:") and mana is None:
                        mana = line[9:].strip()
                    elif line.startswith("Types:") and types is None:
                        types = line[6:].strip()
                    elif line.startswith("AI:"):
                        # e.g. "AI:RemoveDeck:All"; DeckHas/DeckHints are
                        # separate keys and deliberately ignored.
                        body = line[3:]
                        for kind in REMOVE_KINDS:
                            if body.strip() == f"RemoveDeck:{kind}":
                                kinds.append(kind)
                        stats[f"ai_line:{body.strip()}"] += 1
        except OSError:
            stats["unreadable"] += 1
            continue

        stats["files_parsed"] += 1
        if not names:
            stats["no_name_line"] += 1
            continue
        if kinds:
            stats["scripts_with_removedeck"] += 1
            stats[f"removedeck_{kinds[0]}"] += 1

        rec = {
            "primary": names[0],
            "faces": names,
            "kind": kinds[0] if kinds else "",
            "mv": parse_mana_cost(mana or ""),
            "is_creature": int("creature" in (types or "").lower()),
        }
        records.append(rec)
    return records, stats


# ---------------------------------------------------------- draft rankings

def read_rankings() -> tuple[dict[str, list[float]], Counter]:
    """``rank_key -> [normalized rank per set]`` from ``res/draft/rankings/*.rnk``.

    Mirrors ``ReadDraftRankings.readFile``: '//' comments skipped, parsing stops
    at the first blank line, fields are ``#rank|Name|Rarity|Set``, and each
    set's normalizer is the largest rank seen for that set.
    """
    stats: Counter = Counter()
    raw: list[tuple[str, str, int]] = []       # (edition, name_key, rank)
    set_sizes: dict[str, int] = {}

    files = sorted(RANKINGS_DIR.glob("*.rnk"))
    stats["rnk_files"] = len(files)
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n").rstrip("\r")
                if not line:
                    break                      # Forge stops at the first blank
                if line.startswith("//"):
                    continue
                parts = line.split("|")
                if len(parts) < 4:
                    stats["malformed"] += 1
                    continue
                try:
                    rank = int(parts[0].strip()[1:])
                except ValueError:
                    stats["bad_rank"] += 1
                    continue
                edition = parts[3].strip()
                key = rank_key(parts[1].strip())
                raw.append((edition, key, rank))
                set_sizes[edition] = max(set_sizes.get(edition, 0), rank)

    by_name: dict[str, list[float]] = defaultdict(list)
    for edition, key, rank in raw:
        size = set_sizes.get(edition, 0)
        if size <= 0:
            continue
        by_name[key].append(rank / size)
    stats["ranking_entries"] = len(raw)
    stats["ranking_editions"] = len(set_sizes)
    stats["ranking_distinct_names"] = len(by_name)
    return dict(by_name), stats


# ------------------------------------------------------------------- assembly

def build_rows(records: list[dict], ranks: dict[str, list[float]]) -> tuple[list[dict], Counter]:
    stats: Counter = Counter()
    rows: dict[str, dict] = {}

    def emit(name: str, rec: dict) -> None:
        name = name.strip()
        if not name or name in rows:
            return
        vals = ranks.get(rank_key(name), [])
        if vals:
            stats["rows_with_rank"] += 1
        rows[name] = {
            "name": name,
            "ai_remove_deck": int(bool(rec["kind"])),
            "ai_remove_deck_kind": rec["kind"],
            "draft_rank": f"{sum(vals) / len(vals):.6f}" if vals else "",
            "draft_rank_best": f"{min(vals):.6f}" if vals else "",
            "draft_rank_n_sets": len(vals),
            "mv": "" if rec["mv"] is None else f"{rec['mv']:g}",
            "is_creature": rec["is_creature"],
        }

    for rec in records:
        emit(rec["primary"], rec)
        # Split / DFC / adventure cards are referenced elsewhere by the joined
        # "A // B" spelling; give that form its own row with the same hints.
        if len(rec["faces"]) == 2:
            emit(" // ".join(rec["faces"]), rec)
            stats["joined_face_rows"] += 1

    matched_keys = {rank_key(n) for n in rows}
    stats["ranking_names_unmatched"] = sum(
        1 for k in ranks if k not in matched_keys)
    return list(rows.values()), stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None, help="CSV output path")
    ap.add_argument("--limit", type=int, default=None,
                    help="max card scripts to scan (default: all)")
    ap.add_argument("--smoke", action="store_true",
                    help="scan 2000 scripts, write forge_hints_smoke.csv")
    args = ap.parse_args()

    limit = args.limit if args.limit is not None else (2000 if args.smoke else None)
    out = args.out or (SCRATCH / "forge_hints_smoke.csv" if args.smoke
                       else DEFAULT_OUT)

    print(f"cardsfolder : {CARDSFOLDER}")
    print(f"rankings    : {RANKINGS_DIR}")
    ranks, rstats = read_rankings()
    print(f"rankings    : {rstats['rnk_files']} .rnk files, "
          f"{rstats['ranking_entries']} entries, "
          f"{rstats['ranking_editions']} editions, "
          f"{rstats['ranking_distinct_names']} distinct names "
          f"(malformed {rstats['malformed']}, bad rank {rstats['bad_rank']})")

    records, cstats = scan_card_scripts(limit)
    print(f"card scripts: {cstats['files_parsed']} parsed of "
          f"{cstats['files_found']} found"
          + (f" (limit {limit})" if limit else "")
          + f"; {cstats['no_name_line']} with no Name: line")

    rows, bstats = build_rows(records, ranks)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name", "ai_remove_deck", "ai_remove_deck_kind", "draft_rank",
            "draft_rank_best", "draft_rank_n_sets", "mv", "is_creature"])
        w.writeheader()
        w.writerows(rows)

    n_rem = sum(r["ai_remove_deck"] for r in rows)
    n_rank = sum(1 for r in rows if r["draft_rank"] != "")
    n_creat = sum(r["is_creature"] for r in rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")
    print(f"  ai_remove_deck=1        : {n_rem} "
          f"({100 * n_rem / max(len(rows), 1):.1f}%)")
    for kind in REMOVE_KINDS:
        n = sum(1 for r in rows if r["ai_remove_deck_kind"] == kind)
        print(f"    RemoveDeck:{kind:<13}: {n}")
    print(f"  draft_rank present      : {n_rank} "
          f"({100 * n_rank / max(len(rows), 1):.1f}%)")
    print(f"  is_creature=1           : {n_creat}")
    print(f"  mv present              : {sum(1 for r in rows if r['mv'] != '')}")
    print(f"  joined 'A // B' rows    : {bstats['joined_face_rows']}")
    print(f"  ranking names with no card script: "
          f"{bstats['ranking_names_unmatched']}")

    both = [r for r in rows if r["ai_remove_deck"] and r["draft_rank"] != ""]
    print(f"  ranked AND blacklisted  : {len(both)}")


if __name__ == "__main__":
    main()
