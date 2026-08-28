"""T0: observational score landscape.

Samples decks of every builder method from the match corpora, scores them with
the production scorer, computes per-deck macro features, and writes a CSV for
downstream statistical cuts.
"""

from __future__ import annotations

import gzip
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

import probe_lib as pl

OUT = pl.SCRATCH / "t0_decks.csv"
PER_METHOD_CAP = 4000

MATCH_FILES = [
    "matches-b07/match-outcomes-gen0.txt.gz",
    "matches-b07/match-outcomes-gen1-vs-0.txt.gz",
    "matches-b07/match-outcomes-gen2-vs-0-1.txt.gz",
    "matches-b07/match-outcomes-gen3-vs-0-2.txt",
    "matches-b07/match-outcomes-gen3-vs-forge-best.txt",
    "matches-b07/match-outcomes-gen4-vs-forge-best-gen3.txt",
    "matches-b07/match-outcomes-gen5-vs-gen4-forge.txt",
]


def iter_match_lines(path: Path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(";")
            if len(parts) >= 10:
                yield parts


def collect_decks():
    """Dedup by content; cap per method. Returns list of (method, set, deck)."""
    seen: set[str] = set()
    per_method: Counter = Counter()
    out = []
    for rel in MATCH_FILES:
        path = pl.YDATA / rel
        if not path.exists():
            print("missing:", path)
            continue
        for parts in iter_match_lines(path):
            set_code = parts[2]
            for m_idx, d_idx in ((3, 5), (4, 6)):
                method = parts[m_idx]
                if per_method[method] >= PER_METHOD_CAP:
                    continue
                deck = parts[d_idx].split("|")
                key = hashlib.md5(
                    ("|".join(sorted(deck)) + set_code).encode()
                ).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                per_method[method] += 1
                out.append((method, set_code, deck))
    print("collected:", dict(per_method))
    return out


def deck_features(probe: pl.Probe, deck: list[str], text_cache: dict) -> dict | None:
    nonbasic = [c for c in deck if c.lower() not in pl.BASIC_LAND_NAMES]
    mats, feats = [], []
    for name in nonbasic:
        e = probe.embedding(name)
        if e is None:
            return None
        mats.append(e)
        if name not in text_cache:
            t = probe.locator.load_text(name)
            text_cache[name] = pl.card_features(t.text) if t else None
        feats.append(text_cache[name])
    if any(f is None for f in feats):
        return None

    m = np.stack(mats)
    det = m[:, -pl.layout.FEATURE_COUNT:]
    pips = det[:, 1:6]  # W U B R G pip counts
    is_land = det[:, pl.layout.IS_LAND] > 0.5
    spells = ~is_land
    mv = det[:, pl.layout.MANA_VALUE]

    pip_sum = pips.sum(axis=0)
    colors_present = int((pip_sum > 0).sum())
    n_spells = int(spells.sum())
    if n_spells == 0:
        return None
    spell_mv = mv[spells]
    sorted_pips = np.sort(pip_sum)[::-1]
    total_pips = float(pip_sum.sum())

    return {
        "n_cards": len(nonbasic),
        "n_spells": n_spells,
        "n_nonbasic_lands": int(is_land.sum()),
        "n_colors": colors_present,
        "avg_mv": float(spell_mv.mean()),
        "mv_le2": int((spell_mv <= 2).sum()),
        "mv_3_4": int(((spell_mv >= 3) & (spell_mv <= 4)).sum()),
        "mv_ge5": int((spell_mv >= 5).sum()),
        "mv_ge6": int((spell_mv >= 6).sum()),
        "primary_pip_share": float(sorted_pips[0] / total_pips) if total_pips else 0.0,
        "splash_pips": float(sorted_pips[2:].sum()),
        "n_creatures": sum(1 for f in feats if f["is_creature"]),
        "n_removal": sum(1 for f in feats if f["is_removal"]),
        "n_evasion": sum(1 for f in feats if f["has_evasion"]),
        "n_flying": sum(1 for f in feats if f["has_flying"]),
        "n_vanilla": sum(1 for f in feats if f["vanilla"]),
        "n_carddraw": sum(1 for f in feats if f["draws_cards"] and not f["is_creature"]),
        "n_instant": sum(1 for f in feats if f["is_instant"]),
        "n_sorcery": sum(1 for f in feats if f["is_sorcery"]),
        "n_aura": sum(1 for f in feats if f["is_aura"]),
        "n_equipment": sum(1 for f in feats if f["is_equipment"]),
        "sum_power": float(np.nansum([f["power"] or 0 for f in feats])),
        "sum_tough": float(np.nansum([f["toughness"] or 0 for f in feats])),
        "n_dupes": len(nonbasic) - len(set(nonbasic)),
    }


def main():
    probe = pl.Probe()
    wr = pl.load_win_rates()
    decks = collect_decks()

    text_cache: dict = {}
    rows, mats = [], []
    skipped = 0
    for method, set_code, deck in decks:
        try:
            feat = deck_features(probe, deck, text_cache)
        except KeyError:
            feat = None
        if feat is None:
            skipped += 1
            continue
        nonbasic = [c for c in deck if c.lower() not in pl.BASIC_LAND_NAMES]
        labels = [wr.get(c) for c in nonbasic]
        sp = [r["shrunk_score_play"] for r in labels if r and r["shrunk_score_play"] is not None]
        cl = [r["shrunk_cast_lift"] for r in labels if r and r["shrunk_cast_lift"] is not None]
        pr = [r["shrunk_played_rate"] for r in labels if r and r["shrunk_played_rate"] is not None]
        feat.update(
            method=method,
            set_code=set_code,
            mean_score_play=float(np.mean(sp)) if sp else np.nan,
            mean_cast_lift=float(np.mean(cl)) if cl else np.nan,
            mean_played_rate=float(np.mean(pr)) if pr else np.nan,
        )
        rows.append(feat)
        mats.append(probe.deck_matrix(deck))

    print(f"featurized {len(rows)} decks ({skipped} skipped); scoring...")
    scores = probe.score_matrices(mats)
    for r, s in zip(rows, scores):
        r["score"] = float(s)

    import csv
    cols = list(rows[0].keys())
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
