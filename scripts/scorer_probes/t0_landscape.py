"""T0: observational score landscape.

Samples decks of every builder method from the match corpora, scores them with
the production scorer, computes per-deck macro features, and writes a CSV for
downstream statistical cuts.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorer_probes import probe_lib as pl

OUT = pl.SCRATCH / "t0_decks.csv"
PER_METHOD_CAP = 4000


def collect_decks() -> list[pl.MatchDeck]:
    decks = list(pl.sample_match_decks(pl.MATCH_PATHS, PER_METHOD_CAP))
    print("collected:", dict(Counter(d.method for d in decks)))
    return decks


def deck_features(probe: pl.Probe, deck: list[str], text_cache: dict) -> dict | None:
    nonbasic = pl.nonbasic(deck)
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
    pips = det[:, pl.layout.COLOR_PIPS]
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
    for d in decks:
        try:
            feat = deck_features(probe, d.cards, text_cache)
        except KeyError:
            feat = None
        if feat is None:
            skipped += 1
            continue
        labels = [wr.get(c) for c in pl.nonbasic(d.cards)]
        sp = [r["shrunk_score_play"] for r in labels if r and r["shrunk_score_play"] is not None]
        cl = [r["shrunk_cast_lift"] for r in labels if r and r["shrunk_cast_lift"] is not None]
        pr = [r["shrunk_played_rate"] for r in labels if r and r["shrunk_played_rate"] is not None]
        feat.update(
            method=d.method,
            set_code=d.set_code,
            mean_score_play=float(np.mean(sp)) if sp else np.nan,
            mean_cast_lift=float(np.mean(cl)) if cl else np.nan,
            mean_played_rate=float(np.mean(pr)) if pr else np.nan,
        )
        rows.append(feat)
        mats.append(probe.deck_matrix(d.cards))

    print(f"featurized {len(rows)} decks ({skipped} skipped); scoring...")
    scores = probe.score_matrices(mats)
    for r, s in zip(rows, scores):
        r["score"] = float(s)

    cols = list(rows[0].keys())
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
