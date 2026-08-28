"""R2a/R2b — memorization vs generalization, decomposed.

R2a splits the fidelity-vs-honest gap into its two causes. A probe fitted
on a random **half** of the encoder's own train split is evaluated on
(i) the held-out half of train and (ii) the encoder's val split. Both are
probe-honest, so (i) − (ii) is *encoder* memorization and fidelity_cv −
(i) is probe overfit. Reported per head and per ``n_in_deck`` decile,
and normalised by the label's split-half reliability at that n (from
``l_report.md`` §f) into a fraction-of-explainable.

R2b characterises the 171 identical-text equivalence classes against the
corpus (mana value, type mix, line count) so the ≤0.515 text-explainable
bound can be quoted with its representativeness caveat.

Outputs ``output/encoder-probes/r2a_*.csv`` + ``r2ab_summary.json``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl  # noqa: E402

HEADS = ("score_play", "score_draw", "played_rate", "cast_lift")
SEED = 2024
N_DECILES = 10

# l_report.md §f — Spearman-Brown split-half reliability of score_play by
# min-half observation count. Halves are ~n_in_deck/2 each, so the bucket
# edges are doubled to land on the n_in_deck scale.
RELIABILITY_BUCKETS = [
    (50, 200, 0.533), (200, 500, 0.731), (500, 1000, 0.797),
    (1000, 2000, 0.818), (2000, 6000, 0.877), (6000, np.inf, 0.933),
]
RELIABILITY_PLAYED_RATE = 0.988  # corpus-wide; the head is nearly noise-free


def reliability_at(n: float) -> float:
    for lo, hi, r in RELIABILITY_BUCKETS:
        if lo <= n < hi:
            return r
    return RELIABILITY_BUCKETS[0][2] if n < 50 else RELIABILITY_BUCKETS[-1][2]


def wr2(y, pred, w):
    mean = float((w * y).sum() / w.sum())
    ss_res = float((w * (y - pred) ** 2).sum())
    ss_tot = float((w * (y - mean) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def resid_sd(y, pred, w):
    return float(np.sqrt((w * (y - pred) ** 2).sum() / w.sum()))


def main() -> None:
    join = pl.build_join()
    join = join[join["is_primary"]].reset_index(drop=True)
    emb = pl.load_embedding_matrix(list(join["name"]), join)

    is_train = (join["split"] == "train").to_numpy()
    is_val = (join["split"] == "val").to_numpy()
    n_in_deck = join["n_in_deck"].to_numpy(float)

    rng = np.random.default_rng(SEED)
    half = np.zeros(len(join), dtype=bool)
    tr_idx = np.flatnonzero(is_train)
    half[rng.choice(tr_idx, len(tr_idx) // 2, replace=False)] = True   # fit half
    train_holdout = is_train & ~half

    # n_in_deck deciles over the whole joined corpus
    edges = np.quantile(n_in_deck, np.linspace(0, 1, N_DECILES + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    decile = np.clip(np.digitize(n_in_deck, edges[1:-1]), 0, N_DECILES - 1)

    honest = pl.load_probes("honest", True)
    fidelity = pl.load_probes("fidelity", True)

    rows: list[dict] = []
    summary: dict = {"n_fit_half": int(half.sum()),
                     "n_train_holdout": int(train_holdout.sum()),
                     "n_val": int(is_val.sum()), "deciles": list(map(float, edges[1:-1]))}

    for head in HEADS:
        y = pd.to_numeric(join[f"shrunk_{head}"], errors="coerce").to_numpy(float)
        w = join[f"w_{head}"].to_numpy(float)
        have = np.isfinite(y) & (w > 0)

        fit = half & have
        alpha, _ = pl._choose_alpha(emb[fit], y[fit], w[fit])
        coef, b = pl._ridge_solve(emb[fit], y[fit], w[fit], [alpha])[alpha]
        pred_half = emb @ coef + b

        pred_honest = honest.probes[head].predict(emb)
        pred_fid = fidelity.probes[head].predict(emb)

        groups = {
            "train_holdout": train_holdout & have,
            "val": is_val & have,
            "fit_half": fit,
        }
        head_rec: dict = {"alpha_half_probe": float(alpha)}
        for gname, mask in groups.items():
            head_rec[f"halfprobe_r2_{gname}"] = wr2(y[mask], pred_half[mask], w[mask])
            head_rec[f"halfprobe_residsd_{gname}"] = resid_sd(y[mask], pred_half[mask], w[mask])
        head_rec["honest_r2_train"] = wr2(y[is_train & have], pred_honest[is_train & have],
                                          w[is_train & have])
        head_rec["honest_r2_val"] = wr2(y[is_val & have], pred_honest[is_val & have],
                                        w[is_val & have])
        head_rec["fidelity_cv_r2"] = fidelity.probes[head].metrics["cv_r2"]
        head_rec["encoder_memorization_gap"] = (head_rec["halfprobe_r2_train_holdout"]
                                                - head_rec["halfprobe_r2_val"])
        head_rec["probe_overfit_gap"] = (head_rec["fidelity_cv_r2"]
                                         - head_rec["halfprobe_r2_train_holdout"])
        summary[head] = head_rec

        for d in range(N_DECILES):
            in_d = decile == d
            rec = {"head": head, "decile": d + 1,
                   "n_lo": float(n_in_deck[in_d].min()),
                   "n_hi": float(n_in_deck[in_d].max()),
                   "n_median": float(np.median(n_in_deck[in_d]))}
            rel = (RELIABILITY_PLAYED_RATE if head == "played_rate"
                   else reliability_at(rec["n_median"]))
            rec["reliability"] = rel
            for gname, base in (("train_holdout", train_holdout), ("val", is_val)):
                m = in_d & base & have
                if m.sum() < 30:
                    continue
                rec[f"r2_{gname}"] = wr2(y[m], pred_half[m], w[m])
                rec[f"residsd_{gname}"] = resid_sd(y[m], pred_half[m], w[m])
                rec[f"n_{gname}"] = int(m.sum())
                rec[f"frac_explainable_{gname}"] = rec[f"r2_{gname}"] / rel
            m = in_d & is_val & have
            if m.sum() >= 30:
                rec["honest_r2_val"] = wr2(y[m], pred_honest[m], w[m])
            m = in_d & have
            if m.sum() >= 30:
                rec["fidelity_r2_all"] = wr2(y[m], pred_fid[m], w[m])
            rows.append(rec)

    dec = pd.DataFrame(rows)
    dec.to_csv(pl.SCRATCH / "r2a_decile_r2.csv", index=False)

    # ── R2b: are the equivalence classes representative? ────────────────
    classes = pl.equivalence_classes(join)
    member_names = {n for c in classes for n in c["names"]}

    def card_stats(path: str) -> dict:
        lines = [l for l in Path(path).read_text(encoding="utf-8", errors="replace")
                 .splitlines() if l.strip() and not l.startswith("name:")]
        cost = next((l for l in lines if l.startswith("mana cost:")), "")
        mv = 0.0
        for sym in re.findall(r"\{([^}]*)\}", cost):
            mv += int(sym) if sym.isdigit() else (0 if sym.upper() == "X" else 1)
        types = next((l for l in lines if l.startswith("types:")), "")
        ab = [l for l in lines if l.startswith(("static:", "spell[", "activated[",
                                                "triggered", "replacement"))]
        return {"mv": mv, "n_lines": len(lines), "n_ability_lines": len(ab),
                "is_creature": "creature" in types, "is_land": "land" in types,
                "is_instant_sorcery": ("instant" in types or "sorcery" in types)}

    stats = pd.DataFrame([card_stats(p) for p in join["txt_path"]])
    stats["name"] = join["name"].to_numpy()
    stats["n_in_deck"] = n_in_deck
    stats["rarity"] = join["first_rarity"].to_numpy()
    stats["in_class"] = stats["name"].isin(member_names)

    cols = ["mv", "n_lines", "n_ability_lines", "is_creature", "is_land",
            "is_instant_sorcery", "n_in_deck"]
    rep = stats.groupby("in_class")[cols].mean().T
    rep.columns = ["corpus", "class_members"]
    rep["diff"] = rep["class_members"] - rep["corpus"]
    rep.to_csv(pl.SCRATCH / "r2b_class_representativeness.csv")

    rar = (stats.groupby("in_class")["rarity"].value_counts(normalize=True)
           .unstack(0).fillna(0.0))
    rar.columns = ["corpus", "class_members"]
    rar.to_csv(pl.SCRATCH / "r2b_class_rarity.csv")

    var0 = pl.variance_decomposition(classes, join)
    var800 = pl.variance_decomposition(classes, join, min_n=800)
    summary["r2b"] = {
        "n_classes": len(classes),
        "n_members": int(stats["in_class"].sum()),
        "variance_min_n_0": var0,
        "variance_min_n_800": var800,
        "representativeness": rep.to_dict(),
        "rarity_mix": rar.to_dict(),
    }

    with open(pl.SCRATCH / "r2ab_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(json.dumps({h: summary[h] for h in HEADS}, indent=2, default=float))
    print()
    print(rep.to_string())
    print()
    print(dec[dec["head"] == "score_play"].to_string(index=False))


if __name__ == "__main__":
    main()
