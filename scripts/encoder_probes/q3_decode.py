"""R7 — what the 512-dim embedding encodes of the visible card.

Linear decodability, honest split (ridge fitted on the encoder's own train
cards, scored on its held-out val cards): mana value, pip count, power,
toughness, P+T, creature-ness, ability-line count, first-printing year and
first-printing rarity. Plus the two follow-ups the C battery set up: where
``{X}`` costs land on the decoded-MV axis, and whether decoded power
collapses past 8 the way the 12/12 counterfactual sweep did.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q_common as qc  # noqa: E402

REG_TARGETS = (
    ("mv", "mana value", None),
    ("pips_total", "coloured pip count", None),
    ("power", "power", "creature"),
    ("toughness", "toughness", "creature"),
    ("p_plus_t", "power + toughness", "creature"),
    ("n_abilities", "ability-line count", None),
    ("first_year", "first-printing year", None),
)
BIN_TARGETS = (
    ("is_creature", "is a creature"),
    ("is_land", "is a land"),
    ("rare_plus", "first printing rare or mythic"),
)


def main() -> None:
    join, emb = qc.load_frame()
    is_train = (join["split"] == "train").to_numpy()
    is_val = (join["split"] == "val").to_numpy()

    join["p_plus_t"] = (pd.to_numeric(join["power"], errors="coerce")
                        + pd.to_numeric(join["toughness"], errors="coerce"))
    join["rare_plus"] = join["first_rarity"].isin(["rare", "mythic"]).astype(float)
    join["first_year"] = pd.to_numeric(join["first_year"], errors="coerce")
    creature = join["is_creature"].fillna(0).to_numpy(float) > 0

    rows, fits = [], {}
    for col, label, gate in REG_TARGETS:
        y = pd.to_numeric(join[col], errors="coerce").to_numpy(float)
        keep = np.isfinite(y)
        if gate == "creature":
            keep &= creature
        r = qc.honest_ridge(emb, y, is_train, is_val, keep)
        fits[col] = r
        va = r["val_mask"]
        rows.append({
            "target": label, "kind": "R2", "n_val": r["n_val"],
            "val_r2": r["val_r2"], "val_pearson": r["val_pearson"],
            "val_sd": float(np.std(y[va])),
            "resid_sd": float(np.std(y[va] - r["val_pred"])),
        })
        print("decode", col, round(r["val_r2"], 4), flush=True)

    for col, label in BIN_TARGETS:
        y = pd.to_numeric(join[col], errors="coerce").fillna(0).to_numpy(float)
        r = qc.honest_ridge(emb, y, is_train, is_val, np.isfinite(y))
        va = r["val_mask"]
        rows.append({
            "target": label, "kind": "AUC", "n_val": r["n_val"],
            "val_r2": r["val_r2"], "val_pearson": r["val_pearson"],
            "auc": qc.auc(y[va] > 0.5, r["val_pred"]),
            "base_rate": float((y[va] > 0.5).mean()),
        })
        print("decode", col, round(r["val_r2"], 4), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(qc.SCRATCH / "q3_decodability.csv", index=False)

    # ── MV as a classification into 0..8+ ────────────────────────────────
    mv = pd.to_numeric(join["mv"], errors="coerce").to_numpy(float)
    r_mv = fits["mv"]
    va = r_mv["val_mask"]
    mv_true = np.clip(mv[va], 0, 8)
    mv_hat = np.clip(np.round(r_mv["val_pred"]), 0, 8)
    mv_cls = {
        "n_val": int(va.sum()),
        "exact_accuracy": float((mv_hat == mv_true).mean()),
        "within_1_accuracy": float((np.abs(mv_hat - mv_true) <= 1).mean()),
        "majority_baseline": float(pd.Series(mv_true).value_counts(normalize=True).max()),
    }
    by_mv = pd.DataFrame({"true": mv_true, "pred": r_mv["val_pred"][:]}) \
        .groupby("true")["pred"].agg(["count", "mean", "std"]).reset_index()
    by_mv.to_csv(qc.SCRATCH / "q3_mv_by_class.csv", index=False)

    # ── the {X} question: decode MV from a no-X, MV<=7 fit ───────────────
    has_x = pd.to_numeric(join["has_x"], errors="coerce").fillna(0).to_numpy(float) > 0
    fit_keep = np.isfinite(mv) & (~has_x) & (mv <= 7)
    r_nox = qc.honest_ridge(emb, mv, is_train, is_val, fit_keep)
    coef, b = r_nox["coef"], r_nox["intercept"]
    decoded = emb @ coef + b
    x_val = has_x & is_val & np.isfinite(mv)
    nox_val = (~has_x) & is_val & np.isfinite(mv) & (mv <= 7)
    x_rows = []
    for name, m in (("no-X (fit class, MV<=7)", nox_val), ("{X} cards", x_val)):
        x_rows.append({
            "class": name, "n": int(m.sum()),
            "mean_printed_mv": float(np.mean(mv[m])),
            "mean_decoded_mv": float(np.mean(decoded[m])),
            "mean_gap": float(np.mean(decoded[m] - mv[m])),
            "median_decoded_mv": float(np.median(decoded[m])),
        })
    # {X} cards split by their printed (X=0) MV
    for lo, hi in ((0, 1), (2, 2), (3, 3), (4, 9)):
        m = x_val & (mv >= lo) & (mv <= hi)
        if m.sum() < 10:
            continue
        x_rows.append({
            "class": f"{{X}} with printed MV {lo}-{hi}" if lo != hi
                     else f"{{X}} with printed MV {lo}",
            "n": int(m.sum()),
            "mean_printed_mv": float(np.mean(mv[m])),
            "mean_decoded_mv": float(np.mean(decoded[m])),
            "mean_gap": float(np.mean(decoded[m] - mv[m])),
            "median_decoded_mv": float(np.median(decoded[m])),
        })
        # the matched no-X control at the same printed MV
        c = nox_val & (mv >= lo) & (mv <= hi)
        x_rows.append({
            "class": f"  control no-X, MV {lo}-{hi}" if lo != hi
                     else f"  control no-X, MV {lo}",
            "n": int(c.sum()),
            "mean_printed_mv": float(np.mean(mv[c])),
            "mean_decoded_mv": float(np.mean(decoded[c])),
            "mean_gap": float(np.mean(decoded[c] - mv[c])),
            "median_decoded_mv": float(np.median(decoded[c])),
        })
    x_table = pd.DataFrame(x_rows)
    x_table.to_csv(qc.SCRATCH / "q3_x_cost.csv", index=False)

    # ── decoded power vs printed power, real cards, 0..12+ ───────────────
    pw = pd.to_numeric(join["power"], errors="coerce").to_numpy(float)
    r_pw = fits["power"]
    pw_pred_all = emb @ r_pw["coef"] + r_pw["intercept"]
    m = is_val & creature & np.isfinite(pw)
    bins = np.clip(pw[m], 0, 12)
    pw_tab = pd.DataFrame({"printed": bins, "decoded": pw_pred_all[m]}) \
        .groupby("printed")["decoded"].agg(["count", "mean", "std"]).reset_index()
    pw_tab.to_csv(qc.SCRATCH / "q3_power_ladder.csv", index=False)

    summary = {"mv_classification": mv_cls,
               "n_x_cards_val": int(x_val.sum()),
               "mv_nox_fit_val_r2": r_nox["val_r2"]}
    with open(qc.SCRATCH / "q3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(table.round(4).to_string(index=False))
    print(json.dumps(mv_cls, indent=2))
    print(by_mv.round(3).to_string(index=False))
    print(x_table.round(3).to_string(index=False))
    print(pw_tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
