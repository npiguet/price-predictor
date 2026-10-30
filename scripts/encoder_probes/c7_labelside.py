"""C7 — the label-side column the C-series divergence flags are read against.

The counterfactual batteries measure what the *encoder* charges for an
edit. To call a result a divergence we need the matching number from the
labels themselves, computed on the same feature definitions and quoted in
the same label-SD units. This script fits MV/type-controlled weighted
least squares on the labels for every feature the C battery edits:
keywords, the P-T asymmetry, spell-effect phrases, tribes, and type
lines. It also fits the identical regressions on the *predicted* labels,
which separates "the encoder disagrees with the labels" from "the
correlational and counterfactual readings of the same encoder differ".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402
from _lib.stats import wls  # noqa: E402

PHRASES = {
    "destroy target creature.": "ph_destroy_target_creature",
    "exile target creature.": "ph_exile_target_creature",
    "CARDNAME deals N damage to any target.": "ph_damage_any_target",
    "CARDNAME deals N damage to target creature.": "ph_damage_target_creature",
    "target player sacrifices a creature.": "ph_edict",
    "destroy all creatures.": "ph_sweeper",
    "fight": "ph_fight",
    "tap target creature.": "ph_tap_target",
    "counter target spell.": "ph_counterspell",
    "draw a card.": "ph_draw_a_card",
    "draw two cards.": "ph_draw_multi",
    "you gain N life.": "ph_gain_life",
    "+N/+N until end of turn": "ph_pump_eot",
    "create a token": "ph_create_token",
    "add mana": "ph_add_mana",
}


def num(df: pd.DataFrame, col: str, fill: float | None = None) -> np.ndarray:
    v = pd.to_numeric(df[col], errors="coerce")
    return (v if fill is None else v.fillna(fill)).to_numpy(float)


def flag(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].fillna(False).astype(bool).to_numpy()


def effect(df: pd.DataFrame, is_feature: np.ndarray, ycol: str,
           controls: list[np.ndarray], wcol: str = "w_score_play") -> dict:
    """WLS coefficient (HC0 SE) of ``is_feature`` on ``ycol`` over ``controls``."""
    y = num(df, ycol)
    w = df[wcol].to_numpy(float)
    X = np.column_stack([np.ones(len(df)), is_feature.astype(float)] + controls)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1) & (w > 0)
    if ok.sum() < 30 or is_feature[ok].sum() < 10:
        return {"n": int(is_feature[ok].sum()), "beta": np.nan, "se": np.nan}
    fit = wls(X[ok], y[ok], w[ok])
    return {"n": int(is_feature[ok].sum()), "beta": float(fit.beta[1]), "se": float(fit.se[1])}


def mv_controls(df: pd.DataFrame, statline: bool = False) -> list[np.ndarray]:
    """MV and MV²; with ``statline``, power and toughness too (missing read as 0)."""
    mv = num(df, "mv", 0)
    out = [mv, mv ** 2]
    if statline:
        out += [num(df, "power", 0), num(df, "toughness", 0)]
    return out


def with_predictions(j: pd.DataFrame) -> pd.DataFrame:
    j = j.copy()
    P = pl.predict_sd(pl.load_embedding_matrix(j["name"].tolist(), j))
    j["pred_sp"] = P["score_play"].to_numpy()
    j["pred_pr"] = P["played_rate"].to_numpy()
    j["label_sp"] = num(j, "shrunk_score_play") / pl.SD["score_play"]
    j["label_pr"] = num(j, "shrunk_played_rate") / pl.SD["played_rate"]
    return j


def feature_row(family: str, feature: str, df: pd.DataFrame, is_feature: np.ndarray,
                ctrl: list[np.ndarray], with_pr: bool) -> dict:
    lab = effect(df, is_feature, "label_sp", ctrl)
    labpr = (effect(df, is_feature, "label_pr", ctrl, wcol="w_played_rate")["beta"]
             if with_pr else np.nan)
    return {"family": family, "feature": feature, "n": lab["n"],
            "label_sp": lab["beta"], "label_se": lab["se"],
            "pred_sp": effect(df, is_feature, "pred_sp", ctrl)["beta"], "label_pr": labpr}


def main() -> None:
    j = with_predictions(pl.load_frame(text=True))
    crea = flag(j, "is_creature")
    power, tough = num(j, "power"), num(j, "toughness")
    rows = []

    # ── keywords, within creatures with a literal statline ─────────────
    sub = j[crea & np.isfinite(power) & np.isfinite(tough)].reset_index(drop=True)
    smv = num(sub, "mv", 0)
    sp_, st_ = num(sub, "power"), num(sub, "toughness")
    nkw = num(sub, "kw_count", 0)
    ctrl = [smv, smv ** 2, sp_, st_, nkw]
    for kw, col in cc.KEYWORD_FLAGS.items():
        rows.append(feature_row("keyword", kw, sub, flag(sub, col), ctrl, with_pr=True))

    # ── spell-effect phrases, within noncreature spells ────────────────
    spells = j[flag(j, "is_instant") | flag(j, "is_sorcery")].reset_index(drop=True)
    pctrl = mv_controls(spells) + [spells["is_instant"].fillna(False).astype(float).to_numpy()]
    for label, col in PHRASES.items():
        if col in spells.columns:
            rows.append(feature_row("spell effect", label, spells, flag(spells, col), pctrl,
                                    with_pr=True))

    # ── tribes, within creatures, statline + MV controlled ─────────────
    tribe = np.array([(cc.sole_tribe(t) or (None, None))[1] for t in sub["text"]])
    for t in cc.TRIBES:
        is_t = tribe == t
        if is_t.sum() >= 15:
            rows.append(feature_row("tribe", t, sub, is_t, ctrl, with_pr=False))

    # ── type lines ─────────────────────────────────────────────────────
    creatures = j[crea].reset_index(drop=True)
    for label, col in (("artifact creature (vs creature)", "is_artifact"),
                       ("enchantment creature (vs creature)", "is_enchantment")):
        rows.append(feature_row("type", label, creatures, flag(creatures, col),
                                mv_controls(creatures, statline=True), with_pr=False))
    rows.append(feature_row("type", "instant (vs sorcery)", spells,
                            flag(spells, "is_instant"), mv_controls(spells), with_pr=False))

    tab = pd.DataFrame(rows)
    tab.to_csv(pl.SCRATCH / "c7_labelside.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False), flush=True)

    # ── the P-T asymmetry, label side ──────────────────────────────────
    out = {}
    tot = sp_ + st_
    for ycol in ("label_sp", "pred_sp"):
        y = num(sub, ycol)
        w = sub["w_score_play"].to_numpy(float)
        X = np.column_stack([np.ones(len(sub)), sp_ - st_, tot, smv, smv ** 2, nkw])
        m = np.isfinite(tot) & np.isfinite(y) & (w > 0) & (tot >= 4) & (tot <= 8)
        fit = wls(X[m], y[m], w[m])
        out[f"pt_gradient_{ycol}"] = {"beta_per_point_of_P_minus_T": float(fit.beta[1]),
                                      "se": float(fit.se[1]), "n": int(m.sum())}
    # taplands, label side
    lands = j[flag(j, "is_land")].reset_index(drop=True)
    tapped = flag(lands, "ph_enters_tapped")
    for ycol in ("label_sp", "pred_sp"):
        out[f"tapland_{ycol}"] = effect(lands, tapped, ycol, [np.zeros(len(lands))])
    # dies vs etb triggers and the sacrifice-self drawback, label side
    c2 = mv_controls(creatures, statline=True)
    for name, col in (("dies_trigger", "ph_dies_trigger"), ("etb_trigger", "ph_etb"),
                      ("sacrifice_self", "ph_sacrifice_self")):
        for ycol in ("label_sp", "pred_sp"):
            out[f"{name}_{ycol}"] = effect(creatures, flag(creatures, col), ycol, c2)

    (pl.SCRATCH / "c7_labelside.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
