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


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares with HC0 standard errors."""
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    beta = XtX_inv @ (Xw.T @ yw)
    resid = y - X @ beta
    meat = (X * (w * resid ** 2)[:, None]).T @ X
    cov = XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def effect(df: pd.DataFrame, flag: np.ndarray, ycol: str,
           controls: list[np.ndarray], wcol: str = "w_score_play") -> dict:
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(float)
    w = df[wcol].to_numpy(float)
    cols = [np.ones(len(df)), flag.astype(float)] + controls
    X = np.column_stack(cols)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1) & (w > 0)
    if ok.sum() < 30 or flag[ok].sum() < 10:
        return {"n": int(flag[ok].sum()), "beta": np.nan, "se": np.nan}
    beta, se = wls(X[ok], y[ok], w[ok])
    return {"n": int(flag[ok].sum()), "beta": float(beta[1]), "se": float(se[1])}


def main() -> None:
    j = cc.join_table().copy()
    names = j["name"].tolist()
    E = pl.load_embedding_matrix(names, j)
    P = cc.predict_sd(E)
    j["pred_sp"] = P["score_play"].to_numpy()
    j["pred_pr"] = P["played_rate"].to_numpy()
    j["label_sp"] = pd.to_numeric(j["shrunk_score_play"], errors="coerce") / cc.SD["score_play"]
    j["label_pr"] = pd.to_numeric(j["shrunk_played_rate"], errors="coerce") / cc.SD["played_rate"]

    mv = pd.to_numeric(j["mv"], errors="coerce").fillna(0).to_numpy(float)
    crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    power = pd.to_numeric(j["power"], errors="coerce").to_numpy(float)
    tough = pd.to_numeric(j["toughness"], errors="coerce").to_numpy(float)
    base_ctrl = [mv, mv ** 2, crea.astype(float)]

    rows = []

    # ── keywords, within creatures with a literal statline ─────────────
    KW = {
        "flying": "kw_flying", "lifelink": "kw_lifelink",
        "deathtouch": "kw_deathtouch", "vigilance": "kw_vigilance",
        "first strike": "kw_first_strike", "double strike": "kw_double_strike",
        "trample": "kw_trample", "haste": "kw_haste", "reach": "kw_reach",
        "menace": "kw_menace", "defender": "kw_defender",
        "hexproof": "kw_hexproof", "shroud": "kw_shroud",
        "indestructible": "kw_indestructible", "flash": "kw_flash",
        "ward {2}": "kw_ward",
    }
    sub = j[crea & np.isfinite(power) & np.isfinite(tough)].reset_index(drop=True)
    smv = pd.to_numeric(sub["mv"], errors="coerce").fillna(0).to_numpy(float)
    sp_ = pd.to_numeric(sub["power"], errors="coerce").to_numpy(float)
    st_ = pd.to_numeric(sub["toughness"], errors="coerce").to_numpy(float)
    nkw = pd.to_numeric(sub["kw_count"], errors="coerce").fillna(0).to_numpy(float)
    ctrl = [smv, smv ** 2, sp_, st_, nkw]
    for kw, col in KW.items():
        flag = sub[col].fillna(False).astype(bool).to_numpy()
        lab = effect(sub, flag, "label_sp", ctrl)
        prd = effect(sub, flag, "pred_sp", ctrl)
        labpr = effect(sub, flag, "label_pr", ctrl, wcol="w_played_rate")
        rows.append({"family": "keyword", "feature": kw, "n": lab["n"],
                     "label_sp": lab["beta"], "label_se": lab["se"],
                     "pred_sp": prd["beta"], "label_pr": labpr["beta"]})

    # ── spell-effect phrases, within noncreature spells ────────────────
    PH = {
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
    spells = j[(j["is_instant"].fillna(False) | j["is_sorcery"].fillna(False))
               .astype(bool)].reset_index(drop=True)
    pmv = pd.to_numeric(spells["mv"], errors="coerce").fillna(0).to_numpy(float)
    pctrl = [pmv, pmv ** 2,
             spells["is_instant"].fillna(False).astype(float).to_numpy()]
    for label, col in PH.items():
        if col not in spells.columns:
            continue
        flag = spells[col].fillna(False).astype(bool).to_numpy()
        lab = effect(spells, flag, "label_sp", pctrl)
        prd = effect(spells, flag, "pred_sp", pctrl)
        labpr = effect(spells, flag, "label_pr", pctrl, wcol="w_played_rate")
        rows.append({"family": "spell effect", "feature": label, "n": lab["n"],
                     "label_sp": lab["beta"], "label_se": lab["se"],
                     "pred_sp": prd["beta"], "label_pr": labpr["beta"]})

    # ── tribes, within creatures, statline + MV controlled ─────────────
    TRIBES = ["dragon", "angel", "demon", "sphinx", "hydra", "wurm", "giant",
              "zombie", "goblin", "elf", "human", "soldier", "bird", "lizard",
              "spirit", "beast", "wall", "rat"]

    def tribe_of(text: str) -> str | None:
        for l in cc.lines(text):
            if l.startswith("types:"):
                words = l[6:].split()
                if "creature" not in words:
                    return None
                subs = [w for w in words if w not in pl._TYPE_WORDS
                        and w not in pl._SUPERTYPE_WORDS]
                if len(subs) == 1 and subs[0] in TRIBES:
                    return subs[0]
                return None
        return None

    sub_t = sub.copy()
    sub_t["tribe"] = [tribe_of(t) for t in sub_t["text"]]
    for t in TRIBES:
        flag = (sub_t["tribe"] == t).to_numpy()
        if flag.sum() < 15:
            continue
        lab = effect(sub_t, flag, "label_sp", ctrl)
        prd = effect(sub_t, flag, "pred_sp", ctrl)
        rows.append({"family": "tribe", "feature": t, "n": lab["n"],
                     "label_sp": lab["beta"], "label_se": lab["se"],
                     "pred_sp": prd["beta"], "label_pr": np.nan})

    # ── type lines ─────────────────────────────────────────────────────
    art = j["is_artifact"].fillna(False).astype(bool).to_numpy()
    ench = j["is_enchantment"].fillna(False).astype(bool).to_numpy()
    for label, flag, pool in (
        ("artifact creature (vs creature)", art[crea], j[crea].reset_index(drop=True)),
        ("enchantment creature (vs creature)", ench[crea], j[crea].reset_index(drop=True)),
    ):
        pmv2 = pd.to_numeric(pool["mv"], errors="coerce").fillna(0).to_numpy(float)
        pp = pd.to_numeric(pool["power"], errors="coerce").fillna(0).to_numpy(float)
        pt = pd.to_numeric(pool["toughness"], errors="coerce").fillna(0).to_numpy(float)
        c2 = [pmv2, pmv2 ** 2, pp, pt]
        lab = effect(pool, flag, "label_sp", c2)
        prd = effect(pool, flag, "pred_sp", c2)
        rows.append({"family": "type", "feature": label, "n": lab["n"],
                     "label_sp": lab["beta"], "label_se": lab["se"],
                     "pred_sp": prd["beta"], "label_pr": np.nan})
    inst = j["is_instant"].fillna(False).astype(bool).to_numpy()
    sorc = j["is_sorcery"].fillna(False).astype(bool).to_numpy()
    pool = j[inst | sorc].reset_index(drop=True)
    pmv2 = pd.to_numeric(pool["mv"], errors="coerce").fillna(0).to_numpy(float)
    flag = pool["is_instant"].fillna(False).astype(bool).to_numpy()
    lab = effect(pool, flag, "label_sp", [pmv2, pmv2 ** 2])
    prd = effect(pool, flag, "pred_sp", [pmv2, pmv2 ** 2])
    rows.append({"family": "type", "feature": "instant (vs sorcery)", "n": lab["n"],
                 "label_sp": lab["beta"], "label_se": lab["se"],
                 "pred_sp": prd["beta"], "label_pr": np.nan})

    tab = pd.DataFrame(rows)
    tab.to_csv(cc.SCRATCH / "c7_labelside.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False), flush=True)

    # ── the P-T asymmetry, label side ──────────────────────────────────
    out = {}
    ok = np.isfinite(sp_) & np.isfinite(st_)
    tot = sp_ + st_
    dif = sp_ - st_
    for ycol, wcol in (("label_sp", "w_score_play"), ("pred_sp", "w_score_play")):
        y = pd.to_numeric(sub[ycol], errors="coerce").to_numpy(float)
        w = sub[wcol].to_numpy(float)
        X = np.column_stack([np.ones(len(sub)), dif, tot, smv, smv ** 2, nkw])
        m = ok & np.isfinite(y) & (w > 0) & (tot >= 4) & (tot <= 8)
        beta, se = wls(X[m], y[m], w[m])
        out[f"pt_gradient_{ycol}"] = {"beta_per_point_of_P_minus_T": float(beta[1]),
                                      "se": float(se[1]), "n": int(m.sum())}
    # taplands, label side
    land = j["is_land"].fillna(False).astype(bool).to_numpy()
    lpool = j[land].reset_index(drop=True)
    tapped = lpool["ph_enters_tapped"].fillna(False).astype(bool).to_numpy()
    lab = effect(lpool, tapped, "label_sp", [np.ones(len(lpool)) * 0.0])
    prd = effect(lpool, tapped, "pred_sp", [np.ones(len(lpool)) * 0.0])
    out["tapland_label_sp"] = lab
    out["tapland_pred_sp"] = prd
    # dies vs etb triggers, label side
    dies = j["ph_dies_trigger"].fillna(False).astype(bool).to_numpy()
    etb = j["ph_etb"].fillna(False).astype(bool).to_numpy()
    pool = j[crea].reset_index(drop=True)
    pmv2 = pd.to_numeric(pool["mv"], errors="coerce").fillna(0).to_numpy(float)
    c2 = [pmv2, pmv2 ** 2,
          pd.to_numeric(pool["power"], errors="coerce").fillna(0).to_numpy(float),
          pd.to_numeric(pool["toughness"], errors="coerce").fillna(0).to_numpy(float)]
    out["dies_trigger_label_sp"] = effect(pool, dies[crea], "label_sp", c2)
    out["dies_trigger_pred_sp"] = effect(pool, dies[crea], "pred_sp", c2)
    out["etb_trigger_label_sp"] = effect(pool, etb[crea], "label_sp", c2)
    out["etb_trigger_pred_sp"] = effect(pool, etb[crea], "pred_sp", c2)
    # sacrifice-self drawback, label side
    sac = j["ph_sacrifice_self"].fillna(False).astype(bool).to_numpy()
    out["sacrifice_self_label_sp"] = effect(pool, sac[crea], "label_sp", c2)
    out["sacrifice_self_pred_sp"] = effect(pool, sac[crea], "pred_sp", c2)

    (cc.SCRATCH / "c7_labelside.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
