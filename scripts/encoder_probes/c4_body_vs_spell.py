"""C4 (R14) — body vs spell, on matched synthetic shells.

The same effect text E is placed on (a) a sorcery, (a') an instant,
(b) a 2/2 creature's enter-the-battlefield trigger, against (c) the bare
2/2 creature and (d) a do-nothing sorcery ("you gain 1 life."). Cost and
colour are held fixed across every arm of an effect, so (b)-(a) is the
body premium and (b)-(c) / (a)-(d) are the effect's marginal on each
chassis.

These shells are synthetic, so this is a **mechanism probe**: it is gated
by the manifold distance, its nearest real neighbours are reported, and
it is anchored by MV-matched real-card class means computed from the join
table plus the grounding feature flags.
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

# (effect text, colour pip) — colour is held fixed across every arm.
EFFECTS: list[tuple[str, str]] = [
    ("create a 1/1 white soldier creature token.", "W"),
    ("create two 1/1 white soldier creature tokens.", "W"),
    ("you gain 3 life.", "W"),
    ("you gain 7 life.", "W"),
    ("draw a card.", "U"),
    ("draw two cards.", "U"),
    ("scry 2.", "U"),
    ("return target creature to its owner's hand.", "U"),
    ("destroy target artifact.", "G"),
    ("search your library for a basic land card, put it onto the battlefield tapped, then shuffle.", "G"),
    ("put a +1/+1 counter on target creature.", "G"),
    ("target creature gets +2/+2 until end of turn.", "G"),
    ("each opponent loses 2 life.", "B"),
    ("target player discards a card.", "B"),
    ("destroy target creature.", "B"),
    ("CARDNAME deals 2 damage to any target.", "R"),
    ("CARDNAME deals 4 damage to target creature.", "R"),
    ("target creature gains haste until end of turn.", "R"),
]
CONTROL_EFFECT = "you gain 1 life."
GENERICS = ["2", "4"]


def shell(kind: str, cost: str, effect: str) -> str:
    if kind == "sorcery":
        return f"mana cost: {cost}\ntypes: sorcery\nspell[1]: {effect}"
    if kind == "instant":
        return f"mana cost: {cost}\ntypes: instant\nspell[1]: {effect}"
    if kind == "creature_etb":
        return (f"mana cost: {cost}\ntypes: creature human soldier\n"
                f"power toughness: 2/2\ntriggered: when CARDNAME enters, {effect}")
    if kind == "creature_bare":
        return (f"mana cost: {cost}\ntypes: creature human soldier\n"
                f"power toughness: 2/2")
    raise ValueError(kind)


def main() -> None:
    j = cc.join_table()
    keys, ref = pl.corpus_embedding_matrix()
    out: dict = {}

    kinds = ["sorcery", "instant", "creature_etb", "creature_bare"]
    texts, meta = [], []
    for gen in GENERICS:
        for effect, colour in EFFECTS + [(CONTROL_EFFECT, "W")]:
            cost = f"{{{gen}}}{{{colour}}}"
            for kind in kinds:
                texts.append(shell(kind, cost, effect))
                meta.append({"generic": gen, "effect": effect,
                             "colour": colour, "kind": kind})
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    dist, nidx = pl.manifold_distance(emb, ref)
    m = pd.DataFrame(meta)
    m["score_play"] = pred["score_play"].to_numpy()
    m["played_rate"] = pred["played_rate"].to_numpy()
    m["dist"] = dist
    m["nearest"] = [keys[i] for i in nidx]
    m.to_csv(cc.SCRATCH / "c4_shells.csv", index=False)

    piv = m.pivot_table(index=["generic", "effect"], columns="kind",
                        values="score_play")
    piv_pr = m.pivot_table(index=["generic", "effect"], columns="kind",
                           values="played_rate")
    piv_d = m.pivot_table(index=["generic", "effect"], columns="kind",
                          values="dist")

    rows = []
    for gen in GENERICS:
        ctrl = piv.loc[(gen, CONTROL_EFFECT)]
        for effect, _ in EFFECTS:
            r = piv.loc[(gen, effect)]
            rpr = piv_pr.loc[(gen, effect)]
            rows.append({
                "generic": gen, "effect": effect,
                "body_premium_etb_minus_sorcery": r["creature_etb"] - r["sorcery"],
                "effect_on_body_etb_minus_bare": r["creature_etb"] - r["creature_bare"],
                "effect_on_spell_vs_control": r["sorcery"] - ctrl["sorcery"],
                "instant_minus_sorcery": r["instant"] - r["sorcery"],
                "sorcery_sp": r["sorcery"], "etb_sp": r["creature_etb"],
                "bare_sp": r["creature_bare"],
                "body_premium_pr": rpr["creature_etb"] - rpr["sorcery"],
                "max_dist": float(piv_d.loc[(gen, effect)].max()),
            })
    bp = pd.DataFrame(rows)
    bp.to_csv(cc.SCRATCH / "c4_body_premium.csv", index=False)

    agg = bp.groupby("effect")[[
        "body_premium_etb_minus_sorcery", "effect_on_body_etb_minus_bare",
        "effect_on_spell_vs_control", "instant_minus_sorcery",
    ]].mean().reset_index().sort_values(
        "body_premium_etb_minus_sorcery", ascending=False)
    agg.to_csv(cc.SCRATCH / "c4_body_premium_agg.csv", index=False)
    print(agg.to_string(index=False), flush=True)

    v = bp["body_premium_etb_minus_sorcery"].to_numpy()
    lo, hi = cc.bootstrap_ci(v)
    out["body_premium_mean_sd"] = float(v.mean())
    out["body_premium_ci"] = [lo, hi]
    out["body_premium_frac_pos"] = float((v > 0).mean())
    vi = bp["instant_minus_sorcery"].to_numpy()
    out["instant_minus_sorcery_mean_sd"] = float(vi.mean())
    out["instant_minus_sorcery_ci"] = list(cc.bootstrap_ci(vi))
    out["shell_dist_p50"] = float(np.median(dist))
    out["shell_dist_p95"] = float(np.quantile(dist, 0.95))
    out["shell_off_manifold"] = float((dist > cc.MANIFOLD_GATE).mean())
    out["nearest_examples"] = m.sample(8, random_state=1)[
        ["kind", "effect", "dist", "nearest"]].to_dict("records")

    # ── the mana-rock vs mana-dork case ─────────────────────────────────
    rock_shells = {
        "artifact rock {2}": "mana cost: {2}\ntypes: artifact\nactivated[1]: {T}: add {C}.",
        "artifact creature rock {2} 1/1": ("mana cost: {2}\ntypes: artifact creature construct\n"
                                           "power toughness: 1/1\nactivated[1]: {T}: add {C}."),
        "creature dork {2} 1/1": ("mana cost: {2}\ntypes: creature elf druid\n"
                                  "power toughness: 1/1\nactivated[1]: {T}: add {C}."),
        "creature dork colour {2} 1/1": ("mana cost: {2}\ntypes: creature elf druid\n"
                                         "power toughness: 1/1\nactivated[1]: {T}: add {G}."),
        "bare creature {2} 1/1": ("mana cost: {2}\ntypes: creature elf druid\n"
                                  "power toughness: 1/1"),
        "bare artifact {2}": "mana cost: {2}\ntypes: artifact",
        "ramp sorcery {2}": ("mana cost: {2}\ntypes: sorcery\nspell[1]: search your library "
                             "for a basic land card, put it onto the battlefield tapped, "
                             "then shuffle."),
        "enchantment rock {2}": ("mana cost: {2}\ntypes: enchantment\n"
                                 "activated[1]: {T}: add {C}."),
    }
    rtexts = list(rock_shells.values())
    remb = cc.encode(rtexts)
    rpred = cc.predict_sd(remb)
    rdist, rnidx = pl.manifold_distance(remb, ref)
    rock = pd.DataFrame({
        "shell": list(rock_shells),
        "score_play_sd": rpred["score_play"].to_numpy(),
        "played_rate_sd": rpred["played_rate"].to_numpy(),
        "dist": rdist,
        "nearest": [keys[i] for i in rnidx],
    })
    rock.to_csv(cc.SCRATCH / "c4_rock_dork.csv", index=False)
    print(rock.to_string(index=False), flush=True)

    # ── real-card class anchors, MV-matched ─────────────────────────────
    names = j["name"].tolist()
    E = pl.load_embedding_matrix(names, j)
    P = cc.predict_sd(E)
    j = j.copy()
    j["pred_sp"] = P["score_play"].to_numpy()
    j["pred_pr"] = P["played_rate"].to_numpy()
    j["label_sp"] = pd.to_numeric(j["shrunk_score_play"], errors="coerce") / cc.SD["score_play"]
    mv = pd.to_numeric(j["mv"], errors="coerce")
    f = lambda c: j[c].fillna(False).astype(bool)  # noqa: E731
    classes = {
        "mana rock (noncreature artifact, add mana)": f("ph_add_mana") & f("is_artifact") & ~f("is_creature"),
        "mana dork (creature, add mana)": f("ph_add_mana") & f("is_creature"),
        "ramp sorcery (search basic land)": f("ph_search_basic_land") & ~f("is_creature"),
        "token creature (creature that makes tokens)": f("ph_create_token") & f("is_creature"),
        "token spell (noncreature token maker)": f("ph_create_token") & ~f("is_creature"),
        "vanilla-ish creature": f("is_creature") & (pd.to_numeric(j["n_abilities"], errors="coerce").fillna(9) == 0),
        "all creatures": f("is_creature"),
        "all noncreature spells": f("is_instant") | f("is_sorcery"),
        "corpus": pd.Series(True, index=j.index),
    }
    mv_lo, mv_hi = 1, 4
    band = mv.between(mv_lo, mv_hi)
    crows = []
    for name, mask in classes.items():
        sub = j[mask & band]
        if len(sub) < 10:
            continue
        crows.append({"class": name, "n": len(sub), "mean_mv": float(mv[mask & band].mean()),
                      "pred_sp": float(sub["pred_sp"].mean()),
                      "label_sp": float(sub["label_sp"].mean()),
                      "pred_pr": float(sub["pred_pr"].mean())})
    anchors = pd.DataFrame(crows)
    anchors.to_csv(cc.SCRATCH / "c4_class_anchors.csv", index=False)
    print(f"[anchors] MV {mv_lo}-{mv_hi}")
    print(anchors.to_string(index=False), flush=True)

    (cc.SCRATCH / "c4_summary.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
