"""C3 (R12) — the removal / spell-effect ladder.

One base card, many effects: the ``spell[1]:`` line of a real single-line
instant or sorcery has its *content* replaced by each template in turn.
Every arm is one line in the same slot on the same base card, so all
pairwise contrasts are layout-matched and the memorization offset of the
base card cancels.

The aura family does the same to the second ``static:`` line of a real
aura (the first is ``enchant creature``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402

TEMPLATES = [
    "destroy target creature.",
    "destroy target creature with flying.",
    "destroy target creature with power 4 or greater.",
    "exile target creature.",
    "CARDNAME deals 3 damage to any target.",
    "CARDNAME deals 3 damage to target creature.",
    "target player sacrifices a creature.",
    "destroy all creatures.",
    "target creature fights target creature you don't control.",
    "tap target creature.",
    "return target creature to its owner's hand.",
    "counter target spell.",
    "draw two cards.",
    "you gain 4 life.",
    "target creature gets +3/+3 until end of turn.",
]

AURA_TEMPLATES = [
    "enchanted creature can't attack or block.",
    "enchanted creature gets +2/+2.",
    "enchanted creature gets -2/-2.",
    "enchanted creature has flying.",
    "enchanted creature can't block.",
]


def main() -> None:
    j = cc.join_table()
    stripped = [cc.strip_name(t) for t in j["text"]]
    is_sp = (j["is_instant"].fillna(False) | j["is_sorcery"].fillna(False)) \
        .astype(bool).to_numpy()
    mv = pd.to_numeric(j["mv"], errors="coerce").to_numpy(float)
    rng = np.random.default_rng(9)
    out: dict = {}

    def spell_line(s: str) -> int:
        for i, l in enumerate(cc.lines(s)):
            if l.startswith("spell[1]:"):
                return i
        return -1

    sl = np.array([spell_line(s) for s in stripped])
    n_ab = np.array([len(cc.ability_lines(s)) for s in stripped])
    base = np.flatnonzero(is_sp & (sl >= 0) & (n_ab == 1) &
                          np.isfinite(mv) & (mv >= 1) & (mv <= 5))
    if len(base) > 200:
        base = np.sort(rng.choice(base, 200, replace=False))
    print(f"[C3] {len(base)} base spells, {len(TEMPLATES)} arms", flush=True)

    texts = [cc.replace_in_line(stripped[r], sl[r], f"spell[1]: {tmpl}")
             for r in base for tmpl in TEMPLATES]
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    dist = cc.offmanifold(emb)
    nt = len(TEMPLATES)
    S = pred["score_play"].to_numpy().reshape(len(base), nt)
    P = pred["played_rate"].to_numpy().reshape(len(base), nt)
    D = dist.reshape(len(base), nt)

    grand = S.mean(axis=1, keepdims=True)
    rows = []
    ref = TEMPLATES.index("destroy target creature.")
    for i, tmpl in enumerate(TEMPLATES):
        centred = S[:, i] - grand[:, 0]
        vs_ref = S[:, i] - S[:, ref]
        lo, hi = cc.bootstrap_ci(centred)
        rlo, rhi = cc.bootstrap_ci(vs_ref)
        rows.append({
            "effect": tmpl, "n": len(base),
            "value_sp": float(centred.mean()), "ci_lo": lo, "ci_hi": hi,
            "vs_destroy_creature": float(vs_ref.mean()),
            "ref_ci_lo": rlo, "ref_ci_hi": rhi,
            "value_pr": float((P[:, i] - P.mean(axis=1)).mean()),
            "off_manifold": float((D[:, i] > cc.MANIFOLD_GATE).mean()),
        })
    tab = pd.DataFrame(rows).sort_values("value_sp", ascending=False)
    tab.to_csv(cc.SCRATCH / "c3_spell_ladder.csv", index=False)
    print(tab.to_string(index=False), flush=True)

    # instant vs sorcery split of the same ladder
    is_inst = j["is_instant"].fillna(False).astype(bool).to_numpy()[base]
    split_rows = []
    for i, tmpl in enumerate(TEMPLATES):
        c = S[:, i] - grand[:, 0]
        split_rows.append({"effect": tmpl,
                           "instant_n": int(is_inst.sum()),
                           "instant_sp": float(c[is_inst].mean()),
                           "sorcery_n": int((~is_inst).sum()),
                           "sorcery_sp": float(c[~is_inst].mean())})
    pd.DataFrame(split_rows).to_csv(
        cc.SCRATCH / "c3_ladder_by_type.csv", index=False)

    # ── auras ───────────────────────────────────────────────────────────
    is_aura = j["is_aura"].fillna(False).astype(bool).to_numpy()

    def second_static(s: str) -> int:
        st = [i for i, l in enumerate(cc.lines(s)) if l.startswith("static: ")]
        return st[1] if len(st) >= 2 else -1

    ss = np.array([second_static(s) for s in stripped])
    has_ench = np.array(["static: enchant creature" in s for s in stripped])
    abase = np.flatnonzero(is_aura & has_ench & (ss >= 0) & (n_ab <= 3))
    if len(abase) > 200:
        abase = np.sort(rng.choice(abase, 200, replace=False))
    print(f"[C3] {len(abase)} base auras", flush=True)
    atexts = [cc.replace_in_line(stripped[r], ss[r], f"static: {tmpl}")
              for r in abase for tmpl in AURA_TEMPLATES]
    aemb = cc.encode(atexts)
    apred = cc.predict_sd(aemb)
    adist = cc.offmanifold(aemb)
    na = len(AURA_TEMPLATES)
    AS = apred["score_play"].to_numpy().reshape(len(abase), na)
    AP = apred["played_rate"].to_numpy().reshape(len(abase), na)
    AD = adist.reshape(len(abase), na)
    aref = AURA_TEMPLATES.index("enchanted creature gets +2/+2.")
    arows = []
    for i, tmpl in enumerate(AURA_TEMPLATES):
        v = AS[:, i] - AS[:, aref]
        lo, hi = cc.bootstrap_ci(v)
        arows.append({"aura_text": tmpl, "n": len(abase),
                      "vs_plus2_sp": float(v.mean()), "ci_lo": lo, "ci_hi": hi,
                      "frac_pos": float((v > 0).mean()),
                      "vs_plus2_pr": float((AP[:, i] - AP[:, aref]).mean()),
                      "off_manifold": float((AD[:, i] > cc.MANIFOLD_GATE).mean())})
    atab = pd.DataFrame(arows).sort_values("vs_plus2_sp", ascending=False)
    atab.to_csv(cc.SCRATCH / "c3_aura.csv", index=False)
    print(atab.to_string(index=False), flush=True)

    out["ladder_spread_sd"] = float(tab["value_sp"].max() - tab["value_sp"].min())
    out["off_manifold_ladder"] = float((D > cc.MANIFOLD_GATE).mean())
    out["off_manifold_aura"] = float((AD > cc.MANIFOLD_GATE).mean())
    (cc.SCRATCH / "c3_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
