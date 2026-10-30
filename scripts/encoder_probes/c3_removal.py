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
import probe_lib as pl  # noqa: E402

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
    j = pl.load_frame(text=True)
    stripped = list(j["text"])
    is_inst = j["is_instant"].fillna(False).astype(bool).to_numpy()
    is_sp = is_inst | j["is_sorcery"].fillna(False).astype(bool).to_numpy()
    mv = pd.to_numeric(j["mv"], errors="coerce").to_numpy(float)
    rng = np.random.default_rng(9)
    out: dict = {}

    sl = np.array([cc.find_line(cc.lines(s), "spell[1]:") for s in stripped])
    n_ab = np.array([len(cc.ability_lines(s)) for s in stripped])
    base = np.flatnonzero(is_sp & (sl >= 0) & (n_ab == 1) &
                          np.isfinite(mv) & (mv >= 1) & (mv <= 5))
    if len(base) > 200:
        base = np.sort(rng.choice(base, 200, replace=False))
    print(f"[C3] {len(base)} base spells, {len(TEMPLATES)} arms", flush=True)

    L = pl.score_texts([cc.replace_in_line(stripped[r], sl[r], f"spell[1]: {tmpl}")
                        for r in base for tmpl in TEMPLATES]).reshape(len(base), -1)
    grand = L.sp.mean(axis=1)
    ref = TEMPLATES.index("destroy target creature.")
    rows = []
    for i, tmpl in enumerate(TEMPLATES):
        vs_ref = pl.mean_ci(L.sp[:, i] - L.sp[:, ref], "vs_destroy_creature")
        rows.append({
            "effect": tmpl, "n": len(base),
            **pl.mean_ci(L.sp[:, i] - grand, "value_sp"),
            "vs_destroy_creature": vs_ref["vs_destroy_creature"],
            "ref_ci_lo": vs_ref["ci_lo"], "ref_ci_hi": vs_ref["ci_hi"],
            "value_pr": float((L.pr[:, i] - L.pr.mean(axis=1)).mean()),
            "off_manifold": float(L.off[:, i].mean()),
        })
    tab = pd.DataFrame(rows).sort_values("value_sp", ascending=False)
    tab.to_csv(pl.SCRATCH / "c3_spell_ladder.csv", index=False)
    print(tab.to_string(index=False), flush=True)

    # instant vs sorcery split of the same ladder
    inst = is_inst[base]
    split_rows = []
    for i, tmpl in enumerate(TEMPLATES):
        c = L.sp[:, i] - grand
        split_rows.append({"effect": tmpl,
                           "instant_n": int(inst.sum()),
                           "instant_sp": float(c[inst].mean()),
                           "sorcery_n": int((~inst).sum()),
                           "sorcery_sp": float(c[~inst].mean())})
    pd.DataFrame(split_rows).to_csv(
        pl.SCRATCH / "c3_ladder_by_type.csv", index=False)

    # ── auras ───────────────────────────────────────────────────────────
    is_aura = j["is_aura"].fillna(False).astype(bool).to_numpy()

    def second_static(s: str) -> int:
        st = [i for i, ln in enumerate(cc.lines(s)) if ln.startswith("static: ")]
        return st[1] if len(st) >= 2 else -1

    ss = np.array([second_static(s) for s in stripped])
    has_ench = np.array(["static: enchant creature" in s for s in stripped])
    abase = np.flatnonzero(is_aura & has_ench & (ss >= 0) & (n_ab <= 3))
    if len(abase) > 200:
        abase = np.sort(rng.choice(abase, 200, replace=False))
    print(f"[C3] {len(abase)} base auras", flush=True)
    AU = pl.score_texts([cc.replace_in_line(stripped[r], ss[r], f"static: {tmpl}")
                         for r in abase for tmpl in AURA_TEMPLATES]).reshape(len(abase), -1)
    aref = AURA_TEMPLATES.index("enchanted creature gets +2/+2.")
    arows = [pl.contrast_row(AU.sp[:, i] - AU.sp[:, aref], AU.pr[:, i] - AU.pr[:, aref],
                             AU.off[:, i], key="vs_plus2_sp", pr_key="vs_plus2_pr",
                             aura_text=tmpl)
             for i, tmpl in enumerate(AURA_TEMPLATES)]
    atab = pd.DataFrame(arows).sort_values("vs_plus2_sp", ascending=False)
    atab.to_csv(pl.SCRATCH / "c3_aura.csv", index=False)
    print(atab.to_string(index=False), flush=True)

    out["ladder_spread_sd"] = float(tab["value_sp"].max() - tab["value_sp"].min())
    out["off_manifold_ladder"] = float(L.off.mean())
    out["off_manifold_aura"] = float(AU.off.mean())
    (pl.SCRATCH / "c3_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
