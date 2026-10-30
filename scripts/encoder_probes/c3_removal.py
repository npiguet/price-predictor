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


def second_static(text: str) -> int:
    st = [i for i, ln in enumerate(cc.lines(text)) if ln.startswith("static: ")]
    return st[1] if len(st) >= 2 else -1


def main() -> None:
    j = pl.load_frame(text=True)
    C = cc.CardCorpus.of(j)
    stripped = C.text
    is_inst = pl.flag(j, "is_instant")
    is_sp = is_inst | pl.flag(j, "is_sorcery")
    mv = pl.num(j, "mv")
    rng = np.random.default_rng(9)
    out: dict = {}

    sl = np.array([cc.find_line(cc.lines(s), "spell[1]:") for s in stripped])
    base = cc.sample_rows(is_sp & (sl >= 0) & (C.n_abilities == 1)
                          & np.isfinite(mv) & (mv >= 1) & (mv <= 5), 200, rng)
    print(f"[C3] {len(base)} base spells, {len(TEMPLATES)} arms", flush=True)

    L = pl.score_arms(base, TEMPLATES, lambda r, tmpl: cc.replace_in_line(
        stripped[r], sl[r], f"spell[1]: {tmpl}"))
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
    pl.write_table(tab, "c3_spell_ladder.csv")

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

    # auras
    ss = np.array([second_static(s) for s in stripped])
    has_ench = np.array(["static: enchant creature" in s for s in stripped])
    abase = cc.sample_rows(pl.flag(j, "is_aura") & has_ench & (ss >= 0)
                           & (C.n_abilities <= 3), 200, rng)
    print(f"[C3] {len(abase)} base auras", flush=True)
    AU = pl.score_arms(abase, AURA_TEMPLATES, lambda r, tmpl: cc.replace_in_line(
        stripped[r], ss[r], f"static: {tmpl}"))
    atab = pd.DataFrame(pl.contrasts(
        AU, AURA_TEMPLATES, cols=range(len(AURA_TEMPLATES)),
        ref=AURA_TEMPLATES.index("enchanted creature gets +2/+2."), field="aura_text",
        key="vs_plus2_sp", pr_key="vs_plus2_pr")).sort_values("vs_plus2_sp", ascending=False)
    pl.write_table(atab, "c3_aura.csv")

    out["ladder_spread_sd"] = float(tab["value_sp"].max() - tab["value_sp"].min())
    out["off_manifold_ladder"] = float(L.off.mean())
    out["off_manifold_aura"] = float(AU.off.mean())
    pl.write_summary(out, "c3_summary.json")


if __name__ == "__main__":
    main()
