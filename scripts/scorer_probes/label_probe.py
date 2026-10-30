"""Join per-card empirical labels to converted card text properties (read-only).

Writes ``labels_joined.csv`` (the input of ``rarity_probe.py`` and
``human_rank_probe.py``) and prints the label-level slices.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorer_probes import probe_lib as pl

from sealed.domain import card_embedding_layout as layout

OUT_CSV = pl.SCRATCH / "labels_joined.csv"
FEATURES = ["mv", "pips", "colors", "creature", "instant", "sorcery", "artifact", "ench",
            "land", "colorless", "power", "tough", "evasive", "removal", "counter", "draw",
            "lifegain", "scry", "mill", "trick", "aura", "equip", "nkw", "hasx", "hybrid",
            "ntok", "kw_flying", "kw_deathtouch", "kw_lifelink", "kw_first_strike",
            "kw_trample", "kw_haste", "kw_defender", "kw_vigilance", "kw_menace", "kw_flash"]


def props(txt: str, emb: np.ndarray) -> dict:
    """Card properties from lower-cased converted text; mana value, coloured
    pips and colours from the embedding's deterministic block."""
    det = emb[-pl.FC:]
    d = {}
    mc = re.search(r"^mana cost:\s*(.+)$", txt, re.M)
    cost = mc.group(1) if mc else ""
    pips = int(round(float(det[layout.COLOR_PIPS].sum())))
    d["mv"] = int(round(float(det[layout.MANA_VALUE])))
    d["pips"] = pips
    d["colors"] = len(layout.card_colors(emb))
    d["hasx"] = 1 if det[layout.X_COUNT] > 0 else 0
    d["hybrid"] = 1 if re.search(r"\{[wubrg2]/[wubrg]\}", cost) else 0
    d["phyrex"] = 1 if re.search(r"\{[wubrg]/p\}", cost) else 0
    ty = re.search(r"^types:\s*(.+)$", txt, re.M)
    types = ty.group(1) if ty else ""
    d["creature"] = 1 if "creature" in types else 0
    d["instant"] = 1 if "instant" in types else 0
    d["sorcery"] = 1 if "sorcery" in types else 0
    d["artifact"] = 1 if "artifact" in types else 0
    d["ench"] = 1 if "enchantment" in types else 0
    d["land"] = 1 if "land" in types else 0
    d["pw"] = 1 if "planeswalker" in types else 0
    d["colorless"] = 1 if pips == 0 and not d["land"] else 0
    pt = re.search(r"^power toughness:\s*(-?\d+|\*)/(-?\d+|\*)", txt, re.M)
    if pt:
        try:
            d["power"] = int(pt.group(1))
            d["tough"] = int(pt.group(2))
        except ValueError:
            d["power"] = d["tough"] = np.nan
    else:
        d["power"] = d["tough"] = np.nan
    body = txt
    for kw in ["flying", "deathtouch", "lifelink", "first strike", "double strike", "trample",
               "menace", "vigilance", "haste", "hexproof", "reach", "defender", "flash"]:
        d["kw_" + kw.replace(" ", "_")] = 1 if kw in body else 0
    d["evasive"] = 1 if ("flying" in body or "menace" in body or "can't be blocked" in body
                         or "fear" in body or "intimidate" in body or "shadow" in body) else 0
    d["removal"] = 1 if re.search(
        r"destroy target (creature|permanent)|exile target (creature|permanent)"
        r"|deals \d+ damage to (target|any target)|target creature gets -", body) else 0
    d["counter"] = 1 if "counter target spell" in body else 0
    d["draw"] = 1 if re.search(r"draw (a|one|two|three|\w+) card", body) else 0
    d["lifegain"] = 1 if re.search(r"gain \d+ life|gains? \d+ life", body) else 0
    d["scry"] = 1 if "scry" in body else 0
    d["mill"] = 1 if "mill" in body else 0
    d["trick"] = 1 if (d["instant"] and re.search(r"target creature gets \+", body)) else 0
    d["aura"] = 1 if "enchant creature" in body else 0
    d["equip"] = 1 if "equip" in body else 0
    d["nkw"] = sum(v for k, v in d.items() if k.startswith("kw_"))
    d["ntok"] = len(body.split())
    return d


def join_labels() -> pd.DataFrame:
    """One row per labelled card with converted text and an embedding."""
    locator = pl.ConvertedCardLocator(pl.CARDS_PATH)
    rows = []
    for name, rec in pl.load_win_rates().items():
        text = locator.load_text(name)
        emb = locator.load_embedding(name)
        if text is None or emb is None:
            continue

        def g(col, rec=rec):
            return np.nan if rec[col] is None else float(rec[col])

        r = {"name": name, "n": g("wins_when_in_deck") + g("losses_when_in_deck"),
             "sp": g("shrunk_score_play"), "sd": g("shrunk_score_draw"),
             "pr": g("shrunk_played_rate"), "cl": g("shrunk_cast_lift"),
             "rsp": g("raw_score_play")}
        r.update(props(text.text.lower(), emb))
        rows.append(r)
    return pd.DataFrame(rows)


def print_correlations(d: pd.DataFrame) -> None:
    targets = ["sp", "sd", "pr", "cl"]
    print("\n--- Pearson r (n>=200) ---")
    print(f"{'feature':18s}" + "".join(f"{t:>9s}" for t in targets))
    for f in FEATURES:
        line = f"{f:18s}"
        for t in targets:
            sub = d[[f, t]].dropna()
            line += f"{sub[f].corr(sub[t]):9.3f}" if len(sub) > 30 else "      n/a"
        print(line)


def print_groups(d: pd.DataFrame) -> None:
    print("\n--- group means (n>=200) ---")

    def grp(mask, label):
        s = d[mask]
        if len(s) < 15:
            print(f"{label:28s} n={len(s):5d}  (too few)")
            return
        print(f"{label:28s} n={len(s):5d}  sp={s.sp.mean():+.4f} sd={s.sd.mean():+.4f} "
              f"pr={s.pr.mean():.3f} cl={s.cl.mean():+.4f} mv={s.mv.mean():.2f}")

    grp(d.creature == 1, "creature")
    grp((d.instant == 1), "instant")
    grp((d.sorcery == 1), "sorcery")
    grp((d.instant == 1) & (d.removal == 1), "instant removal")
    grp((d.sorcery == 1) & (d.removal == 1), "sorcery removal")
    grp((d.creature == 0) & (d.removal == 1) & (d.land == 0), "noncreature removal (any)")
    grp(d.counter == 1, "counterspell")
    grp(d.trick == 1, "combat trick (instant pump)")
    grp((d.creature == 0) & (d.draw == 1) & (d.land == 0), "noncreature card draw")
    grp((d.artifact == 1) & (d.creature == 0), "noncreature artifact")
    grp(d.colorless == 1, "colorless (nonland)")
    grp(d.land == 1, "land (nonbasic)")
    grp(d.colors >= 2, "multicolor (gold)")
    grp((d.colors == 1) & (d.creature == 1), "mono-color creature")
    grp(d.aura == 1, "aura")
    grp(d.equip == 1, "equipment-ish")
    grp(d.hasx == 1, "X spell")
    grp(d.hybrid == 1, "hybrid cost")
    grp(d.phyrex == 1, "phyrexian cost")
    grp((d.creature == 1) & (d.kw_flying == 1), "creature w/ flying")
    grp((d.creature == 1) & (d.kw_flying == 0) & (d.nkw == 0), "creature no-kw")


def print_mv_slices(d: pd.DataFrame) -> None:
    print("\n--- creature vs noncreature spell, matched MV ---")
    for mv in range(1, 8):
        a = d[(d.creature == 1) & (d.mv == mv)]
        b = d[(d.creature == 0) & (d.land == 0) & (d.mv == mv)]
        if len(a) > 20 and len(b) > 20:
            print(f"MV {mv}: creature sp={a.sp.mean():+.4f} (n={len(a):4d})  "
                  f"noncreature sp={b.sp.mean():+.4f} (n={len(b):4d})"
                  f"   pr: {a.pr.mean():.3f} vs {b.pr.mean():.3f}")

    print("\n--- by MV: played_rate & score_play (creatures only) ---")
    c = d[d.creature == 1]
    for mv in range(0, 9):
        s = c[c.mv == mv]
        if len(s) > 20:
            print(f"MV {mv}: n={len(s):4d} sp={s.sp.mean():+.4f} pr={s.pr.mean():.3f} "
                  f"cl={s.cl.mean():+.4f}")


def main() -> None:
    df = join_labels()
    print("joined:", len(df), " with n>=200:", (df.n >= 200).sum(),
          " n>=1000:", (df.n >= 1000).sum())
    print("n_in_deck quantiles:", df.n.quantile([.1, .25, .5, .75, .9]).round(0).to_dict())
    d = df[(df.n >= 200) & df.sp.notna()].copy()
    print("\n=== N used:", len(d))
    print_correlations(d)
    print_groups(d)
    print_mv_slices(d)
    d.to_csv(OUT_CSV, index=False)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    main()
