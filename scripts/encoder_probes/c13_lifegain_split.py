"""C13 — the gain-life label, split by whether lifegain is the whole card.

The c3 ladder's lifegain arm is a spell whose entire text is "you gain 4
life"; the label-side reference (`c7_labelside.py`'s `ph_gain_life`) fires
on any spell containing the clause, which is mostly riders on otherwise
normal cards. This probe splits the flag into the two populations —
sole-clause lifegain spells and lifegain riders — and fits the same
MV/instant-controlled WLS on the labels and on the encoder's predictions
for each, so the c3 arm can be compared against its actual reference.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402
from c7_labelside import effect  # noqa: E402

SOLE_RE = re.compile(r"spell\[1\]: (?:you gain|target player gains) \d+ life\.")


def main() -> None:
    j = cc.join_table().copy()
    E = pl.load_embedding_matrix(j["name"].tolist(), j)
    P = cc.predict_sd(E)
    j["pred_sp"] = P["score_play"].to_numpy()
    j["label_sp"] = pd.to_numeric(j["shrunk_score_play"],
                                  errors="coerce") / cc.SD["score_play"]

    spells = j[j["is_instant"].fillna(False).astype(bool)
               | j["is_sorcery"].fillna(False).astype(bool)].reset_index(drop=True)
    stripped = [cc.strip_name(t) for t in spells["text"]]

    def is_sole(s: str) -> bool:
        body = [l for l in cc.lines(s)
                if not l.startswith(("mana cost:", "types:"))]
        return len(body) == 1 and bool(SOLE_RE.fullmatch(body[0]))

    gain = spells["ph_gain_life"].fillna(False).astype(bool).to_numpy()
    sole = np.array([is_sole(s) for s in stripped])
    rider = gain & ~sole
    print(f"sole-clause lifegain spells ({int(sole.sum())}): "
          f"{sorted(spells.loc[sole, 'name'])}", flush=True)

    pmv = pd.to_numeric(spells["mv"], errors="coerce").fillna(0).to_numpy(float)
    pctrl = [pmv, pmv ** 2,
             spells["is_instant"].fillna(False).astype(float).to_numpy()]
    rows = []
    for label, flag in (("gain life, any clause", gain),
                        ("gain life, sole clause", sole),
                        ("gain life, rider", rider)):
        lab = effect(spells, flag, "label_sp", pctrl)
        prd = effect(spells, flag, "pred_sp", pctrl)
        w = spells["w_score_play"].to_numpy(float) * flag
        rows.append({
            "population": label, "n": int(flag.sum()),
            "label_sp": lab["beta"], "label_se": lab["se"],
            "pred_sp": prd["beta"],
            "raw_mean_label": float(np.average(
                np.nan_to_num(spells["label_sp"]), weights=w)) if w.sum() else np.nan,
            "raw_mean_pred": float(np.average(
                np.nan_to_num(spells["pred_sp"]), weights=w)) if w.sum() else np.nan,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(cc.SCRATCH / "c13_lifegain_split.csv", index=False)
    print(tab.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
