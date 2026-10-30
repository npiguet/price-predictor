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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402
from c7_labelside import effect, spell_controls, with_predictions  # noqa: E402

SOLE_RE = re.compile(r"spell\[1\]: (?:you gain|target player gains) \d+ life\.")


def main() -> None:
    j = with_predictions(pl.load_frame(text=True))

    spells = j[pl.flag(j, "is_instant") | pl.flag(j, "is_sorcery")].reset_index(drop=True)

    def is_sole(s: str) -> bool:
        body = [ln for ln in cc.lines(s)
                if not ln.startswith(("mana cost:", "types:"))]
        return len(body) == 1 and bool(SOLE_RE.fullmatch(body[0]))

    gain = pl.flag(spells, "ph_gain_life")
    sole = np.array([is_sole(s) for s in spells["text"]])
    rider = gain & ~sole
    print(f"sole-clause lifegain spells ({int(sole.sum())}): "
          f"{sorted(spells.loc[sole, 'name'])}", flush=True)

    pctrl = spell_controls(spells)
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
    pl.write_table(rows, "c13_lifegain_split.csv")


if __name__ == "__main__":
    main()
