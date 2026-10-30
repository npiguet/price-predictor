"""C6 (R15) — spot checks on riders and trigger timing.

(i)   the cantrip rider: ``. draw a card.`` appended *inside* the existing
      ``spell[1]:`` line, against ``. you gain 2 life.`` and ``. scry 1.``
      appended in the same place — no line is added, so the whole family is
      layout-matched and the contrast is rider-vs-rider;
(iii) the drawback rider: one added ``triggered:`` line in every arm, so
      the line-add artifact cancels in ``drawback - control``;
(iv)  trigger timing: ``when CARDNAME dies`` -> ``when CARDNAME enters`` on
      real death-trigger creatures (a one-word substitution).

(ii) — flash — is covered by C1's keyword matrix, where ``flash`` is one of
the sixteen substitutable keywords.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402

RIDERS = [
    ("you gain 2 life.", "control rider (gain 2 life)"),
    ("draw a card.", "cantrip rider (draw a card)"),
    ("scry 1.", "scry 1 rider"),
    ("draw two cards.", "draw two cards rider"),
    ("you lose 2 life.", "drawback rider (lose 2 life)"),
]

END_TRIGGERS = [
    ("you gain 1 life.", "control (gain 1 life)"),
    ("sacrifice CARDNAME.", "drawback (sacrifice CARDNAME)"),
    ("draw a card.", "upside (draw a card)"),
    ("you lose 1 life.", "drawback (lose 1 life)"),
    ("put a +1/+1 counter on CARDNAME.", "upside (+1/+1 counter)"),
]


def vs_base_and_control(S: pl.Scored, labels: list[str], label_key: str,
                        base_key: str, control_key: str) -> pd.DataFrame:
    """Each edited arm (columns 1..) against the unedited card (column 0) and
    against the control arm (column 1)."""
    rows = []
    for i, label in enumerate(labels, start=1):
        vs_ctrl = pl.mean_ci(S.sp[:, i] - S.sp[:, 1], control_key)
        rows.append({label_key: label, "n": S.sp.shape[0],
                     **pl.mean_ci(S.sp[:, i] - S.sp[:, 0], base_key),
                     control_key: vs_ctrl[control_key],
                     "c_ci_lo": vs_ctrl["ci_lo"], "c_ci_hi": vs_ctrl["ci_hi"],
                     "delta_pr": float((S.pr[:, i] - S.pr[:, 0]).mean()),
                     "off_manifold": float(S.off[:, i].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    j = pl.load_frame(text=True)
    C = cc.CardCorpus.of(j)
    stripped = C.text
    is_sp = pl.flag(j, "is_instant") | pl.flag(j, "is_sorcery")
    rng = np.random.default_rng(21)

    # (i) riders appended inside the spell line
    sl = np.array([cc.find_line(cc.lines(s), "spell[1]:") for s in stripped])
    ends_dot = np.array([sl[i] >= 0 and cc.lines(stripped[i])[sl[i]].rstrip().endswith(".")
                         for i in range(len(stripped))])
    sel = cc.sample_rows(is_sp & (sl >= 0) & (C.n_abilities == 1) & ends_dot, 500, rng)
    print(f"[C6] rider base: {len(sel)}", flush=True)

    def with_rider(r: int, rider: str | None) -> str:
        if rider is None:
            return stripped[r]
        base_line = cc.lines(stripped[r])[sl[r]].rstrip()
        return cc.replace_in_line(stripped[r], sl[r], f"{base_line} {rider}")

    S = pl.score_arms(sel, [None] + [rider for rider, _ in RIDERS], with_rider)
    pl.write_table(vs_base_and_control(S, [label for _, label in RIDERS], "rider",
                                       "vs_no_rider_sd", "vs_control_rider_sd"),
                   "c6_riders.csv")

    # (iii) end-step trigger: drawback vs control vs upside
    csel = cc.sample_rows(C.is_creature & (C.n_abilities <= 1) & (C.kw_count <= 1), 500, rng)
    print(f"[C6] end-step-trigger base: {len(csel)}", flush=True)
    S = pl.score_arms(csel, [None] + [clause for clause, _ in END_TRIGGERS],
                      lambda r, clause: stripped[r] if clause is None else cc.add_line(
                          stripped[r], f"triggered: at the beginning of your end step, {clause}"))
    pl.write_table(vs_base_and_control(S, [label for _, label in END_TRIGGERS],
                                       "added_trigger", "vs_no_line_sd", "vs_control_line_sd"),
                   "c6_end_triggers.csv")

    # (iv) dies -> enters on real death triggers
    die_i = np.array([cc.find_line(cc.lines(s), "triggered: when CARDNAME dies,")
                      for s in stripped])
    dsel = np.flatnonzero(C.is_creature & (die_i >= 0))
    print(f"[C6] death-trigger creatures: {len(dsel)}", flush=True)

    def retimed(r: int, when: str | None) -> str:
        if when is None:
            return stripped[r]
        line = cc.lines(stripped[r])[die_i[r]]
        return cc.replace_in_line(stripped[r], die_i[r], line.replace(
            "when CARDNAME dies,", f"when CARDNAME {when},", 1))

    S = pl.score_arms(dsel, [None, "enters", "attacks"], retimed)
    pl.write_table(pl.contrasts(S, ["dies -> enters", "dies -> attacks"]), "c6_timing.csv")


if __name__ == "__main__":
    main()
