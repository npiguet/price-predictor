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


def write(tab: pd.DataFrame, name: str) -> None:
    tab.to_csv(pl.SCRATCH / name, index=False)
    print(tab.to_string(index=False), flush=True)


def main() -> None:
    j = pl.load_frame(text=True)
    stripped = list(j["text"])
    is_crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    is_sp = j[["is_instant", "is_sorcery"]].fillna(0).astype(bool).any(axis=1).to_numpy()
    n_ab = np.array([len(cc.ability_lines(s)) for s in stripped])
    rng = np.random.default_rng(21)

    # ── (i) riders appended inside the spell line ───────────────────────
    sl = np.array([cc.find_line(cc.lines(s), "spell[1]:") for s in stripped])
    ends_dot = np.array([sl[i] >= 0 and cc.lines(stripped[i])[sl[i]].rstrip().endswith(".")
                         for i in range(len(stripped))])
    sel = np.flatnonzero(is_sp & (sl >= 0) & (n_ab == 1) & ends_dot)
    if len(sel) > 500:
        sel = np.sort(rng.choice(sel, 500, replace=False))
    print(f"[C6] rider base: {len(sel)}", flush=True)
    texts = []
    for r in sel:
        base_line = cc.lines(stripped[r])[sl[r]].rstrip()
        texts.append(stripped[r])
        texts += [cc.replace_in_line(stripped[r], sl[r], f"{base_line} {rider}")
                  for rider, _ in RIDERS]
    S = pl.score_texts(texts).reshape(len(sel), 1 + len(RIDERS))
    write(vs_base_and_control(S, [label for _, label in RIDERS], "rider",
                              "vs_no_rider_sd", "vs_control_rider_sd"), "c6_riders.csv")

    # ── (iii) end-step trigger: drawback vs control vs upside ───────────
    kw_count = pd.to_numeric(j["kw_count"], errors="coerce").fillna(0).to_numpy()
    csel = np.flatnonzero(is_crea & (n_ab <= 1) & (kw_count <= 1))
    if len(csel) > 500:
        csel = np.sort(rng.choice(csel, 500, replace=False))
    print(f"[C6] end-step-trigger base: {len(csel)}", flush=True)
    texts = []
    for r in csel:
        texts.append(stripped[r])
        texts += [cc.add_line(stripped[r],
                              f"triggered: at the beginning of your end step, {clause}")
                  for clause, _ in END_TRIGGERS]
    S = pl.score_texts(texts).reshape(len(csel), 1 + len(END_TRIGGERS))
    write(vs_base_and_control(S, [label for _, label in END_TRIGGERS], "added_trigger",
                              "vs_no_line_sd", "vs_control_line_sd"), "c6_end_triggers.csv")

    # ── (iv) dies -> enters on real death triggers ──────────────────────
    die_i = np.array([cc.find_line(cc.lines(s), "triggered: when CARDNAME dies,")
                      for s in stripped])
    dsel = np.flatnonzero(is_crea & (die_i >= 0))
    print(f"[C6] death-trigger creatures: {len(dsel)}", flush=True)
    texts = []
    for r in dsel:
        line = cc.lines(stripped[r])[die_i[r]]
        texts.append(stripped[r])
        texts += [cc.replace_in_line(stripped[r], die_i[r],
                                     line.replace("when CARDNAME dies,", f"when CARDNAME {w},", 1))
                  for w in ("enters", "attacks")]
    S = pl.score_texts(texts).reshape(len(dsel), 3)
    write(pd.DataFrame([
        pl.contrast_row(S.sp[:, i] - S.sp[:, 0], S.pr[:, i] - S.pr[:, 0], S.off[:, i],
                        contrast=label)
        for i, label in enumerate(["dies -> enters", "dies -> attacks"], start=1)]),
        "c6_timing.csv")


if __name__ == "__main__":
    main()
