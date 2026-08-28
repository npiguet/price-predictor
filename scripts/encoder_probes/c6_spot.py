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

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402

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


def main() -> None:
    j = cc.join_table()
    stripped = [cc.strip_name(t) for t in j["text"]]
    is_crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    is_sp = (j["is_instant"].fillna(False) | j["is_sorcery"].fillna(False)) \
        .astype(bool).to_numpy()
    n_ab = np.array([len(cc.ability_lines(s)) for s in stripped])
    rng = np.random.default_rng(21)
    out: dict = {}

    # ── (i) riders appended inside the spell line ───────────────────────
    sl = np.array([next((i for i, l in enumerate(cc.lines(s))
                         if l.startswith("spell[1]:")), -1) for s in stripped])
    ends_dot = np.array([sl[i] >= 0 and cc.lines(stripped[i])[sl[i]].rstrip().endswith(".")
                         for i in range(len(stripped))])
    sel = np.flatnonzero(is_sp & (sl >= 0) & (n_ab == 1) & ends_dot)
    if len(sel) > 500:
        sel = np.sort(rng.choice(sel, 500, replace=False))
    print(f"[C6] rider base: {len(sel)}", flush=True)
    texts = []
    for r in sel:
        ls = cc.lines(stripped[r])
        base_line = ls[sl[r]].rstrip()
        texts.append(stripped[r])
        for rider, _ in RIDERS:
            texts.append(cc.replace_in_line(stripped[r], sl[r],
                                            f"{base_line} {rider}"))
    k = 1 + len(RIDERS)
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    dist = cc.offmanifold(emb)
    S = pred["score_play"].to_numpy().reshape(len(sel), k)
    P = pred["played_rate"].to_numpy().reshape(len(sel), k)
    D = dist.reshape(len(sel), k)
    ctrl = 1  # index of the control rider
    rows = []
    for i, (_, label) in enumerate(RIDERS, start=1):
        v_base = S[:, i] - S[:, 0]
        v_ctrl = S[:, i] - S[:, ctrl]
        lo, hi = cc.bootstrap_ci(v_base)
        clo, chi = cc.bootstrap_ci(v_ctrl)
        rows.append({"rider": label, "n": len(sel),
                     "vs_no_rider_sd": float(v_base.mean()),
                     "ci_lo": lo, "ci_hi": hi,
                     "vs_control_rider_sd": float(v_ctrl.mean()),
                     "c_ci_lo": clo, "c_ci_hi": chi,
                     "delta_pr": float((P[:, i] - P[:, 0]).mean()),
                     "off_manifold": float((D[:, i] > cc.MANIFOLD_GATE).mean())})
    riders = pd.DataFrame(rows)
    riders.to_csv(cc.SCRATCH / "c6_riders.csv", index=False)
    print(riders.to_string(index=False), flush=True)

    # ── (iii) end-step trigger: drawback vs control vs upside ───────────
    kw_count = pd.to_numeric(j["kw_count"], errors="coerce").fillna(0).to_numpy()
    csel = np.flatnonzero(is_crea & (n_ab <= 1) & (kw_count <= 1))
    if len(csel) > 500:
        csel = np.sort(rng.choice(csel, 500, replace=False))
    print(f"[C6] end-step-trigger base: {len(csel)}", flush=True)
    texts = []
    for r in csel:
        texts.append(stripped[r])
        for clause, _ in END_TRIGGERS:
            texts.append(cc.add_line(
                stripped[r],
                f"triggered: at the beginning of your end step, {clause}"))
    k = 1 + len(END_TRIGGERS)
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    dist = cc.offmanifold(emb)
    S = pred["score_play"].to_numpy().reshape(len(csel), k)
    P = pred["played_rate"].to_numpy().reshape(len(csel), k)
    D = dist.reshape(len(csel), k)
    rows = []
    for i, (_, label) in enumerate(END_TRIGGERS, start=1):
        v_base = S[:, i] - S[:, 0]
        v_ctrl = S[:, i] - S[:, 1]
        lo, hi = cc.bootstrap_ci(v_base)
        clo, chi = cc.bootstrap_ci(v_ctrl)
        rows.append({"added_trigger": label, "n": len(csel),
                     "vs_no_line_sd": float(v_base.mean()),
                     "ci_lo": lo, "ci_hi": hi,
                     "vs_control_line_sd": float(v_ctrl.mean()),
                     "c_ci_lo": clo, "c_ci_hi": chi,
                     "delta_pr": float((P[:, i] - P[:, 0]).mean()),
                     "off_manifold": float((D[:, i] > cc.MANIFOLD_GATE).mean())})
    trig = pd.DataFrame(rows)
    trig.to_csv(cc.SCRATCH / "c6_end_triggers.csv", index=False)
    print(trig.to_string(index=False), flush=True)

    # ── (iv) dies -> enters on real death triggers ──────────────────────
    die_i = np.array([
        next((i for i, l in enumerate(cc.lines(s))
              if l.startswith("triggered: when CARDNAME dies,")), -1)
        for s in stripped])
    dsel = np.flatnonzero(is_crea & (die_i >= 0))
    print(f"[C6] death-trigger creatures: {len(dsel)}", flush=True)
    texts = []
    for r in dsel:
        ls = cc.lines(stripped[r])
        line = ls[die_i[r]]
        texts.append(stripped[r])
        texts.append(cc.replace_in_line(
            stripped[r], die_i[r],
            line.replace("when CARDNAME dies,", "when CARDNAME enters,", 1)))
        texts.append(cc.replace_in_line(
            stripped[r], die_i[r],
            line.replace("when CARDNAME dies,",
                         "when CARDNAME attacks,", 1)))
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    dist = cc.offmanifold(emb).reshape(len(dsel), 3)
    S = pred["score_play"].to_numpy().reshape(len(dsel), 3)
    P = pred["played_rate"].to_numpy().reshape(len(dsel), 3)
    rows = []
    for i, label in enumerate(["dies -> enters", "dies -> attacks"], start=1):
        v = S[:, i] - S[:, 0]
        lo, hi = cc.bootstrap_ci(v)
        rows.append({"contrast": label, "n": len(dsel),
                     "mean_sd": float(v.mean()), "ci_lo": lo, "ci_hi": hi,
                     "frac_pos": float((v > 0).mean()),
                     "delta_pr": float((P[:, i] - P[:, 0]).mean()),
                     "off_manifold": float((dist[:, i] > cc.MANIFOLD_GATE).mean())})
    timing = pd.DataFrame(rows)
    timing.to_csv(cc.SCRATCH / "c6_timing.csv", index=False)
    print(timing.to_string(index=False), flush=True)

    (cc.SCRATCH / "c6_summary.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
