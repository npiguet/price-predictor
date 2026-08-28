"""Render the figures embedded in experiments/2026-08-27-scorer-preferences.md.

Reads the staged probe outputs in ``output/scorer-probes/`` (regenerate them
with the t*-scripts if absent) and writes ``2026-08-27-scorer-*.png`` / ``.svg``
into ``experiments/images/``. Requires matplotlib (``pip install matplotlib``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "scorer-probes"
FIG = REPO / "experiments" / "images"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 110,
})

BLUE, ORANGE, GREEN, RED, GRAY = "#3b6fb6", "#e08f3c", "#4a9a62", "#c05555", "#8a8a8a"


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"2026-08-27-scorer-{name}.{ext}", bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
    plt.close(fig)
    print("wrote", name)


def fig_calibration():
    c = json.load(open(OUT / "t7_results.json"))["C_per_set_calibration"]
    cal, platt = c["calibration"], c["platt"]
    d = np.array([r["mean_delta"] for r in cal])
    emp = np.array([r["emp_p"] for r in cal])
    xs = np.linspace(-6, 6, 200)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(xs, 1 / (1 + np.exp(-(platt["intercept"] + platt["slope"] * xs))),
            color=BLUE, label=f"fit: sigmoid({platt['intercept']:+.2f} + {platt['slope']:.2f}·Δ)")
    ax.plot(xs, 1 / (1 + np.exp(-xs)), color=GRAY, ls="--",
            label="training objective: sigmoid(Δ)")
    ax.plot(d, emp, "o", color=ORANGE, ms=6, label="held-out matches (deciles)")
    ax.axhline(0.5, color=GRAY, lw=0.6)
    ax.set_xlabel("score difference  s(A) − s(B)")
    ax.set_ylabel("P(A wins the Bo7 match)")
    ax.set_title("One score unit ≈ +18 winrate points; the raw sigmoid overshoots")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "calibration")


def fig_ablation():
    cond = json.load(open(OUT / "t5_results.json"))["conditions"]
    rows = [("full model", cond["full"]["acc"], BLUE),
            ("text shuffled within deck", cond["text_shuffle"]["acc"], BLUE),
            ("det features → corpus mean", cond["det_mean"]["acc"], ORANGE),
            ("text block → corpus mean", cond["text_mean"]["acc"], ORANGE),
            ("both → corpus mean", cond["both_mean"]["acc"], GRAY)]
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [r[2] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.barh(labels, vals, color=cols, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(0.5, color=RED, lw=1, ls=":")
    ax.text(0.501, -0.45, "coin flip", color=RED, fontsize=8)
    ax.set_xlim(0.45, 0.76)
    ax.set_xlabel("held-out accuracy (4,708 Bo7 matches)")
    ax.set_title("Erasing card text costs 9 points; erasing the 32 hand features costs 2")
    save(fig, "ablation")


def fig_pc_truncation():
    rows = json.load(open(OUT / "t6_results.json"))["p1_pc_truncation"]["rows"]
    ks = [r["k"] for r in rows]
    x = np.arange(len(ks))
    acc = [r["match_acc"] for r in rows]
    rho = [r["spearman_vs_full"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(x, rho, "o-", color=ORANGE, label="ranking agreement with full model (ρ)")
    ax.plot(x, acc, "s-", color=BLUE, label="held-out accuracy")
    ax.axhline(acc[-1], color=BLUE, lw=0.6, ls="--")
    ax.text(0.1, acc[-1] + 0.012, f"full model {acc[-1]:.3f}", color=BLUE, fontsize=8)
    ax.set_xticks(x, [str(k) for k in ks])
    ax.set_xlabel("top-k principal components of the text block kept")
    ax.set_ylabel("")
    ax.set_ylim(0.2, 1.05)
    ax.set_title("Two text directions reproduce held-out accuracy; four saturate it")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "pc-truncation")


def fig_builder_scores():
    df = pd.read_csv(OUT / "t0_decks.csv")
    order = ["random", "forge-8sub", "forge-3sub", "forge-best", "gen1", "gen2a",
             "gen2b1", "gen2ba", "gen3-128", "gen3-256", "gen4-256", "gen4-512", "gen5"]
    m = df.groupby("method")["score"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    cols = [GRAY if x.startswith(("random", "forge")) else BLUE for x in order]
    cols[3] = ORANGE
    ax.barh(order, m.values, color=cols, height=0.62)
    for i, v in enumerate(m.values):
        ax.text(v + (0.06 if v >= 0 else -0.06), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    fb, g5, rd = m["forge-best"], m["gen5"], m["random"]
    ax.set_ylim(-1.9, 13.9)
    ax.annotate("", xy=(fb, -1.1), xytext=(rd, -1.1),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1))
    ax.text((rd + fb) / 2, -1.75, f"coherence: {fb - rd:.1f} units", color=RED,
            ha="center", fontsize=8)
    ax.annotate("", xy=(g5, 13.1), xytext=(fb, 13.1),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1))
    ax.text((fb + g5) / 2, 13.45, f"quality: {g5 - fb:.1f}", color=GREEN,
            ha="center", fontsize=8)
    ax.set_xlabel("mean score of the builder's decks (42,525 decks)")
    ax.set_title("Three quarters of the score range separates incoherent decks from coherent ones")
    save(fig, "builder-scores")


def fig_shape_ladders():
    lad = pd.read_csv(OUT / "t3_ladders.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))

    ax = axes[0]
    b = lad[lad.ladder == "B_creature"].copy()
    b["cc"] = b.realized_x.round().astype(int)
    g = b.groupby("cc")["delta"].agg(["mean", "sem", "count"])
    g = g[g["count"] >= 30]
    ax.errorbar(g.index, g["mean"], yerr=g["sem"], fmt="o-", color=BLUE, capsize=2)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xlabel("creatures among 23 spells")
    ax.set_ylabel("Δscore vs as-built deck")
    ax.set_title("Creature optimum: 19–20;\ntoo few punished harder")

    ax = axes[1]
    c = lad[lad.ladder == "C_curve"].copy()
    c["mv"] = (c.realized_x * 4).round() / 4
    g = c.groupby("mv")["delta"].agg(["mean", "sem", "count"])
    g = g[g["count"] >= 20]
    ax.errorbar(g.index, g["mean"], yerr=g["sem"], fmt="o-", color=BLUE, capsize=2)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xlabel("mean spell mana value")
    ax.set_title("Curve optimum: MV 3.2–3.3;\ncheap punished harder than expensive")

    ax = axes[2]
    width = 0.38
    for off, name, colr, lbl in ((-width / 2, "A_color", BLUE, "any card of a new color"),
                                 (width / 2, "E_splash", ORANGE, "single-pip splash card")):
        sub = lad[lad.ladder == name]
        rungs = sorted(r for r in sub.rung.unique() if r > 0)
        means = [sub[sub.rung == r]["delta"].mean() for r in rungs]
        marg = np.diff([0.0] + means)
        ax.bar(np.array(rungs[:3]) + off, marg[:3], width, color=colr, label=lbl)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xticks([1, 2, 3], ["1st", "2nd", "3rd"])
    ax.set_xlabel("k-th card of the new color")
    ax.set_title("Color penalty is a threshold:\nthe first off-color card pays it")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("")
    fig.tight_layout()
    save(fig, "shape-ladders")


def fig_card_values():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import probe_lib as pl

    df = pd.read_csv(OUT / "t2_card_values.csv")
    df = df[df.v_swap.notna() & (df.det_is_land == 0)].copy()

    loc = pl.ConvertedCardLocator(pl.CARDS_PATH)
    trick, counter, pw, veh, xc = [], [], [], [], []
    for name in df.name:
        t = loc.load_text(name)
        txt = t.text.lower() if t else ""
        types = re.search(r"types:.*", txt)
        types = types.group(0) if types else ""
        mc = re.search(r"mana cost:.*", txt)
        mc = mc.group(0) if mc else ""
        trick.append(bool("instant" in types
                          and re.search(r"gets? \+\d+/\+\d+ until end of turn", txt)))
        counter.append("counter target" in txt and "spell" in txt)
        pw.append("planeswalker" in types)
        veh.append("vehicle" in types)
        xc.append("{x}" in mc)
    df["trick"], df["counter"], df["pw"], df["veh"], df["xc"] = trick, counter, pw, veh, xc

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))

    ax = axes[0]
    buckets = [(0, 1.5, "0–1"), (1.5, 2.5, "2"), (2.5, 3.5, "3"),
               (3.5, 4.5, "4"), (4.5, 5.5, "5"), (5.5, 99, "6+")]
    cre = df[df.is_creature == True]  # noqa: E712
    xs = np.arange(len(buckets))
    for off, mask, colr, lbl in ((-0.19, cre.has_flying == False, GRAY, "ground creature"),  # noqa: E712
                                 (0.19, cre.has_flying == True, BLUE, "flying creature")):  # noqa: E712
        vals = [cre[mask & (cre.mv >= lo) & (cre.mv < hi)].v_swap.mean()
                for lo, hi, _ in buckets]
        ax.bar(xs + off, vals, 0.38, color=colr, label=lbl)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xticks(xs, [b[2] for b in buckets])
    ax.set_xlabel("mana value")
    ax.set_ylabel("mean swap-in value (0 = median deck card)")
    ax.set_title("Expensive preferred at the margin;\nthe flying premium grows with size")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[1]
    ncre = df[df.is_creature == False]  # noqa: E712
    classes = [
        ("creature", cre.v_swap),
        ("planeswalker", df[df.pw].v_swap),
        ("removal (instant or sorcery)",
         ncre[(ncre.is_removal == True)].v_swap),  # noqa: E712
        ("counterspell", df[df.counter].v_swap),
        ("vehicle", df[df.veh].v_swap),
        ("X-cost spell", df[df.xc].v_swap),
        ("card draw", ncre[ncre.draws_cards == True].v_swap),  # noqa: E712
        ("combat trick", df[df.trick].v_swap),
        ("other noncreature", ncre[(ncre.is_removal == False)  # noqa: E712
                                   & (ncre.draws_cards == False)].v_swap),  # noqa: E712
    ]
    classes.sort(key=lambda t: t[1].mean())
    names = [f"{n}  (n={len(v)})" for n, v in classes]
    vals = [v.mean() for _, v in classes]
    cols = [BLUE if v > -0.05 else (ORANGE if v > -0.2 else RED) for v in vals]
    ax.barh(names, vals, color=cols, height=0.62)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("mean swap-in value")
    ax.set_title("Creatures and planeswalkers on top;\ntricks and durdle at the bottom")

    fig.tight_layout()
    save(fig, "card-values")


def fig_det_groups():
    data = json.load(open(OUT / "t5b_results.json"))
    rows = data["groups"]
    subset_of_cost = {"color pips only", "mana value only"}
    rows = sorted(rows, key=lambda r: r["d_acc"])

    labels, vals, errs, cols = [], [], [], []
    for r in rows:
        name = r["group"]
        label = f"{name}  [{r['n_features']}]"
        if name in subset_of_cost:
            label = "   └ " + label
        labels.append(label)
        vals.append(100 * r["d_acc"])
        errs.append(200 * r["paired_se"])
        if name == "all 32":
            cols.append(ORANGE)
        elif abs(r["z"]) >= 2:
            cols.append(BLUE)
        else:
            cols.append(GRAY)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(labels, vals, xerr=errs, color=cols, height=0.62,
            error_kw=dict(lw=1, capsize=2.5, ecolor="#333333"))
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(min(v - e, 0) - 0.06, i, f"{v:+.2f}", va="center", ha="right", fontsize=8)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("change in held-out accuracy when the group is erased (pp; bars ±2 paired SE)")
    ax.set_title("Color pips carry most of the deterministic features' contribution")
    ax.margins(x=0.12)
    save(fig, "det-groups")


def fig_synergy_dose():
    a = json.load(open(OUT / "t4_results.json"))["A_dose_response"]["dose_curve"]
    ks = sorted(int(k) for k in a)
    pay = [a[str(k)]["m_pay"]["mean"] for k in ks]
    pay_se = [a[str(k)]["m_pay"]["se"] for k in ks]
    ctl = [a[str(k)]["m_ctl"]["mean"] for k in ks]
    ctl_se = [a[str(k)]["m_ctl"]["se"] for k in ks]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.errorbar(ks, pay, yerr=pay_se, fmt="o-", color=BLUE, capsize=3,
                label="synergy payoff card")
    ax.errorbar(ks, ctl, yerr=ctl_se, fmt="s-", color=GRAY, capsize=3,
                label="matched control card")
    ax.set_xticks(ks)
    ax.set_xlabel("on-mechanism enablers swapped into the deck")
    ax.set_ylabel("swap-in marginal of the card")
    ax.set_ylim(0, 0.16)
    ax.set_title("Enabler density does not separate a payoff from its control")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "synergy-dose")


if __name__ == "__main__":
    fig_calibration()
    fig_ablation()
    fig_pc_truncation()
    fig_det_groups()
    fig_builder_scores()
    fig_shape_ladders()
    fig_card_values()
    fig_synergy_dose()
    print("done ->", FIG)
