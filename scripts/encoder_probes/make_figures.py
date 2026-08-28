"""Render the figures embedded in experiments/2026-08-28-encoder-preferences.md.

Reads the staged probe outputs in ``output/encoder-probes/`` (regenerate them
with the probe scripts if absent) and writes ``2026-08-28-encoder-*.png`` /
``.svg`` into ``experiments/images/``. Requires matplotlib.
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
OUT = REPO / "output" / "encoder-probes"
FIG = REPO / "experiments" / "images"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 110,
})

# Same palette as scripts/scorer_probes/make_figures.py. The pairs used
# together (blue/orange, blue/red, each vs gray) were checked in OKLab under
# deutan/protan simulation; red carries sign, which bar direction re-encodes.
BLUE, ORANGE, RED, GRAY = "#3b6fb6", "#e08f3c", "#c05555", "#8a8a8a"
LABEL_SD = 0.06181


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"2026-08-28-encoder-{name}.{ext}", bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
    plt.close(fig)
    print("wrote", name)


def _report_table(path: Path, header_frag: str) -> list[list[str]]:
    """Rows of the first markdown table after the line containing header_frag."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if header_frag in ln)
    rows = []
    in_table = False
    for ln in lines[start:]:
        if ln.startswith("|"):
            in_table = True
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):
                rows.append(cells)
        elif in_table:
            break
    return rows


_NUM = re.compile(r"[-+]?\d*\.?\d+")


def _f(cell: str) -> float:
    return float(_NUM.search(cell.replace("−", "-")).group())


def fig_channels():
    rows = _report_table(OUT / "l_report.md", "feature | n cards")[1:]
    names = {
        "creature (vs noncreature)": "creature",
        "combat trick (instant +N/+N eot)": "combat trick",
        "mana rock (noncreature artifact {T}: add)": "mana rock",
        "card-draw spell (noncreature)": "card-draw spell",
        "MV 3-4 (base 0-2)": "MV 3–4", "MV 5-6 (base 0-2)": "MV 5–6",
        "MV 7+ (base 0-2)": "MV 7+",
    }
    feats = [names.get(r[0], r[0]) for r in rows]
    w = np.array([_f(r[7]) for r in rows]) / LABEL_SD
    m = np.array([_f(r[8]) for r in rows]) / LABEL_SD
    d = np.array([_f(r[9]) for r in rows]) / LABEL_SD
    score = np.array([_f(r[5]) for r in rows]) / LABEL_SD
    order = np.argsort(score)
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for vals, color, lab in ((w, BLUE, "builder selection (w)"),
                             (m, GRAY, "castability (m)"),
                             (d, ORANGE, "in-game contribution (d)")):
        v = vals[order]
        left = np.zeros(len(v))
        for other in (w, m, d):
            if other is vals:
                break
            o = other[order]
            left += np.where(np.sign(o) == np.sign(v), o, 0.0)
        ax.barh(y, v, left=left, height=0.66, color=color, label=lab,
                edgecolor="white", linewidth=1.2)
    ax.plot(score[order], y, "o", color="black", ms=5, ls="none",
            label="net label premium")
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_yticks(y, [feats[i] for i in order])
    ax.set_xlabel("contribution to the score_play premium (label SD)")
    ax.set_title("The same premium comes from different places:\n"
                 "removal is builder taste, tricks and expensive cards are cast-gap")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "channels")


def fig_memorization():
    r2 = json.load(open(OUT / "r2ab_summary.json"))["score_play"]
    shuffled = pd.read_csv(OUT / "r1a_val_r2_table.csv", index_col=0).loc["score_play", "full"]
    nameable = next(float(r[5]) for r in
                    json.load(open(OUT / "s_r18b.json"))["same_split"]
                    if r[0] == "score_play" and r[1] == "nameable features (GBM)")
    p0 = {_f(r[0]): _f(r[7])
          for r in _report_table(OUT / "p0_report.md", "min_n | classes")[1:]}
    bound_lo, bound_hi = p0[0], p0[800]
    rel = _f(_report_table(OUT / "l_report.md", "quantity | n cards")[1][3])

    bars = [
        ("label reliability ceiling (split-half)", rel, GRAY),
        ("encoder, its own training cards", r2["honest_r2_train"], BLUE),
        ("identical-text twin bound", bound_lo, GRAY),
        ("encoder, held-out cards", r2["honest_r2_val"], BLUE),
        ("135 nameable features, no encoder", nameable, ORANGE),
        ("encoder, held-out + words shuffled", shuffled, BLUE),
    ]
    bars.sort(key=lambda b: -b[1])
    y = np.arange(len(bars))[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for yi, (lab, v, c) in zip(y, bars):
        ax.barh(yi, v, height=0.62, color=c, edgecolor="white", linewidth=1.2)
        extra = bound_hi - v if lab.startswith("identical-text") else 0
        if extra:
            ax.barh(yi, extra, left=v, height=0.62, color=GRAY, alpha=0.45,
                    edgecolor="white", linewidth=1.2)
            ax.text(bound_hi + 0.012, yi, f"{v:.2f}–{bound_hi:.2f}",
                    va="center", fontsize=9)
        else:
            ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(y, [b[0] for b in bars])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("R² against the score_play label")
    ax.set_title("The train–val gap is memorization: the encoder fits its training\n"
                 "cards past the identical-text twin bound")
    save(fig, "memorization")


def fig_shuffle():
    surv = pd.read_csv(OUT / "r1a_survival_table.csv", index_col=0)
    conds = ["none", "line_order", "within_line", "full"]
    cond_labels = ["intact text", "lines permuted", "words shuffled\nwithin lines",
                   "all words shuffled"]
    y = np.arange(len(conds))[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for off, (head, color, lab) in zip(
            (0.19, -0.19),
            (("score_play", BLUE, "winnability (score_play)"),
             ("played_rate", ORANGE, "cast frequency (played_rate)"))):
        v = surv.loc[head, conds].to_numpy() * 100
        ax.barh(y + off, v, height=0.34, color=color, label=lab,
                edgecolor="white", linewidth=1.2)
        for yi, vi in zip(y + off, v):
            ax.text(vi + 1.2, yi, f"{vi:.0f}%", va="center", fontsize=8.5)
    ax.set_yticks(y, cond_labels)
    ax.set_xlim(0, 112)
    ax.set_xlabel("held-out R², as a share of the intact-text R²")
    ax.set_title("Most winnability knowledge survives destroying word order;\n"
                 "played_rate is the order-sensitive head")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "shuffle")


def fig_keywords():
    scale = pd.read_csv(OUT / "c1_scale.csv").set_index("keyword")
    div = pd.read_csv(OUT / "c_divergence_keywords.csv").set_index("keyword")
    kws = scale.index.tolist()
    kws.sort(key=lambda k: -scale.loc[k, "value_sp"])
    y = np.arange(len(kws))[::-1]

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    for yi, k in zip(y, kws):
        enc = scale.loc[k, "value_sp"]
        lab = div.loc[k, "label_c"]
        ax.plot([enc, lab], [yi, yi], color=GRAY, lw=1.4, zorder=1)
        ax.errorbar(enc, yi, xerr=[[enc - scale.loc[k, "ci_lo"]],
                                   [scale.loc[k, "ci_hi"] - enc]],
                    fmt="o", color=BLUE, ms=6, lw=1.2, capsize=0, zorder=3)
        ax.plot(lab, yi, "o", color=ORANGE, ms=6, zorder=2)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_yticks(y, kws)
    ax.set_xlabel("keyword value (label SD)")
    ax.set_title("The keyword scale: edits and labels agree on flying, deathtouch\n"
                 "and lifelink, and invert on the protection keywords")
    ax.plot([], [], "o", color=BLUE, label="encoder, counterfactual edit")
    ax.plot([], [], "o", color=ORANGE, label="label, matched regression")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    save(fig, "keywords")


_SPELL_SHORT = [
    ("damage to any target", "3 damage to any target"),
    ("fights", "fight"),
    ("exile target creature", "exile a creature"),
    ("destroy target creature with flying", "destroy a flier"),
    ("power 4 or greater", "destroy power 4+"),
    ("destroy target creature", "destroy a creature"),
    ("damage to target creature", "3 damage to a creature"),
    ("sacrifices a creature", "edict"),
    ("+3/+3", "+3/+3 combat trick"),
    ("return target creature", "bounce"),
    ("draw two", "draw two cards"),
    ("counter target spell", "counterspell"),
    ("tap target creature", "tap a creature"),
    ("destroy all creatures", "sweeper"),
    ("gain 4 life", "gain 4 life"),
]


def fig_spells():
    df = pd.read_csv(OUT / "c3_spell_ladder.csv").sort_values("value_sp")

    def short(e):
        return next((s for frag, s in _SPELL_SHORT if frag in e), e[:28])

    y = np.arange(len(df))
    v = df["value_sp"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.barh(y, v, height=0.66, color=np.where(v >= 0, BLUE, RED),
            edgecolor="white", linewidth=1.2)
    ax.errorbar(v, y, xerr=[v - df["ci_lo"], df["ci_hi"] - v],
                fmt="none", ecolor="black", elinewidth=1.0, capsize=0, alpha=0.6)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_yticks(y, [short(e) for e in df["effect"]])
    ax.set_xlabel("predicted score_play vs the same base spells (label SD)")
    ax.set_title("The spell-effect ladder: burn and fight on top,\n"
                 "lifegain text at the bottom")
    save(fig, "spells")


def fig_decode():
    df = pd.read_csv(OUT / "q3_decodability.csv")
    r2 = df[df["kind"] == "R2"].sort_values("val_r2")
    auc = df[df["kind"] == "AUC"].sort_values("auc")
    heads = [("score_play", 0.370), ("cast_lift", 0.469), ("played_rate", 0.608)]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.6, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    y = np.arange(len(r2))
    ax1.barh(y, r2["val_r2"], height=0.62, color=BLUE,
             edgecolor="white", linewidth=1.2)
    for name, v in heads:
        ax1.axvline(v, color=GRAY, lw=1.0, ls="--")
        ax1.text(v, len(r2) - 0.1, f" {name}", rotation=90, va="top",
                 fontsize=8, color=GRAY)
    ax1.set_yticks(y, r2["target"])
    ax1.set_xlabel("decoding R² (held-out cards)")
    ax1.set_title("visible attributes, vs the trained-label\nceilings (dashed)")

    short = {"is a creature": "creature", "is a land": "land",
             "first printing rare or mythic": "rare or mythic"}
    y2 = np.arange(len(auc))
    ax2.barh(y2, auc["auc"], height=0.55, color=BLUE,
             edgecolor="white", linewidth=1.2)
    ax2.axvline(0.5, color=GRAY, lw=1.0, ls="--")
    ax2.set_yticks(y2, [short.get(t, t) for t in auc["target"]])
    fig.subplots_adjust(wspace=0.42)
    ax2.set_xlim(0.4, 1.02)
    ax2.set_xlabel("decoding AUC")
    ax2.set_title("binary attributes\n(0.5 = chance)")
    fig.suptitle("The embedding describes the card better than it judges it", y=1.04)
    save(fig, "decode")


def fig_integers():
    nn = pd.read_csv(OUT / "c2_nn_sweep.csv")
    pw = pd.read_csv(OUT / "q3_power_ladder.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax1.fill_between(nn["N"], nn["ci_lo"], nn["ci_hi"], color=BLUE, alpha=0.18)
    ax1.plot(nn["N"], nn["score_play_sd"], "-o", color=BLUE, ms=5)
    ax1.axhline(0, color=GRAY, lw=0.8)
    ax1.set_xlabel("edited statline N/N on a fixed body")
    ax1.set_ylabel("predicted score_play Δ (label SD)")
    ax1.set_title("counterfactual sweep: 12/12 prices like 0/0")

    ax2.plot([0, 12], [0, 12], ls="--", color=GRAY, lw=1.0, label="printed = decoded")
    ax2.plot(pw["printed"], pw["mean"], "-o", color=BLUE, ms=5,
             label="decoded from embedding")
    ax2.set_xlabel("printed power (real cards)")
    ax2.set_ylabel("decoded power")
    ax2.set_title("decoding: flat past printed 8")
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Integer tokens carry ordinal meaning only where they are common",
                 y=1.04)
    save(fig, "integers")


if __name__ == "__main__":
    fig_channels()
    fig_memorization()
    fig_shuffle()
    fig_keywords()
    fig_spells()
    fig_decode()
    fig_integers()
