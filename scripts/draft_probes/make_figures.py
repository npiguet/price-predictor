"""Render the figures embedded in experiments/2026-08-29-draft-agent-behaviour.md.

Reads the staged probe outputs in ``output/draft-probes/`` (regenerate them with
the d*-scripts if absent) and writes ``2026-08-29-draft-*.png`` / ``.svg`` into
``experiments/images/``. Requires matplotlib (``pip install matplotlib``).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "draft-probes"
FIG = REPO / "experiments" / "images"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 110,
})

BLUE, ORANGE, GREEN, RED, GRAY = "#3b6fb6", "#e08f3c", "#4a9a62", "#c05555", "#8a8a8a"
# Roughly the five colours as Magic prints them, darkened enough to read on white.
PIE = {"W": "#c9b071", "U": "#2f6fb5", "B": "#4b4b55", "R": "#c0392b", "G": "#3f8f4f"}
WUBRG = "WUBRG"
GENS = ["gen1", "gen3", "gen4"]
GEN_LABEL = {"gen1": "gen-1", "gen3": "gen-3", "gen4": "gen-4",
             "gen4b": "gen-4 sibling"}
GEN_COLOUR = {"gen1": GRAY, "gen3": ORANGE, "gen4": BLUE, "gen4b": GREEN}


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"2026-08-29-draft-{name}.{ext}", bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
    plt.close(fig)
    print("wrote", name)


def fig_channels() -> None:
    """Each block against a random substitution of its own size, by pack."""
    rows = [r for r in csv.DictReader((OUT / "d1_placebo_table.csv").open())
            if r["model"] == "gen4"]
    blocks = ["POOL", "PASSED", "TAKEN"]
    labels, obs, plc = [], [], []
    for b in blocks:
        for p in ("1", "2", "3"):
            r = next(x for x in rows if x["block"] == b and x["pack"] == p)
            labels.append(f"{b}  pack {p}\n({float(r['mean_size']):.0f} cards)")
            obs.append(float(r["observed_flip"]))
            plc.append(float(r["placebo_flip"]))
    y = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.barh(y + 0.19, obs, height=0.36, color=[BLUE] * 3 + [GRAY] * 6,
            label="blanking the block")
    ax.barh(y - 0.19, plc, height=0.36, color="none", edgecolor=RED, hatch="///",
            label="blanking that many random cards")
    for yi, o, p in zip(y, obs, plc):
        ax.text(max(o, p) + 0.008, yi, f"×{o / p:.1f}", va="center", fontsize=9,
                color=BLUE if o / p > 1.5 else GRAY)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("share of picks that change")
    ax.set_xlim(0, 0.56)
    # Below the axes: the bars fill the plot area at every height, so any
    # in-axes corner collides with one of them.
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.14))
    ax.set_title("Erasing what the others took moves fewer picks than erasing "
                 "noise\nof the same size; only the seat's own pool clears its "
                 "placebo")
    save(fig, "channels")


def fig_colour_chain() -> None:
    """Where each colour ranks at every step from Forge's games to the pick."""
    games = np.array(load("d11_winratecolour.json")
                     ["fits"]["+ Forge draft rank"]["beta"][:5])
    reward = np.array(load("d10_rewardcolour.json")["fits"]["colour only"]["beta"][:5])
    p1p1 = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader((OUT / "d2_p1p1_values.csv").open(encoding="utf-8")):
        for gen in ("gen1", "gen4"):
            v = r.get(f"p1p1_{gen}")
            if v and r["colours"]:
                for c in r["colours"]:
                    if c in PIE:
                        p1p1[gen][c].append(float(v))
    series = [
        ("Forge's\nself-play games", games),
        ("the scorer's\nreward", reward),
        ("gen-4's\nfirst pick", np.array([np.mean(p1p1["gen4"][c]) for c in WUBRG])),
        ("gen-1's\nfirst pick", np.array([np.mean(p1p1["gen1"][c]) for c in WUBRG])),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = [0, 1, 2, 3.55]
    for i, c in enumerate(WUBRG):
        ranks = [1 + list(np.argsort(-v)).index(i) for _, v in series]
        ax.plot(xs[:3], ranks[:3], "-o", color=PIE[c], lw=2.4, ms=9,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.plot(xs[3], ranks[3], "o", color=PIE[c], ms=9,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.text(xs[0] - 0.12, ranks[0], c, ha="right", va="center",
                fontsize=11, fontweight="bold", color=PIE[c])
        ax.text(xs[-1] + 0.14, ranks[-1], c, ha="left", va="center",
                fontsize=11, fontweight="bold", color=PIE[c])
    ax.axvline(2.78, color=GRAY, lw=0.8, ls=":")
    ax.text(3.55, 5.42, "control: never trained on a game result",
            fontsize=8, color=GRAY, va="bottom", ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([s for s, _ in series], fontsize=9)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(5.5, 0.5)
    ax.set_ylabel("rank among the five colours")
    ax.set_xlim(-0.5, 4.1)
    ax.grid(axis="x", visible=False)
    ax.set_title("White and green carry through the whole chain;\nblue and red "
                 "swap places between the games and the models")
    save(fig, "colour-chain")


def fig_pick_order() -> None:
    """Agreement of each generation's pick order with the models below it."""
    sp = load("d2_pickorder.json")["part3_attribution"]["spearman"]
    x = np.arange(3)
    lines = [("the scorer's card values", "scorer_v_swap", BLUE, "-"),
             ("encoder PC2 (winnability)", "text_pc2", GREEN, "--"),
             ("encoder PC1 (played rate)", "text_pc1", ORANGE, "--")]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for label, key, col, ls in lines:
        v = [sp[g][key]["spearman"] for g in GENS]
        ax.plot(x, v, ls, color=col, marker="o", lw=2.2, ms=7, label=label)
        ax.text(2.06, v[-1], f"{v[-1]:.2f}", color=col, va="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([GEN_LABEL[g] for g in GENS])
    ax.set_xlim(-0.15, 2.45)
    ax.set_ylim(0.1, 0.78)
    ax.set_ylabel("Spearman with the first-pick card ranking")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Every generation's pick order moves toward the two models\n"
                 "underneath it")
    save(fig, "pick-order")


def fig_leverage() -> None:
    """Where in a pack the generations disagree."""
    lev = load("d3_exchange.json")["leverage"]
    by_pick: dict[int, list] = defaultdict(list)
    for r in lev:
        by_pick[r["pick"]].append(r)
    picks = sorted(by_pick)
    kl = [float(np.mean([r["kl_gen1_gen4"] for r in by_pick[p]])) for p in picks]
    over = [float(np.mean([r["disagree_over_baseline"] for r in by_pick[p]]))
            for p in picks]
    raw = [float(np.mean([r["disagree_gen1_gen4"] for r in by_pick[p]]))
           for p in picks]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(picks, raw, "-o", color=GRAY, lw=1.8, ms=5,
            label="picks where gen-1 and gen-4 differ")
    ax.plot(picks, over, "-o", color=BLUE, lw=2.4, ms=6,
            label="the same, divided by what chance would give")
    ax.set_xlabel("pick within the pack")
    ax.set_ylabel("share of picks")
    ax.set_ylim(0, 0.68)
    ax.set_xticks(range(1, 16, 2))
    ax2 = ax.twinx()
    ax2.plot(picks, kl, "--", color=ORANGE, lw=2.0,
             label="KL(gen-1 ‖ gen-4)")
    ax2.set_ylabel("KL (nats)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(ORANGE)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="lower left")
    ax.set_title("The generations diverge where the pack still offers a choice,\n"
                 "and agree once it is down to the last few cards")
    save(fig, "leverage")


def fig_commitment() -> None:
    """The pull toward the pool's colours, and the pool's share of the tokens."""
    m = load("d4_commitment.json")["models"]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for gen in ("gen1", "gen3", "gen4", "gen4b"):
        rows = m.get(gen)
        if not rows:
            continue
        ax.plot([r["clock"] for r in rows], [r["slope"] for r in rows], "o",
                color=GEN_COLOUR[gen], lw=2.0, ms=5, label=GEN_LABEL[gen],
                ls="--" if gen == "gen4b" else "-")
    ax.set_xlabel("pick number in the draft")
    ax.set_ylabel("logit pull per unit of the pool's own colour")
    ax.set_ylim(0, 22)
    ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)
    ax2 = ax.twinx()
    rows = m["gen4"]
    ax2.fill_between([r["clock"] for r in rows],
                     [r["pool_token_share"] for r in rows], color=GRAY, alpha=0.13)
    ax2.set_ylabel("pool's share of the tokens (shaded)", color=GRAY)
    ax2.set_ylim(0, 0.42)
    ax2.tick_params(axis="y", colors=GRAY)
    ax2.grid(False)
    ax.set_title("Commitment to the pool's colours nearly triples across a draft,\n"
                 "and gen-1 hardens as much as gen-4")
    save(fig, "commitment")


def fig_duplicates() -> None:
    """The redundancy bonus: dose ladder, and by how alike the two cards are."""
    m = load("d8_duplicates.json")["models"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for gen in ("gen1", "gen3", "gen4", "gen4b"):
        d = m.get(gen)
        if not d:
            continue
        ks = [1, 2, 3]
        a1.errorbar(ks, [d[str(k)]["effect_mean"] for k in ks],
                    yerr=[d[str(k)]["effect_se"] for k in ks], fmt="o",
                    color=GEN_COLOUR[gen], lw=2.0, ms=6, capsize=3,
                    ls="--" if gen == "gen4b" else "-", label=GEN_LABEL[gen])
        buckets = d["1"]["by_similarity"]
        a2.plot([0.5 * (b["cos_lo"] + b["cos_hi"]) for b in buckets],
                [b["mean"] for b in buckets], "o", color=GEN_COLOUR[gen],
                lw=2.0, ms=6, ls="--" if gen == "gen4b" else "-")
    a1.axhline(0, color=GRAY, lw=0.8)
    a1.plot([1, 2, 3], [m["gen4"][str(k)]["placebo_mean"] for k in (1, 2, 3)],
            ":", color=RED, lw=1.8, label="a third card nobody copied")
    a1.set_xticks([1, 2, 3])
    a1.set_xlabel("copies already in the pool")
    a1.set_ylabel("change in the card's centred logit")
    a1.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    a1.set_title("Owning a copy raises the card's value")
    a2.set_xlabel("how alike the two candidate cards are (cosine)")
    a2.set_ylabel("effect of one copy")
    a2.set_ylim(0, None)
    a2.set_title("Part of it is resemblance, but not all")
    fig.suptitle("The redundancy bonus is largest in gen-1, the one generation "
                 "that never trained against the reward", fontsize=11)
    fig.tight_layout()
    save(fig, "duplicates")


def fig_context() -> None:
    """What a linear probe recovers from the trunk's summary token."""
    m = load("d7_contextprobe.json")["models"]
    xs = np.arange(len(m["gen4"]))
    lab = [f"{r['pick_lo']}–{r['pick_hi']}" for r in m["gen4"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(xs, [r["colour_auc"] for r in m["gen4"]], "-o", color=GREEN, lw=2.4,
            ms=6, label="the two colours the seat finishes in (AUC)")
    ax.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.set_ylim(0.42, 1.0)
    ax.set_ylabel("AUC", color=GREEN)
    ax.tick_params(axis="y", colors=GREEN)
    ax2 = ax.twinx()
    for gen, col in (("gen4", BLUE), ("gen1", ORANGE)):
        ax2.plot(xs, [r["reward_r2"] for r in m[gen]], "-s", color=col, lw=2.0,
                 ms=5, label=f"{GEN_LABEL[gen]}: the seat's final score (R²)")
    ax2.set_ylabel("R²")
    ax2.set_ylim(0, 0.72)
    ax2.grid(False)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab)
    ax.set_xlabel("picks made so far")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="lower right")
    ax.set_title("The colours are settled by pick 10, and gen-1 — the one "
                 "generation\nwhose value head was trained — reads the final "
                 "score better")
    save(fig, "context")


if __name__ == "__main__":
    fig_channels()
    fig_colour_chain()
    fig_pick_order()
    fig_leverage()
    fig_commitment()
    fig_duplicates()
    fig_context()
