"""Draw the charts of the gen-2 effect-model records from their CSV files.

    python scripts/effect_gen2_report/make_charts.py

Reads ``experiments/2026-09-18-effect-model-gen2/data/*.csv`` (written by
``build_data.py``) and writes one PNG per chart into the ``charts/`` folder beside
it. Every chart reads the CSVs alone, so the charts can be redrawn without the model
runs. Checkpoints or runs missing from the data are left out of each chart, which is
how the 256-wide runs join the charts as they land.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "experiments" / "2026-09-18-effect-model-gen2" / "data"
CHARTS = DATA.parent / "charts"

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 110, "legend.frameon": False,
})
GRAY = "#8a8a8a"
WIDTH_COLOUR = {64: "#3b6fb6", 128: "#e08f3c", 256: "#4a9a62"}
# A run's phase in its width's line: the arm, run A (full rate), run B (a tenth).
PHASE_SHADE = {"arm": 0.45, "A": 0.75, "B": 1.0}
LINES = {64: ("e64-lethality", "A64", "B64"), 128: ("e128-lethality", "A128", "B128"),
         256: ("e256-lethality", "A256", "B256")}
NO_COPIES = {64: "e64", 128: "e128"}
CHECKPOINTS = ["e64 (earlier inputs)", "e64", "e128", "e64-lethality", "e128-lethality",
               "e256-lethality", "A64 best", "A64 last", "B64 best", "B64 last", "A128 best",
               "A128 last", "B128 best", "B128 last", "A256 best", "A256 last", "B256 best",
               "B256 last"]
SHORT = {"e64 (earlier inputs)": "e64 early", "e64-lethality": "e64-L",
         "e128-lethality": "e128-L", "e256-lethality": "e256-L"}
WIDTH_OF = {"e64 (earlier inputs)": 64, "e64": 64, "e128": 128, "e64-lethality": 64,
            "e128-lethality": 128, "e256-lethality": 256, "A64": 64, "B64": 64, "A128": 128,
            "B128": 128, "A256": 256, "B256": 256}


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def shade(colour: str, amount: float) -> tuple:
    """``colour`` blended toward white; 1 keeps it, 0 is white."""
    rgb = np.array(matplotlib.colors.to_rgb(colour))
    return tuple(1 - amount * (1 - rgb))


def run_colour(run: str) -> tuple:
    width = WIDTH_OF.get(run.split(" ")[0], 64)
    phase = "B" if run.startswith("B") else "A" if run.startswith("A") else "arm"
    return shade(WIDTH_COLOUR[width], PHASE_SHADE[phase])


def present(names, available) -> list[str]:
    return [n for n in names if n in set(available)]


def short(name: str) -> str:
    return SHORT.get(name, name)


def save(fig, name: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", name)


def rule_staircase(ax, offset: float = 0.0) -> None:
    """Outline the cells where the hit reaches the toughness (row ≥ column)."""
    xs, ys = [-0.5], [-0.5]
    for k in range(8):
        xs += [k + 0.5, k + 0.5]
        ys += [k - 0.5 + offset, k + 0.5 + offset]
    ax.plot(xs, ys, color="black", lw=1.2)


def _oklch_to_srgb(lightness: float, chroma: float, hue_deg: float):
    """One OKLCh colour as sRGB in 0..1, or None when no screen can show it."""
    a = chroma * np.cos(np.radians(hue_deg))
    b = chroma * np.sin(np.radians(hue_deg))
    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    linear = np.array([
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076381450 * s_,
    ])
    if linear.min() < -1e-4 or linear.max() > 1 + 1e-4:
        return None
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * linear ** (1 / 2.4) - 0.055)


def _death_colourmap() -> matplotlib.colors.ListedColormap:
    """Perceived lightness rises evenly from 0 to 1; the hue flips from blue to orange at 0.5.

    Built in OKLCh, a perceptually uniform colour space, so the two hues meet at the
    same perceived brightness and the break marks the even-odds line without a jump in
    brightness. Chroma is lowered only where a colour falls outside the sRGB gamut.
    """
    colours = []
    for v in np.linspace(0, 1, 256):
        lightness = 0.22 + 0.75 * v
        hue = 255.0 if v < 0.5 else 62.0
        chroma = 0.16
        rgb = _oklch_to_srgb(lightness, chroma, hue)
        while rgb is None:
            chroma -= 0.002
            rgb = _oklch_to_srgb(lightness, chroma, hue)
        colours.append(tuple(rgb))
    return matplotlib.colors.ListedColormap(colours, name="death")


DEATH_CMAP = _death_colourmap()


def colourbar(fig, axes) -> None:
    sm = matplotlib.cm.ScalarMappable(cmap=DEATH_CMAP, norm=matplotlib.colors.Normalize(0, 1))
    fig.colorbar(sm, ax=axes, shrink=0.7, label="predicted death")


def heatmap(ax, grid: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    ax.imshow(grid, cmap=DEATH_CMAP, vmin=0, vmax=1, origin="upper")
    ax.set_xticks(range(8), range(1, 9))
    ax.set_yticks(range(8), range(1, 9))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)


# ── training ───────────────────────────────────────────────────────────────


def training_losses() -> None:
    ep = csv("epochs.csv")
    widths = [w for w in (64, 128, 256) if LINES[w][0] in set(ep.run)]
    metrics = ("card_disjoint", "game_disjoint", "train")
    fig, axes = plt.subplots(len(metrics), len(widths), figsize=(4.3 * len(widths), 9.4),
                             sharey="row", squeeze=False)
    limits = {"card_disjoint": (3.4, 4.6), "game_disjoint": (2.75, 3.7), "train": (2.7, 4.7)}
    for col, width in enumerate(widths):
        for row, metric in enumerate(metrics):
            ax = axes[row][col]
            no_copies = NO_COPIES.get(width)
            if no_copies in set(ep.run):
                d = ep[ep.run == no_copies]
                ax.plot(d.epoch, d[metric], color=GRAY, lw=1, alpha=0.8,
                        label=f"{no_copies}, no copies")
            for run in LINES[width]:
                d = ep[ep.run == run].sort_values("epoch")
                if d.empty:
                    continue
                ax.plot(d.epoch, d[metric], "o-", ms=2.5, lw=1, color=run_colour(run),
                        label=run)
                best = d.loc[d.card_disjoint.idxmin()]
                ax.plot(best.epoch, best[metric], "*", ms=11, color=run_colour(run),
                        mec="black", mew=0.6)
            ax.set_ylim(*limits[metric])
            ax.set_xlim(left=5)
            if row == 0:
                done = LINES[width][0] in set(csv("checkpoints.csv").run)
                ax.set_title(f"e width {width}" + ("" if done else " (still training)"))
                ax.legend(fontsize=8, loc="upper right")
            if col == 0:
                ax.set_ylabel({"train": "training loss"}.get(
                    metric, metric.replace("_", "-") + " loss"))
            if row == len(metrics) - 1:
                ax.set_xlabel("epoch")
    fig.suptitle("Validation and training losses by epoch; a star marks each run's lowest "
                 "card-disjoint epoch, the one it keeps", y=1.0)
    save(fig, "training-losses")


def loss_noise() -> None:
    """Spread of each run's epoch losses around its own five-epoch moving mean."""
    ep = csv("epochs.csv")
    rows = []
    for run, d in ep.groupby("run"):
        d = d.sort_values("epoch")
        d = d[d.epoch >= 10]
        if len(d) < 8 or run == "gen-1":
            continue
        for metric in ("card_disjoint", "game_disjoint", "train"):
            trend = d[metric].rolling(5, center=True).mean()
            resid = (d[metric] - trend).dropna()
            step = d[metric].diff().abs().dropna()
            rows.append(dict(run=run, metric=metric, sd=resid.std(), step=step.median()))
    df = pd.DataFrame(rows)
    order = [r for r in ["e64 (earlier inputs)", "e64", "e128", "e64-lethality",
                         "e128-lethality", "e256-lethality", "A64", "A128", "A256", "B64",
                         "B128", "B256"] if r in set(df.run)]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(order))
    for i, (metric, label) in enumerate((("card_disjoint", "card-disjoint"),
                                         ("game_disjoint", "game-disjoint"),
                                         ("train", "training"))):
        vals = [df[(df.run == r) & (df.metric == metric)].sd.iloc[0] for r in order]
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=label,
               color=["#3b6fb6", "#e08f3c", GRAY][i])
    ax.set_xticks(x, [short(r) for r in order], rotation=30, ha="right")
    ax.set_ylabel("standard deviation around\nthe 5-epoch moving mean")
    ax.set_title("Epoch-to-epoch noise in each loss, from epoch 10 on; "
                 "B64 and B128 train at a tenth of the learning rate")
    ax.legend()
    save(fig, "loss-noise")


def probes_vs_loss() -> None:
    ck = csv("checkpoints.csv")
    lad = csv("probe_ladder.csv")
    m = lad[(lad.rung == "3") & (lad.probe == "model") & (lad.target != "damage_taken")]
    ck = ck.set_index("checkpoint")
    for stratum in ("card-disjoint", "game-disjoint"):
        ck[f"probe_{stratum}"] = m[m.stratum == stratum].groupby("checkpoint").score.mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (loss, label) in zip(axes, (("card_disjoint", "card-disjoint loss"),
                                        ("game_disjoint", "game-disjoint loss"))):
        for name, r in ck.iterrows():
            colour = WIDTH_COLOUR[int(r.e_width)]
            marker = "o" if r.lethality_copies else "s"
            ax.plot(r[loss], r["probe_game-disjoint"], marker, color=colour, ms=7,
                    mfc="white" if not r.lethality_copies else colour, mew=1.4)
            ax.annotate(short(name), (r[loss], r["probe_game-disjoint"]), fontsize=7,
                        xytext=(4, 3), textcoords="offset points", color="#333")
        for run in ("A64", "B64", "A128", "B128", "A256", "B256"):
            if f"{run} best" in ck.index and f"{run} last" in ck.index:
                b, la = ck.loc[f"{run} best"], ck.loc[f"{run} last"]
                ax.annotate("", (la[loss], la["probe_game-disjoint"]),
                            (b[loss], b["probe_game-disjoint"]),
                            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))
        ax.set_xlabel(f"{label} (lower is better)")
    axes[0].set_ylabel("mean of the model's own probe scores,\ngame-disjoint records")
    fig.suptitle("Probes against each validation loss; arrows go from a run's kept epoch to "
                 "its last (open squares: no lethality copies)", y=1.0)
    save(fig, "probes-vs-loss")


def field_explained() -> None:
    fe = csv("fields_by_epoch.csv")
    fields = [("gate", "affected gate"), ("zone_outcome", "zone outcome"),
              ("keywords_gained", "keywords gained"), ("blocker_legal", "blocking legality"),
              ("types_gained", "types gained"), ("poison_delta", "poison")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.8), sharex=True)
    for ax, (field, label) in zip(axes.flat, fields):
        for width in (64, 128, 256):
            for run in LINES[width]:
                d = fe[(fe.run == run) & (fe.field == field)].sort_values("epoch")
                if not d.empty:
                    ax.plot(d.epoch, d.explained_pct, lw=1, color=run_colour(run), label=run)
        ax.set_title(label)
        ax.axhline(0, color=GRAY, lw=0.6)
        ax.set_ylim(-40, 100)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of the constant\nprediction's loss removed")
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.suptitle("Card-disjoint explained deviance by field, along the lethality-copy lines",
                 y=1.0)
    save(fig, "fields-explained")


# ── the death rule ─────────────────────────────────────────────────────────


def damage_grids() -> None:
    g = csv("damage_grids.csv")
    g = g[g.record_set == "probe set"]
    chosen = present(["e64", "e64-lethality", "e128-lethality", "e256-lethality", "A64 best",
                      "A64 last", "B64 last", "A128 best", "B128 last", "A256 best", "A256 last",
                      "B256 last"], g.checkpoint)
    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.1 * rows), squeeze=False)
    for i, (ax, name) in enumerate(zip(axes.flat, chosen)):
        d = g[g.checkpoint == name].pivot(index="damage", columns="toughness",
                                          values="predicted_death")
        heatmap(ax, d.values, short(name), "toughness", "damage" if i % cols == 0 else "")
        rule_staircase(ax)
    for ax in list(axes.flat)[len(chosen):]:
        ax.axis("off")
    fig.subplots_adjust(hspace=0.45)
    colourbar(fig, axes)
    fig.suptitle("Predicted death of a damage spell's target, 20 probe records per cell; "
                 "below the black line the rule says it dies", y=1.0)
    save(fig, "damage-grids")


def toughness_sweep_records() -> None:
    """One band per damage group: lowest to highest record at each toughness, median line.

    Activated pings, a permanent's paid ability dealing 1 damage (Prodigal Sorcerer's
    {T}), form their own group, apart from 1-damage spells and triggers.
    """
    sr = csv("sweep_records.csv")
    meta = csv("toughness_sweep_records.csv").set_index("record")
    meta["group"] = [("1 activated" if k == "activated" and d == 1 else str(d))
                     for d, k in zip(meta.damage, meta.kind)]
    groups = [("1", 1, "#3b6fb6", "1 damage, spell or trigger"),
              ("1 activated", 1, "#8e5bb5", "1 damage, activated ping"),
              ("2", 2, "#4a9a62", "2 damage"), ("3", 3, "#e08f3c", "3 damage"),
              ("5", 5, "#c05555", "5 damage")]
    grid = [["e64", "e64-lethality", "A64 best", "A64 last", "B64 best", "B64 last"],
            ["e128", "e128-lethality", "A128 best", "A128 last", "B128 best", "B128 last"],
            [None, "e256-lethality", "A256 best", "A256 last", "B256 best", "B256 last"]]
    columns = ["no copies", "arm with copies", "A run, kept epoch", "A run, last epoch",
               "B run, kept epoch", "B run, last epoch"]
    available = set(sr.checkpoint)
    fig, axes = plt.subplots(len(grid), len(columns),
                             figsize=(2.9 * len(columns), 2.8 * len(grid)), sharey=True,
                             squeeze=False)
    empty = []
    for r, row in enumerate(grid):
        for c, name in enumerate(row):
            ax = axes[r][c]
            if name not in available:
                ax.axis("off")
                empty.append(ax)
                continue
            d = sr[(sr.checkpoint == name) & (sr.sweep == "toughness")]
            d = d.join(meta[["group"]], on="record")
            ax.axhline(0.5, color="black", lw=0.8)
            for group, damage, colour, label in groups:
                g = d[d.group == group].groupby("value").prediction
                n = d[d.group == group].record.nunique()
                ax.fill_between(g.min().index, g.min(), g.max(), color=colour, alpha=0.16, lw=0)
                ax.plot(g.median().index, g.median(), color=colour, lw=1.6,
                        label=f"{label} ({n})")
                ax.axvline(damage + 0.5, color=colour, lw=0.8, ls=":")
            ax.set_title(short(name), fontsize=9.5)
            ax.set_xticks(range(1, 9))
            ax.tick_params(labelsize=8)
            ax.set_ylim(0, 1)
            if r == len(grid) - 1 or grid[r + 1][c] not in available:
                ax.set_xlabel("target's toughness", fontsize=8.5)
        first = next((c for c, name in enumerate(row) if name in available), None)
        if first is not None:
            axes[r][first].set_ylabel(f"e width {64 * 2 ** r}\npredicted death", fontsize=9)
            axes[r][first].yaxis.set_tick_params(labelleft=True)
    for c, label in enumerate(columns):
        axes[0][c].annotate(label, (0.5, 1.22), xycoords="axes fraction", ha="center",
                            fontsize=9.5, fontweight="bold")
    handles, labels = axes[0][0].get_legend_handles_labels()
    # The last empty slot is the bottom-right one, away from every row's axis label.
    (empty[-1] if empty else axes[0][-1]).legend(handles, labels, fontsize=8, loc="center",
                                                 title="ability (records)", title_fontsize=8)
    fig.subplots_adjust(hspace=0.45, wspace=0.12)
    fig.suptitle("Toughness sweep: band from lowest to highest record at each toughness, line at"
                 " the median; the rule says each group stops dying at its dotted line", y=0.99)
    save(fig, "toughness-sweep-records")


def boundary_by_epoch() -> None:
    b = csv("boundary_by_epoch.csv")
    ck = csv("checkpoints.csv")
    widths = [w for w in (64, 128, 256) if LINES[w][1] in set(b.run)]
    fig, axes = plt.subplots(1, len(widths), figsize=(5.6 * len(widths), 3.9), sharey=True,
                             squeeze=False)
    for ax, width in zip(axes[0], widths):
        arm, a, bb = LINES[width]
        for metric, colour, label in (("lethal_by_1", "#c05555", "lethal by 1 point"),
                                      ("exactly_lethal", "#e08f3c", "exactly lethal"),
                                      ("short_by_1", "#3b6fb6", "short by 1 point")):
            for run, ls, lw in ((arm, ":", 1.0), (a, "-", 0.9), (bb, "-", 2.0)):
                d = b[b.run == run].sort_values("epoch")
                ax.plot(d.epoch, d[metric], marker="o", ms=2.5, ls=ls, lw=lw, color=colour,
                        alpha=0.55 if run == a else 1.0,
                        label=label if run == bb else None)
        start_b = b[b.run == bb].epoch.min()
        if not np.isnan(start_b):
            ax.axvspan(start_b - 0.5, b[b.run == bb].epoch.max() + 0.5, color=GRAY, alpha=0.12)
            ax.text(start_b, 0.03, " run B: a tenth of the\n learning rate", fontsize=8,
                    color="#555")
        for _, r in ck[ck.run.isin([arm, a, bb]) & (ck.role == "best")].iterrows():
            ax.axvline(r.epoch, color="black", lw=0.7, ls="--")
        ax.set_title(f"e width {width}: {arm}, then {a}, then {bb}")
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="black", lw=0.9, zorder=1)
    axes[0][0].set_ylabel("predicted death given affected,\n40 training records")
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Where the lethal boundary sits, epoch by epoch (dotted: the arm; thin: the A run;"
                 " thick: the B run); dashed lines mark the epochs each run keeps", y=1.02)
    save(fig, "boundary-by-epoch")


def damage_sweep() -> None:
    """The damage sweep, over quadrants that say whether a prediction is on the rule's side.

    The rule kills a toughness-4 target from 4 damage up, so a curve is on the right
    side below one half left of the step and above one half right of it, however soft.
    """
    s = csv("sweeps.csv")
    s = s[s.sweep == "damage_spell"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    right, wrong = "#4a9a62", "#c05555"
    for (x0, x1), (y0, y1), colour in (((0.5, 3.5), (0, 0.5), right),
                                       ((3.5, 8.5), (0.5, 1), right),
                                       ((0.5, 3.5), (0.5, 1), wrong),
                                       ((3.5, 8.5), (0, 0.5), wrong)):
        ax.fill_between([x0, x1], y0, y1, color=colour, alpha=0.10, lw=0)
    ax.text(3.4, 0.02, "right: survives", fontsize=8, color=right, ha="right")
    ax.text(8.3, 0.96, "right: dies", fontsize=8, color=right, ha="right", va="top")
    ax.text(0.7, 0.96, "wrong: predicted to die", fontsize=8, color=wrong, va="top")
    ax.text(8.3, 0.03, "wrong: predicted to survive", fontsize=8, color=wrong, ha="right")
    style = {"arm": ":", "A": "--", "B": "-"}
    for name in present(["e64", "e64-lethality", "A64 last", "B64 best", "B64 last",
                         "e128-lethality", "A128 best", "B128 best", "B128 last",
                         "e256-lethality", "A256 last", "B256 best", "B256 last"],
                        s.checkpoint):
        d = s[s.checkpoint == name]
        run = name.split(" ")[0]
        if name == "e64":
            colour, ls = GRAY, "-"
        else:
            phase = "B" if run.startswith("B") else "A" if run.startswith("A") else "arm"
            colour, ls = WIDTH_COLOUR[WIDTH_OF[run]], style[phase]
        on_rule = bool(((d.value < 4) == (d.prediction < 0.5)).all())
        marker = "s" if name.endswith(" best") else "o"
        ax.plot(d.value, d.prediction, marker=marker, ms=4 if on_rule else 2.5,
                lw=2.0 if on_rule else 1.0, ls=ls, color=colour, alpha=1.0 if on_rule else 0.3,
                zorder=3 if on_rule else 2,
                label=("✓ " if on_rule else "") + short(name)
                + (", no copies" if name == "e64" else ""))
    ax.axvline(3.5, color="black", lw=0.8)
    ax.axhline(0.5, color="black", lw=0.8)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("NumDmg$ of the spell")
    ax.set_ylabel("predicted death, toughness-4 target")
    ax.set_title("Damage sweep: the rule kills from 4 damage up. Bold, ✓: every point on the "
                 "rule's side\n(dotted: arm, dashed: A run, solid: B run; squares: a run's kept "
                 "epoch)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, "damage-sweep")


# ── combat ─────────────────────────────────────────────────────────────────


def combat_grids() -> None:
    c = csv("combat_grids.csv")
    name = "B64 last"  # the checkpoint the gen-2 record's combat section describes
    panels = [("none", "blocker", "Blocker's death, no keyword\n(rule: dies below the line)",
               "attacker's power", "blocker's toughness", True),
              ("none", "attacker", "Attacker's death, no keyword\n(rule: dies below the line)",
               "blocker's power", "attacker's toughness", True),
              ("attacker-deathtouch", "blocker",
               "Blocker's death, attacker has deathtouch\n(rule: dies on every cell)",
               "attacker's power", "blocker's toughness", False),
              ("attacker-first-strike", "attacker",
               "Attacker's death in the first-strike step,\nattacker has first strike "
               "(rule: never dies)", "blocker's power", "attacker's toughness", False)]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.0))
    for ax, (variant, reads, title, ylabel, xlabel, staircase) in zip(axes.flat, panels):
        d = c[(c.checkpoint == name) & (c.variant == variant) & (c.reads == reads)]
        grid = d.pivot(index="striker_power", columns="struck_toughness",
                       values="predicted_death").values
        heatmap(ax, grid, title, xlabel, ylabel)
        ax.title.set_fontsize(10)
        if staircase:
            rule_staircase(ax)
    fig.subplots_adjust(hspace=0.42, wspace=0.3)
    colourbar(fig, axes)
    fig.suptitle(f"Combat grids of {short(name)}: predicted death over 40 real one-on-one "
                 "blocks per cell", y=0.97)
    save(fig, "combat-grids")


def combat_keywords() -> None:
    c = csv("combat_grids.csv")
    pw, tg = c.striker_power, c.struck_toughness
    lethal, short_ = pw >= tg, pw < tg
    second = (pw < tg) & (tg <= 2 * pw)
    # Each label names the creature whose death is read, the creature holding the keyword,
    # and the cells averaged: the ones where the keyword changes the rule's answer.
    checks = [
        ("Blocker's death\nblocker has indestructible · lethal hits",
         "blocker-indestructible", "blocker", lethal, 1, 0),
        ("Blocker's death\nattacker has deathtouch · hits short of lethal",
         "attacker-deathtouch", "blocker", short_, 0, 1),
        ("Blocker's death\nattacker's double strike, second hit · lethal only as two",
         "attacker-double-strike-second-hit", "blocker", second, 0, 1),
        ("Attacker's death, first-strike step\nattacker has first strike · blocker's lethal hits",
         "attacker-first-strike", "attacker", lethal, 1, 0),
        ("Blocker's death, first-strike step\nblocker has first strike · attacker's lethal hits",
         "blocker-first-strike", "blocker", lethal, 1, 0),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for i, (label, variant, reads, cells, rule_before, rule_after) in enumerate(checks):
        y = len(checks) - 1 - i
        plain = c[(c.variant == "none") & (c.reads == reads) & cells]
        kw = c[(c.variant == variant) & (c.reads == reads) & cells]
        p = plain.groupby("checkpoint").predicted_death.mean()
        k = kw.groupby("checkpoint").predicted_death.mean()
        common = p.index.intersection(k.index)
        for ck in common:
            ax.plot([p[ck], k[ck]], [y + 0.12, y - 0.12], color=GRAY, lw=0.5, alpha=0.6)
        ax.plot(p[common], np.full(len(common), y + 0.12), "o", color="#8fb3e0", ms=5,
                label="without the keyword" if i == 0 else None)
        ax.plot(k[common], np.full(len(common), y - 0.12), "o", color="#c05555", ms=5,
                label="with the keyword" if i == 0 else None)
        ax.plot(rule_after, y - 0.12, "|", color="black", ms=16, mew=2.2,
                label="the rule, with the keyword" if i == 0 else None)
    ax.set_yticks(range(len(checks)), [x[0] for x in checks][::-1], fontsize=8.5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("mean predicted death over the cells the keyword changes, "
                  "one dot per checkpoint")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.set_title("Does a keyword move the predicted death where the rule says?")
    save(fig, "combat-keywords")


def gate2_heatmap() -> None:
    g = csv("gate2.csv")
    order = present(CHECKPOINTS, g.checkpoint)
    kws = ["lifelink", "indestructible", "trample", "first_strike", "deathtouch", "infect",
           "double_strike"]
    g = g[~((g.checkpoint == "e64 (earlier inputs)")
            & g.keyword.isin(["first_strike", "double_strike"]))]  # scored on the wrong step
    n = g.groupby("keyword").qualifying_records.median().astype(int)
    grid = np.array([[g[(g.checkpoint == ck) & (g.keyword == kw)].agreement_pct.mean()
                      for ck in order] for kw in kws])
    fig, ax = plt.subplots(figsize=(10.2, 3.6))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=40, vmax=100, aspect="auto")
    for i in range(len(kws)):
        for j in range(len(order)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=7.5)
    labels = []
    for kw in kws:
        half = 196 * np.sqrt(0.8 * 0.2 / n[kw])
        labels.append(f"{kw.replace('_', ' ')}  (n={n[kw]}, ±{half:.0f})")
    ax.set_yticks(range(len(kws)), labels)
    ax.set_xticks(range(len(order)), [short(o) for o in order], rotation=35, ha="right")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="direction agreement %", shrink=0.8)
    ax.set_title("Gate 2: how often a keyword moves the prediction the way it moves the real "
                 "outcome (± is the 95% sampling half-width at 80%)")
    save(fig, "gate2-keywords")


# ── where knowledge lives ──────────────────────────────────────────────────


def ladder() -> None:
    lad = csv("probe_ladder.csv")
    lad = lad[(lad.stratum == "card-disjoint")
              & (((lad.probe == "mlp") & lad.rung.isin(["0", "1", "2"]))
                 | ((lad.probe == "model") & (lad.rung == "3")))]
    targets = [("affordable", "affordable"), ("pt_until_end_of_turn", "pump lasts until EOT"),
               ("legal_target", "legal target"),
               ("legal_target_protected", "legal target, protected"),
               ("fires", "trigger fires"), ("affected", "card affected"), ("dies", "dies"),
               ("may_block", "may block"), ("damage_taken", "damage taken (R²)")]
    gen2 = present(CHECKPOINTS[3:], lad.checkpoint)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    styles = {"0": ("board alone", "#bbbbbb", "s"), "1": ("board + e, no trunk", "#3b6fb6", "o"),
              "2": ("probe on the trunk's vector", "#e08f3c", "D"),
              "3": ("model's own answer (output head)", "#c05555", "*")}
    for i, (target, label) in enumerate(targets):
        y = len(targets) - 1 - i
        for k, (rung, (name, colour, marker)) in enumerate(styles.items()):
            d = lad[(lad.target == target) & (lad.rung == rung) & lad.checkpoint.isin(gen2)]
            if d.empty:
                continue
            med, lo, hi = d.score.median(), d.score.min(), d.score.max()
            yy = y + (k - 1.5) * 0.15
            ax.plot([lo, hi], [yy, yy], color=colour, lw=1.5, alpha=0.6)
            ax.plot(med, yy, marker, color=colour, ms=7 if marker != "*" else 10,
                    label=name if i == 0 else None)
    ax.set_yticks(range(len(targets)), [t[1] for t in targets][::-1])
    ax.set_xlim(-0.2, 1.0)
    ax.set_xlabel("probe score on card-disjoint records (AUC; R² for damage taken): median "
                  "over gen-2 checkpoints with copies, bar = range")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Where the knowledge lives: probes on the board alone, on e, on the trunk's "
                 "vector, and the model's own answer")
    save(fig, "probe-ladder")


def line_properties() -> None:
    lad = csv("probe_ladder.csv")
    d = lad[(lad.rung == "1") & (lad.probe == "mlp") & lad.stratum.isin(["trained", "held-out"])]
    targets = [("tap_cost", "tap cost"), ("sacrifice_cost", "sacrifice cost"),
               ("instant_speed", "instant speed"), ("repeatable", "repeatable"),
               ("cost_total", "total mana cost (R²)"), ("damage", "damage amount (R²)"),
               ("counters_placed", "counters placed (R²)"), ("power_change", "power change (R²)"),
               ("toughness_change", "toughness change (R²)")]
    targets = [t for t in targets if t[0] in set(d.target)]
    groups = [("gen-1", "gen-1"), ("e64", "e64"),
              ("B64 last", "B64 last"), ("B128 last", "B128 last"), ("B256 last", "B256 last")]
    groups = [g for g in groups if g[0] in set(d.checkpoint)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, stratum in zip(axes, ("trained", "held-out")):
        width = 0.8 / len(groups)
        for j, (ck, label) in enumerate(groups):
            vals = [d[(d.checkpoint == ck) & (d.target == t) & (d.stratum == stratum)].score
                    .mean() for t, _ in targets]
            ax.barh(np.arange(len(targets)) + (j - len(groups) / 2) * width, vals, width,
                    label=label, color=[GRAY, "#8fb3e0", "#1f3f73", "#e08f3c",
                                        "#4a9a62"][j % 5])
        ax.set_yticks(range(len(targets)), [t[1] for t in targets])
        ax.set_xlim(-0.3, 1.02)
        ax.axvline(0, color=GRAY, lw=0.6)
        ax.set_title(f"{stratum} lines")
        ax.invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(groups),
               bbox_to_anchor=(0.5, -0.06), fontsize=8.5)
    fig.suptitle("Line properties read from e alone by an MLP probe (AUC, or R² for amounts)",
                 y=1.0)
    save(fig, "line-properties")


def board_abilities() -> None:
    b = csv("board_abilities.csv")
    panels = [("attacker_legal", ["legality-static", "legality-keyword", "legality-none"],
               "attack legality"),
              ("blocker_legal", ["legality-static", "legality-keyword", "legality-none"],
               "block legality"),
              ("zone_outcome", ["resolution-replacement", "resolution-static",
                                "resolution-none"], "zone outcome")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (field, subsets, title) in zip(axes, panels):
        for i, subset in enumerate(subsets):
            d = b[(b.field == field) & (b.subset == subset)]
            ax.bar(i, d.loss.median(), color=["#c05555", "#3b6fb6", GRAY][i], alpha=0.7)
            ax.plot(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.2, 0.2, len(d)),
                    d.loss, "o", color="black", ms=2.5)
        ax.set_xticks(range(3), [s.split("-")[1] for s in subsets])
        ax.set_title(title)
    axes[0].set_ylabel("field loss (median bar; one dot per checkpoint)")
    fig.suptitle("A restriction the card slots cannot show: a static (red) against a keyword "
                 "(blue) or nothing (grey)", y=1.02)
    save(fig, "board-abilities")


def ablation() -> None:
    a = csv("ablation.csv")
    a = a[a.replacement == "noise"]
    fields = ["gate", "keywords_gained", "power_delta", "damage_taken", "life_delta",
              "counters_delta_p1p1"]
    gen2 = present(CHECKPOINTS[3:], a.checkpoint)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for k, (scope, colour, label) in enumerate((("all", GRAY, "every slot"),
                                                ("act", "#c05555", "[ACT] only"),
                                                ("cards", "#3b6fb6", "card slots only"))):
        vals = [a[(a.field == f) & (a.scope == scope) & a.checkpoint.isin(gen2)]
                .loss_increase.median() for f in fields]
        ax.bar(np.arange(len(fields)) + (k - 1) * 0.27, vals, 0.27, color=colour, label=label)
    ax.set_xticks(range(len(fields)), ["affected\ngate", "keywords\ngained", "power\nchange",
                                       "damage\ntaken", "life\nchange", "+1/+1\ncounters"])
    ax.set_ylabel("loss increase, median over\ngen-2 checkpoints with copies")
    ax.legend()
    ax.set_title("Replacing e with matched noise: the trunk reads the acting ability's e, "
                 "not the board's")
    save(fig, "ablation")


def width_summary() -> None:
    ck = csv("checkpoints.csv")
    g3 = csv("gate3.csv").set_index("checkpoint")
    lad = csv("probe_ladder.csv")
    m = lad[(lad.rung == "3") & (lad.probe == "model") & (lad.target != "damage_taken")]
    probe = m.groupby(["checkpoint", "stratum"]).score.mean().unstack()
    ck = ck[ck.lethality_copies > 0].set_index("checkpoint")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    metrics = [(ck.card_disjoint, "card-disjoint loss"), (ck.game_disjoint, "game-disjoint loss"),
               (probe.loc[ck.index, "game-disjoint"], "probe mean, game-disjoint"),
               (g3.loc[g3.index.intersection(ck.index), "participation_ratio"],
                "participation ratio of e")]
    for ax, (series, title) in zip(axes, metrics):
        for name, v in series.items():
            w = int(ck.loc[name, "e_width"])
            run = ck.loc[name, "run"]
            phase = "B" if run.startswith("B") else "A" if run.startswith("A") else "arm"
            offset = {"arm": -0.24, "A": -0.04, "B": 0.16}[phase]
            offset += 0.07 if ck.loc[name, "role"] == "last" else 0
            ax.plot(list(WIDTH_COLOUR).index(w) + offset, v,
                    "^" if ck.loc[name, "role"] == "last" else "o", color=run_colour(run), ms=6)
        ax.set_xticks(range(3), ["64", "128", "256"])
        ax.set_xlabel("width of e")
        ax.set_title(title)
    fig.suptitle("Every checkpoint trained with lethality copies, by width (light: arm, "
                 "mid: the A run, dark: the B run; triangles: a run's last epoch)", y=1.03)
    save(fig, "width")


def other_sweeps() -> None:
    s = csv("sweeps.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for name in present(["e64", "e64-lethality", "B64 last", "B128 last", "B256 last"],
                        s.checkpoint):
        d = s[(s.checkpoint == name) & (s.sweep == "affordability") & (s.value <= 8)]
        base = name.split(" ")[0]
        ax.plot(d.value, d.prediction, marker="o", ms=3, lw=1.2, label=short(name),
                color=GRAY if name == "e64" else run_colour(base))
    items = s[(s.sweep == "affordability") & (s.value <= 8)].groupby("value").records.max()
    for v, n in items.items():
        ax.text(v, -0.06, str(n), ha="center", fontsize=7, color="#555")
    ax.set_ylim(-0.1, 1)
    ax.set_xlabel("untapped mana given to the acting player (items per point below)")
    ax.set_ylabel("predicted affordable verdict")
    ax.set_title("Affordability sweep")
    ax.legend(fontsize=8)
    for name in present(CHECKPOINTS, s.checkpoint) + (["gen-1"] if "gen-1" in set(s.checkpoint)
                                                      else []):
        d = s[(s.checkpoint == name) & (s.sweep == "board_size")]
        ax2.plot(d.value, d.prediction, lw=1, color=GRAY if name == "gen-1" else "#3b6fb6",
                 alpha=0.9 if name == "gen-1" else 0.35)
    ax2.plot([0, 8], [0, 8], color="black", lw=0.8, ls="--")
    ax2.text(1.75, 2.35, "one death\nper creature", fontsize=8)
    ax2.text(5.5, 0.35, "gen-1", fontsize=8, color=GRAY)
    ax2.text(4.5, 1.3, "every gen-2 checkpoint", fontsize=8, color="#3b6fb6")
    ax2.set_xlabel("opposing creatures on the board")
    ax2.set_ylabel("predicted deaths summed over the board")
    ax2.set_title("Board-size sweep, a sweeper in the acting slot")
    ax2.set_ylim(0, 3)
    save(fig, "other-sweeps")


EVASION_ROWS = ["none", "flying", "horsemanship", "fear", "intimidate", "shadow",
                "unblockable", "protection-blocker", "protection-other"]
EVASION_COLUMNS = ["none", "flying", "reach", "horsemanship", "shadow", "artifact", "black",
                   "shares-colour"]


def evasion_grid() -> None:
    """Predicted block legality per attacker keyword and blocker trait, rule cells outlined."""
    g = csv("evasion_grid.csv")
    chosen = present(["e64-lethality", "B64 last", "B128 last"], g.checkpoint)
    fig, axes = plt.subplots(1, len(chosen), figsize=(5.4 * len(chosen), 5.4), squeeze=False)
    for ax, name in zip(axes[0], chosen):
        d = g[g.checkpoint == name]
        pred = d.pivot(index="attacker_keyword", columns="blocker_trait",
                       values="predicted_may_block").loc[EVASION_ROWS, EVASION_COLUMNS]
        rule = d.pivot(index="attacker_keyword", columns="blocker_trait",
                       values="rule_may_block").loc[EVASION_ROWS, EVASION_COLUMNS]
        ax.imshow(pred.values, cmap="viridis", vmin=0, vmax=0.8, aspect="auto")
        for i in range(len(EVASION_ROWS)):
            for j in range(len(EVASION_COLUMNS)):
                value = pred.values[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                        color="black" if value > 0.45 else "white")
                if rule.values[i, j] >= 0.5:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="#ff4d4d", lw=2.2))
        ax.set_xticks(range(len(EVASION_COLUMNS)), EVASION_COLUMNS, rotation=40, ha="right")
        ax.set_yticks(range(len(EVASION_ROWS)), EVASION_ROWS)
        ax.set_xlabel("trait given to the blocker")
        ax.set_title(short(name))
        ax.grid(False)
    axes[0][0].set_ylabel("keyword granted to the attacker")
    for ax in axes[0][1:]:
        ax.set_yticklabels([])
    fig.subplots_adjust(wspace=0.06)
    fig.suptitle("Predicted chance the blocker may block, 40 real blocks per cell; red outlines "
                 "mark the cells where the rule allows the block", y=1.0)
    save(fig, "evasion-grid")


def evasion_ranking() -> None:
    """Per attacker keyword: AUC of allowed over forbidden blocks, one dot per checkpoint."""
    r = csv("evasion_ranking.csv")
    rows = [k for k in ["none", "flying", "horsemanship", "fear", "intimidate", "shadow",
                        "protection-other", "all rows"] if k in set(r.attacker_keyword)]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, keyword in enumerate(rows):
        y = len(rows) - 1 - i
        d = r[r.attacker_keyword == keyword].dropna(subset=["auc"])
        for _, row in d.iterrows():
            run = row.checkpoint.split(" ")[0]
            colour = run_colour(run) if run in WIDTH_OF else GRAY
            ax.plot(row.auc, y + np.random.default_rng(hash(row.checkpoint) % 1000)
                    .uniform(-0.15, 0.15), "o", color=colour, ms=5)
    ax.axvline(0.5, color="black", lw=0.8)
    ax.set_yticks(range(len(rows)), rows[::-1])
    ax.set_xlim(0.15, 1.0)
    ax.set_xlabel("AUC: allowed blocks ranked above forbidden ones (0.5 = chance), one dot per "
                  "checkpoint\n(blue 64-wide, orange 128-wide, green 256-wide; darker = later "
                  "in training)")
    ax.set_title("Does the model rank the blocks each evasion keyword allows above the ones "
                 "it forbids?")
    save(fig, "evasion-ranking")


def target_flips() -> None:
    """Predicted legality of a legal target before and after breaking its one condition."""
    f = csv("target_flips.csv")
    flips = (f.groupby("flip").targets.max().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for i, flip in enumerate(flips.index):
        y = len(flips) - 1 - i
        d = f[f.flip == flip]
        for _, row in d.iterrows():
            ax.plot([row.predicted_legal_before, row.predicted_legal_after],
                    [y + 0.12, y - 0.12], color=GRAY, lw=0.5, alpha=0.6)
        ax.plot(d.predicted_legal_before, np.full(len(d), y + 0.12), "o", color="#8fb3e0",
                ms=5, label="before the flip" if i == 0 else None)
        ax.plot(d.predicted_legal_after, np.full(len(d), y - 0.12), "o", color="#c05555",
                ms=5, label="after the flip" if i == 0 else None)
    ax.axvline(0, color="black", lw=2.2, ymax=0.02)
    ax.set_yticks(range(len(flips)),
                  [f"{flip} ({n} targets)" for flip, n in flips.items()][::-1])
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("mean predicted legality of the target, one dot per checkpoint "
                  "(the rule: legal before, illegal after)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("Breaking the one condition a target restriction states")
    save(fig, "target-flips")


def main() -> None:
    for chart in (training_losses, loss_noise, probes_vs_loss, field_explained, damage_grids,
                  toughness_sweep_records, boundary_by_epoch, damage_sweep, combat_grids,
                  combat_keywords, gate2_heatmap, ladder, line_properties, board_abilities,
                  ablation, width_summary, other_sweeps, evasion_grid, evasion_ranking,
                  target_flips):
        chart()


if __name__ == "__main__":
    main()
