"""Draw the charts of the gen-2 effect-model records from their CSV files.

    python scripts/effect_gen2_report/make_charts.py

Reads ``experiments/2026-09-18-effect-model-gen2/data/*.csv`` (written by
``build_data.py``) and writes one PNG per chart into the ``charts/`` folder beside
it. Every chart reads the CSVs alone, so the charts can be redrawn without the model
runs. Checkpoints or runs missing from the data are left out of each chart, which is
how the 256-wide runs join the charts as they land.

One visual language runs through every chart. The width of ``e`` is the colour (64
blue, 128 orange, 256 green) and nothing else. The stage of training is the lightness
plus a second channel: the marker shape for a point (arm ▲, A run ■, B run ●) and the
line style for a line (arm dotted, A run dashed, B run solid). A run's kept epoch is a
filled marker and its last epoch a hollow one. Arms trained without lethality copies
are grey.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "experiments" / "2026-09-18-effect-model-gen2" / "data"
CHARTS = DATA.parent / "charts"

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 110, "legend.frameon": False, "axes.titlelocation": "left",
    "axes.titleweight": "bold", "figure.titleweight": "bold",
})
GRAY = "#8a8a8a"
INK, INK_SOFT = "#222222", "#555555"
WIDTH_COLOUR = {64: "#3b6fb6", 128: "#e08f3c", 256: "#4a9a62"}
# A run's phase in its width's line: the arm, run A (full rate), run B (a tenth).
PHASE_SHADE = {"arm": 0.62, "A": 0.82, "B": 1.0}
STAGE_MARKER = {"arm": "^", "A": "s", "B": "o"}
STAGE_LINE = {"arm": ":", "A": "--", "B": "-"}
STAGE_NAME = {"arm": "arm", "A": "A run, full rate", "B": "B run, a tenth of the rate"}
NO_COPIES_GRAY = "#9a9a9a"
EARLIER_GRAY = "#c8c8c8"
GEN1_GRAY = "#4d4d4d"
LINES = {64: ("e64-lethality", "A64", "B64"), 128: ("e128-lethality", "A128", "B128"),
         256: ("e256-lethality", "A256", "B256")}
NO_COPIES = {64: "e64", 128: "e128"}
CHECKPOINTS = ["e64 (earlier inputs)", "e64", "e128", "e64-lethality", "e128-lethality",
               "e256-lethality", "A64 best", "A64 last", "B64 best", "B64 last", "A128 best",
               "A128 last", "B128 best", "B128 last", "A256 best", "A256 last", "B256 best",
               "B256 last"]
WIDTH_OF = {"e64 (earlier inputs)": 64, "e64": 64, "e128": 128, "e64-lethality": 64,
            "e128-lethality": 128, "e256-lethality": 256, "A64": 64, "B64": 64, "A128": 128,
            "B128": 128, "A256": 256, "B256": 256}
#: Diverging map for a change of either sign: red below the neutral grey, purple above.
#: It stays off the three width hues and off a red-green pair.
DIVERGING_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "diverging", ["#a3302b", "#e3a39b", "#f4f4f4", "#b5a3d6", "#46277f"])
#: Sequential map for "better", kept off the three width hues.
BETTER_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "better", ["#f4f1f8", "#cdbfe0", "#8f73bd", "#4b2a86", "#2a1550"])
#: Diverging map for "e against a random vector": red below zero, purple above.
GAIN_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "gain", ["#b2342f", "#e8a49c", "#f3f3f3", "#b9a6d8", "#4b2a86"])
#: Non-width colours for categories that are not checkpoints (rungs, scopes, distances).
PURPLE, PURPLE_SOFT, SLATE = "#5b3a8e", "#b39ddb", "#3d3d3d"


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def shade(colour: str, amount: float) -> tuple:
    """``colour`` blended toward white; 1 keeps it, 0 is white."""
    rgb = np.array(matplotlib.colors.to_rgb(colour))
    return tuple(1 - amount * (1 - rgb))


def darker(colour, amount: float = 0.3) -> tuple:
    """``colour`` blended toward black by ``amount``."""
    return tuple((1 - amount) * np.array(matplotlib.colors.to_rgb(colour)))


def run_of(checkpoint: str) -> str:
    return checkpoint if checkpoint in WIDTH_OF else checkpoint.split(" ")[0]


def stage_of(name: str) -> str:
    """``arm``, ``A`` or ``B`` for a run or checkpoint; ``none`` without copies; ``gen-1``."""
    run = run_of(name)
    if run == "gen-1":
        return "gen-1"
    if run in ("e64 (earlier inputs)", "e64", "e128"):
        return "none"
    return "B" if run.startswith("B") else "A" if run.startswith("A") else "arm"


def run_colour(run: str) -> tuple | str:
    """The line colour of a run: its width's hue at its stage's lightness, grey without copies."""
    stage = stage_of(run)
    if stage == "gen-1":
        return GEN1_GRAY
    if stage == "none":
        return EARLIER_GRAY if run_of(run) == "e64 (earlier inputs)" else NO_COPIES_GRAY
    return shade(WIDTH_COLOUR[WIDTH_OF[run_of(run)]], PHASE_SHADE[stage])


def checkpoint_marker(checkpoint: str, ms: float = 6.5) -> dict:
    """``plot`` keywords for one checkpoint: width colour, stage shape, kept filled, last hollow.

    ``B256 best`` is run B256's own lowest epoch, which it did not keep (it keeps run
    A256's epoch 53), and is half filled.
    """
    stage = stage_of(checkpoint)
    colour = run_colour(checkpoint)
    if stage in ("none", "gen-1"):
        edge = "#666666" if stage == "none" else GEN1_GRAY
        marker = "^" if stage == "none" else "X"
    else:
        edge = shade(WIDTH_COLOUR[WIDTH_OF[run_of(checkpoint)]], 1.0)
        marker = STAGE_MARKER[stage]
    style = dict(marker=marker, color=colour, mec=edge, mew=1.2, ms=ms * (0.92 if marker == "s"
                                                                          else 1.0), ls="none")
    if checkpoint.endswith(" last") and not is_merged(checkpoint):
        style["mfc"] = "white"
    elif checkpoint == "B256 best":
        style.update(fillstyle="left", mfcalt="white")
    return style


def run_line(run: str) -> dict:
    """``plot`` keywords for a run's line: colour by width and stage, style by stage."""
    stage = stage_of(run)
    return dict(color=run_colour(run), ls=STAGE_LINE.get(stage, "-"),
                lw=1.0 if stage in ("none", "gen-1") else 1.5)


def present(names, available) -> list[str]:
    return [n for n in names if n in set(available)]


def _merged_pairs() -> dict[str, str]:
    """``{"X best": "X last"}`` for each run whose last epoch is its kept one."""
    ck = csv("checkpoints.csv").set_index("checkpoint")
    out = {}
    for name in ck.index:
        best = name.replace(" last", " best")
        if name.endswith(" last") and best in ck.index and ck.loc[best, "epoch"] == ck.loc[
                name, "epoch"]:
            out[best] = name
    return out


MERGED = None


def is_merged(checkpoint: str) -> bool:
    """Whether ``checkpoint`` is one half of a run whose last epoch is its kept one."""
    global MERGED
    if MERGED is None:
        MERGED = _merged_pairs()
    return checkpoint in MERGED or checkpoint in MERGED.values()


def short(name: str) -> str:
    """The name the records use for a checkpoint; a merged run's kept epoch reads "= last"."""
    if name.endswith(" best") and is_merged(name):
        return f"{name} = last"
    return name


def gen2(available, copies_only: bool = False) -> list[str]:
    """The gen-2 checkpoints in ``available``, in lineage order, each merged run once."""
    names = [n for n in present(CHECKPOINTS, available)
             if not (n.endswith(" last") and is_merged(n))]
    return [n for n in names if stage_of(n) != "none"] if copies_only else names


def gen2_checkpoints(available) -> list[tuple[int, str, str]]:
    """``(width, checkpoint, label)`` by width of e, in lineage order, each merged run once."""
    pairs = merged_labels(present(CHECKPOINTS, available))
    return sorted(((WIDTH_OF[run_of(n)], n, label) for n, label in pairs),
                  key=lambda t: t[0])


def save(fig, name: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / f"{name}.png", bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def offsets(names: list[str], spread: float = 0.3) -> dict[str, float]:
    """A fixed vertical offset per checkpoint, earliest on top, so a row reads in order."""
    if len(names) == 1:
        return {names[0]: 0.0}
    return dict(zip(names, np.linspace(spread, -spread, len(names))))


def tag(ax, x: float, y: float, text: str, dx: float = 8, dy: float = 8,
        ha: str = "left", colour: str = INK) -> None:
    """A small label with a leader line, for a checkpoint the text names."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                color=colour, ha=ha, va="center", zorder=6,
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=0.6, shrinkA=0, shrinkB=3))


def checkpoint_handles(names, fill: bool = True) -> list:
    """Legend entries explaining the markers of ``names``: widths, stages, kept and last."""
    stages = {stage_of(n) for n in names}
    widths = sorted({WIDTH_OF[run_of(n)] for n in names if stage_of(n) in ("arm", "A", "B")})
    handles = [Line2D([], [], marker="s", ls="none", ms=8, color=WIDTH_COLOUR[w],
                      label=f"{w}-wide e") for w in widths]
    handles += [Line2D([], [], marker=STAGE_MARKER[s], ls="none", ms=7, mfc=INK_SOFT,
                       mec=INK_SOFT, label=STAGE_NAME[s]) for s in ("arm", "A", "B")
                if s in stages]
    if "none" in stages:
        handles.append(Line2D([], [], marker="^", ls="none", ms=7, color=NO_COPIES_GRAY,
                              mec="#666666", label="arm without copies"))
    if "e64 (earlier inputs)" in names:
        handles.append(Line2D([], [], marker="^", ls="none", ms=7, color=EARLIER_GRAY,
                              mec="#666666", label="e64 (earlier inputs)"))
    if "gen-1" in stages:
        handles.append(Line2D([], [], marker="X", ls="none", ms=7, color=GEN1_GRAY,
                              label="gen-1"))
    if fill and any(n.endswith(" last") and not is_merged(n) for n in names):
        handles += [Line2D([], [], marker="o", ls="none", ms=7, mfc=INK_SOFT, mec=INK_SOFT,
                           label="kept epoch: filled"),
                    Line2D([], [], marker="o", ls="none", ms=7, mfc="white", mec=INK_SOFT,
                           mew=1.2, label="last epoch: hollow")]
    return handles


def line_handles(widths, stages=("arm", "A", "B"), no_copies: bool = True) -> list:
    """Legend entries for run lines: one per width hue and one per stage's line style."""
    handles = [Line2D([], [], color=WIDTH_COLOUR[w], lw=2.5, label=f"{w}-wide e")
               for w in widths]
    handles += [Line2D([], [], color=INK_SOFT, ls=STAGE_LINE[s], lw=1.5, label=STAGE_NAME[s])
                for s in stages]
    if no_copies:
        handles.append(Line2D([], [], color=NO_COPIES_GRAY, lw=1.2, label="arm without copies"))
    return handles


def strip(ax, y: float, values: pd.Series, names: list[str], spread: float = 0.3,
          muted: bool = False, ms: float = 6.5) -> dict[str, tuple[float, float]]:
    """One marker per checkpoint at ``y`` plus its fixed offset; returns each point."""
    where = offsets(names, spread)
    points = {}
    for name in names:
        if name not in values.index or pd.isna(values[name]):
            continue
        style = checkpoint_marker(name, ms)
        if muted:
            style.update(color=EARLIER_GRAY, mec=GRAY, alpha=0.8)
            if style.get("mfc") not in (None, "white"):
                style["mfc"] = EARLIER_GRAY
        ax.plot(values[name], y + where[name], zorder=4, **style)
        points[name] = (values[name], y + where[name])
    return points


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


def colourbar(fig, axes, **kwargs) -> None:
    sm = matplotlib.cm.ScalarMappable(cmap=DEATH_CMAP, norm=matplotlib.colors.Normalize(0, 1))
    fig.colorbar(sm, ax=axes, shrink=kwargs.pop("shrink", 0.7), label="predicted death",
                 **kwargs)


def heatmap(ax, grid: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    ax.imshow(grid, cmap=DEATH_CMAP, vmin=0, vmax=1, origin="upper")
    ax.set_xticks(range(8), range(1, 9))
    ax.set_yticks(range(8), range(1, 9))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)


def death_rule() -> np.ndarray:
    """The rule's damage-by-toughness grid: 1 where the damage reaches the toughness."""
    return np.array([[1.0 if p >= t else 0.0 for t in range(1, 9)] for p in range(1, 9)])


# ── training ───────────────────────────────────────────────────────────────


def training_losses() -> None:
    ep = csv("epochs.csv")
    widths = [w for w in (64, 128, 256) if LINES[w][0] in set(ep.run)]
    metrics = ("card_disjoint", "game_disjoint", "train")
    fig, axes = plt.subplots(len(metrics), len(widths), figsize=(4.3 * len(widths), 9.4),
                             sharey="row", sharex=True, squeeze=False)
    limits = {"card_disjoint": (3.4, 4.6), "game_disjoint": (2.75, 3.7), "train": (2.6, 4.7)}
    last_epoch = int(ep[ep.run != "gen-1"].epoch.max())
    for col, width in enumerate(widths):
        for row, metric in enumerate(metrics):
            ax = axes[row][col]
            no_copies = NO_COPIES.get(width)
            if no_copies in set(ep.run):
                d = ep[ep.run == no_copies]
                ax.plot(d.epoch, d[metric], **run_line(no_copies), alpha=0.9)
            for run in LINES[width]:
                d = ep[ep.run == run].sort_values("epoch")
                if d.empty:
                    continue
                ax.plot(d.epoch, d[metric], marker="o", ms=2, **run_line(run))
                best = d.loc[d.card_disjoint.idxmin()]
                stage = stage_of(run)
                ax.plot(best.epoch, best[metric], STAGE_MARKER[stage], ms=9,
                        color=WIDTH_COLOUR[width], mec="black", mew=0.8, zorder=5)
            ax.set_ylim(*limits[metric])
            ax.set_xlim(5, last_epoch + 1)
            if row == 0:
                done = LINES[width][0] in set(csv("checkpoints.csv").run)
                ax.set_title(f"{width}-wide e" + ("" if done else " (still training)"),
                             color=WIDTH_COLOUR[width])
            if col == 0:
                ax.set_ylabel({"train": "training loss"}.get(
                    metric, metric.replace("_", "-") + " loss"))
            if row == len(metrics) - 1:
                ax.set_xlabel("epoch")
    handles = [Line2D([], [], color=NO_COPIES_GRAY, lw=1.2, label="arm without copies")]
    handles += [Line2D([], [], color=INK_SOFT, ls=STAGE_LINE[s], lw=1.5, label=STAGE_NAME[s])
                for s in ("arm", "A", "B")]
    handles += [Line2D([], [], marker="o", ls="none", ms=8, mfc=INK_SOFT, mec="black",
                       label="large marker: the run's lowest card-disjoint epoch")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.025),
               fontsize=8.5)
    fig.suptitle("Validation and training losses by epoch; the large marker is each run's lowest "
                 "card-disjoint epoch (▲ arm, ■ A run, ● B run)", x=0.01, ha="left", y=0.95)
    fig.subplots_adjust(bottom=0.08)
    save(fig, "training-losses")


def loss_noise_table() -> pd.DataFrame:
    """Standard deviation of each run's epoch losses around its own five-epoch moving mean.

    Read from epoch 10 on, where every run's trend has flattened; ``run`` × ``metric``.
    """
    ep = csv("epochs.csv")
    rows = []
    for run, d in ep.groupby("run"):
        d = d.sort_values("epoch")
        d = d[d.epoch >= 10]
        if len(d) < 8 or run == "gen-1":
            continue
        for metric in ("card_disjoint", "game_disjoint", "train"):
            trend = d[metric].rolling(5, center=True).mean()
            rows.append(dict(run=run, metric=metric, sd=(d[metric] - trend).dropna().std()))
    return pd.DataFrame(rows).pivot(index="run", columns="metric", values="sd")


def loss_noise() -> None:
    """The swing per stage and width, with run A's swing over run B's written at each B."""
    sd = loss_noise_table()
    widths = [w for w in (64, 128, 256) if all(r in sd.index for r in LINES[w])]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    for ax, (metric, label) in zip(axes, (("card_disjoint", "card-disjoint loss"),
                                          ("game_disjoint", "game-disjoint loss"),
                                          ("train", "training loss"))):
        for k, run in enumerate(r for r in ("e64 (earlier inputs)", "e64", "e128")
                                if r in sd.index):
            ax.plot(-0.45 + 0.08 * k, sd.loc[run, metric], **checkpoint_marker(run, 6))
        order = sorted(widths, key=lambda w: -sd.loc[LINES[w][2], metric])
        for width in widths:
            runs = LINES[width]
            xs = np.arange(3) + (widths.index(width) - 1) * 0.06
            ys = [sd.loc[r, metric] for r in runs]
            ax.plot(xs, ys, color=WIDTH_COLOUR[width], lw=1.4, zorder=2)
            for x, y, run in zip(xs, ys, runs):
                ax.plot(x, y, **checkpoint_marker(run, 7))
            ratio = ys[1] / ys[2]
            dy = 7 * (1 - order.index(width)) if len(order) == 3 else 0
            ax.annotate(f"÷{ratio:.1f}", (xs[2], ys[2]), xytext=(9, dy), textcoords="offset points",
                        fontsize=8.5, color=WIDTH_COLOUR[width], va="center", fontweight="bold")
        ax.set_yscale("log")
        ax.set_xticks([-0.37, 0, 1, 2], ["no\ncopies", "arm", "A run\nfull rate",
                                         "B run\na tenth"])
        ax.set_xlim(-0.6, 2.55)
        ax.set_title(label)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("standard deviation around the\nfive-epoch moving mean (log scale)")
    ticks = [0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]
    axes[0].set_yticks(ticks, [f"{t:g}" for t in ticks])
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    names = [r for r in ("e64 (earlier inputs)", "e64", "e128") if r in sd.index]
    fig.legend(handles=checkpoint_handles(names + [r for w in widths for r in LINES[w]],
                                          fill=False),
               loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    fig.suptitle("Epoch-to-epoch swing of each loss, from epoch 10 on; ÷ is how many times "
                 "smaller the B run's swing is than the A run's.\nRuns B64, B128 and B256 train "
                 "at a tenth of the learning rate", x=0.01, ha="left", y=1.06)
    save(fig, "loss-noise")


def merged_last_epochs(ck: pd.DataFrame) -> pd.DataFrame:
    """``ck`` without each "X last" row whose epoch is its "X best" row's.

    A run whose last epoch is also its lowest-loss one keeps a single checkpoint, so
    its two rows are one point; the kept row is relabelled "X best = last".
    """
    ck = ck.copy()
    for name in list(ck.index):
        if not name.endswith(" last"):
            continue
        best = name.replace(" last", " best")
        if best in ck.index and ck.loc[best, "epoch"] == ck.loc[name, "epoch"]:
            ck = ck.drop(index=name).rename(index={best: f"{best} = last"})
    return ck


def merged_labels(names: list[str]) -> list[tuple[str, str]]:
    """``(checkpoint, label)`` for ``names``, a run's last epoch dropped when it is its best."""
    return [(n, short(n)) for n in names if not (n.endswith(" last") and is_merged(n))]


#: Label offsets in points, per panel and checkpoint or run, around the crowded corners:
#: ``(dx, dy, ha)``. A run's label sits beside its arrow's midpoint.
PROBE_LABEL_OFFSETS = {
    ("card_disjoint", "e64"): (-8, 3, "right"),
    ("card_disjoint", "e128"): (-8, 0, "right"),
    ("card_disjoint", "e64 (earlier inputs)"): (-8, 0, "right"),
    ("card_disjoint", "e64-lethality"): (8, -3, "left"),
    ("card_disjoint", "e128-lethality"): (8, 0, "left"),
    ("card_disjoint", "e256-lethality"): (8, -6, "left"),
    ("card_disjoint", "A256 best"): (0, 10, "center"),
    ("card_disjoint", "A64"): (55, 8, "center"),
    ("card_disjoint", "B64"): (58, -9, "center"),
    ("card_disjoint", "B256"): (-4, 9, "center"),
    ("card_disjoint", "A128"): (12, 0, "left"),
    ("card_disjoint", "B128"): (-4, 9, "center"),
    ("game_disjoint", "e64"): (8, 4, "left"),
    ("game_disjoint", "e128"): (-8, 0, "right"),
    ("game_disjoint", "e64 (earlier inputs)"): (-8, 0, "right"),
    ("game_disjoint", "e64-lethality"): (8, 0, "left"),
    ("game_disjoint", "e128-lethality"): (8, -6, "left"),
    ("game_disjoint", "e256-lethality"): (-8, 0, "right"),
    ("game_disjoint", "A256 best"): (-8, 2, "right"),
    ("game_disjoint", "A64"): (10, 6, "left"),
    ("game_disjoint", "B64"): (0, 9, "center"),
    ("game_disjoint", "B256"): (-8, 6, "right"),
    ("game_disjoint", "A128"): (10, -2, "left"),
    ("game_disjoint", "B128"): (-10, 0, "right"),
}


def probes_vs_loss() -> None:
    ck = csv("checkpoints.csv")
    lad = csv("probe_ladder.csv")
    m = lad[(lad.rung == "3") & (lad.probe == "model") & (lad.target != "damage_taken")]
    ck = ck.set_index("checkpoint")
    ck["probe"] = m[m.stratum == "game-disjoint"].groupby("checkpoint").score.mean()
    # A checkpoint whose probes have not run yet has no point to draw.
    names = [n for n in gen2(ck.dropna(subset=["probe"]).index)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.4), sharey=True)
    for ax, (loss, label) in zip(axes, (("card_disjoint", "card-disjoint loss"),
                                        ("game_disjoint", "game-disjoint loss"))):
        for run in ("A64", "B64", "A128", "B128", "A256", "B256"):
            best, last = f"{run} best", f"{run} last"
            if best in names and last in names:
                b, la = ck.loc[best], ck.loc[last]
                ax.annotate("", (la[loss], la.probe), (b[loss], b.probe),
                            arrowprops=dict(arrowstyle="-|>", color=run_colour(run), lw=1.4,
                                            shrinkA=5, shrinkB=5, mutation_scale=11))
                mid = ((b[loss] + la[loss]) / 2, (b.probe + la.probe) / 2)
                dx, dy, ha = PROBE_LABEL_OFFSETS.get((loss, run), (0, 7, "center"))
                ax.annotate(run, mid, xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                            ha=ha, va="center", fontweight="bold",
                            color=shade(WIDTH_COLOUR[WIDTH_OF[run]], 1.0))
        for name in names:
            r = ck.loc[name]
            ax.plot(r[loss], r.probe, zorder=4, **checkpoint_marker(name, 7.5))
            if stage_of(name) in ("arm", "none"):
                dx, dy, ha = PROBE_LABEL_OFFSETS.get((loss, name), (8, 0, "left"))
                ax.annotate(name, (r[loss], r.probe), xytext=(dx, dy),
                            textcoords="offset points", fontsize=7.5, color=INK_SOFT,
                            ha=ha, va="center")
            elif name == "A256 best":
                dx, dy, ha = PROBE_LABEL_OFFSETS.get((loss, name), (0, -11, "center"))
                ax.annotate("A256 best = last", (r[loss], r.probe), xytext=(dx, dy),
                            textcoords="offset points", fontsize=7.5, ha=ha, va="center",
                            color=shade(WIDTH_COLOUR[256], 1.0), fontweight="bold")
        ax.set_xlabel(f"{label} (lower is better, to the left)")
        ax.set_title(f"against the {label}")
    axes[0].set_ylabel("mean of the model's own probe scores\n(AUC, game-disjoint records; "
                       "higher is better)")
    axes[0].legend(handles=checkpoint_handles(names), fontsize=7.5, loc="lower left", ncol=2)
    fig.suptitle("Probes against each validation loss: each arrow goes from a run's lowest "
                 "card-disjoint epoch to its last", x=0.01, ha="left", y=0.96)
    save(fig, "probes-vs-loss")


def field_explained() -> None:
    fe = csv("fields_by_epoch.csv")
    fields = [("gate", "affected gate"), ("zone_outcome", "zone outcome"),
              ("keywords_gained", "keywords gained"), ("blocker_legal", "blocking legality"),
              ("types_gained", "types gained"), ("poison_delta", "poison")]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharex=True)
    widths = set()
    for ax, (field, label) in zip(axes.flat, fields):
        low = 0
        for width in (64, 128, 256):
            for run in LINES[width]:
                d = fe[(fe.run == run) & (fe.field == field)].sort_values("epoch")
                if not d.empty:
                    ax.plot(d.epoch, d.explained_pct, **run_line(run))
                    low = min(low, d.explained_pct.min())
                    widths.add(width)
        ax.set_title(label)
        ax.axhline(0, color=INK_SOFT, lw=0.8)
        ax.set_ylim(min(-40, 10 * np.floor(low / 10) - 5), 100)
    axes[1][2].text(0.98, 0.04, "below 0: worse than\na constant prediction",
                    transform=axes[1][2].transAxes, ha="right", fontsize=7.5, color=INK_SOFT)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of the constant\nprediction's loss removed")
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    fig.legend(handles=line_handles(sorted(widths), no_copies=False), loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.05), fontsize=8.5)
    fig.suptitle("Card-disjoint explained deviance by field, along the three lethality-copy lines",
                 x=0.01, ha="left", y=0.96)
    save(fig, "fields-explained")


# ── the death rule ─────────────────────────────────────────────────────────

#: The toughness-sweep grid's layout: one row per width, one column per stage.
STAGE_GRID = [["e64", "e64-lethality", "A64 best", "A64 last", "B64 best", "B64 last"],
              ["e128", "e128-lethality", "A128 best", "A128 last", "B128 best", "B128 last"],
              [None, "e256-lethality", "A256 best", "A256 last", "B256 best", "B256 last"]]
STAGE_COLUMNS = ["no copies", "arm with copies", "A run, kept epoch", "A run, last epoch",
                 "B run, kept epoch", "B run, last epoch"]


def damage_grids() -> None:
    """The damage grid of every checkpoint, laid out as the toughness sweep's grid."""
    g = csv("damage_grids.csv")
    g = g[g.record_set == "probe set"]
    rule = death_rule()
    grids, errors = {}, {}
    for name in set(g.checkpoint):
        grids[name] = g[g.checkpoint == name].pivot(index="damage", columns="toughness",
                                                    values="predicted_death").values
        errors[name] = float(np.abs(grids[name] - rule).mean())
    shown = [n for row in STAGE_GRID for n in row if n in grids and not
             (n.endswith(" last") and is_merged(n))]
    lowest = min(shown, key=errors.get)
    fig, axes = plt.subplots(3, 6, figsize=(15.2, 8.6), squeeze=False)
    for r, row in enumerate(STAGE_GRID):
        for c, name in enumerate(row):
            ax = axes[r][c]
            if name is None or name not in grids:
                ax.axis("off")
                continue
            if name.endswith(" last") and is_merged(name):
                ax.axis("off")
                ax.text(0.5, 0.5, "last epoch is\nthe kept epoch", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8.5, color=INK_SOFT, style="italic")
                continue
            heatmap(ax, grids[name], "", "", "")
            rule_staircase(ax)
            ax.set_title(f"{short(name)}\nerror {errors[name]:.3f}", fontsize=9,
                         fontweight="bold" if name == lowest else "normal",
                         loc="center")
            ax.tick_params(labelsize=7, length=0)
            if c == 0 or (r == 2 and c == 1):
                ax.set_ylabel(f"{64 * 2 ** r}-wide e\ndamage", color=WIDTH_COLOUR[64 * 2 ** r],
                              fontweight="bold")
            if r == 2 or name in ("e128",):
                ax.set_xlabel("toughness", fontsize=8.5)
    for c, label in enumerate(STAGE_COLUMNS):
        axes[0][c].annotate(label, (0.5, 1.3), xycoords="axes fraction", ha="center",
                            fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(hspace=0.5, wspace=0.12)
    colourbar(fig, axes, shrink=0.5, pad=0.015)
    fig.suptitle("Predicted death of a damage spell's target, 20 probe records per cell. The rule "
                 "kills in every cell below the black staircase, the diagonal included "
                 "(damage ≥ toughness).\nError: mean absolute difference from the rule's 0 or 1 "
                 f"over the 64 cells; the lowest, {short(lowest)}'s, in bold",
                 x=0.01, ha="left", y=1.02, fontsize=10.5)
    save(fig, "damage-grids")


def toughness_sweep_records() -> None:
    """One band per damage group: lowest to highest record at each toughness, median line.

    Activated pings, a permanent's paid ability dealing 1 damage (Prodigal Sorcerer's
    {T}), form their own group, apart from 1-damage spells and triggers. Both 1-damage
    groups stop dying at the same toughness, so their dotted rule lines share one place
    and are drawn as one line of alternating dashes.
    """
    sr = csv("sweep_records.csv")
    meta = csv("toughness_sweep_records.csv").set_index("record")
    meta["group"] = [("1 activated" if k == "activated" and d == 1 else str(d))
                     for d, k in zip(meta.damage, meta.kind)]
    groups = [("1", 1, "#3b6fb6", "1, spell or trigger", (0, (4, 4))),
              ("1 activated", 1, "#8e5bb5", "1, activated ping", (4, (4, 4))),
              ("2", 2, "#4a9a62", "2", ":"), ("3", 3, "#e08f3c", "3", ":"),
              ("5", 5, "#c05555", "5", ":")]
    available = set(sr.checkpoint)
    fig, axes = plt.subplots(len(STAGE_GRID), len(STAGE_COLUMNS),
                             figsize=(2.9 * len(STAGE_COLUMNS), 2.8 * len(STAGE_GRID)),
                             sharey=True, squeeze=False)
    empty = []
    for r, row in enumerate(STAGE_GRID):
        for c, name in enumerate(row):
            ax = axes[r][c]
            if name not in available:
                ax.axis("off")
                empty.append(ax)
                continue
            d = sr[(sr.checkpoint == name) & (sr.sweep == "toughness")]
            d = d.join(meta[["group"]], on="record")
            ax.axhline(0.5, color="black", lw=0.8)
            for group, damage, colour, label, dashes in groups:
                g = d[d.group == group].groupby("value").prediction
                n = d[d.group == group].record.nunique()
                ax.fill_between(g.min().index, g.min(), g.max(), color=colour, alpha=0.16, lw=0)
                ax.plot(g.median().index, g.median(), color=colour, lw=1.6,
                        label=f"{label} ({n})")
                ax.axvline(damage + 0.5, color=colour, lw=1.0, ls=dashes)
            ax.set_title(name, fontsize=9.5, loc="center", fontweight="normal")
            ax.set_xticks(range(1, 9))
            ax.tick_params(labelsize=8)
            ax.set_ylim(0, 1)
            if r == len(STAGE_GRID) - 1 or STAGE_GRID[r + 1][c] not in available:
                ax.set_xlabel("target's toughness", fontsize=8.5)
        first = next((c for c, name in enumerate(row) if name in available), None)
        if first is not None:
            axes[r][first].set_ylabel(f"e width {64 * 2 ** r}\npredicted death", fontsize=9)
            axes[r][first].yaxis.set_tick_params(labelleft=True)
    for c, label in enumerate(STAGE_COLUMNS):
        axes[0][c].annotate(label, (0.5, 1.22), xycoords="axes fraction", ha="center",
                            fontsize=9.5, fontweight="bold")
    handles, labels = axes[0][0].get_legend_handles_labels()
    # The empty bottom-left slot holds the legend, kept clear of the next panel's axis label.
    (empty[-1] if empty else axes[0][-1]).legend(
        handles, labels, fontsize=8, loc="center left", bbox_to_anchor=(-0.12, 0.5),
        title="damage, ability (records)", title_fontsize=8, alignment="left")
    fig.subplots_adjust(hspace=0.45, wspace=0.12)
    fig.suptitle("Toughness sweep: band from lowest to highest record at each toughness, line at"
                 " the median; the rule says each group stops dying at its dotted line", y=0.99,
                 fontweight="normal")
    save(fig, "toughness-sweep-records")


#: The three distances from the lethal boundary, dark for the hit that should kill.
BOUNDARY_SERIES = (("lethal_by_1", "#2e1a5e", "lethal by 1 point", "v"),
                   ("exactly_lethal", "#7d5bb0", "exactly lethal", "o"),
                   ("short_by_1", "#b9a2d8", "short by 1 point", "^"))


def boundary_by_epoch() -> None:
    """Rows per width; the arm and A run on the left, the B run on the right.

    The metric is the predicted death given that the hit creature is affected, the
    death head alone, on 40 qualifying training records read at every epoch.
    """
    b = csv("boundary_by_epoch.csv")
    ck = csv("checkpoints.csv")
    widths = [w for w in (64, 128, 256) if LINES[w][1] in set(b.run)]
    fig, axes = plt.subplots(len(widths), 2, figsize=(12, 2.75 * len(widths)), sharey=True,
                             squeeze=False)
    a_range = b[b.run.str.startswith(("A", "e"))].epoch
    b_range = b[b.run.str.startswith("B")].epoch
    for r, width in enumerate(widths):
        arm, a, bb = LINES[width]
        for ax, runs in ((axes[r][0], (arm, a)), (axes[r][1], (bb,))):
            ax.axhspan(0.5, 1, color="#f2f2f2", lw=0, zorder=0)
            ax.axhline(0.5, color="black", lw=0.9, zorder=1)
            for metric, colour, label, marker in BOUNDARY_SERIES:
                for run in runs:
                    d = b[b.run == run].sort_values("epoch")
                    if d.empty:
                        continue
                    ax.plot(d.epoch, d[metric], marker=marker, ms=3.5, color=colour,
                            ls=STAGE_LINE[stage_of(run)], lw=1.6 if run != arm else 1.2,
                            label=label if run == runs[-1] else None)
            for _, k in ck[ck.run.isin(runs) & (ck.role == "best")].iterrows():
                ax.axvline(k.epoch, color=WIDTH_COLOUR[width], lw=1.0, ls="--", zorder=1)
            ax.set_ylim(0, 1)
        kept_a = ck[(ck.run == a) & (ck.role == "best")].epoch
        left, right = axes[r][0], axes[r][1]
        left.set_xlim(a_range.min() - 1, a_range.max() + 1)
        right.set_xlim(b_range.min() - 1, b_range.max() + 1)
        arm_epochs = b[b.run == arm].epoch
        if not arm_epochs.empty:
            left.axvspan(left.get_xlim()[0], arm_epochs.max() + 0.5, color=shade(
                WIDTH_COLOUR[width], 0.12), lw=0, zorder=0)
            left.text(arm_epochs.min(), 0.04, f"arm {arm}", fontsize=7.5, color=INK_SOFT)
        else:
            left.text(left.get_xlim()[0] + 1, 0.04, f"arm {arm}: epochs not read",
                      fontsize=7.5, color=INK_SOFT)
        left.set_title(f"{width}-wide e: arm, then run {a} at the full rate",
                       color=WIDTH_COLOUR[width], fontsize=10)
        origin = f", from {a}'s epoch {int(kept_a.iloc[0])}" if not kept_a.empty else ""
        right.set_title(f"run {bb} at a tenth of the rate{origin}", color=WIDTH_COLOUR[width],
                        fontsize=10)
        left.set_ylabel("predicted death\ngiven affected")
        if r == len(widths) - 1:
            left.set_xlabel("epoch")
            right.set_xlabel("epoch")
    handles = [Line2D([], [], color=c, marker=m, ms=4, lw=1.6, label=lab)
               for _, c, lab, m in BOUNDARY_SERIES]
    handles += [Line2D([], [], color=INK_SOFT, ls=STAGE_LINE[s], lw=1.4, label=STAGE_NAME[s])
                for s in ("arm", "A", "B")]
    handles.append(Line2D([], [], color=INK_SOFT, ls="--", lw=1.0,
                          label="vertical: the run's lowest card-disjoint epoch"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06),
               fontsize=8.5)
    fig.suptitle("Where the lethal boundary sits, epoch by epoch, on 40 training records: a sharp "
                 "boundary at the rule puts\n“lethal by 1” near 1 and “short by 1” near 0, on "
                 "either side of the even-chance line (shaded above it)", x=0.01, ha="left",
                 y=0.99)
    fig.subplots_adjust(hspace=0.42, wspace=0.06)
    save(fig, "boundary-by-epoch")


def _sweep_panel(ax, s: pd.DataFrame, names: list[str]) -> None:
    """One damage-sweep panel over grey wrong-side quadrants; ✓ marks a curve on the rule."""
    for (x0, x1), (y0, y1) in (((0.5, 3.5), (0.5, 1)), ((3.5, 8.5), (0, 0.5))):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#e4e4e4", lw=0, zorder=0))
    ax.text(0.7, 0.96, "wrong side:\npredicted to die", fontsize=7.5, color=INK_SOFT, va="top")
    ax.text(8.3, 0.04, "wrong side:\npredicted to survive", fontsize=7.5, color=INK_SOFT,
            ha="right")
    for name in names:
        d = s[s.checkpoint == name].sort_values("value")
        on_rule = bool(((d.value < 4) == (d.prediction < 0.5)).all())
        line = run_line(name)
        style = checkpoint_marker(name, 5.5 if on_rule else 4.5)
        style.update(line, lw=2.2 if on_rule else 1.0)
        if stage_of(name) in ("none", "gen-1"):
            # The grey runs without copies tell apart by their dashes instead.
            style["ls"] = {"e64": "-", "e128": (0, (5, 2)), "gen-1": "-"}.get(name, (0, (1, 1.5)))
            style["lw"] = 1.4
        ax.plot(d.value, d.prediction, **style,
                alpha=1.0 if on_rule else 0.55, zorder=3 if on_rule else 2,
                label=("✓ " if on_rule else "    ") + short(name))
    ax.axvline(3.5, color="black", lw=0.8)
    ax.axhline(0.5, color="black", lw=0.8)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, 9))
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=2.6)


def damage_sweep() -> None:
    """The damage sweep of every checkpoint, one panel per width and one for no copies.

    The rule kills a toughness-4 target from 4 damage up, so a curve is on the right
    side below one half left of the step and above one half right of it, however soft.
    """
    s = csv("sweeps.csv")
    s = s[s.sweep == "damage_spell"]
    available = set(s.checkpoint)
    panels = [("without lethality copies", None, present(
        ["gen-1", "e64 (earlier inputs)", "e64", "e128"], available))]
    for width in (64, 128, 256):
        names = [n for n in gen2(available, copies_only=True) if WIDTH_OF[run_of(n)] == width]
        if names:
            panels.append((f"{width}-wide e, with copies", width, names))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True, sharey=True)
    for ax, (title, width, names) in zip(axes.flat, panels):
        _sweep_panel(ax, s, names)
        ax.set_title(title, color=WIDTH_COLOUR.get(width, INK))
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    for ax in axes[1]:
        ax.set_xlabel("NumDmg$ of the spell")
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted death,\ntoughness-4 target")
    fig.subplots_adjust(wspace=0.55, hspace=0.25)
    fig.suptitle("Damage sweep: the rule kills from 4 damage up. ✓ and bold: every point on the "
                 "rule's side of an even chance\n(dotted ▲ arm, dashed ■ A run, solid ● B run; "
                 "filled the kept epoch, hollow the last)", x=0.01, ha="left", y=0.98)
    save(fig, "damage-sweep")


# ── combat ─────────────────────────────────────────────────────────────────


def combat_grids() -> None:
    """Plain grids on the death map; keyword grids as their change from the plain ones."""
    c = csv("combat_grids.csv")
    name = "B64 last"  # the checkpoint the gen-2 record's combat section describes

    def grid(variant: str, reads: str) -> np.ndarray:
        d = c[(c.checkpoint == name) & (c.variant == variant) & (c.reads == reads)]
        return d.pivot(index="striker_power", columns="struck_toughness",
                       values="predicted_death").values

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.2))
    for ax, reads, title, ylabel, xlabel in (
            (axes[0][0], "blocker", "Blocker's death, no keyword", "attacker's power",
             "blocker's toughness"),
            (axes[0][1], "attacker", "Attacker's death, no keyword", "blocker's power",
             "attacker's toughness")):
        heatmap(ax, grid("none", reads), title + "\n(rule: dies below the staircase)",
                xlabel, ylabel)
        rule_staircase(ax)
    norm = matplotlib.colors.TwoSlopeNorm(0, -1, 1)
    for ax, variant, reads, title, ylabel, xlabel in (
            (axes[1][0], "attacker-deathtouch", "blocker",
             "Attacker has deathtouch: change in\nthe blocker's death (rule: +1 above\n"
             "the staircase, 0 below)", "attacker's power", "blocker's toughness"),
            (axes[1][1], "attacker-first-strike", "attacker",
             "Attacker has first strike, first-strike\nstep: change in its death (rule: −1\n"
             "below the staircase, 0 above)", "blocker's power", "attacker's toughness")):
        diff = grid(variant, reads) - grid("none", reads)
        im = ax.imshow(diff, cmap=DIVERGING_CMAP, norm=norm, origin="upper")
        for i in range(8):
            for j in range(8):
                text = f"{diff[i, j]:+.1f}".replace("-", "−")
                ax.text(j, i, "0" if text[1:] == "0.0" else text, ha="center", va="center",
                        fontsize=6.5, color="white" if abs(diff[i, j]) > 0.6 else INK)
        rule_staircase(ax)
        ax.set_xticks(range(8), range(1, 9))
        ax.set_yticks(range(8), range(1, 9))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9.5)
        ax.grid(False)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    colourbar(fig, axes[0], shrink=0.85, pad=0.03)
    fig.colorbar(im, ax=axes[1], shrink=0.85, pad=0.03,
                 label="change in predicted death\nwith the keyword")
    fig.suptitle(f"Combat grids of {name}: predicted death over 40 real one-on-one blocks per "
                 "cell.\nRead with the base record's damage assignment, stale after the edit: the "
                 "first-strike and power-axis\nreadings may understate what the model knows",
                 x=0.01, ha="left", y=0.99)
    save(fig, "combat-grids")


#: Each combat keyword check: label, variant, creature read, cells, and the rule's answer.
#: The cells are those where the keyword changes the rule's answer.
def _combat_checks(c: pd.DataFrame) -> list[tuple]:
    pw, tg = c.striker_power, c.struck_toughness
    lethal, short_ = pw >= tg, pw < tg
    second = (pw < tg) & (tg <= 2 * pw)
    return [
        ("indestructible blocker\nblocker's death on lethal hits",
         "blocker-indestructible", "blocker", lethal, 0),
        ("deathtouch attacker\nblocker's death on hits short of lethal",
         "attacker-deathtouch", "blocker", short_, 1),
        ("double-strike attacker, second hit\nblocker's death, lethal only as two",
         "attacker-double-strike-second-hit", "blocker", second, 1),
        ("first-strike attacker, first-strike step\nattacker's death on the blocker's lethal "
         "hits", "attacker-first-strike", "attacker", lethal, 0),
        ("first-strike blocker, first-strike step\nblocker's death on the attacker's lethal hits",
         "blocker-first-strike", "blocker", lethal, 0),
    ]


def combat_keywords() -> None:
    """How far each keyword moves the predicted death toward the rule's answer."""
    c = csv("combat_grids.csv")
    checks = _combat_checks(c)
    names = gen2(c.checkpoint)
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    for i, (label, variant, reads, cells, rule) in enumerate(checks):
        y = len(checks) - 1 - i
        plain = c[(c.variant == "none") & (c.reads == reads) & cells]
        kw = c[(c.variant == variant) & (c.reads == reads) & cells]
        p = plain.groupby("checkpoint").predicted_death.mean()
        k = kw.groupby("checkpoint").predicted_death.mean()
        frac = ((k - p) / (rule - p)).dropna()
        ax.axhspan(y - 0.42, y + 0.42, color="#f6f6f6", lw=0, zorder=0)
        points = strip(ax, y, frac, names, spread=0.3)
        if variant == "attacker-double-strike-second-hit":
            for name, dy in (("A64 last", 10), ("B64 last", -12)):
                if name in points:
                    tag(ax, *points[name], name, dx=10, dy=dy)
        if variant.endswith("first-strike") and "e64 (earlier inputs)" in points:
            tag(ax, *points["e64 (earlier inputs)"], "e64 (earlier inputs)", dx=10, dy=6)
    for x, label in ((0, "keyword ignored"), (1, "the rule's full change")):
        ax.axvline(x, color="black", lw=1.0)
        ax.text(x, len(checks) - 0.45, label, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_yticks(range(len(checks)), [x[0] for x in checks][::-1], fontsize=8.5)
    ax.set_ylim(-0.5, len(checks) - 0.1)
    ax.set_xlim(-0.15, 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of the rule's change the keyword produces: (with − without) / "
                  "(rule − without),\nmean predicted death over the cells where the keyword "
                  "changes the rule's answer")
    ax.legend(handles=checkpoint_handles(names), fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Does a keyword move the predicted death as far as the rule says? One marker "
                 "per checkpoint", pad=16)
    save(fig, "combat-keywords")


def gate2_heatmap() -> None:
    """Gate 2 as a strip per keyword, the band showing what sampling alone would spread."""
    g = csv("gate2.csv")
    names = gen2(g.checkpoint)
    kws = ["lifelink", "indestructible", "trample", "first_strike", "deathtouch", "infect",
           "double_strike"]
    g = g[~((g.checkpoint == "e64 (earlier inputs)")
            & g.keyword.isin(["first_strike", "double_strike"]))]  # scored on the wrong step
    n = g.groupby("keyword").qualifying_records.median().astype(int)
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    for i, kw in enumerate(kws):
        y = len(kws) - 1 - i
        d = g[g.keyword == kw].set_index("checkpoint").agreement_pct
        d = d[~d.index.duplicated()]
        med = d[[x for x in names if x in d.index]].median()
        half = 196 * np.sqrt(med / 100 * (1 - med / 100) / n[kw])
        ax.add_patch(Rectangle((med - half, y - 0.4), 2 * half, 0.8, color="#dedede", lw=0,
                               zorder=1))
        ax.plot([med, med], [y - 0.4, y + 0.4], color=GRAY, lw=1.0, zorder=2)
        points = strip(ax, y, d, names, spread=0.32)
        if kw == "double_strike":
            for name, dy in (("B64 best", 12), ("B64 last", 10)):
                if name in points:
                    tag(ax, *points[name], name, dx=12, dy=dy,
                        ha="left")
        if kw == "first_strike" and "A256 best" in points:
            tag(ax, *points["A256 best"], "A256 best = last", dx=-12, dy=-8, ha="right")
    ax.axvline(70, color="black", lw=1.2)
    ax.text(70, len(kws) - 0.4, " gate passes from 70%", fontsize=8, va="bottom")
    ax.set_yticks(range(len(kws)),
                  [f"{kw.replace('_', ' ')}\n{n[kw]:,} records" for kw in kws][::-1])
    ax.set_ylim(-0.6, len(kws) - 0.1)
    ax.set_xlim(40, 101)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("direction agreement, %: how often a keyword moves the prediction the way it "
                  "moves the real outcome")
    handles = checkpoint_handles(names) + [Patch(color="#dedede", label="median ± 95% sampling "
                                                 "half-width")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Gate 2 by keyword, one marker per checkpoint: a spread wider than the grey band "
                 "is the model's, not sampling's", pad=16)
    save(fig, "gate2-keywords")


# ── where knowledge lives ──────────────────────────────────────────────────

LADDER_TARGETS = {"affordable": "affordable", "pt_until_end_of_turn": "pump lasts until EOT",
                  "legal_target": "legal target",
                  "legal_target_protected": "legal target, protected",
                  "fires": "trigger fires", "affected": "card affected", "dies": "dies",
                  "may_block": "may block", "damage_taken": "damage taken"}
#: The rungs, bottom to top of the model, with each one's mark.
RUNGS = [("0", "rung 0: board alone", dict(marker="s", color=GRAY, mec=GRAY)),
         ("1w", "rung 1w: board + random vector of e's width",
          dict(marker="o", color="white", mec=GRAY, mew=1.3)),
         ("1", "rung 1: board + e", dict(marker="o", color=PURPLE, mec=PURPLE)),
         ("1o", "rung 1o: board + the true values (oracle)",
          dict(marker="D", color="white", mec=PURPLE, mew=1.3)),
         ("2", "rung 2: probe on the trunk's vector", dict(marker="D", color=SLATE, mec=SLATE)),
         ("3", "rung 3: the model's own answer", dict(marker="*", color="black", mec="black"))]
#: A share is read only where the oracle rung closes at least this much of the gap.
ORACLE_FLOOR = 0.5


def ladder_table() -> pd.DataFrame:
    """Card-disjoint rungs per target and checkpoint, with the shares the record defines.

    MLP probes for rungs 0 to 2, the model's own head for rung 3; the gen-2 checkpoints
    trained with lethality copies, a run whose last epoch is its kept one counted once.
    ``share`` is (1 − 0)/(3 − 0); ``share_w`` and ``share_o`` put rungs 1w and 1o in the
    numerator.
    """
    lad = csv("probe_ladder.csv")
    lad = lad[(lad.stratum == "card-disjoint")
              & (((lad.probe == "mlp") & lad.rung.isin(["0", "1", "1w", "1o", "2"]))
                 | ((lad.probe == "model") & (lad.rung == "3")))]
    lad = lad[lad.checkpoint.isin(gen2(lad.checkpoint, copies_only=True))]
    t = lad.pivot_table(index=["target", "checkpoint"], columns="rung", values="score")
    gap = t["3"] - t["0"]
    for col, rung in (("share", "1"), ("share_w", "1w"), ("share_o", "1o")):
        t[col] = (t[rung] - t["0"]) / gap
    t.loc[gap <= 0, ["share", "share_w", "share_o"]] = np.nan
    return t


def ladder() -> None:
    """(a) every rung per target; (b) the share e carries, rows sorted by its median."""
    t = ladder_table()
    names = gen2(t.index.get_level_values("checkpoint"))
    med = t.groupby("target").median()
    # The share is defined only where the model's answer beats the board alone.
    defined = med["3"] > med["0"]
    med.loc[~defined, ["share", "share_w", "share_o"]] = np.nan
    auc = [x for x in LADDER_TARGETS if x in med.index and x != "damage_taken"]
    auc = sorted(auc, key=lambda x: -np.nan_to_num(med.loc[x, "share"], nan=-9))
    flagged = {x for x in med.index if defined[x] and med.loc[x, "share_o"] < ORACLE_FLOOR
               and med.loc[x, "share"] < 0.9}
    fig = plt.figure(figsize=(13.2, 9.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[len(auc), 1.25], width_ratios=[1, 1.05],
                            hspace=0.2, wspace=0.09)
    rung_ax = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]
    share_ax = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])]
    groups = [(auc, rung_ax[0], share_ax[0]), (["damage_taken"], rung_ax[1], share_ax[1])]
    step = 0.13
    for targets, ax, sax in groups:
        for i, target in enumerate(targets):
            y = len(targets) - 1 - i
            if target in flagged:
                for a in (ax, sax):
                    a.axhspan(y - 0.47, y + 0.47, color="#fbf3dc", lw=0, zorder=0)
            d = t.loc[target]
            for k, (rung, _, style) in enumerate(RUNGS):
                yy = y + (2.5 - k) * step
                ax.plot([d[rung].min(), d[rung].max()], [yy, yy], color=style["mec"], lw=1.4,
                        alpha=0.55, zorder=2, solid_capstyle="butt")
                ax.plot(d[rung].median(), yy, ls="none", ms=11 if rung == "3" else 6.5,
                        zorder=3, **style)
            share = d["share"]
            if defined[target]:
                strip(sax, y, share.clip(-0.4, 1.25), names, spread=0.3, ms=6)
                for col, mark in (("share_w", dict(marker="|", ms=16, mew=2.2, color=GRAY)),
                                  ("share_o", dict(marker="D", ms=7, mfc="white", mec=PURPLE,
                                                   mew=1.4, color=PURPLE))):
                    v = med.loc[target, col]
                    sax.plot(np.clip(v, -0.37, 1.22), y, ls="none", zorder=5, **mark)
                    if v < -0.37:
                        sax.annotate(f"{v:.2f}", (-0.37, y), xytext=(4, -9),
                                     textcoords="offset points", fontsize=7, color=INK_SOFT)
                closes = med.loc[target, "share_o"]
                closes = f"{closes:.0%}" if closes > 0 else "none"
                note = (f"oracle rung closes {closes} of the gap:\n"
                        "share not interpreted" if target in flagged else "")
                sax.text(1.24, y - 0.44, note, ha="right", va="bottom", fontsize=7.2,
                         color="#8a6a00", style="italic")
            else:
                sax.text(0.42, y, "no share: the model scores below the board alone",
                         ha="center", va="center", fontsize=8, color=INK_SOFT, style="italic")
        ax.set_yticks(range(len(targets)), [LADDER_TARGETS[x] for x in targets][::-1])
        ax.set_ylim(-0.55, len(targets) - 0.45)
        sax.set_ylim(*ax.get_ylim())
        sax.set_yticks(range(len(targets)), [""] * len(targets))
        for a in (ax, sax):
            a.grid(axis="y", visible=False)
        sax.set_xlim(-0.4, 1.26)
        for x in (0, 1):
            sax.axvline(x, color="black", lw=0.9, zorder=1)
    rung_ax[0].set_xlim(0.5, 1.005)
    rung_ax[0].axvline(0.5, color="black", lw=1.5)
    rung_ax[0].text(0.503, -0.5, "chance", fontsize=8, color=INK_SOFT, va="bottom")
    rung_ax[0].set_xlabel("AUC")
    rung_ax[1].set_xlim(-0.3, 0.8)
    rung_ax[1].axvline(0, color="black", lw=0.9)
    rung_ax[1].set_xlabel("R² (0 = the target's mean)")
    share_ax[0].set_xticklabels([])
    share_ax[0].text(0.0, len(auc) - 0.5, "board alone", ha="center", va="bottom", fontsize=8)
    share_ax[0].text(1.0, len(auc) - 0.5, "the model's answer", ha="center", va="bottom",
                     fontsize=8)
    share_ax[1].set_xlabel("share carried by e: (rung 1 − rung 0) / (rung 3 − rung 0)")
    rung_ax[0].set_title("(a) Each rung: median over the checkpoints, line = range", pad=18)
    share_ax[0].set_title("(b) The share e carries, one marker per checkpoint", pad=18)
    handles = [Line2D([], [], ls="none", ms=11 if r == "3" else 7, label=lab, **style)
               for r, lab, style in RUNGS]
    handles += [Line2D([], [], marker="|", ls="none", ms=12, mew=2.2, color=GRAY,
                       label="(b) the share a random vector of e's width closes, median"),
                Line2D([], [], marker="D", ls="none", ms=7, mfc="white", mec=PURPLE, mew=1.4,
                       label="(b) the share the oracle rung closes, median"),
                Patch(color="#fbf3dc", label=f"oracle closes under {ORACLE_FLOOR:.0%} of the "
                                             "gap: by the record's rule, share not read")]
    fig.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0.03, -0.045),
               fontsize=8)
    fig.legend(handles=checkpoint_handles(names), loc="lower left", ncol=6,
               bbox_to_anchor=(0.03, -0.085), fontsize=8)
    fig.suptitle("Where the knowledge lives, card-disjoint records, the 14 gen-2 checkpoints "
                 "trained with lethality copies", x=0.01, ha="left", y=0.95)
    save(fig, "probe-ladder")


LINE_TARGETS = [("tap_cost", "tap cost"), ("sacrifice_cost", "sacrifice cost"),
                ("instant_speed", "instant speed"), ("repeatable", "repeatable")]
LINE_AMOUNTS = [("cost_total", "total mana cost"), ("damage", "damage amount"),
                ("counters_placed", "counters placed"), ("power_change", "power change"),
                ("toughness_change", "toughness change")]


def line_properties() -> None:
    """Trained-line score to held-out-line score per checkpoint, the random vector beside."""
    lad = csv("probe_ladder.csv")
    d = lad[(lad.probe == "mlp") & lad.stratum.isin(["trained", "held-out"])
            & lad.rung.isin(["1", "1w"])]
    names = present(["gen-1", "e64-lethality", "e128-lethality", "e256-lethality", "B64 last",
                     "B128 last", "B256 last"], d.checkpoint)
    score = d.groupby(["target", "checkpoint", "stratum", "rung"]).score.mean()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.2),
                             gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.38})
    for ax, targets, lo, chance in ((axes[0], LINE_TARGETS, 0.42, 0.5),
                                    (axes[1], LINE_AMOUNTS, -0.55, 0.0)):
        targets = [t for t in targets if t[0] in set(d.target)]
        for i, (target, label) in enumerate(targets):
            y = len(targets) - 1 - i
            ax.axhspan(y - 0.45, y + 0.45, color="#f6f6f6" if i % 2 else "white", lw=0,
                       zorder=0)
            where = offsets(names, 0.36)
            for name in names:
                yy = y + where[name]
                trained = score.get((target, name, "trained", "1"))
                held = score.get((target, name, "held-out", "1"))
                width = score.get((target, name, "held-out", "1w"))
                colour = run_colour(name)
                ax.plot([trained, held], [yy, yy], color=colour, lw=1.6, zorder=2)
                ax.plot(trained, yy, "o", ms=3.5, color=colour, zorder=3)
                ax.plot(max(held, lo), yy, zorder=4, **checkpoint_marker(name, 6.5))
                ax.plot(max(width, lo), yy, "|", ms=7, mew=1.8, color=GRAY, zorder=3)
        ax.set_yticks(range(len(targets)), [t[1] for t in targets][::-1], fontsize=9.5,
                      fontweight="bold")
        ax.set_ylim(-0.5, len(targets) - 0.5)
        ax.set_xlim(lo, 1.01)
        ax.axvline(chance, color="black", lw=1.2)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("AUC (0.5 = chance)")
    axes[1].set_xlabel("R² (0 = the amount's mean)")
    axes[0].set_title("Yes/no properties")
    axes[1].set_title("Amounts")
    handles = checkpoint_handles(names)
    handles += [Line2D([], [], marker="o", ms=3.5, color=INK_SOFT, ls="-", lw=1.6,
                       label="small dot: trained lines; large marker: held-out lines"),
                Line2D([], [], marker="|", ms=8, mew=1.8, color=GRAY, ls="none",
                       label="random vector of the same width, held-out lines")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06),
               fontsize=8.5)
    fig.suptitle("Line properties read from e alone by an MLP probe: from trained lines (dot) to "
                 "lines the model never saw (marker)", x=0.01, ha="left", y=0.99)
    save(fig, "line-properties")


#: Each restriction of the board-abilities ablation and the control it is divided by.
BOARD_RATIOS = [("attacker_legal", "legality-static", "attack legality: a static restricts"),
                ("attacker_legal", "legality-keyword", "attack legality: a keyword restricts"),
                ("blocker_legal", "legality-static", "block legality: a static restricts"),
                ("blocker_legal", "legality-keyword", "block legality: a keyword restricts"),
                ("zone_outcome", "resolution-replacement",
                 "zone outcome: a replacement elsewhere"),
                ("zone_outcome", "resolution-static", "zone outcome: a static elsewhere")]


def board_abilities() -> None:
    """Each checkpoint's loss where a line restricts, over its loss where nothing does."""
    b = csv("board_abilities.csv")
    names = gen2(b.checkpoint)
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    for i, (field, subset, label) in enumerate(BOARD_RATIOS):
        y = len(BOARD_RATIOS) - 1 - i
        control = subset.split("-")[0] + "-none"
        d = b[b.field == field].drop_duplicates(["checkpoint", "subset"]).pivot(
            index="checkpoint", columns="subset", values="loss")
        ratio = d[subset] / d[control]
        ax.axhspan(y - 0.42, y + 0.42, color="#f6f6f6", lw=0, zorder=0)
        strip(ax, y, ratio, names, spread=0.28)
        r = ratio[[n for n in names if n in ratio.index]]
        ax.text(1.0, y + 0.03, f" ×{r.min():.2g} to ×{r.max():.2g}", fontsize=8,
                color=INK_SOFT, va="center", transform=ax.get_yaxis_transform(), ha="left")
    ax.set_xscale("log")
    ax.axvline(1, color="black", lw=1.2)
    ax.text(1, len(BOARD_RATIOS) - 0.45, "as costly as no restriction", fontsize=8,
            ha="center", va="bottom")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4], ["×0.1", "×0.25", "×0.5", "×1", "×2", "×4"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(len(BOARD_RATIOS)), [x[2] for x in BOARD_RATIOS][::-1])
    ax.set_ylim(-0.5, len(BOARD_RATIOS) - 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("field loss on the restricted records ÷ field loss on the unrestricted control, "
                  "same checkpoint (log scale)")
    ax.legend(handles=checkpoint_handles(names), fontsize=8, loc="upper left",
              bbox_to_anchor=(1.12, 1.0))
    ax.set_title("A restriction the card slots cannot show: how many times its loss is the "
                 "unrestricted loss, per checkpoint", pad=16)
    save(fig, "board-abilities")


ABLATION_FIELDS = [("gate", "affected gate"), ("keywords_gained", "keywords gained"),
                   ("power_delta", "power change"), ("damage_taken", "damage taken"),
                   ("life_delta", "life change"), ("counters_delta_p1p1", "+1/+1 counters")]
ABLATION_SCOPES = [("all", "every slot", dict(marker="o", color=GRAY, mec=GRAY)),
                   ("act", "[ACT] slot only", dict(marker="D", color=PURPLE, mec=PURPLE)),
                   ("cards", "card slots only", dict(marker="s", color="white", mec=INK,
                                                     mew=1.1))]


def ablation() -> None:
    """Loss increase per field, scope and replacement, one marker per checkpoint."""
    a = csv("ablation.csv")
    names = gen2(a.checkpoint, copies_only=True)
    a = a[a.checkpoint.isin(names)]
    replacements = [r for r in ("noise", "api-mean", "nearest") if r in set(a.replacement)]
    titles = {"noise": "matched noise", "api-mean": "the mean e of its API type",
              "nearest": "the e of the nearest other text"}
    fig, axes = plt.subplots(1, len(replacements), figsize=(4.1 * len(replacements), 5.0),
                             sharey=True, sharex=True, squeeze=False)
    for ax, rep in zip(axes[0], replacements):
        for i, (field, _) in enumerate(ABLATION_FIELDS):
            y = len(ABLATION_FIELDS) - 1 - i
            ax.axhspan(y - 0.45, y + 0.45, color="#f6f6f6" if i % 2 else "white", lw=0,
                       zorder=0)
            for k, (scope, _, style) in enumerate(ABLATION_SCOPES):
                d = a[(a.replacement == rep) & (a.field == field) & (a.scope == scope)]
                yy = y + (1 - k) * 0.26
                ax.plot(d.loss_increase, np.full(len(d), yy), ls="none", ms=4.5, alpha=0.75,
                        zorder=3, **style)
                med = d.loss_increase.median()
                if abs(med) < 0.1:
                    text = f"{med:+.3f}" if abs(med) >= 0.0005 else "0.000"
                    ax.annotate(text, (d.loss_increase.max(), yy), xytext=(8, 0),
                                textcoords="offset points", fontsize=6.8, va="center",
                                color=INK_SOFT)
        ax.axvline(0, color="black", lw=0.9)
        ax.set_title(f"e replaced with\n{titles[rep]}", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("loss increase")
    axes[0][0].set_yticks(range(len(ABLATION_FIELDS)), [x[1] for x in ABLATION_FIELDS][::-1])
    axes[0][0].set_ylim(-0.5, len(ABLATION_FIELDS) - 0.5)
    axes[0][0].set_xlim(-0.08, 2.05)
    handles = [Line2D([], [], ls="none", ms=6, label=lab, **style)
               for _, lab, style in ABLATION_SCOPES]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12),
               fontsize=8.5, title="where e is replaced (numbers: median, where under 0.1)",
               title_fontsize=8.5)
    fig.suptitle(f"Replacing e: the trunk reads the acting ability's e, not the board's "
                 f"(one marker per checkpoint, the {len(names)} gen-2 checkpoints with copies)",
                 x=0.01, ha="left", y=1.02)
    save(fig, "ablation")


def width_summary() -> None:
    """Slope chart per measure: stage across, one line per width, kept solid, last dashed."""
    ck = csv("checkpoints.csv").set_index("checkpoint")
    g3 = csv("gate3.csv").set_index("checkpoint")
    lad = csv("probe_ladder.csv")
    m = lad[(lad.rung == "3") & (lad.probe == "model") & (lad.target != "damage_taken")]
    probe = m.groupby(["checkpoint", "stratum"]).score.mean().unstack()["game-disjoint"]
    sd = loss_noise_table()
    measures = [(ck.card_disjoint, "card-disjoint loss", "lower is better", "card_disjoint"),
                (ck.game_disjoint, "game-disjoint loss", "lower is better", "game_disjoint"),
                (probe, "probe mean, game-disjoint", "higher is better", None),
                (g3.participation_ratio, "participation ratio of e", "directions e uses", None)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0))
    for ax, (series, title, sense, noise) in zip(axes, measures):
        tops = []
        for width in (64, 128, 256):
            arm, a, b = LINES[width]
            colour = WIDTH_COLOUR[width]
            kept = [arm, f"{a} best", f"{b} best"]
            last = [arm, f"{a} last", f"{b} last"]
            if not all(n in series.index for n in kept):
                continue
            ax.plot(range(3), [series[n] for n in kept], color=colour, lw=1.8, zorder=2)
            ax.plot(range(3), [series[n] for n in last], color=colour, lw=1.0, ls="--",
                    zorder=2)
            for x, (k, la) in enumerate(zip(kept, last)):
                ax.plot(x, series[k], zorder=4, **checkpoint_marker(k, 7))
                if la != k and not is_merged(la):
                    ax.plot(x, series[la], zorder=4, **checkpoint_marker(la, 7))
            tops.append(width)
        if noise:
            for x, stage in enumerate(("arm", "A", "B")):
                runs = [LINES[w][x] for w in tops if LINES[w][x] in sd.index]
                values = [series[LINES[w][0] if x == 0 else f"{LINES[w][x]} best"] for w in tops]
                swing = float(np.median([sd.loc[r, noise] for r in runs]))
                leader = min(values)
                ax.add_patch(Rectangle((x - 0.22, leader), 0.44, swing, color="#d9d9d9", lw=0,
                                       zorder=1))
        ax.set_xticks(range(3), ["arm", "A run\nfull rate", "B run\na tenth"])
        ax.set_xlim(-0.35, 2.35)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{title}\n", fontsize=10)
        ax.text(0, 1.02, sense, transform=ax.transAxes, fontsize=8.5, color=INK_SOFT)
    handles = [Line2D([], [], color=WIDTH_COLOUR[w], lw=2.5, label=f"{w}-wide e")
               for w in (64, 128, 256)]
    handles += [Line2D([], [], color=INK_SOFT, lw=1.8, marker="o", label="kept epoch (filled)"),
                Line2D([], [], color=INK_SOFT, lw=1.0, ls="--", marker="o", mfc="white",
                       label="last epoch (hollow)"),
                Patch(color="#d9d9d9", label="one standard deviation of the epoch-to-epoch "
                                             "swing, above the stage's lowest kept loss")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1),
               fontsize=8.5)
    fig.suptitle("The three widths at each stage of training, every checkpoint trained with "
                 "lethality copies (▲ arm, ■ A run, ● B run; B256's ◐ is its own lowest epoch, "
                 "55)", x=0.01, ha="left", y=1.06)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "width")


def other_sweeps() -> None:
    s = csv("sweeps.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    names = gen2(s[s.sweep == "affordability"].checkpoint)
    for name in names:
        d = s[(s.checkpoint == name) & (s.sweep == "affordability") & (s.value <= 8)]
        ax.plot(d.value, d.prediction, marker=checkpoint_marker(name)["marker"], ms=3,
                **run_line(name), alpha=0.9)
    items = s[(s.sweep == "affordability") & (s.value <= 8)].groupby("value").records.max()
    for v, n in items.items():
        ax.text(v, -0.06, str(n), ha="center", fontsize=7, color=INK_SOFT)
    ax.axhline(0.5, color="black", lw=1.2)
    ax.text(0.1, 0.51, "even chance", fontsize=8, va="bottom")
    top = s[(s.sweep == "affordability") & (s.value <= 8) & s.checkpoint.isin(names)]
    top = top.loc[top.prediction.idxmax()]
    if top.prediction >= 0.5:
        tag(ax, top.value, top.prediction, f"{top.checkpoint}: {top.prediction:.3f}", dx=-14,
            dy=14, ha="right")
    ax.set_ylim(-0.1, 1)
    ax.set_xlabel("untapped mana given to the acting player (items per point below)")
    ax.set_ylabel("predicted affordable verdict")
    ax.set_title("Affordability sweep, every gen-2 checkpoint")
    board = present(CHECKPOINTS, s[s.sweep == "board_size"].checkpoint)
    for name in gen2(board) + (["gen-1"] if "gen-1" in set(s.checkpoint) else []):
        d = s[(s.checkpoint == name) & (s.sweep == "board_size")]
        ax2.plot(d.value, d.prediction, **{**run_line(name), "lw": 1.8 if name == "gen-1"
                                           else 1.1}, alpha=0.85)
    ax2.plot([0, 8], [0, 8], color="black", lw=1.0, ls="--")
    ax2.plot([0, 8], [0, 1.6], color=INK_SOFT, lw=1.0, ls=(0, (1, 2)))
    ax2.text(1.75, 2.35, "one death per creature\n(the rule)", fontsize=8)
    ax2.text(7.9, 1.72, "a fifth of a death per creature", fontsize=8, ha="right",
             color=INK_SOFT)
    ax2.text(5.5, 0.2, "gen-1", fontsize=8, color=GEN1_GRAY)
    ax2.set_xlabel("opposing creatures on the board")
    ax2.set_ylabel("predicted deaths summed over the board")
    ax2.set_title("Board-size sweep, a sweeper in the acting slot")
    ax2.set_ylim(0, 3)
    widths = sorted({WIDTH_OF[run_of(n)] for n in names if stage_of(n) != "none"})
    fig.legend(handles=line_handles(widths) + [Line2D([], [], color=GEN1_GRAY, lw=1.8,
                                                      label="gen-1")],
               loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.07), fontsize=8.5)
    save(fig, "other-sweeps")


EVASION_ROWS = ["none", "flying", "horsemanship", "fear", "intimidate", "shadow",
                "unblockable", "protection-blocker", "protection-other"]
EVASION_COLUMNS = ["flying", "reach", "horsemanship", "shadow", "artifact", "black",
                   "shares-colour"]


def evasion_lifts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Median lift per cell, the count of checkpoints beating the no-keyword row, the rule.

    A lift is the predicted chance of a block with the blocker trait divided by the chance
    with no trait, in the same attacker row and checkpoint; the gen-2 record's lift table
    reports its median over checkpoints, a run whose last epoch is its kept one once.
    """
    g = csv("evasion_grid.csv")
    names = gen2(g.checkpoint)
    lifts = {}
    for name in names:
        p = g[g.checkpoint == name].pivot(index="attacker_keyword", columns="blocker_trait",
                                          values="predicted_may_block")
        lifts[name] = p.div(p["none"], axis=0).loc[EVASION_ROWS, EVASION_COLUMNS]
    stack = np.stack([lifts[n].values for n in names])
    median = pd.DataFrame(np.median(stack, axis=0), EVASION_ROWS, EVASION_COLUMNS)
    beats = pd.DataFrame((stack > stack[:, [0], :]).sum(axis=0), EVASION_ROWS, EVASION_COLUMNS)
    rule = g[g.checkpoint == names[-1]].pivot(index="attacker_keyword", columns="blocker_trait",
                                              values="rule_may_block")
    return median, beats, rule.loc[EVASION_ROWS, EVASION_COLUMNS]


def evasion_grid() -> None:
    """One heatmap of the median lift; the no-keyword row is what each column compares to."""
    median, beats, rule = evasion_lifts()
    n = len(gen2(csv("evasion_grid.csv").checkpoint))
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    norm = matplotlib.colors.TwoSlopeNorm(0, -2, 4)
    im = ax.imshow(np.log2(median.values), cmap=DIVERGING_CMAP, norm=norm, aspect="auto")
    for i, row in enumerate(EVASION_ROWS):
        for j, col in enumerate(EVASION_COLUMNS):
            v = median.loc[row, col]
            dark = abs(np.log2(v)) > 2.2
            text = f"×{v:.1f}" if v >= 1.95 else f"×{v:.2f}"
            ax.text(j, i - (0.1 if i else 0), text, ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white" if dark else INK)
            if i:
                ax.text(j, i + 0.24, f"{beats.loc[row, col]}/{n}", ha="center", va="center",
                        fontsize=6.8, color="white" if dark else INK_SOFT)
            allowed = rule.loc[row, col]
            if allowed > 0:
                ax.add_patch(plt.Polygon([(j + 0.5, i - 0.5), (j + 0.18, i - 0.5),
                                          (j + 0.5, i - 0.18)], closed=True, lw=1.0,
                                         facecolor=INK if allowed >= 0.5 else "white",
                                         edgecolor=INK))
    ax.axhline(0.5, color="white", lw=4)
    ax.set_xticks(range(len(EVASION_COLUMNS)), EVASION_COLUMNS, rotation=30, ha="right")
    ax.set_yticks(range(len(EVASION_ROWS)), ["none (the reference row)"] + EVASION_ROWS[1:])
    ax.tick_params(length=0)
    ax.set_xlabel("trait given to the blocker")
    ax.set_ylabel("keyword granted to the attacker")
    ax.grid(False)
    bar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02,
                       label="lift: chance with the trait ÷ chance with no trait, same row")
    ticks = [-2, -1, 0, 1, 2, 3, 4]
    bar.set_ticks(ticks, labels=[f"×{2 ** t:g}" for t in ticks])
    ax.set_title("Evasion: how much a blocker trait lifts the predicted chance of a block,\n"
                 f"median over {n} checkpoints. Compare each cell with the top row's: n/{n} "
                 "counts\nthe checkpoints where the trait lifts more against the keyword than "
                 "against a plain attacker.\n◥ the rule allows the block; ◥ hollow: only when "
                 "the attacker is black too", fontsize=10, pad=10)
    save(fig, "evasion-grid")


def evasion_ranking() -> None:
    """Per attacker keyword: AUC of allowed over forbidden blocks, one marker per checkpoint."""
    r = csv("evasion_ranking.csv")
    rows = [k for k in ["none", "flying", "horsemanship", "fear", "intimidate", "shadow",
                        "protection-other", "all rows"] if k in set(r.attacker_keyword)]
    names = gen2(r.checkpoint) + (["gen-1"] if "gen-1" in set(r.checkpoint) else [])
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    for i, keyword in enumerate(rows):
        y = len(rows) - 1 - i
        d = r[r.attacker_keyword == keyword].dropna(subset=["auc"]).set_index("checkpoint").auc
        ax.axhspan(y - 0.42, y + 0.42, color="#f0f0f0" if keyword == "all rows" else "#f8f8f8",
                   lw=0, zorder=0)
        strip(ax, y, d[~d.index.duplicated()], names, spread=0.3)
    ax.axvline(0.5, color="black", lw=1.2)
    ax.text(0.5, len(rows) - 0.45, "chance", ha="center", va="bottom", fontsize=8)
    ax.set_yticks(range(len(rows)), [("pooled over every row" if k == "all rows" else k)
                                     for k in rows][::-1])
    ax.set_ylim(-0.5, len(rows) - 0.1)
    ax.set_xlim(0.15, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC: allowed blocks ranked above forbidden ones (0.5 = chance)")
    ax.legend(handles=checkpoint_handles(names), fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Does the model rank the blocks each evasion keyword allows above the ones it "
                 "forbids?", pad=16)
    save(fig, "evasion-ranking")


#: Rows resting on fewer items than this are drawn grey: too few to weigh.
FEW = 20


def target_flips() -> None:
    """Predicted legality of a legal target before and after breaking its one condition."""
    f = csv("target_flips.csv")
    names = gen2(f.checkpoint)
    flips = f.groupby("flip").targets.max().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for i, flip in enumerate(flips.index):
        y = len(flips) - 1 - i
        d = f[f.flip == flip].set_index("checkpoint")
        d = d[~d.index.duplicated()]
        few = flips[flip] < FEW
        before, after = d.predicted_legal_before, d.predicted_legal_after
        ax.annotate("", (after[names].median(), y), (before[names].median(), y),
                    arrowprops=dict(arrowstyle="-|>", color="#cfcfcf", lw=6,
                                    mutation_scale=16, shrinkA=0, shrinkB=0), zorder=1)
        where = offsets(names, 0.3)
        for name in names:
            ax.plot(before[name], y + where[name], "|", ms=6, mew=1.4, color=GRAY, zorder=2)
        points = strip(ax, y, after, names, spread=0.3, muted=few)
        if flip == "to the opponent" and "e64 (earlier inputs)" in points:
            tag(ax, *points["e64 (earlier inputs)"], "e64 (earlier inputs)", dx=12, dy=8)
        if flip == "loses land" and "e64" in points:
            tag(ax, *points["e64"], "e64", dx=-12, dy=-6, ha="right")
    for x, label in ((0, "rule after\nthe flip"), (1, "rule before")):
        ax.axvline(x, color="black", lw=1.2)
        ax.text(x, len(flips) - 0.45, label, ha="center", va="bottom", fontsize=8)
    labels = []
    for flip, n in flips.items():
        labels.append(f"{flip}\n{n:,} targets" + ("  (too few to weigh)" if n < FEW else ""))
    ax.set_yticks(range(len(flips)), labels[::-1])
    for tick, (flip, n) in zip(ax.get_yticklabels()[::-1], flips.items()):
        tick.set_color(GRAY if n < FEW else INK)
    ax.set_ylim(-0.5, len(flips) + 0.15)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean predicted legality of the target: grey ticks before the flip, markers "
                  "after; the arrow joins the medians")
    ax.legend(handles=checkpoint_handles(names), fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Breaking the one condition a target restriction states: by the rule the "
                 "target turns illegal", pad=22)
    save(fig, "target-flips")


PIP_CONFIGURATIONS = ["surplus", "one short", "pip missing", "pip missing, +2",
                      "all off colour"]
PIP_LABELS = {
    "surplus": "exact + 2 off-colour lands",
    "one short": "exact less one land",
    "pip missing": "same count, one coloured\npip's land off-colour",
    "pip missing, +2": "pip missing + 2 off-colour\nlands",
    "all off colour": "same count, every land\noff-colour",
}


def colour_pips() -> None:
    """Change in the affordable verdict from the exact lands, and the two rankings."""
    c = csv("colour_pips.csv")
    r = csv("colour_pips_ranking.csv")
    names = gen2(c.checkpoint)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                                      gridspec_kw={"width_ratios": [1.5, 1]})
    labels = []
    for i, name in enumerate(PIP_CONFIGURATIONS):
        y = len(PIP_CONFIGURATIONS) - 1 - i
        d = c[c.configuration == name]
        affordable = not d.empty and bool(d.rule_affordable.iloc[0] >= 0.5)
        labels.append(PIP_LABELS[name] + ("\nrule: affordable" if affordable
                                          else "\nrule: not affordable"))
        left.axhspan(y - 0.44, y + 0.44, color="#f6f6f6", lw=0, zorder=0)
        strip(left, y, -d.set_index("checkpoint").below_exact_mean, names, spread=0.3)
    left.axvline(0, color="black", lw=1.2)
    left.text(0, len(PIP_CONFIGURATIONS) - 0.45, "the exact lands", ha="center", va="bottom",
              fontsize=8)
    left.set_yticks(range(len(PIP_CONFIGURATIONS)), labels[::-1], fontsize=8.5)
    left.set_ylim(-0.5, len(PIP_CONFIGURATIONS) - 0.05)
    left.grid(axis="y", visible=False)
    left.set_xlabel("predicted affordable, minus the exact lands' prediction,\nmean over "
                    "300 decisions")
    left.set_title("Change from the exact lands")
    groups = [("amount only", "exact against\none short"),
              ("colours only", "exact, surplus against\nthe three off-colour rows"),
              ("every configuration", "every\nconfiguration")]
    for i, (prefix, label) in enumerate(groups):
        d = r[r["items"].str.startswith(prefix)].set_index("checkpoint").auc
        right.axhspan(i - 0.42, i + 0.42, color="#f6f6f6", lw=0, zorder=0)
        strip(right, i, d[~d.index.duplicated()], names, spread=0.3)
    right.axvline(0.5, color="black", lw=1.2)
    right.text(0.5, len(groups) - 0.45, "chance", ha="center", va="bottom", fontsize=8)
    right.set_yticks(range(len(groups)), [g[1] for g in groups], fontsize=8.5)
    right.set_ylim(-0.5, len(groups) - 0.05)
    right.set_xlim(0.4, 1.0)
    right.grid(axis="y", visible=False)
    right.set_xlabel("AUC: affordable ranked above unaffordable\n(0.5 = chance)")
    right.set_title("Ranking by amount and by colour")
    fig.legend(handles=checkpoint_handles(names), loc="lower center", ncol=12,
               bbox_to_anchor=(0.5, -0.1), fontsize=8, columnspacing=1.0)
    fig.suptitle("Does the model check the colour of the mana, or only the amount? One marker "
                 "per checkpoint", x=0.01, ha="left", y=1.0)
    save(fig, "colour-pips")


ZONE_NAMES = {"died": "graveyard", "exiled": "exile", "to_hand": "hand",
              "library_top": "library", "blinked": "battlefield", "countered": "countered"}


def outcome_zone(outcome: str) -> str:
    return ZONE_NAMES.get(outcome, outcome)


def zone_outcomes() -> None:
    """How often the most probable destination of a moved card is the right one."""
    z = csv("zone_outcomes.csv")
    names = gen2(z.checkpoint)
    sizes = z.groupby(["zone_before", "outcome"]).cards.max()
    sizes = sizes[sizes >= 9]
    order = {"battlefield": 0, "graveyard": 1, "hand": 2}
    rows = sorted(sizes.index, key=lambda k: (order.get(k[0], 9), -sizes[k]))
    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    for i, (zone, outcome) in enumerate(rows):
        y = len(rows) - 1 - i
        d = z[(z.zone_before == zone) & (z.outcome == outcome)].set_index("checkpoint")
        d = d[~d.index.duplicated()]
        few = sizes[(zone, outcome)] < FEW
        ax.axhspan(y - 0.42, y + 0.42, color="#f6f6f6", lw=0, zorder=0)
        strip(ax, y, d.top_right, names, spread=0.3, muted=few)
        wrong = d.loc[[n for n in names if n in d.index], "predicted_instead"].dropna()
        if not wrong.empty:
            top = wrong.value_counts()
            where = ZONE_NAMES.get(top.index[0], top.index[0])
            text = f"→ {where} (on {top.iloc[0]} of {len(names)})"
            ax.text(1.02, y, text, transform=ax.get_yaxis_transform(), fontsize=8,
                    va="center", color=GRAY if few else INK)
    ax.text(1.02, len(rows) - 0.45, "most common wrong\ndestination", fontsize=8,
            transform=ax.get_yaxis_transform(), va="bottom", fontweight="bold")
    labels = [f"{zone} → " + ("battlefield, a blink" if zone == outcome_zone(outcome) else
                              outcome_zone(outcome)) + f" ({sizes[(zone, outcome)]} cards)"
              for zone, outcome in rows]
    ax.set_yticks(range(len(rows)), labels[::-1])
    for tick, key in zip(ax.get_yticklabels()[::-1], rows):
        tick.set_color(GRAY if sizes[key] < FEW else INK)
    for boundary in range(1, len(rows)):
        if rows[boundary][0] != rows[boundary - 1][0]:
            ax.axhline(len(rows) - boundary - 0.5, color=INK_SOFT, lw=0.8)
    ax.axvline(1, color="black", lw=1.2)
    ax.set_ylim(-0.5, len(rows) - 0.1)
    ax.set_xlim(0, 1.015)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of moved cards whose most probable destination is the right one "
                  "(staying put excluded)")
    ax.legend(handles=checkpoint_handles(names), fontsize=8, loc="upper left",
              bbox_to_anchor=(1.2, 1.0))
    ax.set_title(f"Where does the model send a card the resolution moved? Grey rows: fewer than "
                 f"{FEW} cards", pad=16)
    save(fig, "zone-outcomes")


def zone_amounts() -> None:
    """Predicted cards at the affected player as the script's NumCards$ is set 1 to 6."""
    a = csv("zone_amounts.csv")
    names = gen2(a.checkpoint)
    amounts = list(range(1, 7))
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True)
    for ax, api in zip(axes, ("Draw", "Discard", "Mill")):
        d = a[a.api == api].set_index("checkpoint")
        d = d[~d.index.duplicated()]
        for name in names:
            if name in d.index:
                ax.plot(amounts, [d.loc[name, f"amount_{k}"] for k in amounts],
                        **run_line(name), alpha=0.9)
        ax.plot(amounts, amounts, color="black", lw=1.2)
        ax.text(4.95, 5.85, "the script's\nnumber", fontsize=8, ha="right", va="top")
        if api == "Discard" and "e64 (earlier inputs)" in d.index:
            tag(ax, 6, d.loc["e64 (earlier inputs)", "amount_6"], "e64 (earlier inputs)",
                dx=-10, dy=22, ha="right")
        if api == "Mill" and "A256 best" in d.index:
            tag(ax, 6, d.loc["A256 best", "amount_6"], "A256 best = last", dx=10, dy=0,
                ha="left")
        n = int(d.records.max()) if not d.empty else 0
        ax.set_title(f"{api}, {n} records")
        ax.set_xlabel("NumCards$ written into the script")
        ax.set_xticks(amounts)
    axes[0].set_ylabel("predicted cards at the affected player")
    widths = sorted({WIDTH_OF[run_of(n)] for n in names if stage_of(n) != "none"})
    fig.legend(handles=line_handles(widths), loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.08), fontsize=8.5)
    fig.suptitle("Does the predicted count follow the number in the script? One line per "
                 "checkpoint", x=0.01, ha="left", y=1.0)
    save(fig, "zone-amounts")


def zone_who() -> None:
    """Share of the predicted cards still at the originally targeted player after retargeting."""
    w = csv("zone_who.csv")
    names = gen2(w.checkpoint)
    apis = [api for api in ("Draw", "Discard", "Mill") if api in set(w.api)]
    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    before_all = []
    for i, api in enumerate(apis):
        y = len(apis) - 1 - i
        d = w[w.api == api].set_index("checkpoint")
        d = d[~d.index.duplicated()]
        before = d.targeted_before / (d.targeted_before + d.other_before)
        after = d.targeted_after / (d.targeted_after + d.other_after)
        before_all.append(before[[n for n in names if n in before.index]].median())
        ax.axhspan(y - 0.42, y + 0.42, color="#f6f6f6", lw=0, zorder=0)
        points = strip(ax, y, after, names, spread=0.3)
        for name, dx, dy, ha in {"Discard": [("e128-lethality", -14, 14, "right"),
                                             ("B64 last", 12, -6, "left")],
                                 "Mill": [("e128-lethality", -14, -16, "right"),
                                          ("B256 last", 12, -6, "left")]}.get(api, []):
            if name in points:
                tag(ax, *points[name], name, dx=dx, dy=dy, ha=ha)
    ax.axvline(0, color="black", lw=1.2)
    ax.text(0, len(apis) - 0.45, "the rule", ha="center", va="bottom", fontsize=8)
    n = w.groupby("api").records.max()
    ax.set_yticks(range(len(apis)), [f"{api}\n{n[api]} records" for api in apis][::-1])
    ax.set_ylim(-0.5, len(apis) - 0.1)
    ax.set_xlim(-0.03, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("after the target is handed to the other player: share of the predicted cards "
                  "still at the originally targeted player")
    ax.legend(handles=checkpoint_handles(names), fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.05))
    ax.set_title("Does the effect follow the player target? Before the flip the targeted player "
                 f"holds {min(before_all):.0%} to {max(before_all):.0%} of the predicted cards "
                 "(median per effect)", pad=16, fontsize=10)
    save(fig, "zone-who")


# ── run lineage ────────────────────────────────────────────────────────────


def run_lineage() -> None:
    """Every gen-2 run as a bar over its epochs, with the checkpoints probed on it.

    An A run resumes its arm after the arm's last epoch; a B run starts from its A run's
    kept epoch at a tenth of the learning rate. Run B256 never went below the checkpoint it
    started from, so it keeps run A256's epoch 53, and ``B256 best`` is its own lowest
    epoch, 55, probed but not kept.
    """
    runs = csv("runs.csv").set_index("run")
    ck = csv("checkpoints.csv")
    rows = [(64, "e64 (earlier inputs)", "first arm, before the input corrections"),
            (64, "e64", "first arm again, corrected inputs"),
            (64, "e64-lethality", "e64 with lethality copies"),
            (64, "A64", "resumes the arm, full learning rate"),
            (64, "B64", "A64's kept epoch at a tenth of the rate"),
            (128, "e128", "e64 at width 128"),
            (128, "e128-lethality", "e128 with lethality copies"),
            (128, "A128", "resumes the arm, full learning rate"),
            (128, "B128", "A128's kept epoch at a tenth of the rate"),
            (256, "e256-lethality", "e128-lethality at width 256"),
            (256, "A256", "resumes the arm, full learning rate"),
            (256, "B256", "A256's kept epoch at a tenth of the rate")]
    rows = [r for r in rows if r[1] in runs.index]
    ys, y, last_width = {}, 0.0, None
    for width, run, _ in rows:
        if last_width is not None and width != last_width:
            y += 0.6
        ys[run] = y
        y += 1.0
        last_width = width
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    height = 0.36
    for width, run, note in rows:
        r, yy = runs.loc[run], ys[run]
        first, last = int(r.first_epoch), int(r.last_epoch)
        if run == "e64 (earlier inputs)":
            face, edge, hatch = "white", NO_COPIES_GRAY, "////"
        elif r.lethality_copies == 0:
            face, edge, hatch = shade(NO_COPIES_GRAY, 0.5), NO_COPIES_GRAY, None
        else:
            face, edge, hatch = run_colour(run), shade(WIDTH_COLOUR[width], 1.0), None
        ax.add_patch(Rectangle((first - 0.5, yy - height / 2), last - first + 1, height,
                               facecolor=face, edgecolor=edge, hatch=hatch, lw=0.8, zorder=2))
        ax.text(-1.0, yy - 0.08, short(run), ha="right", va="bottom", fontsize=9.2,
                fontweight="bold", color=INK)
        ax.text(-1.0, yy - 0.04, note, ha="right", va="top", fontsize=7.4, color=INK_SOFT)
        parent = r.continues if isinstance(r.continues, str) and r.continues else None
        if parent in ys:
            start = first - 1  # the epoch whose weights the run starts from
            ax.annotate("", (first - 0.5, yy), (start, ys[parent] + height / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, shrinkA=0,
                                        shrinkB=0, mutation_scale=10,
                                        connectionstyle="arc3,rad=0.3"), zorder=4)

    def marker(name: str, role: str, width: int | None) -> dict:
        base = NO_COPIES_GRAY if width is None else WIDTH_COLOUR[width]
        style = checkpoint_marker(name, 8.5)
        style.update(color=darker(base, 0.25), mec=darker(base, 0.25), mew=1.6)
        if role == "last":
            style["mfc"] = "white"
        else:
            style.pop("mfc", None)
        return style

    # The probed checkpoints, labelled with the names every other chart uses.
    names = {"B256 best": "B256 best, not kept"}
    for _, c in ck.iterrows():
        yy = ys.get(c.run)
        if c.checkpoint == "A256 last" or yy is None:
            continue
        width = None if runs.loc[c.run].lethality_copies == 0 else int(c.e_width)
        ax.plot(c.epoch, yy, zorder=5, **marker(c.checkpoint, c.role, width))
        ax.annotate(names.get(c.checkpoint, short(c.checkpoint)), (c.epoch, yy + height / 2),
                    xytext=(3, 5), textcoords="offset points", ha="left", va="bottom",
                    fontsize=7.8, color=INK, zorder=6)
    if "B256" in ys and "A256 best" in set(ck.checkpoint):
        kept = int(ck[ck.checkpoint == "A256 best"].epoch.iloc[0])
        ax.plot(kept, ys["B256"], zorder=5, **marker("B256 kept", "best", 256))
        ax.annotate(f"kept: A256's epoch {kept}", (kept, ys["B256"]), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=7.8,
                    color=INK)
    for width in (64, 128, 256):
        members = [ys[run] for w, run, _ in rows if w == width]
        if members:
            ax.text(-23.5, np.mean(members), f"e width {width}", rotation=90, ha="center",
                    va="center", fontsize=9.5, fontweight="bold",
                    color=shade(WIDTH_COLOUR[width], 1.0))
            ax.plot([-22.3, -22.3], [min(members) - 0.3, max(members) + 0.3],
                    color=shade(WIDTH_COLOUR[width], 1.0), lw=2, clip_on=False)
    ax.set_yticks([])
    ax.set_ylim(y - 0.2, -0.8)
    ax.set_xlim(0, 75)
    ax.set_xticks(range(0, 75, 10))
    ax.set_xlabel("epoch")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    handles = [Patch(facecolor="white", edgecolor=NO_COPIES_GRAY, hatch="////",
                     label="earlier inputs"),
               Patch(facecolor=shade(NO_COPIES_GRAY, 0.5), edgecolor=NO_COPIES_GRAY,
                     label="no lethality copies"),
               Line2D([], [], label="arm", **marker("e64-lethality", "best", None)),
               Line2D([], [], label="A run", **marker("A64 best", "best", None)),
               Line2D([], [], label="B run", **marker("B64 best", "best", None)),
               Line2D([], [], label="kept epoch: filled; last epoch: hollow",
                      **marker("B64 last", "last", None)),
               Line2D([], [], color=INK, lw=1, label="starts from")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.36, -0.09), ncol=7,
              fontsize=8, handlelength=1.5, columnspacing=1.1)
    ax.text(-24.5, -1.25, "The seventeen probed checkpoints and the runs they come from; an "
            "arrow marks the weights a continuation starts from", ha="left", va="bottom",
            fontsize=11, color=INK, clip_on=False)
    save(fig, "run-lineage")


# ── scorecard ──────────────────────────────────────────────────────────────


def scorecard_measures() -> pd.DataFrame:
    """One row per checkpoint, one column per game-relevant measure, as the records define them."""
    ck = csv("checkpoints.csv").set_index("checkpoint")
    out = pd.DataFrame(index=ck.index)
    out["card_disjoint"] = ck.card_disjoint
    out["game_disjoint"] = ck.game_disjoint
    lad = csv("probe_ladder.csv")
    own = lad[(lad.rung == "3") & (lad.probe == "model") & (lad.target != "damage_taken")
              & (lad.stratum == "game-disjoint")]
    out["probe_mean"] = own.groupby("checkpoint").score.mean()
    grid = csv("damage_grids.csv")
    grid = grid[grid.record_set == "probe set"]
    rule = (grid.damage >= grid.toughness).astype(float)
    out["grid_error"] = (grid.predicted_death - rule).abs().groupby(grid.checkpoint).mean()
    sweeps = csv("sweeps.csv")
    dmg = sweeps[sweeps.sweep == "damage_spell"]
    out["sweep_on_rule"] = dmg.groupby("checkpoint").apply(
        lambda d: float(((d.value < 4) == (d.prediction < 0.5)).all()))
    records = csv("sweep_records.csv")
    meta = csv("toughness_sweep_records.csv")
    pings = meta[(meta.kind == "activated") & (meta.damage == 1)].record
    hit = records[(records.sweep == "toughness") & records.record.isin(pings)
                  & (records.value == 1)]
    out["ping_death"] = hit.groupby("checkpoint").prediction.median()
    g2 = csv("gate2.csv")
    varying = g2[g2.keyword.isin(["first_strike", "deathtouch", "infect", "double_strike"])]
    # The first run of arm e64 scored first and double strike on the wrong damage step.
    varying = varying[varying.checkpoint != "e64 (earlier inputs)"]
    out["gate2_varying"] = varying.groupby("checkpoint").agreement_pct.mean()
    ev = csv("evasion_ranking.csv")
    out["evasion_auc"] = ev[ev.attacker_keyword == "all rows"].set_index("checkpoint").auc
    pips = csv("colour_pips_ranking.csv")
    out["colour_auc"] = pips[pips["items"].str.startswith("colours only")].set_index(
        "checkpoint").auc
    board = sweeps[sweeps.sweep == "board_size"]
    out["board_slope"] = board.groupby("checkpoint").apply(
        lambda d: np.polyfit(d.value, d.prediction, 1)[0])
    za = csv("zone_amounts.csv")
    amounts = [f"amount_{k}" for k in range(1, 7)]
    za["slope"] = [np.polyfit(range(1, 7), row[amounts].astype(float), 1)[0]
                   for _, row in za.iterrows()]
    out["zone_slope"] = za.groupby("checkpoint").slope.mean()
    who = csv("zone_who.csv")
    who["stays"] = who.targeted_after / (who.targeted_after + who.other_after)
    out["retarget_stays"] = who.groupby("checkpoint").stays.mean()
    flips = csv("target_flips.csv")
    control = flips[flips.flip == "to the opponent"].set_index("checkpoint")
    out["control_flip"] = control.predicted_legal_before - control.predicted_legal_after
    return out


#: (group, [(measure, label, +1 if higher is better, value format)])
SCORECARD_ROWS = [
    ("Validation loss", [
        ("card_disjoint", "card-disjoint loss (texts never trained on)", -1, "{:.2f}"),
        ("game_disjoint", "game-disjoint loss (trained texts, new games)", -1, "{:.2f}"),
    ]),
    ("Knowledge probes", [
        ("probe_mean", "probe mean: model's own answer, 8 questions, AUC", +1, "{:.3f}"),
    ]),
    ("Damage against toughness", [
        ("grid_error", "damage grid: mean error against the rule", -1, "{:.3f}"),
        ("sweep_on_rule", "damage sweep: every point on the rule's side", +1, "check"),
        ("ping_death", "activated 1-damage ping: predicted death at toughness 1", +1,
         "{:.2f}"),
    ]),
    ("Combat", [
        ("gate2_varying", "gate 2 agreement, %: first strike, double strike, deathtouch, "
         "infect", +1, "{:.0f}"),
    ]),
    ("Blocking, targeting, mana", [
        ("evasion_auc", "evasion: allowed blocks ranked over forbidden, AUC", +1, "{:.2f}"),
        ("control_flip", "\"you control\" target handed over: drop in predicted legality",
         +1, "{:.2f}"),
        ("colour_auc", "colour: right-colour lands ranked over wrong-colour, AUC", +1,
         "{:.2f}"),
    ]),
    ("Counting", [
        ("board_slope", "sweeper: predicted deaths per creature added (rule: 1)", +1,
         "{:.2f}"),
        ("zone_slope", "draw, discard, mill: predicted cards per NumCards$ (rule: 1)", +1,
         "{:.2f}"),
        ("retarget_stays", "draw, discard, mill retargeted: share left at the old target "
         "(rule: 0)", -1, "{:.2f}"),
    ]),
]


def scorecard() -> None:
    """Measure × checkpoint, each row coloured by rank within it (darker is better).

    The last row is each checkpoint's mean rank over the measures it has, 1 being the best
    of every row. ``e64 (earlier inputs)`` scored first and double strike on the wrong
    damage step, so it has no gate-2 cell.
    """
    data = scorecard_measures()
    columns = gen2_checkpoints(data.index)
    names = [name for _, name, _ in columns]
    measures = [m for _, group in SCORECARD_ROWS for m in group if m[0] in data.columns]
    values = data.loc[names]

    def level_of(series: pd.Series, direction: int, fmt: str) -> pd.Series:
        if fmt == "check":
            return series.astype(float)
        rank = (-direction * series).rank(method="min")
        return 1 - (rank - 1) / max(series.notna().sum() - 1, 1)

    ranks = pd.DataFrame({key: (-d * values[key]).rank(method="min")
                          for key, _, d, f in measures if f != "check"})
    mean_rank = ranks.mean(axis=1)
    xs, x, prev = [], 0.0, None
    for width, _, _ in columns:
        if prev is not None and width != prev:
            x += 0.35
        xs.append(x)
        x += 1
        prev = width
    layout, y = [], 0.0
    for group, rows in SCORECARD_ROWS:
        rows = [r for r in rows if r[0] in data.columns]
        if not rows:
            continue
        layout.append((y, "group", group))
        y += 0.75
        for row in rows:
            layout.append((y, "measure", row))
            y += 1
        y += 0.2
    layout.append((y + 0.15, "measure", ("mean_rank", "mean rank over the rows above "
                                                      "(1 = best in every row)", -1, "{:.1f}")))
    y += 1.15
    fig, ax = plt.subplots(figsize=(12.6, 0.3 * y + 1.6))
    fig.subplots_adjust(left=0.33, right=0.995, top=0.8, bottom=0.04)
    for yy, kind, payload in layout:
        if kind == "group":
            ax.text(-0.75, yy + 0.05, payload, ha="right", va="center", fontsize=8.8,
                    fontweight="bold", color=INK)
            continue
        key, label, direction, fmt = payload
        row = mean_rank if key == "mean_rank" else values[key]
        level = level_of(row, direction, fmt)
        best = (-direction * row).rank(method="min") == 1
        for name, xx in zip(names, xs):
            v = row.loc[name]
            if pd.isna(v):
                ax.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1, facecolor="white",
                                       edgecolor="#e6e6e6", lw=0.6))
                ax.text(xx, yy, "–", ha="center", va="center", fontsize=8, color=GRAY)
                continue
            lv = level.loc[name]
            ax.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1,
                                   facecolor=BETTER_CMAP(0.04 + 0.86 * lv), edgecolor="white",
                                   lw=1.1))
            text = ("✓" if v >= 0.5 else "✗") if fmt == "check" else fmt.format(v)
            if fmt == "{:.3f}" and text.startswith("0"):
                text = text[1:]
            ax.text(xx, yy, text, ha="center", va="center", fontsize=8.2,
                    fontweight="bold" if best.loc[name] and fmt != "check" else "normal",
                    color="white" if lv > 0.55 else INK)
        arrow = "▼" if direction < 0 else "▲"
        ax.text(-0.75, yy, f"{label}  {arrow}", ha="right", va="center", fontsize=8.2,
                color=INK, fontweight="bold" if key == "mean_rank" else "normal")
    top = -0.55
    for (width, name, label), xx in zip(columns, xs):
        ax.plot(xx, top - 0.35, clip_on=False, **checkpoint_marker(name, 6))
        ax.text(xx, top - 0.85, label, rotation=90, ha="center", va="bottom", fontsize=8.2,
                color=INK, fontweight="bold" if name.startswith("B") else "normal")
    for width in (64, 128, 256):
        members = [xx for (w, _, _), xx in zip(columns, xs) if w == width]
        if members:
            colour = shade(WIDTH_COLOUR[width], 1.0)
            ax.plot([min(members) - 0.4, max(members) + 0.4], [top - 5.6] * 2, color=colour,
                    lw=2, clip_on=False)
            ax.text(np.mean(members), top - 5.8, f"e width {width}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=colour)
    ax.set_xlim(-0.55, xs[-1] + 0.55)
    ax.set_ylim(y - 0.4, top)
    ax.axis("off")
    ax.annotate("No checkpoint leads everywhere: seven different checkpoints top a row; the "
                "256-wide continuations rank highest overall, the arms near the bottom",
                (0.01, top - 7.0), xycoords=("figure fraction", "data"), ha="left",
                va="bottom", fontsize=11, color=INK, annotation_clip=False)
    fig.text(0.01, 0.005, "Each row is coloured by rank among the checkpoints, darker = better "
             "(▲ higher is better, ▼ lower is better); bold = best in the row. – : not "
             "measured.", fontsize=7.8, color=INK_SOFT)
    save(fig, "scorecard")


# ── where the card-disjoint rise comes from ────────────────────────────────

#: (field or group, label, colour)
RISE_GROUPS = [("gate", "affected gate", "#4b2a86"),
               ("poison_delta", "poison", "#b8336a"),
               ("pt", "power & toughness change", "#b9a6d8"),
               ("zone_outcome", "zone outcome", "#8c7a5b"),
               ("other", "the other 42 fields", "#cfcfcf"),
               ("outside", "loss terms outside the 47 fields", "#707070")]


def loss_rise_by_field() -> None:
    """Each continuation run's change in card-disjoint loss, lowest epoch to last, by field.

    The 47 per-field terms of ``fields_by_epoch.csv`` do not sum to gen-2's total
    card-disjoint loss: about 0.4 to 0.6 of it lies in terms outside them, and the change
    in that remainder is drawn as its own segment.
    """
    ep = csv("epochs.csv")
    fe = csv("fields_by_epoch.csv")
    order = [r for w in (64, 128, 256) for r in LINES[w][1:] if r in set(ep.run)]
    rows = []
    for run in order:
        d = ep[ep.run == run]
        low, last = d.loc[d.card_disjoint.idxmin()], d.loc[d.epoch.idxmax()]
        a = fe[(fe.run == run) & (fe.epoch == low.epoch)].set_index("field").card_disjoint_loss
        b = fe[(fe.run == run) & (fe.epoch == last.epoch)].set_index("field").card_disjoint_loss
        delta = (b - a).fillna(0)
        parts = {"gate": delta.get("gate", 0), "poison_delta": delta.get("poison_delta", 0),
                 "pt": delta.get("power_delta", 0) + delta.get("toughness_delta", 0),
                 "zone_outcome": delta.get("zone_outcome", 0)}
        named = {"gate", "poison_delta", "power_delta", "toughness_delta", "zone_outcome"}
        parts["other"] = delta[[f for f in delta.index if f not in named]].sum()
        total = last.card_disjoint - low.card_disjoint
        parts["outside"] = total - delta.sum()
        rows.append(dict(run=run, low=int(low.epoch), last=int(last.epoch), total=total,
                         **parts))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.6, 0.62 * len(df) + 1.9))
    for i, r in df.iterrows():
        y = len(df) - 1 - i
        if r.low == r["last"]:
            ax.text(0.003, y, f"lowest card-disjoint epoch is its last ({r['last']}): no rise",
                    va="center", fontsize=8.2, color=INK_SOFT, style="italic")
        pos, neg = 0.0, 0.0
        for key, label, colour in RISE_GROUPS:
            v = r[key]
            if abs(v) < 1e-9:
                continue
            left = pos if v > 0 else neg + v
            ax.barh(y, abs(v), left=left, height=0.56, color=colour, edgecolor="white", lw=1)
            if abs(v) >= 0.02:
                ax.text(left + abs(v) / 2, y, f"{v:+.3f}", ha="center", va="center",
                        fontsize=7.6, color="white" if key in ("gate", "poison_delta",
                                                              "outside") else INK)
            if v > 0:
                pos += v
            else:
                neg += v
        if r.low != r["last"]:
            ax.plot([r.total, r.total], [y - 0.36, y + 0.36], color=INK, lw=2)
            ax.text(max(pos, r.total) + 0.004, y, f"total {r.total:+.3f}", va="center",
                    fontsize=8.4, color=INK, fontweight="bold")
    ax.set_yticks(range(len(df)),
                  [f"{r.run}   epoch {r.low} → {r['last']}" for _, r in df.iterrows()][::-1])
    for label in ax.get_yticklabels():
        run = label.get_text().split()[0]
        label.set_color(shade(WIDTH_COLOUR[WIDTH_OF[run]], 1.0))
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlim(min(-0.06, df[[k for k, _, _ in RISE_GROUPS]].clip(upper=0).sum(axis=1).min()
                    - 0.01), df.total.max() + 0.07)
    ax.set_xlabel("change in card-disjoint loss from the run's lowest card-disjoint epoch to its "
                  "last; rises right, falls left, black bar = net")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=c, label=label) for _, label, c in RISE_GROUPS],
              loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=3, fontsize=8)
    ax.set_title("More than half of every run's late card-disjoint rise is in the affected gate, "
                 "and poison adds\nthe next-largest part in four runs of five", loc="left",
                 fontsize=10.5, x=-0.2)
    save(fig, "loss-rise-by-field")


# ── e against a random vector of its width ─────────────────────────────────

#: Families in the probes record's order, with a label per target.
GAIN_FAMILIES = [
    ("magnitudes", "Magnitudes", [
        ("dies", "dies", True), ("damage_taken", "damage taken", True),
        ("damage", "damage amount", False), ("damage_per_mana", "damage per mana*", False),
        ("counters_placed", "counters placed", False), ("life_amount", "life amount", False),
        ("power_change", "power change", False), ("toughness_change", "toughness change", False)]),
    ("mana-production", "Mana production", [
        ("mana_amount", "mana amount", False), ("observed_mana", "observed mana", False),
        ("any_colour", "any colour", False), ("produces_w", "produces W", False),
        ("produces_u", "produces U", False), ("produces_b", "produces B", False),
        ("produces_r", "produces R", False), ("produces_g", "produces G", False),
        ("produces_c", "produces C", False), ("restricted", "spending restricted", False),
        ("conditional", "conditional", False)]),
    ("mana-usage", "Mana usage", [
        ("affordable", "affordable", True), ("cost_total", "total mana cost", False),
        ("cost_generic", "generic cost", False), ("cost_w", "W pips", False),
        ("cost_u", "U pips", False), ("cost_b", "B pips", False), ("cost_r", "R pips", False),
        ("cost_g", "G pips", False), ("cost_c", "C pips", False),
        ("cost_reducer", "cost reducer", False), ("cost_tax", "cost tax", False)]),
    ("timing-costs", "Timing and non-mana costs", [
        ("instant_speed", "instant speed", False), ("tap_cost", "tap cost", False),
        ("sacrifice_cost", "sacrifice cost", False), ("life_cost", "life cost", False)]),
    ("interactions", "Interactions", [
        ("fires", "trigger fires", True), ("fires_pair", "A's event fires B's trigger†", False)]),
    ("side", "Side", [
        ("affected", "card affected", True), ("affects_you", "affects its controller", False),
        ("affects_opponent", "affects an opponent", False),
        ("affects_each", "affects each player", False)]),
    ("evasion-blocking", "Evasion and blocking", [("may_block", "may block", True)]),
    ("target-legality", "Target legality", [
        ("legal_target", "legal target", True),
        ("legal_target_protected", "legal target, protected", True)]),
    ("duration-repeatability", "Duration and repeatability", [
        ("pt_until_end_of_turn", "pump lasts until end of turn", True),
        ("until_end_of_turn", "lasts until end of turn", False),
        ("as_counter", "comes as a counter", False), ("repeatable", "repeatable", False)]),
    ("state-dependence", "State dependence", [
        ("spread", "outcome spread across boards", False)]),
]


def e_over_random() -> None:
    """Rung 1 minus rung 1w per target and checkpoint: what e adds over a random vector.

    Rung 1 is the MLP probe on the board plus e (board targets, card-disjoint records) or
    on e alone (line properties, held-out lines); rung 1w replaces e with a fixed random
    vector of the same width. Binary targets are AUC differences, amounts R² differences;
    the colour doubles an AUC difference so both kinds run from 0, no better than the
    random vector, to 1, perfect where the control is at chance.
    """
    lad = csv("probe_ladder.csv")
    lad = lad[(lad.probe == "mlp") & lad.rung.isin(["1", "1w"])]
    kind = lad.groupby("target").kind.first()

    def stratum_for(target: str, board: bool) -> str:
        if board:
            return "card-disjoint"
        have = set(lad[lad.target == target].stratum)
        return "held-out" if "held-out" in have else "all" if "all" in have else "trained"

    columns = [("gen-1", "gen-1", None)] + [(name, label, w) for w, name, label in
                                            gen2_checkpoints(lad.checkpoint)]
    xs, x, prev = [], 0.0, "start"
    for _, _, width in columns:
        if width != prev and prev != "start":
            x += 0.45
        xs.append(x)
        x += 1
        prev = width
    def has_scores(target: str, board: bool) -> bool:
        d = lad[(lad.target == target) & (lad.stratum == stratum_for(target, board))]
        return d.score.notna().any()

    layout = []  # (y, kind of row, payload)
    y = 0.0
    for family, title, targets in GAIN_FAMILIES:
        present_targets = [t for t in targets if t[0] in kind.index and has_scores(t[0], t[2])]
        if not present_targets:
            continue
        layout.append((y, "family", title))
        y += 0.85
        for target, label, board in present_targets:
            layout.append((y, "target", (target, label, board)))
            y += 1
        y += 0.25
    top = -1.25
    head = 10.4  # data rows above the first family: column labels, width names, title
    height = 0.215 * (y - top + head) + 0.9
    fig = plt.figure(figsize=(9.6, height))
    ax = fig.add_axes([0.3, 0.9 / height, 0.69, 1 - 1.0 / height])
    norm = matplotlib.colors.Normalize(-1, 1)
    for yy, row_kind, payload in layout:
        if row_kind == "family":
            ax.text(-0.75, yy + 0.1, payload, ha="right", va="center", fontsize=9,
                    fontweight="bold", color=INK)
            continue
        target, label, board = payload
        stratum = stratum_for(target, board)
        d = lad[(lad.target == target) & (lad.stratum == stratum)]
        piv = d.pivot_table(index="checkpoint", columns="rung", values="score")
        amount = kind[target] == "amount"
        for (name, _, _), xx in zip(columns, xs):
            if name not in piv.index or piv.loc[name].isna().any():
                ax.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1, facecolor="white",
                                       edgecolor="#eeeeee", lw=0.5))
                continue
            diff = piv.loc[name, "1"] - piv.loc[name, "1w"]
            level = diff if amount else 2 * diff
            ax.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1, facecolor=GAIN_CMAP(norm(level)),
                                   edgecolor="white", lw=0.8))
            ax.text(xx, yy, f"{100 * diff:.0f}", ha="center", va="center", fontsize=7.4,
                    color="white" if abs(level) > 0.62 else INK)
        tag = ("board" if board else "line") + (", R²" if amount else "")
        ax.text(-0.75, yy, f"{label}  ({tag})", ha="right", va="center", fontsize=8.2,
                color=INK, fontweight="bold" if target == "dies" else "normal")
        if target == "dies":
            ax.add_patch(Rectangle((xs[1] - 0.5, yy - 0.5), xs[-1] - xs[1] + 1, 1, fill=False,
                                   edgecolor=INK, lw=1.6, zorder=5))
    for (name, label, width), xx in zip(columns, xs):
        colour = shade(WIDTH_COLOUR[width], 1.0) if width else INK_SOFT
        ax.text(xx, top, label, rotation=90, ha="center", va="bottom", fontsize=8,
                color=INK, fontweight="bold" if name.startswith("B") else "normal")
        if name != "gen-1":
            ax.plot(xx, top + 0.45, clip_on=False, **checkpoint_marker(name, 5.5))
        else:
            ax.plot(xx, top + 0.45, "D", color=colour, ms=4.5, clip_on=False)
    for width in (None, 64, 128, 256):
        members = [xx for (_, _, w), xx in zip(columns, xs) if w == width]
        if members:
            ax.text(np.mean(members), top - 6.4, "gen-1" if width is None
                    else f"e width {width}", ha="center", va="bottom", fontsize=8.8,
                    fontweight="bold",
                    color=INK_SOFT if width is None else shade(WIDTH_COLOUR[width], 1.0))
    ax.set_xlim(-0.55, xs[-1] + 0.55)
    ax.set_ylim(y - 0.3, top - head)
    ax.axis("off")
    ax.annotate("e beats a random vector of its width on what a line states, and barely on "
                "what the board decides", (0.01, top - 10.3), xycoords=("figure fraction", "data"),
                ha="left", va="top", fontsize=11, color=INK, annotation_clip=False)
    ax.annotate("Board questions the line settles (damage taken, pump duration, affordable) "
                "gain; deaths, blocks, triggers, targets and\nthe affected gate gain 0.13 AUC "
                "at most. Outlined: on dies at most 0.04, and less than nothing at width 256.",
                (0.01, top - 9.3),
                xycoords=("figure fraction", "data"), ha="left", va="top", fontsize=8.6,
                color=INK_SOFT, annotation_clip=False)
    cax = fig.add_axes([0.04, 0.6 / height, 0.2, 0.12 / height])
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=GAIN_CMAP, norm=norm), cax=cax,
                       orientation="horizontal", ticks=[-1, 0, 1])
    bar.ax.set_xticklabels(["worse than\nrandom", "same as\nrandom", "perfect over\nrandom"],
                           fontsize=7.2)
    bar.ax.set_title("colour: R² gain, or twice the AUC gain", fontsize=7.6, color=INK_SOFT)
    fig.text(0.3, 0.12 / height, "Cells: (rung 1 − rung 1w) × 100, MLP probe; board targets on "
             "card-disjoint records, line properties on held-out lines\n(* trained lines only, "
             "† all interaction pairs). A random vector's R² falls below zero, so an amount's "
             "gain can pass 100. gen-1 is probed on its own corpus.", fontsize=7.4,
             color=INK_SOFT)
    save(fig, "e-over-random")


# ── the activated ping's gate under edits ──────────────────────────────────


def ping_gate_edits() -> None:
    """The gate at the targeted creature of five activated damage records, under six edits."""
    p = csv("ping_gate_edits.csv").set_index("edit")
    labels = {
        "as collected": "as collected",
        "moved to the first main phase": "moved to the first main phase",
        "source untapped": "source untapped",
        "read as a spell: AB$ to SP$, cost dropped":
            "read as a spell: AB\\$ → SP\\$, cost dropped",
        "'any target' read as 'target creature'": "\"any target\" read as \"target creature\"",
        "a healer's 'prevent the next 1 damage' text in its place":
            "control: a healer's \"prevent the next\n1 damage\" text in its place",
    }
    edits = list(p.index)
    hits = [e for e in edits if "prevent" not in e]  # the edits that keep a damage ping
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    # The healer control sits below a gap: its rule is the opposite answer.
    ys = {edit: len(edits) - 1 - i + (0.0 if edit in hits else -0.35)
          for i, edit in enumerate(edits)}
    wording = "'any target' read as 'target creature'"
    if wording in ys:
        ax.axhspan(ys[wording] - 0.42, ys[wording] + 0.42, color="#efe9f7", lw=0, zorder=0)
    if len(hits) < len(edits):
        ax.axhline(min(ys[e] for e in hits) - 0.67, color=GRAY, lw=0.6)
    offsets = np.linspace(-0.18, 0.18, len(p.columns))
    for k, name in enumerate(p.columns):
        style = checkpoint_marker(name, 7)
        ax.plot([p.loc[e, name] for e in hits], [ys[e] + offsets[k] for e in hits],
                color=style["mec"], lw=0.9, alpha=0.45, zorder=2)
        ax.plot([p.loc[e, name] for e in edits], [ys[e] + offsets[k] for e in edits], zorder=3,
                label=short(name), **style)
    ax.axvline(0.5, color=INK, lw=0.8, zorder=1)
    for edit in edits:
        rule = 0.0 if "prevent" in edit else 1.0
        ax.plot(rule, ys[edit], "|", color=INK, ms=15, mew=2.2, zorder=4)
    ax.set_yticks([ys[e] for e in edits], [labels.get(e, e) for e in edits])
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(min(ys.values()) - 0.6, len(edits) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean predicted chance the targeted creature is affected (the gate), "
                  "five activated damage records; black tick = the rule")
    handles, names = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="|", color=INK, ls="none", ms=12, mew=2))
    names.append("the rule")
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=5,
              fontsize=8.3)
    ax.set_title("Only rewording \"any target\" as \"target creature\" puts the gate above 0.9 "
                 "on every checkpoint", loc="left", fontsize=10.5)
    save(fig, "ping-gate-edits")


# ── the block-legality labels ──────────────────────────────────────────────


def planned_block_labels() -> None:
    """How many "may not block" labels name a creature already in the AI's block plan.

    Each bar is one count of ``legality_planned_blocks.csv`` against the total it belongs
    to. The CSV gives no overlaps between the sub-counts, so they are not stacked.
    """
    t = csv("legality_planned_blocks.csv")
    t.columns = ["what", "count"]
    count = {w.strip(): c for w, c in zip(t.what, t["count"])}
    plan, rule, neutral = "#b8336a", "#4b2a86", "#a9a9a9"
    groups = [
        ("forbidden blockers listed", "of the {:,} \"may not block\" labels in real decisions", [
            ("naming a creature the snapshot shows blocking",
             "name a creature the AI already planned to block with", plan),
            ("naming no restricting static", "name a creature no static restricts", neutral),
            ("naming a tapped creature", "name a tapped creature, which the rules forbid",
             rule)]),
        ("blockers records", "of the {:,} real block-legality records", [
            ("whose snapshot shows blocks the AI had already planned",
             "show blocks the AI had already planned on the board", plan)]),
        ("attackers anchoring a record, per game and turn",
         "of the {:,} attackers anchoring a record in a game's turn", [
             ("anchoring more than one real record",
              "anchor more than one \"real\" decision, one per plan stage", plan)]),
    ]
    bars = []
    for total_key, title, parts in groups:
        if total_key not in count:
            continue
        bars.append(("title", title.format(count[total_key]), None, None))
        for key, label, colour in parts:
            if key in count:
                bars.append(("bar", label, count[key] / count[total_key], (count[key], colour)))
    height = 0.42 * len(bars) + 1.1
    fig = plt.figure(figsize=(9.0, height))
    ax = fig.add_axes([0.43, 0.45 / height, 0.5, 1 - 1.15 / height])
    y = 0.0
    for kind, label, share, extra in bars:
        if kind == "title":
            y += 0.2
            ax.text(-0.86, y, label[0].upper() + label[1:], fontsize=9, fontweight="bold",
                    color=INK, va="center")
            y += 0.7
            continue
        n, colour = extra
        ax.barh(y, 1, height=0.62, color="#eeeeee", lw=0)
        ax.barh(y, share, height=0.62, color=colour, lw=0)
        ax.text(share + 0.012, y, f"{share:.0%}", va="center", fontsize=10, fontweight="bold",
                color=INK)
        ax.text(-0.03, y, f"{label}  ({n:,})", va="center", ha="right", fontsize=8.4,
                color=INK)
        y += 0.8
    ax.set_ylim(y - 0.3, -0.3)
    ax.set_xlim(0, 1)
    ax.axis("off")
    fig.text(0.005, 1 - 0.12 / height, "Most \"may not block\" labels gen-2 trained on are the "
             "AI's own block plan; a tapped creature,\na refusal the rules make, is one label in "
             "fourteen", ha="left", va="top", fontsize=10.5, color=INK)
    fig.text(0.005, 0.1 / height, "Game-disjoint stratum, decisions the collector counted as "
             "real. The sub-counts may overlap, so they are not stacked.", fontsize=7.6,
             color=INK_SOFT)
    save(fig, "planned-block-labels")


def main() -> None:
    """Draw every chart, or only the ones named on the command line."""
    only = set(sys.argv[1:])
    for chart in (training_losses, loss_noise, probes_vs_loss, field_explained, damage_grids,
                  toughness_sweep_records, boundary_by_epoch, damage_sweep, combat_grids,
                  combat_keywords, gate2_heatmap, ladder, line_properties, board_abilities,
                  ablation, width_summary, other_sweeps, evasion_grid, evasion_ranking,
                  target_flips, colour_pips, zone_outcomes, zone_amounts, zone_who,
                  run_lineage, scorecard, loss_rise_by_field, e_over_random, ping_gate_edits,
                  planned_block_labels):
        if not only or chart.__name__ in only:
            chart()


if __name__ == "__main__":
    main()
