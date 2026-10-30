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

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import cache

import matplotlib

matplotlib.use("Agg")
import lineage  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _lib.style import BLUE, GRAY, GREEN, ORANGE, RED, use_probe_style  # noqa: E402
from chart_style import DATA, WIDTH_COLOUR, draw_requested, save, shade  # noqa: E402
from lineage import (  # noqa: E402
    CHECKPOINTS,
    LINES,
    NO_COPIES,
    RUN_LIST,
    RUNS,
    WIDTH_OF,
    WIDTHS,
)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

use_probe_style()
plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10.5, "legend.frameon": False,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "figure.titleweight": "bold",
})
INK, INK_SOFT = "#222222", "#555555"
# A run's phase in its width's line: the arm, run A (full rate), run B (a tenth).
PHASE_SHADE = {"arm": 0.62, "A": 0.82, "B": 1.0}
STAGE_MARKER = {"arm": "^", "A": "s", "B": "o"}
STAGE_LINE = {"arm": ":", "A": "--", "B": "-"}
STAGE_NAME = {"arm": "arm", "A": "A run, full rate", "B": "B run, a tenth of the rate"}
NO_COPIES_GRAY = "#9a9a9a"
#: Stage colours for the noisy per-epoch curves, where line styles cannot be followed: one
#: panel per width already gives the width, so every stage is a solid curve of its own colour.
STAGE_CURVE = {"none": NO_COPIES_GRAY, "arm": "#cc79a7", "A": "#d55e00", "B": "#1f1f1f"}
#: Per-epoch curves that must keep the width's hue mark the stage by lightness instead.
STAGE_CURVE_SHADE = {"arm": 0.4, "A": 0.68, "B": 1.0}
EARLIER_GRAY = "#c8c8c8"
GEN1_GRAY = "#4d4d4d"
#: The arms trained without lethality copies, earliest first.
NO_COPY_RUNS = [r.name for r in RUN_LIST if not r.copies]
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


@cache
def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def csv(name: str) -> pd.DataFrame:
    """The data file ``name``, read once; each caller gets a copy it may add columns to."""
    return _csv(name).copy()


def by_checkpoint(d: pd.DataFrame) -> pd.DataFrame:
    """``d`` indexed by checkpoint, a checkpoint read twice kept once."""
    d = d.set_index("checkpoint")
    return d[~d.index.duplicated()]


def darker(colour, amount: float = 0.3) -> tuple:
    """``colour`` blended toward black by ``amount``."""
    return tuple((1 - amount) * np.array(matplotlib.colors.to_rgb(colour)))


def run_of(checkpoint: str) -> str:
    return checkpoint if checkpoint in WIDTH_OF else checkpoint.split(" ")[0]


def stage_of(name: str) -> str:
    """``arm``, ``A`` or ``B`` for a run or checkpoint; ``none`` without copies; ``gen-1``."""
    run = run_of(name)
    return "gen-1" if run == "gen-1" else lineage.stage(RUNS[run])


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
        edge = WIDTH_COLOUR[WIDTH_OF[run_of(checkpoint)]]
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


@cache
def merged_pairs() -> dict[str, str]:
    """``{"X best": "X last"}`` for each run whose last epoch is its kept one."""
    ck = csv("checkpoints.csv").set_index("checkpoint")
    out = {}
    for name in ck.index:
        best = name.replace(" last", " best")
        if name.endswith(" last") and best in ck.index and ck.loc[best, "epoch"] == ck.loc[
                name, "epoch"]:
            out[best] = name
    return out


def is_merged(checkpoint: str) -> bool:
    """Whether ``checkpoint`` is one half of a run whose last epoch is its kept one."""
    return checkpoint in merged_pairs() or checkpoint in merged_pairs().values()


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


def top_down(items) -> list[tuple[int, object]]:
    """``(y, item)`` with the first item on the top row."""
    return [(len(items) - 1 - i, item) for i, item in enumerate(items)]


def band(ax, y: float, colour: str = "#f6f6f6", half: float = 0.42) -> None:
    """The shaded background of one row."""
    ax.axhspan(y - half, y + half, color=colour, lw=0, zorder=0)


def strip_frame(ax, labels: list[str], refs=(), xlim=None, top: float = 0.1,
                bottom: float = -0.5, ref_lw: float = 1.2, **tick_style) -> None:
    """Finish a chart of rows drawn by ``top_down``: the row labels, each reference line
    ``(x, label)`` named above the top row, and no grid across the rows."""
    n = len(labels)
    for x, label in refs:
        ax.axvline(x, color="black", lw=ref_lw)
        ax.text(x, n - 0.45, label, ha="center", va="bottom", fontsize=8)
    ax.set_yticks(range(n), list(labels)[::-1], **tick_style)
    ax.set_ylim(bottom, n - top)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)


@dataclass(frozen=True)
class Callout:
    """A label with a leader line on one checkpoint's marker of a row, drawn if it has one."""

    checkpoint: str
    dx: float
    dy: float
    ha: str = "left"
    text: str | None = None  # the checkpoint's name unless given


@dataclass
class StripRow:
    """One row of a strip panel: a marker per checkpoint at its value."""

    label: str
    values: pd.Series
    band_colour: str | None = "#f6f6f6"
    band_half: float = 0.42
    spread: float = 0.3
    ms: float = 6.5
    muted: bool = False
    alpha: float = 1.0
    callouts: tuple[Callout, ...] = ()
    tick_colour: str | None = None
    #: Drawn after the band and under the markers: ``extra(ax, y)``.
    extra: Callable | None = None


@dataclass
class StripPanel:
    """Rows of one marker per checkpoint, the first row on top, with reference lines."""

    rows: list[StripRow]
    names: list[str]
    refs: tuple = ()
    xlim: tuple | None = None
    top: float = 0.1
    bottom: float = -0.5
    ref_lw: float = 1.2
    tick_style: dict = dataclass_field(default_factory=dict)
    setup: Callable | None = None  # ``setup(ax)`` before the rows
    finish: Callable | None = None  # ``finish(ax)`` after the frame


def draw_strip_panel(ax, panel: StripPanel) -> None:
    """Draw ``panel``'s rows, their callouts and its frame onto ``ax``."""
    if panel.setup:
        panel.setup(ax)
    for y, row in top_down(panel.rows):
        if row.band_colour:
            band(ax, y, row.band_colour, row.band_half)
        if row.extra:
            row.extra(ax, y)
        points = strip(ax, y, row.values, panel.names, row.spread, row.muted, row.ms, row.alpha)
        for c in row.callouts:
            if c.checkpoint in points:
                callout(ax, *points[c.checkpoint], c.text or c.checkpoint, c.dx, c.dy, c.ha)
    strip_frame(ax, [row.label for row in panel.rows], panel.refs, panel.xlim, panel.top,
                panel.bottom, panel.ref_lw, **panel.tick_style)
    for tick, row in zip(ax.get_yticklabels()[::-1], panel.rows):
        if row.tick_colour:
            tick.set_color(row.tick_colour)
    if panel.finish:
        panel.finish(ax)


def strip_chart(name: str, panel: StripPanel, figsize: tuple, xlabel: str, title: str,
                title_pad: float = 16, title_style: dict | None = None,
                legend_anchor=(1.01, 1.0), extra_handles=()) -> None:
    """A strip panel alone on a figure, with the checkpoint legend at its side."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_strip_panel(ax, panel)
    ax.set_xlabel(xlabel)
    side_legend(ax, panel.names, legend_anchor, extra_handles)
    ax.set_title(title, pad=title_pad, **(title_style or {}))
    save(fig, name)


def side_legend(ax, names, anchor=(1.01, 1.0), extra_handles=()) -> None:
    ax.legend(handles=[*checkpoint_handles(names), *extra_handles], fontsize=8,
              loc="upper left", bbox_to_anchor=anchor)


def grouped_positions(groups: list, gap: float) -> list[float]:
    """One position per item, 1 apart, with ``gap`` more wherever the group changes."""
    out, x = [], 0.0
    for i, group in enumerate(groups):
        if i and group != groups[i - 1]:
            x += gap
        out.append(x)
        x += 1
    return out


def width_groups(widths: list, positions: list[float], order=WIDTHS) -> list[tuple]:
    """``(width, its positions)`` for each width in ``order`` that holds any."""
    return [(w, members) for w in order
            if (members := [p for wi, p in zip(widths, positions) if wi == w])]


def offsets(names: list[str], spread: float = 0.3) -> dict[str, float]:
    """A fixed vertical offset per checkpoint, earliest on top, so a row reads in order."""
    if len(names) == 1:
        return {names[0]: 0.0}
    return dict(zip(names, np.linspace(spread, -spread, len(names))))


def callout(ax, x: float, y: float, text: str, dx: float = 8, dy: float = 8,
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


def stage_handles(style, stages=("arm", "A", "B"), label: str = "{}") -> list:
    """One legend entry per training stage drawn as ``style(stage)``, named ``label`` with the
    stage's name filled in."""
    return [Line2D([], [], label=label.format(STAGE_NAME[s]), **style(s)) for s in stages]


def line_handles(widths, stages=("arm", "A", "B"), no_copies: bool = True) -> list:
    """Legend entries for run lines: one per width hue and one per stage's line style."""
    handles = [Line2D([], [], color=WIDTH_COLOUR[w], lw=2.5, label=f"{w}-wide e")
               for w in widths]
    handles += stage_handles(lambda s: dict(color=INK_SOFT, ls=STAGE_LINE[s], lw=1.5), stages)
    if no_copies:
        handles.append(Line2D([], [], color=NO_COPIES_GRAY, lw=1.2, label="arm without copies"))
    return handles


def strip(ax, y: float, values: pd.Series, names: list[str], spread: float = 0.3,
          muted: bool = False, ms: float = 6.5,
          alpha: float = 1.0) -> dict[str, tuple[float, float]]:
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
        elif alpha < 1:
            style["alpha"] = alpha
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
    brightness. A saturated blue still looks brighter than an orange of the same
    lightness, so the chroma tapers toward the break, where both hues meet muted.
    Chroma is lowered further only where a colour falls outside the sRGB gamut.
    """
    colours = []
    for v in np.linspace(0, 1, 256):
        lightness = 0.22 + 0.75 * v
        hue = 255.0 if v < 0.5 else 62.0
        chroma = 0.08 + 0.08 * min(1.0, abs(v - 0.5) / 0.3)
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


def heatmap(ax, grid: np.ndarray, title: str = "", xlabel: str = "", ylabel: str = "",
            cmap=DEATH_CMAP, norm=None):
    """An 8×8 grid indexed from 1; on the predicted-death scale unless ``norm`` is given."""
    scale = dict(norm=norm) if norm is not None else dict(vmin=0, vmax=1)
    image = ax.imshow(grid, cmap=cmap, origin="upper", **scale)
    ax.set_xticks(range(8), range(1, 9))
    ax.set_yticks(range(8), range(1, 9))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)
    return image


@dataclass(frozen=True)
class HeatCell:
    """One cell of a table of coloured numbers."""

    colour: object
    text: str
    white_text: bool  # the colour is dark enough to need it
    bold: bool = False


def heat_row(ax, y: float, xs: list[float], cells: list[HeatCell | None], size: float,
             edge_lw: float, missing_edge: str, missing_lw: float, missing_glyph: str = "") -> None:
    """One row of unit cells at ``xs``; a ``None`` cell is an empty white one."""
    for x, cell in zip(xs, cells):
        if cell is None:
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="white",
                                   edgecolor=missing_edge, lw=missing_lw))
            if missing_glyph:
                ax.text(x, y, missing_glyph, ha="center", va="center", fontsize=8, color=GRAY)
            continue
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=cell.colour,
                               edgecolor="white", lw=edge_lw))
        ax.text(x, y, cell.text, ha="center", va="center", fontsize=size,
                fontweight="bold" if cell.bold else "normal",
                color="white" if cell.white_text else INK)


def column_headers(ax, columns: list[tuple[str, str]], xs: list[float], label_y: float,
                   marker_y: float, marker_ms: float, size: float) -> None:
    """Over each ``(checkpoint, label)`` column, its marker and its label read upward."""
    for (name, label), x in zip(columns, xs):
        ax.plot(x, marker_y, clip_on=False, **checkpoint_marker(name, marker_ms))
        ax.text(x, label_y, label, rotation=90, ha="center", va="bottom", fontsize=size,
                color=INK)


NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen", "twenty"]


def number_word(n: int) -> str:
    """``n`` in words up to twenty, for a title that counts what the data holds."""
    return NUMBER_WORDS[n]


def probe_mean() -> pd.Series:
    """Per checkpoint, the mean of the model's own rung-3 score over the game-disjoint probe
    targets, damage taken left out (its rung 3 is rescored as the amount alone)."""
    lad = csv("probe_ladder.csv")
    own = lad[(lad.rung == "3") & (lad.probe == "model") & (lad.target != "damage_taken")
              & (lad.stratum == "game-disjoint")]
    return own.groupby("checkpoint").score.mean()


def grid_errors(grids: pd.DataFrame) -> pd.Series:
    """Per checkpoint, the mean absolute difference of its probe-set damage grid from the
    rule's 0 or 1 over the 64 cells."""
    grid = grids[grids.record_set == "probe set"]
    rule = (grid.damage >= grid.toughness).astype(float)
    return (grid.predicted_death - rule).abs().groupby(grid.checkpoint).mean()


def on_rule(sweep: pd.DataFrame) -> bool:
    """Whether every point of one checkpoint's damage sweep, which sets a damage to a
    toughness-4 target, is on the rule's side of an even chance: below it under 4 damage,
    above it from 4 up."""
    return bool(((sweep.value < 4) == (sweep.prediction < 0.5)).all())


def ping_predictions() -> pd.DataFrame:
    """Predicted death of a toughness-1 creature under each activated 1-damage ping of the
    toughness sweep: a row per checkpoint, a column per ping record."""
    sr = csv("sweep_records.csv")
    meta = csv("toughness_sweep_records.csv")
    pings = meta[(meta.kind == "activated") & (meta.damage == 1)].record
    d = sr[(sr.sweep == "toughness") & sr.record.isin(pings) & (sr.value == 1)]
    return d.pivot(index="checkpoint", columns="record", values="prediction")


def share_at_old_target(who: pd.DataFrame, when: str) -> pd.Series:
    """Per row of ``zone_who.csv``, the share of the predicted cards at the originally
    targeted player, ``when`` being "before" or "after" the target is handed over."""
    targeted = who[f"targeted_{when}"]
    return targeted / (targeted + who[f"other_{when}"])


#: The first run of arm e64 scored first and double strike on the wrong damage step.
WRONG_STEP_CHECKPOINT = "e64 (earlier inputs)"
WRONG_STEP_KEYWORDS = ["first_strike", "double_strike"]


def death_rule() -> np.ndarray:
    """The rule's damage-by-toughness grid: 1 where the damage reaches the toughness."""
    return np.array([[1.0 if p >= t else 0.0 for t in range(1, 9)] for p in range(1, 9)])


# ── training ───────────────────────────────────────────────────────────────


def training_losses() -> None:
    ep = csv("epochs.csv")
    widths = [w for w in WIDTHS if LINES[w][0] in set(ep.run)]
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
                ax.plot(d.epoch, d[metric], color=STAGE_CURVE["none"], lw=1.3)
            for run in LINES[width]:
                d = ep[ep.run == run].sort_values("epoch")
                if d.empty:
                    continue
                stage = stage_of(run)
                ax.plot(d.epoch, d[metric], color=STAGE_CURVE[stage], lw=1.3)
                best = d.loc[d.card_disjoint.idxmin()]
                ax.plot(best.epoch, best[metric], STAGE_MARKER[stage], ms=9,
                        color=STAGE_CURVE[stage], mec="white", mew=1.0, zorder=5)
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
    handles = [Line2D([], [], color=STAGE_CURVE["none"], lw=1.3, label="arm without copies")]
    handles += stage_handles(lambda s: dict(color=STAGE_CURVE[s], lw=1.3,
                                            marker=STAGE_MARKER[s], ms=7, mec="white"))
    handles += [Line2D([], [], ls="none", label="large marker: the run's lowest "
                       "card-disjoint epoch")]
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
    widths = [w for w in WIDTHS if all(r in sd.index for r in LINES[w])]
    bare = [r for r in NO_COPY_RUNS if r in sd.index]
    bare_x = [-1.1 + 0.25 * k for k in range(len(bare))]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    for ax, (metric, label) in zip(axes, (("card_disjoint", "card-disjoint loss"),
                                          ("game_disjoint", "game-disjoint loss"),
                                          ("train", "training loss"))):
        for x, run in zip(bare_x, bare):
            ax.plot(x, sd.loc[run, metric], **checkpoint_marker(run, 6))
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
        # The arms without copies are named one by one, since they share one grey.
        ax.set_xticks(bare_x + [0, 1, 2], bare + ["arm", "A run\nfull rate", "B run\na tenth"])
        for tick in ax.get_xticklabels()[:len(bare)]:
            tick.set(rotation=90, fontsize=7.5, color=INK_SOFT)
        ax.axvline(-0.35, color=GRAY, lw=0.6)
        ax.text(-0.75, 0.0105, "without\ncopies", ha="center", va="bottom", fontsize=7.5,
                color=INK_SOFT)
        ax.set_xlim(-1.3, 2.55)
        ax.set_title(label)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("standard deviation around the\nfive-epoch moving mean (log scale)")
    ticks = [0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]
    axes[0].set_yticks(ticks, [f"{t:g}" for t in ticks])
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.legend(handles=checkpoint_handles(bare + [r for w in widths for r in LINES[w]],
                                          fill=False),
               loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.29), fontsize=8.5)
    fig.suptitle("Epoch-to-epoch swing of each loss, from epoch 10 on; ÷ is how many times "
                 "smaller the B run's swing is than the A run's.\nRuns B64, B128 and B256 train "
                 "at a tenth of the learning rate", x=0.01, ha="left", y=1.06)
    save(fig, "loss-noise")


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
    ck = csv("checkpoints.csv").set_index("checkpoint")
    ck["probe"] = probe_mean()
    # A checkpoint whose probes have not run yet has no point to draw.
    names = gen2(ck.dropna(subset=["probe"]).index)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.4), sharey=True)
    for ax, (loss, label) in zip(axes, (("card_disjoint", "card-disjoint loss"),
                                        ("game_disjoint", "game-disjoint loss"))):
        for run in (r.name for r in RUN_LIST if r.parent):
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
                            color=WIDTH_COLOUR[WIDTH_OF[run]])
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
                            color=WIDTH_COLOUR[256], fontweight="bold")
        ax.set_xlabel(f"{label} (lower is better, to the left)")
        ax.set_title(f"against the {label}")
    axes[0].set_ylabel("mean of the model's own probe scores\n(AUC, game-disjoint records; "
                       "higher is better)")
    axes[0].legend(handles=checkpoint_handles(names), fontsize=7.5, loc="lower left", ncol=2)
    fig.suptitle("Probes against each validation loss: each arrow goes from a run's lowest "
                 "card-disjoint epoch to its last", x=0.01, ha="left", y=0.96)
    save(fig, "probes-vs-loss")


def field_explained() -> None:
    """Card-disjoint explained deviance per field and epoch, the runs with copies by width.

    The curves are too noisy to follow as dots and dashes, so every curve is solid at one
    thickness: the hue gives the width and the lightness the stage, light arm to dark B run.
    """
    fe = csv("fields_by_epoch.csv")
    fields = [("gate", "affected gate"), ("zone_outcome", "zone outcome"),
              ("keywords_gained", "keywords gained"), ("blocker_legal", "blocking legality"),
              ("types_gained", "types gained"), ("poison_delta", "poison")]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharex=True)
    widths = set()
    for ax, (field, label) in zip(axes.flat, fields):
        low = 0
        for width in WIDTHS:
            for run in LINES[width]:
                d = fe[(fe.run == run) & (fe.field == field)].sort_values("epoch")
                if not d.empty:
                    colour = shade(WIDTH_COLOUR[width], STAGE_CURVE_SHADE[stage_of(run)])
                    ax.plot(d.epoch, d.explained_pct, color=colour, lw=1.3)
                    low = min(low, d.explained_pct.min())
                    widths.add(width)
        ax.set_title(label)
        ax.axhline(0, color=INK_SOFT, lw=0.8)
        bottom = min(-40, 10 * np.floor(low / 10) - 5)
        ax.set_ylim(bottom, 100)
        if bottom < -40:
            ax.set_title(f"{label}  (scale to −{abs(bottom):.0f}, not −40)")
    axes[1][2].text(0.98, 0.04, "below 0: worse than\na constant prediction",
                    transform=axes[1][2].transAxes, ha="right", fontsize=7.5, color=INK_SOFT)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of the constant\nprediction's loss removed")
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    handles = [Line2D([], [], color=shade(WIDTH_COLOUR[w], STAGE_CURVE_SHADE[s]), lw=1.3,
                      label=f"{w}-wide e, {STAGE_NAME[s]}")
               for w in sorted(widths) for s in ("arm", "A", "B")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12),
               fontsize=8.5)
    fig.suptitle("Card-disjoint explained deviance by field, the runs with lethality copies of "
                 "each width",
                 x=0.01, ha="left", y=0.96)
    save(fig, "fields-explained")


# ── the death rule ─────────────────────────────────────────────────────────

#: The toughness-sweep grid's layout: one row per width, one column per stage.
STAGE_GRID = [[NO_COPIES.get(w), arm, f"{a} best", f"{a} last", f"{b} best", f"{b} last"]
              for w, (arm, a, b) in LINES.items()]
STAGE_COLUMNS = ["no copies", "arm with copies", "A run, kept epoch", "A run, last epoch",
                 "B run, kept epoch", "B run, last epoch"]


def stage_column_titles(axes, height: float) -> None:
    """Over the top row of a ``STAGE_GRID`` of panels, the name of each column."""
    for c, label in enumerate(STAGE_COLUMNS):
        axes[0][c].annotate(label, (0.5, height), xycoords="axes fraction", ha="center",
                            fontsize=9.5, fontweight="bold")


def damage_grids() -> None:
    """The damage grid of every checkpoint, laid out as the toughness sweep's grid."""
    all_grids = csv("damage_grids.csv")
    errors = grid_errors(all_grids)
    g = all_grids[all_grids.record_set == "probe set"]
    grids = {name: g[g.checkpoint == name].pivot(index="damage", columns="toughness",
                                                 values="predicted_death").values
             for name in set(g.checkpoint)}
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
            heatmap(ax, grids[name])
            rule_staircase(ax)
            ax.set_title(f"{short(name)}\nerror {errors[name]:.3f}", fontsize=9,
                         fontweight="bold" if name == lowest else "normal",
                         loc="center")
            ax.tick_params(labelsize=7, length=0)
            if c == 0 or (r == 2 and c == 1):
                ax.set_ylabel(f"{WIDTHS[r]}-wide e\ndamage", color=WIDTH_COLOUR[WIDTHS[r]],
                              fontweight="bold")
            if r == 2 or name in ("e128",):
                ax.set_xlabel("toughness", fontsize=8.5)
    stage_column_titles(axes, 1.3)
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
    groups = [("1", 1, BLUE, "1, spell or trigger", (0, (4, 4))),
              ("1 activated", 1, "#8e5bb5", "1, activated ping", (4, (4, 4))),
              ("2", 2, GREEN, "2", ":"), ("3", 3, ORANGE, "3", ":"), ("5", 5, RED, "5", ":")]
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
            axes[r][first].set_ylabel(f"e width {WIDTHS[r]}\npredicted death", fontsize=9)
            axes[r][first].yaxis.set_tick_params(labelleft=True)
    stage_column_titles(axes, 1.22)
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
    widths = [w for w in WIDTHS if LINES[w][1] in set(b.run)]
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
    handles += stage_handles(lambda s: dict(color=INK_SOFT, ls=STAGE_LINE[s], lw=1.4))
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


#: The damage sweep's arms without copies, one grey each, lightest the earliest inputs.
NO_ARMS_GREY = {"e64 (earlier inputs)": "#bdbdbd", "e64": "#8a8a8a", "e128": "#4f4f4f"}


def _sweep_panel(ax, s: pd.DataFrame, names: list[str]) -> None:
    """One damage-sweep panel over grey wrong-side quadrants; ✓ marks a curve on the rule."""
    for (x0, x1), (y0, y1) in (((0.5, 3.5), (0.5, 1)), ((3.5, 8.5), (0, 0.5))):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#e4e4e4", lw=0, zorder=0))
    ax.text(0.7, 0.96, "wrong side:\npredicted to die", fontsize=7.5, color=INK_SOFT, va="top")
    ax.text(8.3, 0.04, "wrong side:\npredicted to survive", fontsize=7.5, color=INK_SOFT,
            ha="right")
    for name in names:
        d = s[s.checkpoint == name].sort_values("value")
        right_side = on_rule(d)
        line = run_line(name)
        style = checkpoint_marker(name, 5.5 if right_side else 4.5)
        style.update(line, lw=2.2 if right_side else 1.0)
        if stage_of(name) == "none":
            # The grey arms without copies are dotted like every arm and tell apart by
            # their shade, lightest the earliest inputs.
            grey = NO_ARMS_GREY.get(name, NO_COPIES_GRAY)
            style.update(ls=":", lw=1.6, color=grey, mec=darker(grey, 0.25))
        elif stage_of(name) == "gen-1":
            style.update(ls="-", lw=1.4)
        ax.plot(d.value, d.prediction, **style,
                alpha=1.0 if right_side else 0.55, zorder=3 if right_side else 2,
                label=("✓ " if right_side else "    ") + short(name))
    ax.axvline(3.5, color="black", lw=0.8)
    ax.axhline(0.5, color="black", lw=0.8)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, 9))
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=2.6)


#: The marks of a damage sweep chart, below its first line.
SWEEP_KEY = ("✓ and bold: every point on the rule's side of an even chance\n(dotted ▲ arm, "
             "dashed ■ A run, solid ● B run; filled the kept epoch, hollow the last)")


def sweep_chart(sweep: str, chart: str, xlabel: str, title: str, title_y: float) -> None:
    """A damage sweep of every checkpoint, one panel per width and one for no copies.

    The rule kills a toughness-4 target from 4 damage up, so a curve is on the right
    side below one half left of the step and above one half right of it, however soft.
    """
    s = csv("sweeps.csv")
    s = s[s.sweep == sweep]
    available = set(s.checkpoint)
    panels = [("without lethality copies", None, present(["gen-1", *NO_COPY_RUNS], available))]
    for width in WIDTHS:
        names = [n for n in gen2(available, copies_only=True) if WIDTH_OF[run_of(n)] == width]
        if names:
            panels.append((f"{width}-wide e, with copies", width, names))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True, sharey=True)
    for ax, (panel, width, names) in zip(axes.flat, panels):
        _sweep_panel(ax, s, names)
        ax.set_title(panel, color=WIDTH_COLOUR.get(width, INK))
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted death,\ntoughness-4 target")
    fig.subplots_adjust(wspace=0.55, hspace=0.25)
    fig.suptitle(title, x=0.01, ha="left", y=title_y)
    save(fig, chart)


def damage_sweep() -> None:
    sweep_chart("damage_spell", "damage-sweep", "NumDmg$ of the spell",
                f"Damage sweep: the rule kills from 4 damage up. {SWEEP_KEY}", 0.98)


def damage_sweep_triggered() -> None:
    """The damage sweep on 20 triggered damage abilities, laid out as the spell's sweep.

    The sweep sets every literal ``NumDmg$`` of the acting text, which on a trigger sits
    in the sub-ability its ``Execute$`` names; the trigger's description keeps the amount
    it was collected with. The target's toughness is fixed at 4, so the rule's step is
    between 3 and 4 damage, as on the spell.
    """
    sweep_chart("damage_triggered", "damage-sweep-triggered",
                "NumDmg$ in the trigger's sub-ability line",
                "Damage sweep on a triggered ability, whose amount sits in a sub-ability's "
                "script line: with copies, every checkpoint's\npredicted death rises with it. "
                f"The rule kills from 4 damage up. {SWEEP_KEY}", 1.0)


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
        im = heatmap(ax, diff, "", xlabel, ylabel, cmap=DIVERGING_CMAP, norm=norm)
        ax.set_title(title, fontsize=9.5)
        for i in range(8):
            for j in range(8):
                text = f"{diff[i, j]:+.1f}".replace("-", "−")
                ax.text(j, i, "0" if text[1:] == "0.0" else text, ha="center", va="center",
                        fontsize=6.5, color="white" if abs(diff[i, j]) > 0.6 else INK)
        rule_staircase(ax)
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
    names = gen2(c.checkpoint)
    rows = []
    for label, variant, reads, cells, rule in _combat_checks(c):
        plain = c[(c.variant == "none") & (c.reads == reads) & cells]
        kw = c[(c.variant == variant) & (c.reads == reads) & cells]
        p = plain.groupby("checkpoint").predicted_death.mean()
        k = kw.groupby("checkpoint").predicted_death.mean()
        callouts = ()
        if variant == "attacker-double-strike-second-hit":
            callouts = (Callout("A64 last", 10, 10), Callout("B64 last", 10, -12))
        if variant.endswith("first-strike"):
            callouts = (Callout("e64 (earlier inputs)", 10, 6),)
        rows.append(StripRow(label, ((k - p) / (rule - p)).dropna(), callouts=callouts))
    panel = StripPanel(rows, names, refs=((0, "keyword ignored"), (1, "the rule's full change")),
                       xlim=(-0.15, 1.12), ref_lw=1.0, tick_style=dict(fontsize=8.5))
    strip_chart("combat-keywords", panel, (10.4, 5.6),
                "share of the rule's change the keyword produces: (with − without) / "
                "(rule − without),\nmean predicted death over the cells where the keyword "
                "changes the rule's answer",
                "Does a keyword move the predicted death as far as the rule says? One marker "
                "per checkpoint")


def gate2_heatmap() -> None:
    """Gate 2 as a strip per keyword, the band showing what sampling alone would spread."""
    g = csv("gate2.csv")
    names = gen2(g.checkpoint)
    kws = ["lifelink", "indestructible", "trample", "first_strike", "deathtouch", "infect",
           "double_strike"]
    g = g[~((g.checkpoint == WRONG_STEP_CHECKPOINT) & g.keyword.isin(WRONG_STEP_KEYWORDS))]
    n = g.groupby("keyword").qualifying_records.median().astype(int)

    def sampling_band(med: float, half: float):
        def draw(ax, y):
            ax.add_patch(Rectangle((med - half, y - 0.4), 2 * half, 0.8, color="#dedede", lw=0,
                                   zorder=1))
            ax.plot([med, med], [y - 0.4, y + 0.4], color=GRAY, lw=1.0, zorder=2)
        return draw

    callouts = {"first_strike": (Callout("A256 best", -12, -8, "right", "A256 best = last"),),
                "double_strike": (Callout("B64 best", 12, 12), Callout("B64 last", 12, 10))}
    rows = []
    for kw in kws:
        d = by_checkpoint(g[g.keyword == kw]).agreement_pct
        med = d[present(names, d.index)].median()
        half = 196 * np.sqrt(med / 100 * (1 - med / 100) / n[kw])
        rows.append(StripRow(f"{kw.replace('_', ' ')}\n{n[kw]:,} records", d, band_colour=None,
                             spread=0.32, callouts=callouts.get(kw, ()),
                             extra=sampling_band(med, half)))

    def gate_line(ax) -> None:
        ax.axvline(70, color="black", lw=1.2)
        ax.text(70, len(kws) - 0.4, " gate passes from 70%", fontsize=8, va="bottom")

    strip_chart("gate2-keywords", StripPanel(rows, names, xlim=(40, 101), bottom=-0.6,
                                             finish=gate_line), (10.4, 5.2),
                "direction agreement, %: how often a keyword moves the prediction the way it "
                "moves the real outcome",
                "Gate 2 by keyword, one marker per checkpoint: a spread wider than the grey band "
                "is the model's, not sampling's",
                extra_handles=[Patch(color="#dedede", label="median ± 95% sampling "
                                                            "half-width")])


# ── where knowledge lives ──────────────────────────────────────────────────

LADDER_TARGETS = {"affordable": "affordable", "pt_until_end_of_turn": "pump lasts until EOT",
                  "legal_target": "legal target",
                  "legal_target_protected": "legal target, protected",
                  "fires": "trigger fires", "affected": "card affected", "dies": "dies",
                  "may_block": "may block", "damage_taken": "damage taken"}
#: The rungs, bottom to top of the model, with each one's mark.
#: Each rung is drawn as its own name, so that no rung borrows a checkpoint's stage shape.
RUNGS = [("0", "rung 0: board alone", dict(marker=r"$\mathsf{0}$", color=GRAY, mec=GRAY)),
         ("1w", "rung 1w: board + random vector of e's width",
          dict(marker=r"$\mathsf{1w}$", color=GRAY, mec=GRAY)),
         ("1", "rung 1: board + e", dict(marker=r"$\mathsf{1}$", color=PURPLE, mec=PURPLE)),
         ("1o", "rung 1o: board + the true values (oracle)",
          dict(marker=r"$\mathsf{1o}$", color=PURPLE_SOFT, mec=PURPLE)),
         ("2", "rung 2: probe on the trunk's vector",
          dict(marker=r"$\mathsf{2}$", color=SLATE, mec=SLATE)),
         ("3", "rung 3: the model's own answer",
          dict(marker=r"$\mathsf{3}$", color="black", mec="black"))]
#: A text marker needs more room than a shape to read: two-character names wider still.
RUNG_MS = {"0": 8, "1": 8, "2": 8, "3": 10, "1w": 13, "1o": 13}
RUNGS_BY_NAME = {rung: style for rung, _, style in RUNGS}
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
                    band(a, y, "#fbf3dc", 0.47)
            d = t.loc[target]
            for k, (rung, _, style) in enumerate(RUNGS):
                yy = y + (2.5 - k) * step
                ax.plot([d[rung].min(), d[rung].max()], [yy, yy], color=style["mec"], lw=1.4,
                        alpha=0.55, zorder=2, solid_capstyle="butt")
                ax.plot(d[rung].median(), yy, ls="none", ms=RUNG_MS[rung], mew=0.3, zorder=3,
                        **style)
            share = d["share"]
            if defined[target]:
                strip(sax, y, share.clip(-0.4, 1.25), names, spread=0.3, ms=6)
                # The shares rungs 1w and 1o close: a tick across the row named by its rung,
                # 1w named above the row and 1o below it so that close ticks stay legible.
                for col, rung, end in (("share_w", "1w", 0.33), ("share_o", "1o", -0.33)):
                    colour = RUNGS_BY_NAME[rung]["mec"]
                    v = med.loc[target, col]
                    x = np.clip(v, -0.37, 1.22)
                    sax.plot([x, x], [y - 0.33, y + 0.33], color=colour, lw=1.6, alpha=0.8,
                             ls="-" if rung == "1w" else (0, (2, 1.2)), zorder=1.5)
                    sax.text(x, y + end, rung if v >= -0.37 else f"{rung} at {v:.2f}",
                             ha="center" if v >= -0.37 else "left", fontsize=7.2,
                             va="bottom" if end > 0 else "top", color=colour,
                             fontweight="bold")
                closes = med.loc[target, "share_o"]
                closes = f"{closes:.0%}" if closes > 0 else "none"
                note = (f"oracle rung closes {closes} of the gap:\n"
                        "share not interpreted" if target in flagged else "")
                sax.text(1.24, y - 0.44, note, ha="right", va="bottom", fontsize=7.2,
                         color="#8a6a00", style="italic")
            else:
                sax.text(0.42, y, "no share: the model scores no better than the board alone",
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
    handles = [Line2D([], [], ls="none", ms=RUNG_MS[r], mew=0.3, label=lab, **style)
               for r, lab, style in RUNGS]
    handles += [Line2D([], [], color=GRAY, lw=1.6,
                       label="(b) 1w: the share rung 1w closes, median"),
                Line2D([], [], color=PURPLE, lw=1.6, ls=(0, (2, 1.2)),
                       label="(b) 1o: the share rung 1o closes, median"),
                Patch(color="#fbf3dc", label=f"share under 0.9 and the oracle closes under "
                                             f"{ORACLE_FLOOR:.0%} of the gap: share not read")]
    fig.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0.03, -0.045),
               fontsize=8)
    fig.legend(handles=checkpoint_handles(names), loc="lower left", ncol=6,
               bbox_to_anchor=(0.03, -0.085), fontsize=8)
    fig.suptitle("Where the knowledge lives, card-disjoint records, the "
                 f"{len(names)} gen-2 checkpoints trained with lethality copies", x=0.01,
                 ha="left", y=0.95)
    save(fig, "probe-ladder")


LINE_TARGETS = [("tap_cost", "tap cost"), ("sacrifice_cost", "sacrifice cost"),
                ("instant_speed", "instant speed"), ("repeatable", "repeatable")]
LINE_AMOUNTS = [("cost_total", "total mana cost"), ("damage", "damage amount"),
                ("counters_placed", "counters placed"), ("power_change", "power change"),
                ("toughness_change", "toughness change")]
#: The probes record's state-dependence label, drawn below the amounts.
LINE_STATE = [("spread", "outcome spread\nacross boards")]


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
                                    (axes[1], LINE_AMOUNTS + LINE_STATE, -0.55, 0.0)):
        targets = [t for t in targets if t[0] in set(d.target)]
        if targets[-1] in LINE_STATE:
            # The outcome spread is a label of the text's outcomes, not a stated amount.
            ax.axhline(0.5, color=INK_SOFT, lw=0.8)
            ax.text(1.0, 0.47, "state dependence", fontsize=7.8, color=INK_SOFT,
                    style="italic", va="top", ha="right")
        for i, (y, (target, _)) in enumerate(top_down(targets)):
            band(ax, y, "#f6f6f6" if i % 2 else "white", half=0.45)
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
        strip_frame(ax, [t[1] for t in targets], xlim=(lo, 1.01), top=0.5, fontsize=9.5,
                    fontweight="bold")
        ax.axvline(chance, color="black", lw=1.2)
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
                ("attacker_legal", "legality-keyword",
                 "attack legality: a keyword restricts\n(in the flags: a control)"),
                ("blocker_legal", "legality-static", "block legality: a static restricts"),
                ("blocker_legal", "legality-keyword",
                 "block legality: a keyword restricts\n(in the flags: a control)"),
                ("zone_outcome", "resolution-replacement",
                 "zone outcome: a replacement elsewhere"),
                ("zone_outcome", "resolution-static", "zone outcome: a static elsewhere")]


def board_abilities() -> None:
    """Each checkpoint's loss where a line restricts, over its loss where nothing does."""
    b = csv("board_abilities.csv")
    names = gen2(b.checkpoint)
    rows = []
    for field_name, subset, label in BOARD_RATIOS:
        control = subset.split("-")[0] + "-none"
        d = b[b.field == field_name].drop_duplicates(["checkpoint", "subset"]).pivot(
            index="checkpoint", columns="subset", values="loss")
        ratio = d[subset] / d[control]
        r = ratio[present(names, ratio.index)]

        def range_note(ax, y, r=r):
            ax.text(1.0, y + 0.03, f" ×{r.min():.2g} to ×{r.max():.2g}", fontsize=8,
                    color=INK_SOFT, va="center", transform=ax.get_yaxis_transform(), ha="left")

        # The keyword rows are controls: the keyword flags show those restrictions.
        keyword = subset == "legality-keyword"
        rows.append(StripRow(label, ratio, band_colour="#efe9f7" if keyword else "#f6f6f6",
                             spread=0.28, tick_colour=INK_SOFT if keyword else INK,
                             extra=range_note))

    def log_axis(ax) -> None:
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4], ["×0.1", "×0.25", "×0.5", "×1", "×2", "×4"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    panel = StripPanel(rows, names, refs=((1, "as costly as no restriction"),), setup=log_axis)
    strip_chart("board-abilities", panel, (10.2, 4.6),
                "field loss on the restricted records ÷ field loss on the unrestricted control, "
                "same checkpoint (log scale)",
                "On attack legality a restriction the board snapshot does not list raises every "
                "checkpoint's loss,\nand one the keyword flags show lowers it (shaded purple: "
                "the flag rows, controls)", legend_anchor=(1.12, 1.0))


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
        for i, (y, (field, _)) in enumerate(top_down(ABLATION_FIELDS)):
            band(ax, y, "#f6f6f6" if i % 2 else "white", half=0.45)
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
    probe = probe_mean()
    sd = loss_noise_table()
    measures = [(ck.card_disjoint, "card-disjoint loss", "lower is better", "card_disjoint"),
                (ck.game_disjoint, "game-disjoint loss", "lower is better", "game_disjoint"),
                (probe, "probe mean, game-disjoint", "higher is better; no noise estimate,\n"
                 "and the axis spans a few hundredths", None),
                (g3.participation_ratio, "participation ratio of e",
                 "directions e uses; no noise estimate", None)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0))
    for ax, (series, title, sense, noise) in zip(axes, measures):
        tops = []
        for width in WIDTHS:
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
        ax.set_title(f"{title}\n\n", fontsize=10)
        ax.text(0, 1.02, sense, transform=ax.transAxes, fontsize=8.2, color=INK_SOFT,
                va="bottom")
    handles = line_handles(WIDTHS, stages=(), no_copies=False)
    handles += [Line2D([], [], color=INK_SOFT, lw=1.8, marker="o", label="kept epoch (filled)"),
                Line2D([], [], color=INK_SOFT, lw=1.0, ls="--", marker="o", mfc="white",
                       label="last epoch (hollow)"),
                Patch(color="#d9d9d9", label="one standard deviation of the epoch-to-epoch "
                                             "swing, above the stage's lowest kept loss")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1),
               fontsize=8.5)
    fig.suptitle("The three widths at each stage of training, every checkpoint trained with "
                 "lethality copies (▲ arm, ■ A run, ● B run; B256's ◐ is its own lowest epoch, "
                 f"{ck.epoch['B256 best']})", x=0.01, ha="left", y=1.13)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "width")


def other_sweeps() -> None:
    s = csv("sweeps.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    names = gen2(s[s.sweep == "affordability"].checkpoint)
    affordability = s[(s.sweep == "affordability") & (s.value <= 8)]
    for name in names:
        d = affordability[affordability.checkpoint == name]
        ax.plot(d.value, d.prediction, marker=checkpoint_marker(name)["marker"], ms=3,
                **run_line(name), alpha=0.9)
    for v, n in affordability.groupby("value").records.max().items():
        ax.text(v, -0.06, str(n), ha="center", fontsize=7, color=INK_SOFT)
    ax.axhline(0.5, color="black", lw=1.2)
    ax.text(0.1, 0.51, "even chance", fontsize=8, va="bottom")
    top = affordability[affordability.checkpoint.isin(names)]
    top = top.loc[top.prediction.idxmax()]
    if top.prediction >= 0.5:
        callout(ax, top.value, top.prediction, f"{top.checkpoint}: {top.prediction:.3f}", dx=-14,
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
    # The rule leaves the scale at three creatures; its value at eight sits at the edge.
    ax2.annotate("the rule at 8 creatures:\n8 deaths, off the scale", (8, 2.98),
                 xytext=(8, 2.05), ha="right", va="center", fontsize=8, color=INK,
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, shrinkA=0, shrinkB=0,
                                 relpos=(1.0, 1.0)))
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


def evasion_lifts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Median lift per cell, the count of checkpoints beating the no-keyword row, the rule,
    and the median of each checkpoint's lift divided by its no-keyword row's lift.

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
    relative = pd.DataFrame(np.median(stack / stack[:, [0], :], axis=0), EVASION_ROWS,
                            EVASION_COLUMNS)
    rule = g[g.checkpoint == names[-1]].pivot(index="attacker_keyword", columns="blocker_trait",
                                              values="rule_may_block")
    return median, beats, rule.loc[EVASION_ROWS, EVASION_COLUMNS], relative


def evasion_grid() -> None:
    """One heatmap of the median lift; the no-keyword row is what each column compares to.

    The text is the absolute lift; the colour is the lift against the reference row's in
    the same column, the comparison every reading of the grid makes, so the reference row
    itself is neutral.
    """
    median, beats, rule, relative = evasion_lifts()
    n = len(gen2(csv("evasion_grid.csv").checkpoint))
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    norm = matplotlib.colors.TwoSlopeNorm(0, -1, 3)
    im = ax.imshow(np.log2(relative.values), cmap=DIVERGING_CMAP, norm=norm, aspect="auto")
    for i, row in enumerate(EVASION_ROWS):
        for j, col in enumerate(EVASION_COLUMNS):
            v = median.loc[row, col]
            dark = np.log2(relative.loc[row, col]) > 1.6
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
                       label="colour: the cell's lift ÷ the top row's lift, same column")
    ticks = [-1, 0, 1, 2, 3]
    bar.set_ticks(ticks, labels=[f"×{2 ** t:g}" for t in ticks])
    ax.set_title("Evasion: how much a blocker trait lifts the predicted chance of a block (the "
                 f"text),\nmedian over {n} checkpoints. The colour compares each lift with the "
                 f"top row's in its column;\nn/{n} counts the checkpoints where the trait lifts "
                 "more against the keyword than against\na plain attacker. ◥ the rule allows the "
                 "block; ◥ hollow: only when the attacker is black too", fontsize=10, pad=10)
    save(fig, "evasion-grid")


def evasion_ranking() -> None:
    """Per attacker keyword: AUC of allowed over forbidden blocks, one marker per checkpoint."""
    r = csv("evasion_ranking.csv")
    keywords = [k for k in ["none", "flying", "horsemanship", "fear", "intimidate", "shadow",
                            "protection-other", "all rows"] if k in set(r.attacker_keyword)]
    names = gen2(r.checkpoint) + (["gen-1"] if "gen-1" in set(r.checkpoint) else [])
    rows = [StripRow("pooled over every row" if k == "all rows" else k,
                     by_checkpoint(r[r.attacker_keyword == k].dropna(subset=["auc"])).auc,
                     band_colour="#f0f0f0" if k == "all rows" else "#f8f8f8")
            for k in keywords]
    panel = StripPanel(rows, names, refs=((0.5, "chance"),), xlim=(0.15, 1.0))
    strip_chart("evasion-ranking", panel, (10.2, 5.0),
                "AUC: allowed blocks ranked above forbidden ones (0.5 = chance)",
                "Does the model rank the blocks each evasion keyword allows above the ones it "
                "forbids?")


#: Rows resting on fewer items than this are drawn faint: too few to weigh.
FEW = 20


def target_flips() -> None:
    """The change in predicted legality of a legal target when its one condition breaks.

    One marker per checkpoint at its own after − before, so a checkpoint's two readings
    never need pairing by eye; the rule's change is −1, from legal to illegal.
    """
    f = csv("target_flips.csv")
    names = gen2(f.checkpoint)
    flips = f.groupby("flip").targets.max().sort_values(ascending=False)
    callouts = {"to the opponent": (Callout("e64 (earlier inputs)", 12, 8),),
                "loses land": (Callout("e64", 12, 8),)}
    rows = []
    for flip, n in flips.items():
        d = by_checkpoint(f[f.flip == flip])
        few = n < FEW
        change = d.predicted_legal_after - d.predicted_legal_before
        median = change[present(names, change.index)].median()

        def median_bar(ax, y, median=median, few=few):
            ax.plot([median, median], [y - 0.4, y + 0.4], color="#bdbdbd", lw=5, zorder=1,
                    solid_capstyle="butt", alpha=0.45 if few else 1)

        rows.append(StripRow(f"{flip}\n{n:,} targets" + ("  (too few to weigh)" if few else ""),
                             change, band_colour=None, alpha=0.35 if few else 1.0,
                             callouts=callouts.get(flip, ()),
                             tick_colour=INK_SOFT if few else INK, extra=median_bar))
    panel = StripPanel(rows, names, refs=((-1, "the rule:\nlegal to illegal"), (0, "no change")),
                       xlim=(-1.08, 0.22), top=-0.15)
    strip_chart("target-flips", panel, (10.2, 4.8),
                "change in the target's mean predicted legality, after the flip − before; "
                "grey bar = the median; faint = too few to weigh",
                "Breaking the one condition a target restriction states: by the rule the "
                "target turns illegal", title_pad=22)


PIP_CONFIGURATIONS = ["surplus", "one short", "pip missing", "pip missing, +2",
                      "all off colour"]
#: Each row is named as the probes record's table names it, then described.
PIP_LABELS = {
    "surplus": "surplus: exact + 2 off-colour lands",
    "one short": "one short: exact less one land",
    "pip missing": "pip missing: same count, one\ncoloured pip's land off-colour",
    "pip missing, +2": "pip missing, +2: pip missing\n+ 2 off-colour lands",
    "all off colour": "all off colour: same count,\nevery land off-colour",
}


def colour_pips() -> None:
    """Change in the affordable verdict from the exact lands, and the two rankings."""
    c = csv("colour_pips.csv")
    r = csv("colour_pips_ranking.csv")
    names = gen2(c.checkpoint)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                                      gridspec_kw={"width_ratios": [1.5, 1]})
    configuration_rows = []
    for name in PIP_CONFIGURATIONS:
        d = c[c.configuration == name]
        affordable = not d.empty and bool(d.rule_affordable.iloc[0] >= 0.5)
        configuration_rows.append(StripRow(
            PIP_LABELS[name] + ("\nrule: affordable" if affordable else "\nrule: not affordable"),
            -d.set_index("checkpoint").below_exact_mean, band_half=0.44))
    draw_strip_panel(left, StripPanel(configuration_rows, names, refs=((0, "the exact lands"),),
                                      top=0.05, tick_style=dict(fontsize=8.5)))
    left.set_xlabel("predicted affordable, minus the exact lands' prediction,\nmean over "
                    "300 decisions")
    left.set_title("Change from the exact lands")
    groups = [("every configuration", "every\nconfiguration"),
              ("colours only", "exact, surplus against\nthe three off-colour rows"),
              ("amount only", "exact against\none short")]
    ranking_rows = [StripRow(label, by_checkpoint(r[r["items"].str.startswith(prefix)]).auc)
                    for prefix, label in groups]
    draw_strip_panel(right, StripPanel(ranking_rows, names, refs=((0.5, "chance"),),
                                       xlim=(0.4, 1.0), top=0.05, tick_style=dict(fontsize=8.5)))
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
    keys = sorted(sizes.index, key=lambda k: (order.get(k[0], 9), -sizes[k]))
    rows = []
    for zone, outcome in keys:
        d = by_checkpoint(z[(z.zone_before == zone) & (z.outcome == outcome)])
        few = sizes[(zone, outcome)] < FEW
        wrong = d.loc[present(names, d.index), "predicted_instead"].dropna()

        def wrong_destination(ax, y, wrong=wrong, few=few):
            if wrong.empty:
                return
            top = wrong.value_counts()
            where = ZONE_NAMES.get(top.index[0], top.index[0])
            ax.text(1.02, y, f"→ {where} (on {top.iloc[0]} of {len(names)})",
                    transform=ax.get_yaxis_transform(), fontsize=8, va="center",
                    color=GRAY if few else INK)

        label = f"{zone} → " + ("battlefield, a blink" if zone == outcome_zone(outcome) else
                                outcome_zone(outcome)) + f" ({sizes[(zone, outcome)]} cards)"
        rows.append(StripRow(label, d.top_right, muted=few, tick_colour=GRAY if few else INK,
                             extra=wrong_destination))

    def dividers(ax) -> None:
        ax.text(1.02, len(keys) - 0.45, "most common wrong\ndestination", fontsize=8,
                transform=ax.get_yaxis_transform(), va="bottom", fontweight="bold")
        for boundary in range(1, len(keys)):
            if keys[boundary][0] != keys[boundary - 1][0]:
                ax.axhline(len(keys) - boundary - 0.5, color=INK_SOFT, lw=0.8)
        ax.axvline(1, color="black", lw=1.2)

    strip_chart("zone-outcomes", StripPanel(rows, names, xlim=(0, 1.015), finish=dividers),
                (11.6, 6.0),
                "share of moved cards whose most probable destination is the right one "
                "(staying put excluded)",
                f"Where does the model send a card the resolution moved? Grey rows: fewer than "
                f"{FEW} cards", legend_anchor=(1.2, 1.0))


def zone_amounts() -> None:
    """Predicted cards at the affected player as the script's NumCards$ is set 1 to 6.

    Seventeen curves per panel tangle, so each checkpoint's curve is drawn faint and each
    stage's median over its checkpoints bold, in the stage's line style.
    """
    a = csv("zone_amounts.csv")
    names = gen2(a.checkpoint)
    amounts = list(range(1, 7))
    columns = [f"amount_{k}" for k in amounts]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True)
    for ax, api in zip(axes, ("Draw", "Discard", "Mill")):
        d = by_checkpoint(a[a.api == api])
        for name in names:
            if name in d.index:
                ax.plot(amounts, [d.loc[name, c] for c in columns], **run_line(name),
                        alpha=0.35)
        for kind in ("none", "arm", "A", "B"):
            members = [n for n in names if n in d.index and stage_of(n) == kind]
            if members:
                ax.plot(amounts, d.loc[members, columns].median().values, zorder=4,
                        color=NO_COPIES_GRAY if kind == "none" else INK,
                        ls=STAGE_LINE.get(kind, "-"), lw=2.4)
        ax.plot(amounts, amounts, color="black", lw=1.2)
        ax.text(4.95, 5.85, "the script's\nnumber", fontsize=8, ha="right", va="top")
        if api == "Discard" and "e64 (earlier inputs)" in d.index:
            callout(ax, 6, d.loc["e64 (earlier inputs)", "amount_6"], "e64 (earlier inputs)",
                    dx=-10, dy=22, ha="right")
        if api == "Mill" and "A256 best" in d.index:
            callout(ax, 6, d.loc["A256 best", "amount_6"], "A256 best = last", dx=10, dy=0,
                    ha="left")
        n = int(d.records.max()) if not d.empty else 0
        ax.set_title(f"{api}, {n} records")
        ax.set_xlabel("NumCards$ written into the script")
        ax.set_xticks(amounts)
    axes[0].set_ylabel("predicted cards at the affected player")
    widths = sorted({WIDTH_OF[run_of(n)] for n in names if stage_of(n) != "none"})
    handles = [Line2D([], [], color=WIDTH_COLOUR[w], lw=1.5, alpha=0.5,
                      label=f"{w}-wide e, one checkpoint") for w in widths]
    handles += stage_handles(lambda s: dict(color=INK, ls=STAGE_LINE[s], lw=2.4),
                             label="median, {}")
    handles.append(Line2D([], [], color=NO_COPIES_GRAY, lw=2.4,
                          label="median, arms without copies"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.14),
               fontsize=8.5)
    fig.suptitle("Does the predicted count follow the number in the script? Faint: one curve "
                 "per checkpoint; bold: each stage's median", x=0.01, ha="left", y=1.0)
    save(fig, "zone-amounts")


def zone_who() -> None:
    """Share of the predicted cards still at the originally targeted player after retargeting."""
    w = csv("zone_who.csv")
    names = gen2(w.checkpoint)
    apis = [api for api in ("Draw", "Discard", "Mill") if api in set(w.api)]
    n = w.groupby("api").records.max()
    callouts = {"Discard": (Callout("e128-lethality", -14, 14, "right"),
                            Callout("B64 last", 12, -6)),
                "Mill": (Callout("e128-lethality", -14, -16, "right"),
                         Callout("B256 last", 12, -6))}
    rows, before_all = [], []
    for api in apis:
        d = by_checkpoint(w[w.api == api])
        before = share_at_old_target(d, "before")
        before_all.append(before[present(names, before.index)].median())
        rows.append(StripRow(f"{api}\n{n[api]} records",
                             share_at_old_target(d, "after"),
                             callouts=callouts.get(api, ())))
    strip_chart("zone-who", StripPanel(rows, names, refs=((0, "the rule"),), xlim=(-0.03, 1.0)),
                (10.2, 3.8),
                "after the target is handed to the other player: share of the predicted cards "
                "still at the originally targeted player",
                "Does the effect follow the player target? Before the flip the targeted player "
                f"holds {min(before_all):.0%} to {max(before_all):.0%} of the predicted cards "
                "(median per effect)", title_style=dict(fontsize=10), legend_anchor=(1.01, 1.05))


# ── run lineage ────────────────────────────────────────────────────────────

#: What each run is, beneath its name on the lineage chart.
RUN_NOTES = {"e64 (earlier inputs)": "first arm, before the input corrections",
             "e64": "first arm again, corrected inputs",
             "e128": "e64 at width 128",
             "e64-lethality": "e64 with lethality copies",
             "e128-lethality": "e128 with lethality copies",
             "e256-lethality": "e128-lethality at width 256"}


def run_note(run) -> str:
    if run.name in RUN_NOTES:
        return RUN_NOTES[run.name]
    if lineage.stage(run) == "A":
        return "resumes the arm, full learning rate"
    return f"{run.parent}'s kept epoch at a tenth of the rate"


def run_lineage() -> None:
    """Every gen-2 run as a bar over its epochs, with the checkpoints probed on it.

    An A run resumes its arm after the arm's last epoch; a B run starts from its A run's
    kept epoch at a tenth of the learning rate. Run B256 never went below the checkpoint it
    started from, so it keeps run A256's epoch 53, and ``B256 best`` is its own lowest
    epoch, 55, probed but not kept.
    """
    runs = csv("runs.csv").set_index("run")
    ck = csv("checkpoints.csv")
    rows = [(r.width, r.name, run_note(r)) for r in sorted(RUN_LIST, key=lambda r: r.width)
            if r.name in runs.index]
    positions = grouped_positions([width for width, _, _ in rows], 0.6)
    ys = {run: yy for (_, run, _), yy in zip(rows, positions)}
    y = positions[-1] + 1
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
            face, edge, hatch = run_colour(run), WIDTH_COLOUR[width], None
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

    # The probed checkpoints, labelled with the names every other chart uses; an arm's
    # checkpoint is its row's name, so it is labelled by its epoch alone.
    names = {"B256 best": "B256 best, not kept"}
    for _, c in ck.iterrows():
        yy = ys.get(c.run)
        if c.checkpoint == "A256 last" or yy is None:
            continue
        width = None if runs.loc[c.run].lethality_copies == 0 else int(c.e_width)
        ax.plot(c.epoch, yy, zorder=5, **marker(c.checkpoint, c.role, width))
        label = (f"epoch {c.epoch}" if c.checkpoint == c.run
                 else names.get(c.checkpoint, short(c.checkpoint)))
        below = c.checkpoint == "B256 best"  # clear of the kept epoch's label beside it
        # The y axis runs downward: yy − height / 2 is the bar's top edge.
        ax.annotate(label, (c.epoch, yy + (height if below else -height) / 2),
                    xytext=(3, -2 if below else 2), textcoords="offset points", ha="left",
                    va="top" if below else "bottom", fontsize=7.8, color=INK, zorder=6)
    if "B256" in ys and "A256 best" in set(ck.checkpoint):
        kept = int(ck[ck.checkpoint == "A256 best"].epoch.iloc[0])
        ax.plot(kept, ys["B256"], zorder=5, **marker("B256 kept", "best", 256))
        ax.annotate(f"kept: A256's epoch {kept}", (kept, ys["B256"]), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=7.8,
                    color=INK)
    for width, members in width_groups([w for w, _, _ in rows], positions):
        ax.text(-23.5, np.mean(members), f"e width {width}", rotation=90, ha="center",
                va="center", fontsize=9.5, fontweight="bold", color=WIDTH_COLOUR[width])
        ax.plot([-22.3, -22.3], [min(members) - 0.3, max(members) + 0.3],
                color=WIDTH_COLOUR[width], lw=2, clip_on=False)
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
    ax.text(-24.5, -1.25, f"The {number_word(len(gen2(ck.checkpoint)))} probed checkpoints and "
            "the runs they come from; an arrow marks the weights a continuation starts from",
            ha="left", va="bottom",
            fontsize=11, color=INK, clip_on=False)
    save(fig, "run-lineage")


# ── scorecard ──────────────────────────────────────────────────────────────


def scorecard_measures() -> pd.DataFrame:
    """One row per checkpoint, one column per game-relevant measure, as the records define them."""
    ck = csv("checkpoints.csv").set_index("checkpoint")
    out = pd.DataFrame(index=ck.index)
    out["card_disjoint"] = ck.card_disjoint
    out["game_disjoint"] = ck.game_disjoint
    out["probe_mean"] = probe_mean()
    out["grid_error"] = grid_errors(csv("damage_grids.csv"))
    sweeps = csv("sweeps.csv")
    dmg = sweeps[sweeps.sweep == "damage_spell"]
    out["sweep_on_rule"] = dmg.groupby("checkpoint").apply(lambda d: float(on_rule(d)))
    out["ping_death"] = ping_predictions().median(axis=1)
    g2 = csv("gate2.csv")
    varying = g2[g2.keyword.isin(["first_strike", "deathtouch", "infect", "double_strike"])]
    # A mean over a checkpoint's other keywords alone would not be comparable.
    varying = varying[varying.checkpoint != WRONG_STEP_CHECKPOINT]
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
    counts = np.arange(1, 7)
    centred = counts - counts.mean()
    slopes = za[[f"amount_{k}" for k in counts]].astype(float).to_numpy() @ centred / (
        centred @ centred)
    out["zone_slope"] = pd.Series(slopes, index=za.checkpoint).groupby(level=0).mean()
    who = csv("zone_who.csv")
    out["retarget_stays"] = share_at_old_target(who, "after").groupby(who.checkpoint).mean()
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


#: Each measure's distance toward the rule: (value with no knowledge, the rule's value).
#: A blind predictor's AUC is 0.5, its gate-2 agreement 50%, its death grid off by 0.5
#: everywhere (it says one half), its counts flat and its retargeted share all at the old
#: target. The two losses have no rule and are scaled within their row.
SCORECARD_SCALE = {"probe_mean": (0.5, 1.0), "grid_error": (0.5, 0.0), "sweep_on_rule": (0, 1),
                   "ping_death": (0.0, 1.0), "gate2_varying": (50, 100),
                   "evasion_auc": (0.5, 1.0), "control_flip": (0.0, 1.0),
                   "colour_auc": (0.5, 1.0), "board_slope": (0.0, 1.0),
                   "zone_slope": (0.0, 1.0), "retarget_stays": (1.0, 0.0)}


def level_of(key: str, series: pd.Series, direction: int) -> pd.Series:
    """Each value's share of the way from a blind predictor to the rule, 0 to 1; a row with
    no rule runs from its worst to its best."""
    if key in SCORECARD_SCALE:
        blind, rule = SCORECARD_SCALE[key]
        return ((series - blind) / (rule - blind)).clip(0, 1)
    if key == "toward_rule":
        return series
    good = direction * series
    return (good - good.min()) / max(good.max() - good.min(), 1e-9)


def scorecard_layout(data: pd.DataFrame) -> tuple[list, float]:
    """``(y, "group" | "measure", payload)`` per line of the card, and the y below the last."""
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
    layout.append((y + 0.15, "measure", ("toward_rule", "mean distance toward the rule, rows "
                                                        "below the losses (1 = the rule)", +1,
                                         "{:.2f}")))
    return layout, y + 1.15


def scorecard_row_cells(row: pd.Series, level: pd.Series, direction: int, fmt: str,
                        names: list[str]) -> list[HeatCell | None]:
    """The cells of one measure over ``names``; bold marks each that prints as the best."""
    def shown(v: float) -> str:
        text = ("✓" if v >= 0.5 else "✗") if fmt == "check" else fmt.format(v)
        return text[1:] if fmt == "{:.3f}" and text.startswith("0") else text

    top_value = row.max() if direction > 0 else row.min()
    return [None if pd.isna(row.loc[name]) else HeatCell(
        BETTER_CMAP(0.04 + 0.86 * level.loc[name]), shown(row.loc[name]),
        level.loc[name] > 0.55,
        bold=shown(row.loc[name]) == shown(top_value) and fmt != "check") for name in names]


def scorecard() -> None:
    """Measure × checkpoint, each cell coloured by how far it goes from a blind predictor
    toward the rule (darker is closer to the rule).

    A rank colouring would give a row whose checkpoints differ by noise the same contrast
    as one where they differ by half the scale; the distance toward the rule keeps a row
    where every checkpoint sits near chance pale. The two losses have no rule and are
    coloured from the row's worst to its best. The last row is each checkpoint's mean
    distance toward the rule over the rows that have one. Bold marks every cell that
    prints the same as the row's best. ``e64 (earlier inputs)`` scored first and double
    strike on the wrong damage step, so it has no gate-2 cell.
    """
    data = scorecard_measures()
    columns = gen2_checkpoints(data.index)
    names = [name for _, name, _ in columns]
    measures = [m for _, group in SCORECARD_ROWS for m in group if m[0] in data.columns]
    values = data.loc[names]
    toward_rule = pd.DataFrame({key: level_of(key, values[key], d)
                                for key, _, d, _ in measures if key in SCORECARD_SCALE})
    toward_rule = toward_rule.mean(axis=1)
    widths = [width for width, _, _ in columns]
    xs = grouped_positions(widths, 0.35)
    layout, y = scorecard_layout(data)
    fig, ax = plt.subplots(figsize=(12.6, 0.3 * y + 1.6))
    fig.subplots_adjust(left=0.33, right=0.995, top=0.8, bottom=0.04)
    leaders = set()
    for yy, kind, payload in layout:
        if kind == "group":
            ax.text(-0.75, yy + 0.05, payload, ha="right", va="center", fontsize=8.8,
                    fontweight="bold", color=INK)
            continue
        key, label, direction, fmt = payload
        row = toward_rule if key == "toward_rule" else values[key]
        cells = scorecard_row_cells(row, level_of(key, row, direction), direction, fmt, names)
        heat_row(ax, yy, xs, cells, 8.2, 1.1, "#e6e6e6", 0.6, missing_glyph="–")
        if key != "toward_rule" and fmt != "check":
            leaders |= {name for name, cell in zip(names, cells) if cell and cell.bold}
        arrow = "▼" if direction < 0 else "▲"
        ax.text(-0.75, yy, f"{label}  {arrow}", ha="right", va="center", fontsize=8.2,
                color=INK, fontweight="bold" if key == "toward_rule" else "normal")
    top = -0.55
    column_headers(ax, [(name, label) for _, name, label in columns], xs, top - 0.85,
                   top - 0.35, 6, 8.2)
    for width, members in width_groups(widths, xs):
        colour = WIDTH_COLOUR[width]
        ax.plot([min(members) - 0.4, max(members) + 0.4], [top - 5.6] * 2, color=colour,
                lw=2, clip_on=False)
        ax.text(np.mean(members), top - 5.8, f"e width {width}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=colour)
    ax.set_xlim(-0.55, xs[-1] + 0.55)
    ax.set_ylim(y - 0.4, top)
    ax.axis("off")
    arms = [name for name in names if stage_of(name) in ("none", "arm")]
    last = toward_rule.nsmallest(len(arms)).index
    ax.annotate(f"No checkpoint leads everywhere: {number_word(len(leaders))} different "
                "checkpoints top a row; the 256-wide continuations come closest to the rule "
                f"overall,\nand {number_word(len(set(arms) & set(last)))} of the "
                f"{number_word(len(arms))} arms come last",
                (0.01, top - 7.0), xycoords=("figure fraction", "data"), ha="left",
                va="bottom", fontsize=11, color=INK, annotation_clip=False)
    fig.text(0.01, 0.005, "Colour: the share of the way from a blind predictor (AUC 0.5, "
             "agreement 50%, a flat count, a constant one half) to the rule, darker = closer; "
             "the two losses, which have no rule,\nrun from the row's worst to its best. "
             "▲ higher is better, ▼ lower is better; bold = prints the same as the row's best. "
             "– : not measured.", fontsize=7.8, color=INK_SOFT)
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
    in that remainder is drawn as its own segment. The bars stack rises and falls apart,
    so the affected gate's share is printed against the net change, the black tick.
    """
    ep = csv("epochs.csv")
    fe = csv("fields_by_epoch.csv")
    order = [r for w in WIDTHS for r in LINES[w][1:] if r in set(ep.run)]
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
            ax.text(max(pos, r.total) + 0.004, y + 0.1, f"net {r.total:+.3f}", va="center",
                    fontsize=8.4, color=INK, fontweight="bold")
            ax.text(max(pos, r.total) + 0.004, y - 0.2, f"affected gate {r.gate / r.total:.0%} "
                    "of net", va="center", fontsize=7.6, color=RISE_GROUPS[0][2])
    ax.set_yticks(range(len(df)),
                  [f"{r.run}   epoch {r.low} → {r['last']}" for _, r in df.iterrows()][::-1])
    for label in ax.get_yticklabels():
        run = label.get_text().split()[0]
        label.set_color(WIDTH_COLOUR[WIDTH_OF[run]])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlim(min(-0.06, df[[k for k, _, _ in RISE_GROUPS]].clip(upper=0).sum(axis=1).min()
                    - 0.01), df.total.max() + 0.07)
    ax.set_xlabel("change in card-disjoint loss from the run's lowest card-disjoint epoch to its "
                  "last; rises right, falls left, black tick = net change")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=c, label=label) for _, label, c in RISE_GROUPS],
              loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=3, fontsize=8)
    ax.set_title("The affected gate's rise is more than half of every run's net card-disjoint "
                 "rise; poison's comes\nnext, or level with it in run A128, in every run but A64",
                 loc="left",
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


#: The board questions whose answer the line states, against the ones the board decides.
LINE_STATES_ANSWER = {"damage_taken", "pt_until_end_of_turn", "affordable"}


class GainProbes:
    """The rung 1 and rung 1w probe scores, and the gain of e over a random vector."""

    def __init__(self) -> None:
        lad = csv("probe_ladder.csv")
        self.lad = lad[(lad.probe == "mlp") & lad.rung.isin(["1", "1w"])]
        self.kind = self.lad.groupby("target").kind.first()

    def stratum_for(self, target: str, board: bool) -> str:
        if board:
            return "card-disjoint"
        have = set(self.lad[self.lad.target == target].stratum)
        return "held-out" if "held-out" in have else "all" if "all" in have else "trained"

    def gains(self, target: str, board: bool) -> pd.Series:
        d = self.lad[(self.lad.target == target)
                     & (self.lad.stratum == self.stratum_for(target, board))]
        piv = d.pivot_table(index="checkpoint", columns="rung", values="score")
        if not {"1", "1w"} <= set(piv.columns):
            return pd.Series(dtype=float)
        return (piv["1"] - piv["1w"]).dropna()


def gain_heatmap_layout(blocks: list) -> tuple[list, float]:
    """``(y, "block" | "family" | "target", payload)`` per line of the heatmap, and the y
    below the last."""
    layout = []
    y = 0.0
    for block_title, rows, span in blocks:
        if not rows:
            continue
        layout.append((y, "block", block_title))
        y += 1.1
        family = None
        for fam, fam_title, target, label, board in rows:
            if fam != family:
                layout.append((y, "family", fam_title))
                y += 0.8
                family = fam
            layout.append((y, "target", (target, label, span)))
            y += 1
        y += 0.6
    return layout, y


def draw_board_gains(sax, gp: GainProbes, board_auc: list, board_r2: list,
                     names: list[str]) -> None:
    """The board questions: one marker per checkpoint, the AUC ones sorted by median gain,
    the three whose answer the line states shaded, the R² ones below."""
    rows_y = []
    yy = 0.0
    for r in board_auc:
        rows_y.append((yy, r))
        yy += 1
    if board_r2:
        yy += 0.6
        rule_y = yy - 0.8
        for r in board_r2:
            rows_y.append((yy, r))
            yy += 1
    for yy, (_, _, target, label, _) in rows_y:
        if target in LINE_STATES_ANSWER:
            band(sax, yy, "#efe9f7", 0.45)
        strip(sax, yy, gp.gains(target, True), names, spread=0.32, ms=5.5)
    if board_r2:
        sax.axhline(rule_y, color=GRAY, lw=0.6)
    sax.set_yticks([yy for yy, _ in rows_y],
                   [label + ("  (R²)" if gp.kind[target] == "amount" else "")
                    for _, (_, _, target, label, _) in rows_y])
    for tick, (_, (_, _, target, _, _)) in zip(sax.get_yticklabels(), rows_y):
        tick.set_fontweight("bold" if target == "dies" else "normal")
    sax.set_ylim(rows_y[-1][0] + 0.55, -0.55)
    sax.axvline(0, color=INK, lw=0.8)
    sax.grid(axis="y", visible=False)
    sax.set_xlabel("rung 1 − rung 1w: AUC gain, or R² gain for damage taken")
    sax.set_title("Board questions, card-disjoint records, sorted by median gain; shaded: the "
                  "line states the answer", fontsize=9, fontweight="normal", color=INK_SOFT)
    sax.legend(handles=checkpoint_handles(names), fontsize=7.6, loc="upper left",
               bbox_to_anchor=(1.01, 1.02))


def draw_line_gains(ax, gp: GainProbes, layout: list, columns: list, xs: list[float],
                    norm, top: float) -> None:
    """The line properties: one heatmap block per kind of answer, a column per checkpoint."""
    for yy, row_kind, payload in layout:
        if row_kind == "block":
            ax.text(-0.75 - 9.0, yy + 0.15, payload, ha="left", va="center", fontsize=9.4,
                    fontweight="bold", color=INK)
            continue
        if row_kind == "family":
            ax.text(-0.75, yy + 0.1, payload, ha="right", va="center", fontsize=8.4,
                    fontweight="bold", color=INK_SOFT)
            continue
        target, label, span = payload
        diff = gp.gains(target, False)
        cells = [None if name not in diff.index else HeatCell(
            GAIN_CMAP(norm(diff[name] / span)), f"{100 * diff[name]:.0f}",
            abs(diff[name] / span) > 0.62) for name, _, _ in columns]
        heat_row(ax, yy, xs, cells, 7.4, 0.8, "#eeeeee", 0.5)
        ax.text(-0.75, yy, label, ha="right", va="center", fontsize=8.2, color=INK)
    column_headers(ax, [(name, label) for name, label, _ in columns], xs, top, top + 0.45, 5.5,
                   8)


def e_over_random() -> None:
    """Rung 1 minus rung 1w per target and checkpoint: what e adds over a random vector.

    Rung 1 is the MLP probe on the board plus e (board targets, card-disjoint records) or
    on e alone (line properties, held-out lines); rung 1w replaces e with a fixed random
    vector of the same width. Binary targets are AUC differences, amounts R² differences.
    The board questions sit on top as a strip, one marker per checkpoint, sorted by their
    median gain, with the three whose answer the line states shaded. The line properties
    sit below as a heatmap split by kind, so that a colour always means one unit: the
    yes/no block runs to the AUC's largest possible gain, 0.5, and the amount block to 1.
    """
    gp = GainProbes()
    kind = gp.kind
    targets = [(family, title, t, label, board) for family, title, rows in GAIN_FAMILIES
               for t, label, board in rows if t in kind.index and not gp.gains(t, board).empty]
    board_auc = sorted([r for r in targets if r[4] and kind[r[2]] == "binary"],
                       key=lambda r: gp.gains(r[2], True).median(), reverse=True)
    board_r2 = [r for r in targets if r[4] and kind[r[2]] == "amount"]
    blocks = [("Yes/no line properties: AUC gain × 100",
               [r for r in targets if not r[4] and kind[r[2]] == "binary"], 0.5),
              ("Amounts a line states: R² gain × 100",
               [r for r in targets if not r[4] and kind[r[2]] == "amount"], 1.0)]
    columns = [("gen-1", "gen-1", None)] + [(name, label, w) for w, name, label in
                                            gen2_checkpoints(gp.lad.checkpoint)]
    widths = [width for _, _, width in columns]
    xs = grouped_positions(widths, 0.45)
    layout, y = gain_heatmap_layout(blocks)
    unit = 0.215  # inches per heatmap row
    top, head = -1.25, 7.0  # column labels and width names above the first block
    strip_rows = len(board_auc) + (len(board_r2) + 0.6 if board_r2 else 0)
    heights = dict(title=0.85, strip=0.36 * strip_rows + 0.2, strip_foot=0.8,
                   heat=unit * (y - top + head), foot=1.0)
    height = sum(heights.values())
    fig = plt.figure(figsize=(9.6, height))
    strip_bottom = heights["foot"] + heights["heat"] + heights["strip_foot"]
    sax = fig.add_axes([0.3, strip_bottom / height, 0.52, heights["strip"] / height])
    draw_board_gains(sax, gp, board_auc, board_r2, ["gen-1"] + gen2(gp.lad.checkpoint))
    ax = fig.add_axes([0.3, heights["foot"] / height, 0.69, heights["heat"] / height])
    norm = matplotlib.colors.Normalize(-1, 1)
    draw_line_gains(ax, gp, layout, columns, xs, norm, top)
    for width, members in width_groups(widths, xs, (None, *WIDTHS)):
        ax.text(np.mean(members), top - 6.4, "gen-1" if width is None else f"e width {width}",
                ha="center", va="bottom", fontsize=8.8, fontweight="bold",
                color=INK_SOFT if width is None else WIDTH_COLOUR[width])
    ax.set_xlim(-0.55, xs[-1] + 0.55)
    ax.set_ylim(y - 0.6, top - head)
    ax.axis("off")
    largest_board_gain = max(gp.gains(r[2], True).max() for r in board_auc
                             if r[2] not in LINE_STATES_ANSWER)
    fig.text(0.01, 1 - 0.12 / height, "e adds far more than a random vector of its width on "
             f"what a line states, and at most {largest_board_gain:.2f} of AUC\non what the "
             "board decides: deaths, blocks, triggers, targets and the affected gate",
             ha="left", va="top", fontsize=10.8, color=INK, fontweight="bold")
    cax = fig.add_axes([0.07, 0.62 / height, 0.18, 0.1 / height])
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=GAIN_CMAP, norm=norm), cax=cax,
                       orientation="horizontal", ticks=[-1, 0, 1])
    bar.ax.set_xticklabels(["−50\n−100", "0\n0", "+50\n+100"], fontsize=7.2)
    bar.ax.set_title("cell colour: gain × 100", fontsize=7.4, color=INK_SOFT)
    bar.ax.text(-0.13, -0.55, "AUC\nR²", transform=bar.ax.transAxes, ha="right", va="top",
                fontsize=7.2, color=INK_SOFT)
    fig.text(0.3, 0.12 / height, "Cells: (rung 1 − rung 1w) × 100, MLP probe, line properties on "
             "held-out lines (* trained lines only, † all interaction\npairs). A random vector's "
             "R² falls below zero, so an amount's gain can pass 100. gen-1 is probed on its own "
             "corpus.", fontsize=7.4, color=INK_SOFT)
    save(fig, "e-over-random")


# ── the activated ping's gate under edits ──────────────────────────────────


def ping_gate_edits() -> None:
    """The affected gate at the targeted creature of five activated damage records, under six
    edits, each applied alone to the record as collected."""
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
        band(ax, ys[wording], "#efe9f7")
    if len(hits) < len(edits):
        ax.axhline(min(ys[e] for e in hits) - 0.67, color=GRAY, lw=0.6)
    offsets = np.linspace(0.2, -0.2, len(p.columns))
    base = "as collected"
    for k, name in enumerate(p.columns):
        style = checkpoint_marker(name, 7)
        # Each edit is applied alone to the record as collected: a dumbbell from that value.
        for edit in edits:
            if edit == base or base not in p.index:
                continue
            y = ys[edit] + offsets[k]
            ax.plot([p.loc[base, name], p.loc[edit, name]], [y, y], color=style["mec"], lw=1.1,
                    alpha=0.5, zorder=2)
            ax.plot(p.loc[base, name], y, "|", color=GRAY, ms=7, mew=1.3, zorder=2)
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
    ax.set_xlabel("mean predicted chance the targeted creature is affected (the affected gate), "
                  "five activated damage records; black tick = the rule")
    handles, names = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="|", color=GRAY, ls="none", ms=8, mew=1.3))
    names.append("as collected, before the edit")
    handles.append(Line2D([], [], marker="|", color=INK, ls="none", ms=12, mew=2))
    names.append("the rule")
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3,
              fontsize=8.3)
    ax.set_title("Only rewording \"any target\" as \"target creature\" puts the affected gate "
                 "above 0.9 on every checkpoint read", loc="left", fontsize=10.5)
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
         "of the {:,} attackers anchoring a record in a game's turn, where the AI's blocking "
         "code asks\nabout each attacker again at every stage of filling in its plan", [
             ("anchoring more than one real record",
              "anchor more than one \"real\" decision", plan)]),
    ]
    bars = []
    for total_key, title, parts in groups:
        if total_key not in count:
            continue
        bars.append(("title", title.format(count[total_key]), None, None))
        for key, label, colour in parts:
            if key in count:
                bars.append(("bar", label, count[key] / count[total_key], (count[key], colour)))
    extra_lines = sum(label.count("\n") for kind, label, _, _ in bars if kind == "title")
    height = 0.42 * (len(bars) + extra_lines) + 1.45
    fig = plt.figure(figsize=(9.0, height))
    ax = fig.add_axes([0.43, 0.8 / height, 0.5, 1 - 1.5 / height])
    y = 0.0
    for kind, label, share, extra in bars:
        if kind == "title":
            y += 0.2 + 0.2 * label.count("\n")
            ax.text(-0.86, y, label[0].upper() + label[1:], fontsize=9, fontweight="bold",
                    color=INK, va="center")
            y += 0.7 + 0.2 * label.count("\n")
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
    fig.text(0.005, 1 - 0.12 / height, "In the game-disjoint validation stratum, most \"may not "
             "block\" labels are the AI's own block plan;\na tapped creature, a refusal the rules "
             "make, is one label in fourteen", ha="left", va="top", fontsize=10.5, color=INK)
    fig.legend(handles=[Patch(color=plan, label="the AI's block plan"),
                        Patch(color=rule, label="a refusal the rules make"),
                        Patch(color=neutral, label="no static ability restricts the creature")],
               loc="lower left", bbox_to_anchor=(0.0, 0.3 / height), ncol=3, fontsize=8)
    fig.text(0.005, 0.1 / height, "Decisions the collector counted as real. The sub-counts may "
             "overlap, so they are not stacked.", fontsize=7.6, color=INK_SOFT)
    save(fig, "planned-block-labels")


# ── what a linear probe reads from e: gen-1 against gen-2 ──────────────────

#: The embedding-probe targets transcribed for gen-1, with each one's label.
EMBEDDING_ROWS = [("API type", "API type", "accuracy"), ("line kind", "line kind", "accuracy"),
                  ("cost_mana", "mana cost", "R²"), ("amt_damage", "damage amount", "R²"),
                  ("amt_counters", "counters amount", "R²"), ("amt_life", "life amount", "R²"),
                  ("amt_cards", "cards amount", "R²"), ("amt_power", "power amount", "R²")]


def embedding_probes_gen1() -> None:
    """Linear probes on e, all texts, for gen-1 and every gen-2 checkpoint; the
    participation ratio beneath.

    Gen-1's row in ``embedding_probes.csv`` holds exactly the eight targets drawn here.
    """
    e = csv("embedding_probes.csv")
    g3 = csv("gate3.csv").set_index("checkpoint")
    names = ["gen-1"] + gen2(e.checkpoint)
    score = e.pivot_table(index="target", columns="checkpoint", values="score")
    fig = plt.figure(figsize=(11.2, 7.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[len(EMBEDDING_ROWS), 1.25], hspace=0.32)
    ax, pax = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

    def gen1_against_gen2(values: pd.Series):
        def note(a, y: float) -> None:
            gen2_values = values[[n for n in names if n != "gen-1"]]
            a.text(1.01, y, f"gen-1 {values['gen-1']:.2f}\ngen-2 {gen2_values.min():.2f} to "
                   f"{gen2_values.max():.2f}", transform=a.get_yaxis_transform(), fontsize=7.8,
                   color=INK_SOFT, va="center", linespacing=1.25)
        return note

    rows = [StripRow(f"{label}, {metric}", score.loc[target],
                     band_colour="#f6f6f6" if i % 2 else "white", band_half=0.45, spread=0.36,
                     ms=6, extra=gen1_against_gen2(score.loc[target]))
            for i, (target, label, metric) in enumerate(EMBEDDING_ROWS)]
    draw_strip_panel(ax, StripPanel(rows, names, xlim=(0, 1), top=0.5))
    ax.set_xlabel("linear probe on e, all texts: accuracy for the two kinds, R² for the "
                  "cost and the amounts")
    ax.set_title("What a linear probe reads from e", pad=10)
    pr = g3.participation_ratio
    draw_strip_panel(pax, StripPanel(
        [StripRow("participation ratio", pr, band_colour=None, ms=6,
                  extra=gen1_against_gen2(pr))], names, xlim=(0, 13), top=0.5))
    pax.set_xlabel("participation ratio: the number of equally weighted directions that "
                   "would give e's spread of variance")
    pax.set_title("How many directions e spreads over", pad=8)
    fig.legend(handles=checkpoint_handles(names), loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.04), fontsize=8.3)
    fig.suptitle("A linear probe reads a line's kind, cost and amounts better from every gen-2 "
                 "checkpoint's e than from gen-1's,\nand gen-2's e spreads over about two to "
                 "three times as many directions", x=0.01, ha="left", y=1.0)
    save(fig, "embedding-probes")


# ── the first arm against gen-1 on the shared loss ─────────────────────────


def constant_predictor_losses(fe: pd.DataFrame) -> pd.Series:
    """Each field's constant-predictor loss on the card-disjoint sample, ``field``.

    The loss and the explained share satisfy loss = constant × (1 − explained / 100),
    and the constant is the same at every epoch, so each epoch gives one estimate and the
    median over epochs absorbs the rounding of the explained share to whole percent.
    Epochs explaining 97% or more, where that rounding dominates, are left out. A field
    whose loss stays under 0.002 at every epoch, or whose explained share is never
    defined, has no labels in practice: 0.
    """
    d = fe.copy()
    d["constant"] = d.card_disjoint_loss / (1 - d.explained_pct / 100)
    d.loc[(d.explained_pct >= 97) | (d.card_disjoint_loss < 0.002), "constant"] = np.nan
    out = d.groupby("field").constant.median()
    by_field = d.groupby("field")
    empty = (by_field.card_disjoint_loss.max() < 0.002) | by_field.explained_pct.apply(
        lambda x: x.isna().all())
    out[empty & out.isna()] = 0.0
    return out


#: Readable names for the output fields gen1_vs_first_arm names.
FIELD_LABELS = {"gate": "affected gate", "counters_delta_p1p1": "+1/+1 counters",
                "pt_duration": "pt_duration", "zone_outcome": "zone outcome",
                "poison_delta": "poison", "power_delta": "power change",
                "toughness_delta": "toughness change", "life_delta": "life change"}


def gen1_vs_first_arm() -> None:
    """(a) the 47 field terms summed, by epoch; (b) each field's gap at epoch 18.

    Epoch 18 is the epoch of the gen-2 record's table; the gap swings from epoch to epoch
    as much as panel (a) shows, so (b) is one epoch's split, not a mean.
    """
    fe = csv("fields_by_epoch.csv")
    epoch = 18
    first, gen1 = "e64 (earlier inputs)", "gen-1"
    fig = plt.figure(figsize=(13.4, 7.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.62)
    ax, bax = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    styles = {gen1: dict(color=GEN1_GRAY, lw=1.8, ls="-", marker="X", ms=4),
              first: dict(color=EARLIER_GRAY, lw=1.8, ls=(0, (4, 2)), marker="^", ms=4,
                          mec="#666666"),
              "e64": dict(color=NO_COPIES_GRAY, lw=1.8, ls="-", marker="^", ms=4,
                          mec="#666666")}
    labels = {gen1: "gen-1", first: "arm e64 (earlier inputs)", "e64": "arm e64"}
    for run, style in styles.items():
        d = fe[fe.run == run].groupby("epoch").card_disjoint_loss.sum()
        ax.plot(d.index, d.values, label=labels[run], **style)
    ax.axvline(epoch, color=INK_SOFT, lw=0.8, ls=":")
    ax.text(epoch + 0.5, 4.68, f"epoch {epoch}: panel (b)", fontsize=7.8, color=INK_SOFT,
            va="top")
    ax.set_xlim(0, fe[fe.run == gen1].epoch.max() + 1)
    ax.set_ylim(2.8, 4.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("card-disjoint loss, the 47 output-field terms summed")
    ax.legend(fontsize=8.3, loc="lower left")
    ax.set_title("(a) The shared part of the loss, by epoch", pad=10)

    at = fe[fe.epoch == epoch].pivot(index="field", columns="run", values="card_disjoint_loss")
    gap = (at[first] - at[gen1]).fillna(0)
    const = {run: constant_predictor_losses(fe[fe.run == run]) for run in (gen1, first)}
    mana = [f for f in gap.index if f.startswith("mana_delta_")]
    # The affected gate is drawn whatever its gap: its constant-predictor loss fell.
    rows = [(f, [f], gap[f]) for f in gap.index
            if f not in mana and (abs(gap[f]) >= 0.01 or f == "gate")]
    rows.append(("mana deltas, six fields", mana, gap[mana].sum()))
    shown = {f for _, fs, _ in rows for f in fs}
    rest = [f for f in gap.index if f not in shown]
    rows.sort(key=lambda r: -r[2])
    rows.append((f"the other {len(rest)} fields", rest, gap[rest].sum()))
    larger, similar, smaller = PURPLE, "#bdbdbd", "#e3a39b"
    added = {"larger": 0.0, "other": 0.0}  # the trailing fields' loss, by colour
    for i, (label, fields, value) in enumerate(rows):
        y = len(rows) - 1 - i
        c1 = const[gen1].reindex(fields).sum(min_count=1)
        c2 = const[first].reindex(fields).sum(min_count=1)
        if label.startswith("the other"):
            colour, note = "#e8e8e8", ""
        else:
            ratio = np.inf if c1 == 0 and c2 > 0 else c2 / c1 if c1 else 1.0
            colour = larger if ratio >= 1.25 else smaller if ratio <= 0.8 else similar
            note = f"{c1:.2f} → {c2:.2f}"
        if value > 0:
            added["larger" if colour == larger else "other"] += value
        bax.barh(y, value, height=0.66, color=colour, edgecolor="white", lw=0.8)
        name = FIELD_LABELS.get(label, label.replace("counters_delta_", "counters: ")
                                .replace("_", " "))
        bax.text(-0.103, y, name, ha="right", va="center", fontsize=8.2)
        bax.text(1.03, y, note, ha="left", va="center", fontsize=7.6, color=INK_SOFT,
                 transform=bax.get_yaxis_transform())
        bax.text(value + (0.003 if value >= 0 else -0.003), y, f"{value:+.3f}",
                 ha="left" if value >= 0 else "right", va="center", fontsize=7.2, color=INK)
    total = gap.sum()
    share = added["larger"] / (added["larger"] + added["other"])
    bax.axvline(0, color=INK, lw=0.9)
    bax.set_yticks([])
    bax.set_ylim(-0.6, len(rows) - 0.4)
    bax.set_xlim(-0.1, 0.125)
    bax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
    bax.grid(axis="y", visible=False)
    bax.text(1.03, len(rows) - 0.4, "constant-predictor\nloss, gen-1 → gen-2", fontsize=7.6,
             color=INK_SOFT, va="bottom", transform=bax.get_yaxis_transform())
    bax.set_xlabel(f"arm e64 (earlier inputs) minus gen-1, card-disjoint loss at epoch {epoch}"
                   "\n(right: gen-2 trails; left: gen-2 leads)")
    bax.set_title(f"(b) The gap of {total:+.2f} at epoch {epoch}, field by field", pad=10,
                  x=-0.3)
    bax.legend(handles=[Patch(color=larger, label="gen-2's constant-predictor loss is a "
                              "quarter or more higher, or new"),
                        Patch(color=similar, label="within a fifth either way"),
                        Patch(color=smaller, label="a fifth or more lower, or gone"),
                        Patch(color="#e8e8e8", label="fields under 0.01 either way, summed")],
               loc="upper center", fontsize=7.8, bbox_to_anchor=(0.3, -0.13), ncol=2)
    fig.suptitle("Arms e64 (earlier inputs) and e64 sat above gen-1 on the 47 output-field "
                 "terms at every epoch from 4 on but one. At epoch 18,\nfields whose "
                 "constant-predictor loss "
                 f"gen-2's corpus made larger or new carry {share:.0%} of the loss the "
                 "trailing fields add", x=0.01, ha="left", y=1.01)
    save(fig, "gen1-vs-first-arm")


# ── the share e carries against how much the trunk relies on it ────────────

#: Board question of the ladder, the output field the [ACT]-slot ablation reads for it,
#: its label and its colour. The affordable verdict and trigger firing are verdict bits
#: with no ablation field; may block has no share.
SHARE_RELIANCE = [("pt_until_end_of_turn", "pt_duration", "pump lasts until end of turn\n"
                   "(pt_duration)", "#4b2a86"),
                  ("affected", "gate", "card affected\n(the affected gate)", "#b8336a"),
                  ("dies", "zone_outcome", "dies\n(zone outcome)", "#8c7a5b"),
                  ("damage_taken", "damage_taken", "damage taken\n(damage_taken)", SLATE)]


#: Label offsets in points for the crowded medians: ``(dx, dy, ha)``.
SHARE_LABEL_OFFSETS = {"dies": (-30, 52, "center"), "affected": (60, -48, "left"),
                       "pt_until_end_of_turn": (40, -62, "left"),
                       "damage_taken": (95, 22, "left")}


def share_vs_reliance() -> None:
    """Median share carried by e against the loss the trunk loses without the acting e.

    Share: the probes record's (rung 1 − rung 0) / (rung 3 − rung 0), MLP probes,
    card-disjoint records. Reliance: the loss increase when matched noise replaces e at
    the ``[ACT]`` slot alone, divided by the field's own loss. Both over the gen-2
    checkpoints trained with lethality copies, a run whose last epoch is its kept one
    counted once.
    """
    t = ladder_table()
    names = gen2(t.index.get_level_values("checkpoint"))
    a = csv("ablation.csv")
    a = a[(a.replacement == "noise") & (a.scope == "act") & a.checkpoint.isin(names)]
    a = a.assign(reliance=a.loss_increase / a.base_loss)
    rel = a.pivot_table(index="checkpoint", columns="field", values="reliance")
    med = t.groupby("target").median()
    fig, ax = plt.subplots(figsize=(10.4, 6.6))
    for target, field, label, colour in SHARE_RELIANCE:
        share = t.loc[target, "share"].reindex(names)
        r = rel[field].reindex(names)
        read = not (med.loc[target, "share_o"] < ORACLE_FLOOR and med.loc[target, "share"] < 0.9)
        ax.plot(share, r, "o", ms=4.2, color=colour, alpha=0.45, mec="none", zorder=3)
        mx, my = share.median(), r.median()
        ax.plot(mx, my, "o", ms=14, mec=colour, mew=2.2, color=colour if read else "white",
                zorder=5)
        dx, dy, ha = SHARE_LABEL_OFFSETS.get(target, (14, 0, "left"))
        ax.annotate(f"{label}\nshare {mx:.2f}, +{my:.0%}", (mx, my), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8.2, va="center", ha=ha, color=INK,
                    zorder=6, arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=0.6,
                                                  shrinkA=2, shrinkB=9))
    ax.set_yscale("log")
    ticks = [0.25, 0.5, 1, 2, 4, 8]
    ax.set_yticks(ticks, [f"+{t:.0%}" for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.25, 9)
    ax.set_xlim(-0.05, 1.45)
    for x in (0, 1):
        ax.axvline(x, color="black", lw=0.9, zorder=1)
    ax.text(0.0, 9.3, "board alone", ha="center", va="bottom", fontsize=8)
    ax.text(1.0, 9.3, "the model's answer", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("share carried by e: (rung 1 − rung 0) / (rung 3 − rung 0), MLP probes, "
                  "card-disjoint records")
    ax.set_ylabel("reliance: loss increase when matched noise replaces e at the [ACT] slot\n"
                  "alone, as a share of the field's own loss (log scale)")
    corner = dict(fontsize=7.8, color=INK_SOFT, style="italic", linespacing=1.3)
    ax.text(0.02, 7.6, "low share, high reliance: the trunk does work e could carry;\n"
            "the probes record's rule moves it into e", va="top", **corner)
    ax.text(1.43, 0.28, "low reliance: the field leans little\non the acting ability's e",
            ha="right", va="bottom", **corner)
    ax.text(1.43, 7.6, "high share, high reliance:\ne carries it and the trunk reads it",
            ha="right", va="top", **corner)
    handles = [Line2D([], [], marker="o", ls="none", ms=11, color=INK_SOFT, mec=INK_SOFT,
                      label="median over the checkpoints, share interpreted"),
               Line2D([], [], marker="o", ls="none", ms=11, color="white", mec=INK_SOFT,
                      mew=2, label="median, share not interpreted: the oracle rung closes "
                                   "under half the gap,\nso the rule asks for a stronger "
                                   "probe first"),
               Line2D([], [], marker="o", ls="none", ms=4.5, color=INK_SOFT, alpha=0.5,
                      label="one checkpoint")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, -0.1), ncol=2,
               fontsize=8)
    fig.text(0.06, -0.135, "Not drawn: may block, whose share is undefined because the model "
             "scores no better than the board alone; the affordable verdict and trigger "
             "firing, which have no ablation field;\nthe two legal-target questions, which read "
             "the affected gate, so their reliance is card affected's.", fontsize=7.8,
             color=INK_SOFT)
    ax.set_title("Of the questions drawn, only pump duration has a share the probes record's "
                 "rule can read: e carries\nthe answer and the trunk relies on it; deaths and "
                 "affected cards need a stronger probe first "
                 f"({len(names)} gen-2 checkpoints)", pad=22)
    save(fig, "share-vs-reliance")


# ── the noise pilots ───────────────────────────────────────────────────────


def values_above(ax, xs, ys) -> None:
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:g}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=7.6)


def noise_pilot() -> None:
    """The three stage-0 pilots: validation measures as a change from ratio 0.1, then the
    cache measures and the value head beside gen-1's shipped cache."""
    p = csv("noise_pilot.csv").set_index("measure")
    g3 = csv("gate3.csv").set_index("checkpoint")
    ratios = ["0.05", "0.1", "0.2"]
    xs = np.arange(3)
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.9),
                             gridspec_kw={"width_ratios": [1.5, 1, 1, 1, 1], "wspace": 0.42})
    val = [("card-disjoint loss, epoch 3", "card-disjoint loss", "o"),
           ("game-disjoint loss, epoch 3", "game-disjoint loss", "s"),
           ("gate F1, epoch 3", "the affected gate's F1", "D"),
           ("zone accuracy, epoch 3", "zone accuracy", "^")]
    ax = axes[0]
    for k, (row, label, marker) in enumerate(val):
        v = p.loc[row, [f"ratio {r}" for r in ratios]].astype(float).values
        change = 100 * (v / v[1] - 1)
        ax.plot(xs + (k - 1.5) * 0.04, change, marker=marker, ms=5, lw=1.2,
                color=[PURPLE, PURPLE_SOFT, SLATE, GRAY][k], label=label)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhspan(-2, 2, color="#f0f0f0", lw=0, zorder=0)
    ax.set_ylim(-6, 6)
    ax.set_ylabel("% change from ratio 0.1")
    ax.set_title("validation, epoch 3", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", ncol=1)
    ax.text(2.25, -5.6, "grey band: ±2%", fontsize=7.5, color=INK_SOFT, ha="right")
    cache = [("top_component_share", "top component's share\nof e's variance", 0.30,
              "gate 3's limit"),
             ("participation_ratio", "participation ratio of e", None, None),
             ("mean_pairwise_cosine", "mean pairwise cosine of e", None, None)]
    for ax, (col, label, limit, limit_label) in zip(axes[1:4], cache):
        v = [g3.loc[f"noise pilot {r}", col] for r in ratios]
        ax.plot(xs, v, color=PURPLE, lw=1.4, marker="o", ms=6, zorder=3)
        ax.axhline(g3.loc["gen-1", col], color=GEN1_GRAY, lw=1.0, ls="--")
        ax.text(2.3, g3.loc["gen-1", col], f"gen-1's cache\n{g3.loc['gen-1', col]:g}",
                fontsize=7.3, color=GEN1_GRAY, ha="right", va="bottom")
        if limit is not None:
            ax.axhline(limit, color=INK, lw=1.0)
            ax.text(2.3, 0.02, f"solid line: {limit_label}, {limit:.2f}", fontsize=7.3,
                    color=INK, va="bottom", ha="right")
        values_above(ax, xs, v)
        top = max(max(v), g3.loc["gen-1", col], limit or 0)
        ax.set_ylim(0, top * 1.3)
        ax.set_title(label, fontsize=10)
    v = p.loc["value-head training loss, epoch 3", [f"ratio {r}" for r in ratios]].astype(float)
    ax = axes[4]
    ax.plot(xs, v.values, color=PURPLE, lw=1.4, marker="o", ms=6)
    values_above(ax, xs, v.values)
    ax.set_ylim(0, 2.6)
    ax.set_title("value-head training loss,\nepoch 3", fontsize=10)
    for ax in axes:
        ax.set_xticks(xs, ratios)
        ax.set_xlim(-0.4, 2.4)
        ax.set_xlabel("noise ratio")
        ax.grid(axis="x", visible=False)
        ax.axvspan(0.75, 1.25, color=shade(PURPLE, 0.1), lw=0, zorder=0)
    fig.suptitle("Noise pilots, three epochs on gen-1's corpus: the losses differ by about 2%, "
                 "while ratio 0.1 (shaded) gives e the smallest top\ncomponent and the most "
                 "directions, 0.2 breaks gate 3's limit, and the value head worsens as the "
                 "noise grows", x=0.01, ha="left", y=1.08)
    save(fig, "noise-pilot")


# ── activated pings at a toughness-1 creature ──────────────────────────────


def ping_deaths() -> None:
    """Predicted death of a toughness-1 creature under each of the four activated 1-damage
    pings of the toughness sweep, per checkpoint, the median marked."""
    values = ping_predictions()
    order = [n for n in NO_COPY_RUNS if n in values.index]
    order += sorted(gen2(values.index, copies_only=True), key=lambda n: WIDTH_OF[run_of(n)])
    widths = [None if stage_of(n) == "none" else WIDTH_OF[run_of(n)] for n in order]
    ys = grouped_positions(widths, 0.6)
    fig, ax = plt.subplots(figsize=(8.6, 0.33 * len(order) + 1.8))
    ax.axvspan(0, 0.5, color="#ececec", lw=0, zorder=0)
    ax.text(0.25, 0.012, "wrong side of an even chance:\npredicted to survive", ha="center",
            va="bottom", fontsize=7.6, color=INK_SOFT, transform=ax.transAxes)
    for name, y in zip(order, ys):
        ax.plot(values.loc[name], np.full(values.shape[1], y), "|", ms=8, mew=1.4,
                color=run_colour(name), zorder=3)
        ax.plot(values.loc[name].median(), y, zorder=4, **checkpoint_marker(name, 7.5))
        ax.text(1.02, y, f"{values.loc[name].median():.2f}", va="center", fontsize=7.8,
                color=INK_SOFT, transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color=INK, lw=0.9)
    ax.set_yticks(ys, [short(n) for n in order], fontsize=8.3)
    for label, width in zip(ax.get_yticklabels(), widths):
        if width is not None:
            label.set_color(WIDTH_COLOUR[width])
    ax.set_ylim(ys[-1] + 0.6, -1.0)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.text(1.02, 1.0, "median", fontsize=7.8, color=INK_SOFT, va="bottom",
            transform=ax.transAxes)
    ax.set_xlabel("predicted death of the toughness-1 target: marker = median of the four "
                  "pings, ticks = each ping")
    ax.set_title("Activated 1-damage pings at a toughness-1 creature: every 64-wide checkpoint "
                 "but\ne64 (earlier inputs), and arm e256-lethality, predict on the median ping "
                 "that it\nsurvives; every 128-wide checkpoint and run A256, that it dies",
                 fontsize=10.5, pad=16)
    save(fig, "ping-deaths")


# ── lethal and non-lethal hits in the corpus ───────────────────────────────


def lethal_hits() -> None:
    """Each kind of damage in 40 training shards, split into lethal and not lethal."""
    t = csv("lethal_hits_in_corpus.csv")
    t.columns = ["what", "lethal", "not_lethal"]
    labels = {"a spell or ability's damage to a targeted creature":
              "a spell or ability's damage\nto a targeted creature",
              "its damage to a creature it did not target":
              "a spell or ability's damage\nto a creature it did not target",
              "-X/-X on a targeted creature": "-X/-X on a targeted creature",
              "combat damage to a creature": "combat damage to a creature"}
    fig, ax = plt.subplots(figsize=(9.0, 0.62 * len(t) + 1.4))
    lethal, short_c = PURPLE, PURPLE_SOFT
    for i, r in t.iterrows():
        y = len(t) - 1 - i
        n = r.lethal + r.not_lethal
        share = r.lethal / n
        ax.barh(y, share, height=0.6, color=lethal, edgecolor="white", lw=1)
        ax.barh(y, 1 - share, left=share, height=0.6, color=short_c, edgecolor="white", lw=1)
        ax.text(share / 2, y, f"{share:.0%}  ({r.lethal:,} hits)", ha="center", va="center",
                fontsize=8.4, color="white")
        ax.text(share + (1 - share) / 2, y, f"{1 - share:.0%}  ({r.not_lethal:,} hits)",
                ha="center", va="center", fontsize=8.4, color=INK)
        ax.text(1.015, y, f"{n:,} hits", ha="left", va="center", fontsize=8.4, color=INK)
    ax.text(1.015, len(t) - 0.45, "total", ha="left", va="bottom", fontsize=8.4,
            color=INK_SOFT)
    ax.set_yticks(range(len(t)), [labels.get(w, w) for w in t.what][::-1], fontsize=8.6)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of the row's hits")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=lethal, label="lethal: the creature died"),
                       Patch(color=short_c, label="not lethal: the creature survived")],
              loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, fontsize=8.4)
    ax.set_axisbelow(True)
    ax.set_title("Training data, not model predictions: hits on a creature recorded in 40 "
                 "training shards (2,000 records\neach), by kind of hit, split by whether the "
                 "creature died in Forge's game. A hit is one creature taking\ndamage or −X/−X "
                 "from one event", fontsize=10.5)
    save(fig, "lethal-hits")


# ── rule families in the rarity table ──────────────────────────────────────


def rule_families() -> None:
    """Distinct texts and games per rule family, on a log scale; a family with none is
    written in place of its bar."""
    t = csv("rule_families.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 0.42 * len(t) + 1.6), sharey=True)
    for ax, (col, label) in zip(axes, (("distinct texts", "distinct texts"),
                                       ("games", "games"))):
        for i, r in t.iterrows():
            y = len(t) - 1 - i
            v = r[col]
            if v > 0:
                ax.barh(y, v, height=0.62, color=PURPLE, lw=0)
                ax.text(v * 1.12, y, f"{v:,}", va="center", fontsize=8.2, color=INK)
            else:
                ax.text(1.12, y, "0", va="center", fontsize=8.2, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(1, 2e6 if col == "games" else 6e4)
        ax.set_xticks([10 ** k for k in range(6 if col == "games" else 5)])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(-0.6, len(t) - 0.4)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"{label} (log scale)")
        ax.grid(axis="y", visible=False)
        ax.set_title(label, fontsize=10)
    axes[0].set_yticks(range(len(t)), list(t["rule family"])[::-1])
    fig.suptitle("Rule families in the rarity table span nearly three orders of magnitude, in "
                 "texts and in games; cypher has none", x=0.01, ha="left", y=1.02)
    save(fig, "rule-families")


def main() -> None:
    """Draw every chart, or only the ones named on the command line."""
    draw_requested([
        training_losses, loss_noise, probes_vs_loss, field_explained, damage_grids,
        toughness_sweep_records, boundary_by_epoch, damage_sweep, combat_grids,
        combat_keywords, gate2_heatmap, ladder, line_properties, board_abilities,
        ablation, width_summary, other_sweeps, evasion_grid, evasion_ranking,
        target_flips, colour_pips, zone_outcomes, zone_amounts, zone_who,
        run_lineage, scorecard, loss_rise_by_field, e_over_random, ping_gate_edits,
        planned_block_labels, damage_sweep_triggered, embedding_probes_gen1,
        gen1_vs_first_arm, share_vs_reliance, noise_pilot, ping_deaths, lethal_hits,
        rule_families])


if __name__ == "__main__":
    main()
