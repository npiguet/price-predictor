"""Chart the ability-effect model's training logs. Three charts, one script.

Reads the per-epoch summary lines that ``train-effect-model`` writes to each
run's ``train.log`` (a line like ``epoch 40 | train 4.0311 | card-disjoint
2.9648 | game-disjoint 2.6789 | gate F1 0.824 | zone acc 0.978 | deviance
0.014``, immediately followed by a ``  fields: ...`` line breaking the
card-disjoint loss down per output field) and renders:

  1  losses-by-stratum -- train / card-disjoint / game-disjoint loss per
     epoch, one small-multiple panel per model variant (full, identity,
     taxonomy), sharing a y-axis so the panels compare directly.
  2  per-field -- explained-% per epoch for the full model's largest-floor
     fields, one small-multiple panel per field (a shared panel would need
     ten colours, which the categorical palette does not have to give).
  3  per-field-paired -- explained-% at each run's own best epoch, full vs
     identity, same field set as (2), as paired bars.

"Explained %" is what the training log itself reports next to each field's
card-disjoint loss: ``1 - loss / floor``, where ``floor`` is a constant
(base-rate) predictor's loss on that field. 0% means the head does no better
than the base rate; negative means worse; the log prints ``n/a`` where the
field never varies in the validation sample, and those epochs are simply
absent from that field's line.

The sparse field group (keywords, counters, colors/types gained-or-lost,
control/attachment/duration changes) switches on at the first step of epoch
3 -- a curriculum boundary, not a regression -- so every chart marks it with
a dashed vertical line and every panel's loss is only comparable to another
epoch on the same side of it. Epoch-1 training loss (~15-16, an average over
the epoch's own warmup) dwarfs every later value; the loss charts clip the
y-axis to the range that matters and annotate the off-scale point instead of
letting it flatten the rest of the chart.

Colours are the first three slots of the reference categorical palette
(blue/orange/aqua), the ones validated all-pairs in both light and dark mode
(worst pair CVD Delta E 9.2 light / 9.4 dark, normal-vision 24.0 light / 20.9
dark) -- this script never puts more than three colours on one panel.

Usage
-----
    python scripts/plot_effect_model_training.py \\
        --runs-dir models/effects/runs --prefix 2026-09-17 \\
        --out-dir experiments/images --name-prefix 2026-09-19-effect-model-
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

# Palette: first three slots of the validated reference categorical order
# (references/palette.md in the dataviz skill), light-surface hex values.
# These three clear the all-pairs floor in both modes, so they are safe
# together on one panel without a secondary encoding.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, RULE, BG = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"
GRID = "#e1e0d9"

VARIANTS = (
    ("full-textless-corpus", "full model", BLUE),
    ("identity", "identity baseline", ORANGE),
    ("taxonomy", "taxonomy baseline", AQUA),
)
STRATA = (("train", "train", BLUE), ("cd", "card-disjoint", ORANGE), ("gd", "game-disjoint", AQUA))
CURRICULUM_EPOCH = 2.5  # sparse fields turn on at the first step of epoch 3

EPOCH_RE = re.compile(
    r"epoch (\d+) \| train ([\d.]+) \| card-disjoint ([\d.]+) \| "
    r"game-disjoint ([\d.]+) \| gate F1 ([\d.]+) \| zone acc ([\d.]+) \| "
    r"deviance ([\d.]+)"
)
FIELD_RE = re.compile(r"(\w+) ([\d.]+) \(([-\d]+%|n/a)\)")

# The ten fields with the largest constant-predictor floor at the full
# model's best epoch (floor recovered as loss / (1 - pct/100)), i.e. the
# fields carrying the most loss mass -- see chart 2 and chart 3.
TOP_FIELDS = (
    "gate",
    "life_delta",
    "damage_taken",
    "power_delta",
    "toughness_delta",
    "zone_outcome",
    "keywords_gained",
    "blocker_legal",
    "cards_drawn",
    "min_blockers",
)


@dataclass
class Epoch:
    epoch: int
    train: float
    cd: float
    gd: float
    f1: float
    zone: float
    dev: float
    fields: dict = field(default_factory=dict)  # name -> (loss, pct or None)


def parse_log(path: Path) -> list[Epoch]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    for i, line in enumerate(lines):
        m = EPOCH_RE.search(line)
        if not m:
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        fields = {}
        if nxt.strip().startswith("fields:"):
            for name, loss, pct in FIELD_RE.findall(nxt):
                fields[name] = (float(loss), None if pct == "n/a" else int(pct[:-1]))
        rows.append(
            Epoch(
                epoch=int(m[1]),
                train=float(m[2]),
                cd=float(m[3]),
                gd=float(m[4]),
                f1=float(m[5]),
                zone=float(m[6]),
                dev=float(m[7]),
                fields=fields,
            )
        )
    return rows


def best_epoch(rows: list[Epoch]) -> Epoch:
    return min(rows, key=lambda r: r.cd)


def style_axis(ax) -> None:
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def chart_losses(runs: dict[str, list[Epoch]], out_stub: Path) -> None:
    """Chart 1: train / card-disjoint / game-disjoint loss, one panel per variant."""
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.4), sharey=True, facecolor=BG)
    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.06, right=0.985, wspace=0.08)
    all_vals = [
        getattr(r, s)
        for rows in runs.values()
        for r in rows
        if r.epoch >= 2
        for s in ("train", "cd", "gd")
    ] + [r.cd for rows in runs.values() for r in rows if r.epoch == 1] + [
        r.gd for rows in runs.values() for r in rows if r.epoch == 1
    ]
    y_lo, y_hi = min(all_vals) - 0.25, max(all_vals) + 0.4

    for ax, (key, title, _color) in zip(axes, VARIANTS):
        rows = runs[key]
        style_axis(ax)
        ax.set_title(title, fontsize=11.5, color=INK, fontweight="bold", loc="left")
        for skey, _slabel, color in STRATA:
            xs = [r.epoch for r in rows]
            ys = [getattr(r, skey) for r in rows]
            ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", zorder=3)
        ax.axvline(CURRICULUM_EPOCH, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(0.5, max(r.epoch for r in rows) + 0.5)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.set_xlabel("epoch", fontsize=9.5, color=INK2)

        # Epoch-1 train loss is far off this scale (~15-16); the line is
        # clipped at the axis top, and an offset, arrowed annotation gives
        # the real value instead of stretching the axis to fit it.
        e1 = rows[0]
        ax.annotate(
            f"train {e1.train:.1f}",
            xy=(1, y_hi),
            xytext=(7, y_hi - 0.55),
            fontsize=8,
            color=BLUE,
            ha="left",
            va="top",
            arrowprops={
                "arrowstyle": "-",
                "color": BLUE,
                "linewidth": 0.8,
                "shrinkA": 0,
                "shrinkB": 2,
            },
        )

    axes[0].set_ylabel("loss", fontsize=9.5, color=INK2)
    handles = [
        Line2D([0], [0], color=color, linewidth=2, label=slabel) for _k, slabel, color in STRATA
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        labelcolor=INK2,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Loss by stratum, per training run",
        x=0.03,
        y=0.965,
        ha="left",
        fontsize=13,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.88,
        "Dashed line: sparse-field curriculum turns on at epoch 3, so loss is not comparable\n"
        "across it. Epoch-1 training loss is off-scale; annotated instead of stretching the axis.",
        fontsize=9,
        color=INK2,
        va="top",
    )
    _save(fig, out_stub)


def chart_per_field(full_rows: list[Epoch], out_stub: Path) -> None:
    """Chart 2: explained-% per epoch, full model, one panel per field."""
    fig, axes = plt.subplots(2, 5, figsize=(15.5, 6.6), sharey=True, facecolor=BG)
    fig.subplots_adjust(top=0.80, bottom=0.09, left=0.035, right=0.965, hspace=0.45, wspace=0.12)
    max_epoch = max(r.epoch for r in full_rows)
    for ax, fname in zip(axes.flat, TOP_FIELDS):
        style_axis(ax)
        xs, ys = [], []
        for r in full_rows:
            v = r.fields.get(fname)
            if v is None or v[1] is None:
                continue
            xs.append(r.epoch)
            ys.append(v[1])
        ax.plot(xs, ys, color=BLUE, linewidth=2, solid_capstyle="round", zorder=3)
        ax.axvline(CURRICULUM_EPOCH, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        ax.set_title(fname, fontsize=10.5, color=INK, fontweight="bold", loc="left")
        ax.set_ylim(-10, 100)
        ax.set_xlim(0.5, max_epoch + 0.5)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.axhline(0, color=RULE, linewidth=0.8, zorder=1)
        final = ys[-1] if ys else None
        if final is not None:
            ax.annotate(
                f"{final}%",
                xy=(xs[-1], final),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=8.5,
                color=INK2,
            )
    for ax in axes[-1]:
        ax.set_xlabel("epoch", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("explained %", fontsize=9, color=INK2)
    fig.suptitle(
        "Per-field explained %, full model (card-disjoint validation)",
        x=0.02,
        y=0.97,
        ha="left",
        fontsize=13,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.885,
        "explained % = 1 - loss / constant-predictor loss, per output field. Fields ordered\n"
        "by loss mass at epoch 39, largest first. Dashed line: curriculum turns on at epoch 3.",
        fontsize=9,
        color=INK2,
        va="top",
    )
    _save(fig, out_stub)


def chart_paired(full_best: Epoch, identity_best: Epoch, out_stub: Path) -> None:
    """Chart 3: explained-% at each run's best epoch, full vs identity, paired bars."""
    fig, ax = plt.subplots(figsize=(9.5, 5.6), facecolor=BG)
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.16, right=0.93)
    style_axis(ax)
    ax.grid(axis="x", visible=False)

    fields = list(TOP_FIELDS)
    full_vals = [full_best.fields[f][1] for f in fields]
    ident_vals = [identity_best.fields[f][1] for f in fields]

    y = list(range(len(fields)))
    h = 0.34
    ax.barh([v + h / 2 for v in y], full_vals, height=h, color=BLUE, zorder=3)
    ax.barh([v - h / 2 for v in y], ident_vals, height=h, color=ORANGE, zorder=3)
    for v, fv, iv in zip(y, full_vals, ident_vals):
        # Labels for a tied pair would otherwise print on top of each other:
        # stagger them left/right of the bar end instead of both flush.
        tied = fv == iv
        ax.text(
            fv + 1,
            v + h / 2 + (0.03 if tied else 0),
            f"{fv}%",
            va="center",
            fontsize=8.5,
            color=INK2,
        )
        ax.text(
            iv + 1,
            v - h / 2 - (0.03 if tied else 0),
            f"{iv}%",
            va="center",
            fontsize=8.5,
            color=INK2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(fields, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.axvline(0, color=RULE, linewidth=0.8, zorder=1)
    ax.set_xlim(min(0, min(ident_vals) - 5), max(max(full_vals), max(ident_vals)) + 10)
    ax.set_xlabel("explained % at the run's own best epoch", fontsize=9.5, color=INK2)

    handles = [
        Line2D([0], [0], color=BLUE, linewidth=8, label="full model"),
        Line2D([0], [0], color=ORANGE, linewidth=8, label="identity baseline"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        labelcolor=INK2,
        bbox_to_anchor=(0.55, 0.01),
    )
    fig.suptitle(
        "Full vs identity: where reading the ability text helps",
        x=0.03,
        y=0.965,
        ha="left",
        fontsize=13,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.885,
        f"full model best epoch {full_best.epoch}; "
        f"identity baseline best epoch {identity_best.epoch}.",
        fontsize=9,
        color=INK2,
        va="top",
    )
    _save(fig, out_stub)


def _save(fig, out_stub: Path) -> None:
    out_stub.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stub.with_suffix(".svg"), facecolor=BG)
    fig.savefig(out_stub.with_suffix(".png"), facecolor=BG, dpi=200)
    plt.close(fig)
    print(f"wrote {out_stub.with_suffix('.svg')}")
    print(f"wrote {out_stub.with_suffix('.png')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("models/effects/runs"))
    ap.add_argument("--prefix", default="2026-09-17")
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/images"))
    ap.add_argument("--name-prefix", default="2026-09-19-effect-model-")
    args = ap.parse_args()

    runs = {}
    for key, _title, _color in VARIANTS:
        log = args.runs_dir / f"{args.prefix}-{key}" / "train.log"
        runs[key] = parse_log(log)

    full_rows = runs["full-textless-corpus"]
    identity_rows = runs["identity"]
    full_best = best_epoch(full_rows)
    identity_best = best_epoch(identity_rows)

    chart_losses(runs, args.out_dir / f"{args.name_prefix}losses-by-stratum")
    chart_per_field(full_rows, args.out_dir / f"{args.name_prefix}per-field")
    chart_paired(full_best, identity_best, args.out_dir / f"{args.name_prefix}per-field-paired")


if __name__ == "__main__":
    main()
