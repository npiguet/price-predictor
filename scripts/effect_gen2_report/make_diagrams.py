"""Draw the schematic diagrams of the gen-2 effect-model and knowledge-probe records.

    python scripts/effect_gen2_report/make_diagrams.py

Writes one PNG per diagram into ``experiments/2026-09-18-effect-model-gen2/charts/``,
beside the charts ``make_charts.py`` draws. The diagrams read no data: every box,
arrow and label is fixed here, from the two records and the code under
``src/effects/``, so they are redrawn only when the model or the collector changes.
"""

from __future__ import annotations

from functools import partial

import chart_style
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from chart_style import WIDTH_COLOUR, draw_requested
from chart_style import shade as tint
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.monospace": ["Consolas", "DejaVu Sans Mono"],
    "font.size": 10,
})

# Ink and surfaces. Text always wears these, never a meaning colour.
INK = "#1f2430"
SOFT = "#4b5263"
MUTED = "#7b8191"
RULE = "#c9ccd3"
PANEL = "#f4f5f7"
WHITE = "#ffffff"
# Meanings that recur across the diagrams. None of them shares a hue with the
# widths above: rose, violet, crimson and slate.
PROBE = "#c23b6e"        # where a probe reads, and the card-disjoint stratum
ABLATE = "#6b4fb3"       # what an ablation replaces, gen-3's changes, the game-disjoint stratum
DEFECT = "#b3261e"       # a collector defect in the gen-2 corpus
TRAIN_ONLY = "#8a8f9c"   # parts that exist only while training
STRUCT = "#39475c"       # model structure: encoder, trunk


save = partial(chart_style.save, pad_inches=0.12)


def canvas(width: float, height: float, scale: float = 0.62, bottom: float = 0.0):
    """A blank figure whose data units are equal on both axes."""
    # At the saving dpi, so text measured on the canvas is the text that is saved:
    # hinting rounds glyph advances differently at different resolutions.
    fig = plt.figure(figsize=(width * scale, (height - bottom) * scale), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(bottom, height)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, *, fc=WHITE, ec=RULE, lw=1.0, ls="-", radius=0.18, z=2):
    """A rounded rectangle with its lower-left corner at ``(x, y)``."""
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                           fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(patch)
    return patch


def text(ax, x, y, s, *, size=10, colour=INK, weight="normal", ha="center",
         va="center", mono=False, style="normal", z=6, **kw):
    family = "monospace" if mono else "sans-serif"
    # Forge's script keys end in "$", which matplotlib would read as mathtext.
    return ax.text(x, y, s.replace("$", r"\$"), fontsize=size, color=colour,
                   fontweight=weight, ha=ha, va=va, family=family, fontstyle=style,
                   zorder=z, **kw)


def arrow(ax, start, end, *, colour=SOFT, lw=1.3, ls="-", head=9, rad=0.0,
          shrink=2.0, z=4, style="-|>"):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=head,
                            color=colour, lw=lw, ls=ls, shrinkA=shrink, shrinkB=shrink,
                            connectionstyle=f"arc3,rad={rad}", zorder=z,
                            capstyle="round", joinstyle="round")
    ax.add_patch(patch)
    return patch


def badge(ax, x, y, label, colour=PROBE, *, r=0.26, size=9.5, z=8):
    """A filled circle carrying a short label, such as a probe rung."""
    ax.add_patch(Circle((x, y), r, fc=colour, ec=WHITE, lw=1.4, zorder=z))
    text(ax, x, y - 0.01, label, size=size, colour=WHITE, weight="bold", z=z + 1)


def tag(ax, x, y, label, colour, *, size=8.5, ha="left", z=7):
    """A small pill naming the kind of a part, such as "training only"."""
    return ax.text(x, y, label, fontsize=size, color=colour, ha=ha, va="center",
                   fontweight="bold", zorder=z,
                   bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.5",
                             fc=tint(colour, 0.12), ec="none"))


# ── 1. the model, and where each probe reads it ─────────────────────────


def _slot(ax, x, y, w, h, label, body=None, *, fc=PANEL, ec=STRUCT, lw=1.0):
    """One token of the trunk's input: a bold name and, below it, what it holds."""
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, radius=0.16)
    if body is None:
        text(ax, x + w / 2, y + h / 2, label, size=12, weight="bold", style="italic")
        return
    text(ax, x + w / 2, y + h - 0.36, label, size=9.3, weight="bold", mono=True)
    text(ax, x + w / 2, y + h - 0.68, body, size=8.1, colour=SOFT, va="top",
         linespacing=1.28)


def _width_bars(ax, x, y, step=0.3, unit=0.0115):
    """Three bars, one per width of ``e``, each as long as its width."""
    for i, width in enumerate((64, 128, 256)):
        yy = y - i * step
        ax.add_patch(FancyBboxPatch((x, yy), width * unit, 0.17,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=WIDTH_COLOUR[width], ec="none", zorder=5))
        text(ax, x + width * unit + 0.12, yy + 0.08, str(width), size=8.3, colour=SOFT,
             ha="left")


def model_and_probes() -> None:
    W, H = 24.2, 16.3
    fig, ax = canvas(W, H, scale=0.56, bottom=1.45)
    dashed = (0, (4, 2.5))

    # ── top row: script → tokenizer → encoder → e ──
    ty = 14.3
    box(ax, 0.3, ty - 0.95, 5.0, 1.9, fc=PANEL, ec="none")
    text(ax, 2.8, ty + 0.55, "an ability's Forge script", size=10.3, weight="bold")
    text(ax, 2.8, ty - 0.12, "SP$ DealDamage | NumDmg$ 3\n| ValidTgts$ Creature",
         size=8.8, mono=True, colour=SOFT, linespacing=1.35)
    text(ax, 2.8, ty - 1.25, "gen-2 reads the whole chain, joined by [SEG]", size=8.2,
         colour=MUTED, style="italic")
    badge(ax, 5.3, ty + 0.95, "1o")

    box(ax, 6.1, ty - 0.7, 2.7, 1.4, ec=STRUCT, lw=1.2)
    text(ax, 7.45, ty + 0.25, "tokenizer", size=10.3, weight="bold")
    text(ax, 7.45, ty - 0.27, "word, role and\nnumber of each token", size=8.1,
         colour=SOFT, linespacing=1.25)

    box(ax, 9.6, ty - 0.95, 3.6, 1.9, fc=tint(STRUCT, 0.10), ec=STRUCT, lw=1.3)
    text(ax, 11.4, ty + 0.4, "ability encoder", size=10.3, weight="bold")
    text(ax, 11.4, ty - 0.32, "transformer, 4 layers,\nwidth 256, pooled at [CLS]",
         size=8.1, colour=SOFT, linespacing=1.25)

    ex = 14.1
    text(ax, ex, ty + 0.62, "e", size=17, weight="bold", style="italic", ha="left")
    text(ax, ex + 0.5, ty + 0.58, "one vector per ability line", size=8.3, colour=MUTED,
         ha="left")
    _width_bars(ax, ex, ty + 0.0)
    badge(ax, ex + 3.95, ty + 0.25, "1")
    badge(ax, ex + 4.62, ty + 0.25, "1w")

    arrow(ax, (5.3, ty), (6.1, ty))
    arrow(ax, (8.8, ty), (9.6, ty))
    arrow(ax, (13.2, ty), (13.95, ty))

    # The MLM head reads the encoder's token outputs, not e.
    box(ax, 9.6, 11.15, 3.6, 1.15, ec=TRAIN_ONLY, ls=dashed, lw=1.1)
    text(ax, 11.4, 11.95, "MLM head", size=9.2, weight="bold")
    text(ax, 11.4, 11.5, "masked words, from the\nencoder's token outputs", size=7.9,
         colour=SOFT, linespacing=1.2)
    arrow(ax, (11.4, ty - 0.95), (11.4, 12.3), colour=TRAIN_ONLY, ls=(0, (3, 2)))

    # Noise sits on the way from e to everything that reads it in training.
    nx, nw, ny, nh = 14.1, 3.6, 11.15, 1.15
    box(ax, nx, ny, nw, nh, ec=TRAIN_ONLY, ls=dashed, lw=1.1)
    text(ax, nx + nw / 2, ny + 0.8, "+ Gaussian noise", size=9.2, weight="bold")
    text(ax, nx + nw / 2, ny + 0.36, "covariance r²Σ of e itself", size=7.9,
         colour=SOFT)
    arrow(ax, (ex + 1.0, ty - 0.82), (ex + 1.0, ny + nh), colour=SOFT)

    hx, hw, hy, hh = 18.55, 5.35, 10.75, 1.95
    box(ax, hx, hy, hw, hh, ec=TRAIN_ONLY, ls=dashed, lw=1.1)
    text(ax, hx + 0.22, hy + hh - 0.33, "heads that read e alone", size=9.2,
         weight="bold", ha="left")
    for i, (name, what) in enumerate([
        ("value", "amounts and costs the script states"),
        ("API type", "the script's effect, e.g. DealDamage"),
        ("parameter keys", "which keys the script sets"),
    ]):
        yy = hy + hh - 0.78 - i * 0.4
        text(ax, hx + 0.22, yy, name, size=8.1, weight="bold", ha="left", colour=SOFT)
        text(ax, hx + 1.75, yy, what, size=8.1, ha="left", colour=SOFT)
    arrow(ax, (nx + nw, ny + nh / 2), (hx, ny + nh / 2), colour=TRAIN_ONLY)
    tag(ax, hx + hw, hy + hh + 0.32, "dashed: training only", TRAIN_ONLY, ha="right")

    # ── middle row: the trunk's input slots ──
    sy, sh = 7.95, 1.9
    text(ax, 0.3, 11.35, "trunk input: one token per slot", size=10.3,
         weight="bold", ha="left")
    badge(ax, 0.3, sy + sh, "0")
    _slot(ax, 0.3, sy, 2.0, sh, "[GLOBAL]", "turn, phase,\nstep, kind of\nrecord")
    _slot(ax, 2.45, sy, 2.55, sh, "[ACT]", "acting line's e\n+ X, targets,\nmodes, "
          "outcome;\nempty in combat", fc=WHITE, lw=1.4)
    _slot(ax, 5.15, sy, 1.9, sh, "[PLAYER]", "life, cards,\nmana by\ncolour")
    _slot(ax, 7.2, sy, 1.9, sh, "[PLAYER]", "life, cards,\nmana by\ncolour")
    # One object's block: [CARD], then one slot per spell or activated line.
    bx0, bx1 = 9.25, 13.55
    box(ax, bx0 - 0.1, sy - 1.55, bx1 - bx0 + 0.2, sh + 1.65, fc=tint(STRUCT, 0.05),
        ec=RULE, lw=0.9, radius=0.22, z=1)
    _slot(ax, bx0, sy, 2.2, sh, "[CARD]", "types, P/T,\ndamage, keyword\nflags, "
          "combat")
    e_slots = [(11.6, 0.9), (12.6, 0.9)]
    for x, w in e_slots:
        _slot(ax, x, sy, w, sh, "e", fc=tint(STRUCT, 0.16))
    text(ax, bx0 + 0.08, sy - 0.18, "one block per object in the snapshot:\n[CARD], "
         "then the e of each spell or\nactivated line it prints or an\nattachment "
         "grants it", size=8.0, colour=SOFT, ha="left", va="top", linespacing=1.27)
    # The ability kinds a block never holds.
    gx = 13.85
    for i, kind in enumerate(("static", "trigger", "replace-\nment")):
        x = gx + i * 0.95
        box(ax, x, sy + 0.2, 0.85, sh - 0.4, fc=WHITE, ec=DEFECT, ls=(0, (2, 1.6)),
            lw=1.0, radius=0.12)
        text(ax, x + 0.425, sy + sh - 0.48, "×", size=12, colour=DEFECT, weight="bold")
        text(ax, x + 0.425, sy + sh / 2 - 0.3, kind, size=7.6, colour=DEFECT,
             rotation=90, linespacing=1.0)
    text(ax, gx, sy - 0.18, "never in a block: the\nmodel sees a static,\ntrigger or "
         "replacement\nonly through what it\nhas done to the board", size=8.0,
         colour=DEFECT, ha="left", va="top", linespacing=1.27)

    # The e bus: from the noise, down to [ACT] and into every ability slot.
    bus_y = 10.55
    feed = tint(STRUCT, 0.6)
    ax.plot([nx + nw / 2, nx + nw / 2, 3.72], [ny, bus_y, bus_y], color=feed, lw=1.3,
            zorder=3, solid_joinstyle="round")
    for x in (3.72, *(x + w / 2 for x, w in e_slots)):
        arrow(ax, (x, bus_y), (x, sy + sh), colour=feed, lw=1.3, head=8, shrink=0)
    text(ax, 4.0, bus_y + 0.22, "each line's e: the acting line's into [ACT], each "
         "object's own lines into its block", size=8.1, colour=MUTED, ha="left", style="italic")

    # ── the trunk ──
    tr_y, tr_h = 4.75, 1.15
    box(ax, 0.3, tr_y, 13.25, tr_h, fc=tint(STRUCT, 0.10), ec=STRUCT, lw=1.3)
    text(ax, 6.93, tr_y + tr_h / 2, "trunk: transformer, 6 layers, width 256 — "
         "every slot attends to every other", size=10.2, weight="bold")
    for x in (1.3, 3.72, 6.1, 8.15):
        arrow(ax, (x, sy), (x, tr_y + tr_h), colour=feed, lw=1.1, head=7, shrink=1)
    arrow(ax, (12.4, sy - 1.45), (12.4, tr_y + tr_h), colour=feed, lw=1.1, head=7,
          shrink=1)
    badge(ax, 13.55, tr_y + tr_h / 2, "2")

    # ── output heads ──
    oy, oh = 1.55, 2.4
    heads = [
        (0.3, 3.55, "created-objects head", "at [GLOBAL]: the tokens\nand cards the "
         "ability\nmakes"),
        (4.05, 3.85, "verdict head", "at [ACT]: can play,\naffordable, has a legal\n"
         "target, cost paid,\ntrigger fired"),
        (8.1, 5.45, "per-entity head", "at every [CARD] and [PLAYER]:\naffected? then "
         "how: died or\nmoved, damage taken, P/T,\ncounters, keywords, life, cards\n"
         "drawn; a legal target, may\nattack, may block"),
    ]
    for x, w, title, body in heads:
        box(ax, x, oy, w, oh, ec=STRUCT, lw=1.1)
        text(ax, x + 0.22, oy + oh - 0.35, title, size=9.6, weight="bold", ha="left")
        text(ax, x + 0.22, oy + oh - 0.7, body, size=8.1, colour=SOFT, ha="left",
             va="top", linespacing=1.28)
        arrow(ax, (x + w / 2, tr_y), (x + w / 2, oy + oh), colour=STRUCT, lw=1.1, head=8,
              shrink=1)
    badge(ax, 13.55, oy + oh, "3")

    # ── side panel: the ladder and the ablation ──
    px, py, pw, ph = 17.1, 1.55, 6.8, 8.95
    box(ax, px, py, pw, ph, fc=PANEL, ec="none")
    text(ax, px + 0.3, py + ph - 0.45, "Read-out ladder", size=11, weight="bold",
         ha="left")
    text(ax, px + 0.3, py + ph - 0.85, "a small probe fitted to each input; rung 3 is "
         "the\nmodel's own answer", size=8.1, colour=MUTED, ha="left", va="top",
         linespacing=1.25)
    rungs = [
        ("0", "the trunk's input features,\nevery e zeroed"),
        ("1", "rung 0 + the acting e and the target's\npooled e (a partner's too), no trunk"),
        ("1w", "rung 1 with each e swapped for a\nfixed random vector of its width"),
        ("1o", "rung 0 + the true values the\nscript states (the oracle)"),
        ("2", "the trunk's output at [ACT] or\nat the target's [CARD] slot"),
        ("3", "the model's own prediction"),
    ]
    yy = py + ph - 1.8
    for label, body in rungs:
        badge(ax, px + 0.6, yy, label, r=0.25, size=8.8)
        text(ax, px + 1.1, yy, body, size=8.4, colour=SOFT, ha="left", linespacing=1.25)
        yy -= 0.66 if "\n" in body else 0.5
    text(ax, px + 0.3, yy - 0.05, "e's share = (rung 1 − rung 0) / (rung 3 − rung 0)",
         size=8.6, weight="bold", ha="left")

    yy -= 0.7
    text(ax, px + 0.3, yy, "Ablation", size=11, weight="bold", ha="left")
    text(ax, px + 0.3, yy - 0.3, "in the frozen model, swap e for noise, its API\ntype's "
         "mean or the nearest other text's e:", size=8.1,
         colour=MUTED, ha="left", va="top", linespacing=1.25)
    mini = [("every slot", {1, 5, 6}), ("[ACT] only", {1}), ("card slots only", {5, 6})]
    cells = (("game", 0.55), ("act", 0.55), ("pl.", 0.45), ("pl.", 0.45),
             ("card", 0.55), ("e", 0.3), ("e", 0.3))
    yy -= 1.15
    xx = px + 0.35
    for name, w in cells:
        text(ax, xx + w / 2, yy, name, size=7.2, colour=MUTED)
        xx += w + 0.07
    yy -= 0.4
    for name, hit in mini:
        xx = px + 0.35
        for j, (_, w) in enumerate(cells):
            on = j in hit
            box(ax, xx, yy - 0.15, w, 0.3, fc=ABLATE if on else WHITE,
                ec=ABLATE if on else RULE, lw=0.8, radius=0.06, z=5)
            xx += w + 0.07
        text(ax, xx + 0.15, yy, name, size=8.4, colour=SOFT, ha="left")
        yy -= 0.42

    save(fig, "diagram-model-and-probes")


# ── 2. the validation strata ────────────────────────────────────────────


def _game_tile(ax, x, y, dest: str | None, *, w=0.62, h=0.44):
    """One game: a tile of record dots, outlined by the stratum it goes to."""
    colour = {"card": PROBE, "game": ABLATE}.get(dest, RULE)
    box(ax, x, y, w, h, fc=WHITE, ec=colour, lw=1.3 if dest else 0.8, radius=0.08, z=3)
    for i in range(3):
        dot = PROBE if dest == "card" and i == 1 else tint(STRUCT, 0.35)
        ax.add_patch(Circle((x + 0.15 + i * 0.16, y + h / 2), 0.055, fc=dot, ec="none",
                            zorder=4))


def validation_strata() -> None:
    W, H = 22.0, 12.4
    fig, ax = canvas(W, H, scale=0.56, bottom=0.2)
    STRATUM = {"card": PROBE, "game": ABLATE, "train": STRUCT}

    # ── the corpus: whole games ──
    text(ax, 0.3, 11.95, "the corpus, one tile per game", size=10.5, weight="bold",
         ha="left")
    text(ax, 0.3, 11.5, "each dot a record; a rose dot names a held-out card",
         size=8.3, colour=MUTED, ha="left")
    where = {(0, 2): "card", (1, 4): "card", (3, 1): "card", (4, 5): "card",
             (6, 3): "card", (2, 0): "game", (5, 2): "game"}
    for r in range(8):
        for c in range(6):
            _game_tile(ax, 0.3 + c * 0.74, 10.4 - r * 0.6, where.get((r, c)))
    text(ax, 0.3, 5.95, "outline: rose, the card-disjoint stratum;\nviolet, hashed to the "
         "game-disjoint one", size=8.3, colour=MUTED, ha="left", va="top", linespacing=1.3)
    text(ax, 0.3, 5.0, "Held-out texts are picked by a hash of each\ntext's masked "
         "template; a card carrying one\nis a held-out card. Training games are "
         "played\nwithout any held-out card, and a separate\ncollection plays them.",
         size=8.3, colour=SOFT, ha="left", va="top", linespacing=1.3)

    # ── the two questions, asked of a whole game ──
    def question(x, y, w, h, body):
        box(ax, x, y, w, h, fc=PANEL, ec=STRUCT, lw=1.1, radius=0.3)
        text(ax, x + w / 2, y + h / 2, body, size=9.0, linespacing=1.3)

    q1 = (5.9, 9.4, 4.3, 1.75)
    q2 = (5.9, 6.0, 4.3, 1.75)
    question(*q1, "does any record of the game\nname a held-out card, acting\nor "
             "anywhere on the board?")
    question(*q2, "does the game's id hash\ninto a small fixed share?")
    arrow(ax, (4.75, q1[1] + q1[3] / 2), (q1[0], q1[1] + q1[3] / 2), colour=SOFT)
    arrow(ax, (q1[0] + q1[2] / 2, q1[1]), (q2[0] + q2[2] / 2, q2[1] + q2[3]),
          colour=SOFT)
    text(ax, q1[0] + q1[2] / 2 + 0.15, q1[1] - 0.6, "no", size=8.6, colour=MUTED,
         ha="left", weight="bold")

    # ── the three destinations ──
    dx, dw = 11.6, 5.1
    dests = [
        ("card", 9.25, 2.05, "card-disjoint stratum",
         "games holding a held-out card, whose\nability texts the model never "
         "trained\non, in games it never trained on"),
        ("game", 5.85, 2.05, "game-disjoint stratum",
         "games whose ability texts the model\ntrained on, in games it never "
         "trained on"),
        ("train", 3.05, 1.35, "training", "every other game"),
    ]
    for key, y, dh, title, body in dests:
        colour = STRATUM[key]
        box(ax, dx, y, dw, dh, fc=tint(colour, 0.08), ec=colour, lw=1.4)
        ax.add_patch(FancyBboxPatch((dx, y), 0.16, dh,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=colour, ec="none", zorder=4))
        text(ax, dx + 0.4, y + dh - 0.4, title, size=10.3, weight="bold", ha="left")
        text(ax, dx + 0.4, y + dh - 0.8, body, size=8.4, colour=SOFT, ha="left",
             va="top", linespacing=1.3)
    arrow(ax, (q1[0] + q1[2], q1[1] + q1[3] / 2), (dx, 9.25 + 2.05 / 2),
          colour=STRATUM["card"], lw=1.5)
    text(ax, (q1[0] + q1[2] + dx) / 2, q1[1] + q1[3] / 2 + 0.25, "yes", size=8.6,
         colour=STRATUM["card"], weight="bold")
    arrow(ax, (q2[0] + q2[2], q2[1] + q2[3] / 2), (dx, 5.85 + 2.05 / 2),
          colour=STRATUM["game"], lw=1.5)
    text(ax, (q2[0] + q2[2] + dx) / 2, q2[1] + q2[3] / 2 + 0.25, "yes", size=8.6,
         colour=STRATUM["game"], weight="bold")
    arrow(ax, (q2[0] + q2[2] / 2, q2[1]), (dx, 3.05 + 1.35 / 2), colour=STRATUM["train"],
          lw=1.5, rad=0.3)
    text(ax, q2[0] + q2[2] / 2 - 0.15, q2[1] - 0.5, "no", size=8.6, colour=MUTED,
         weight="bold", ha="right")

    # ── what the two validation strata feed ──
    ox, ow, oh = 17.55, 4.25, 2.55
    top, bottom = 9.25 + 2.05, 5.85
    ax.plot([dx + dw + 0.1, dx + dw + 0.3, dx + dw + 0.3, dx + dw + 0.1],
            [top - 0.1, top - 0.1, bottom + 0.1, bottom + 0.1], color=SOFT, lw=1.1,
            zorder=3, solid_joinstyle="round")
    outs = [
        (top - oh, "validation sample", "a fixed sample of each stratum,\n2,048 "
         "records, the same every\nepoch: the two validation losses.\nThe card-"
         "disjoint one picks\nthe epoch a run keeps."),
        (bottom, "frozen probe set", "probe games drawn from both\nstrata and "
         "frozen once per\ncorpus under a digest: every\ncheckpoint is probed on "
         "the\nsame items."),
    ]
    for y, title, body in outs:
        arrow(ax, (dx + dw + 0.3, (top + bottom) / 2), (ox, y + oh / 2), colour=SOFT,
              rad=0.0)
        box(ax, ox, y, ow, oh, fc=WHITE, ec=SOFT, lw=1.1)
        text(ax, ox + 0.25, y + oh - 0.35, title, size=9.8, weight="bold", ha="left")
        text(ax, ox + 0.25, y + oh - 0.72, body, size=8.3, colour=SOFT, ha="left",
             va="top", linespacing=1.3)

    # ── the vocabulary, as a table ──
    tx, ty = 0.3, 0.45
    box(ax, tx, ty - 0.25, 10.0, 2.25, fc=PANEL, ec="none")
    cols = [("the game in training?", 5.4), ("its ability texts in training?", 8.35)]
    hy = ty + 1.55
    text(ax, tx + 0.3, hy, "what each stratum tests", size=9.8, weight="bold", ha="left")
    for label, x in cols:
        text(ax, x, hy, label, size=8.3, colour=MUTED)
    rows = [("game-disjoint", STRATUM["game"], "no", "yes"),
            ("card-disjoint", STRATUM["card"], "no", "not the held-out card's")]
    for i, (name, colour, a, b) in enumerate(rows):
        y = hy - 0.6 - i * 0.55
        ax.add_patch(Circle((tx + 0.42, y), 0.1, fc=colour, ec="none", zorder=4))
        text(ax, tx + 0.65, y, name, size=9.0, ha="left", weight="bold")
        text(ax, cols[0][1], y, a, size=9.0, colour=SOFT)
        text(ax, cols[1][1], y, b, size=9.0, colour=SOFT)
    text(ax, 11.6, 2.35, "a whole game goes one way, never single records:\nrecords "
         "of one game share a board, so a held-out\ncard beside a training record "
         "would leak into it", size=8.2, colour=MUTED, ha="left", va="top",
         style="italic", linespacing=1.3)

    save(fig, "diagram-validation-strata")


# ── 3. one combat damage step, and what a gen-2 combat record captures ──


def _hatch_box(ax, x, y, w, h, colour):
    """A box whose fill is diagonal hatching: something the corpus lost."""
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.14",
                           fc="none", ec=colour, lw=0.0, hatch="////", zorder=3)
    patch.set_edgecolor(tint(colour, 0.22))
    ax.add_patch(patch)


def combat_step() -> None:
    W, H = 24.4, 14.0
    fig, ax = canvas(W, H, scale=0.56, bottom=-0.05)
    x0 = 3.7
    # Each stage of a damage step: (x, width, title, detail).
    stages = [
        (4.0, 2.55, "damage\nassigned", "who deals how\nmuch to whom"),
        (6.85, 2.55, "damage\ndealt", "damage events,\nlife lost, poison"),
        (9.7, 2.55, "state-based\ndeaths", "lethal damage:\nthe creature dies"),
        (12.55, 4.5, "priority window", "combat-damage triggers\nand spells resolve, "
         "e.g. \"draw\na card\", the monarch's change"),
        (17.35, 2.55, "step ends", "Forge clears the\nassignment table"),
    ]
    end_x = 19.9
    fy, ry = 9.45, 5.6
    lanes = [("first-strike\ndamage step", fy), ("regular\ndamage step", ry)]
    lane_h = 1.75
    first_damage = 6.85

    text(ax, 0.3, 13.45, "One combat: a first-strike step, then a regular step",
         size=11.5, weight="bold", ha="left")
    text(ax, 0.3, 12.95, "the first-strike step happens only when an attacking or "
         "blocking creature has first strike or double strike; each step is its own "
         "combat record",
         size=8.6, colour=MUTED, ha="left")

    for name, y in lanes:
        box(ax, 0.3, y - 0.15, end_x - 0.3 + 0.2, lane_h + 0.3, fc=PANEL, ec="none",
            radius=0.25, z=1)
        text(ax, 0.6, y + lane_h / 2, name, size=10, weight="bold", ha="left",
             linespacing=1.2)
        for sx, sw, title, detail in stages:
            box(ax, sx, y + 0.12, sw, lane_h - 0.24, fc=WHITE, ec=RULE, lw=0.9,
                radius=0.14)
            text(ax, sx + 0.18, y + lane_h - 0.42, title.replace("\n", " "), size=8.9,
                 weight="bold", ha="left")
            text(ax, sx + 0.18, y + lane_h - 0.72, detail, size=7.8, colour=SOFT,
                 ha="left", va="top", linespacing=1.2)
        for (sx, sw, *_), (nx, *_) in zip(stages, stages[1:]):
            arrow(ax, (sx + sw, y + lane_h / 2), (nx, y + lane_h / 2), colour=MUTED,
                  lw=1.0, head=7, shrink=0.5)
        # The step begins: the phase event, where a probe fork copies the game.
        ax.plot([x0, x0], [y - 0.05, y + lane_h + 0.05], color=SOFT, lw=1.6, zorder=4,
                solid_capstyle="round")
    arrow(ax, (1.2, fy - 0.2), (1.2, ry + lane_h + 0.2), colour=MUTED, lw=1.2, head=8)
    text(ax, 1.4, (fy + ry + lane_h) / 2, "then", size=8.4, colour=MUTED, ha="left",
         style="italic")

    # ── the gen-2 combat record of the regular step ──
    top = ry + lane_h + 0.32
    ax.plot([first_damage, first_damage, end_x, end_x], [top - 0.12, top, top, top - 0.12],
            color=STRUCT, lw=2.0, zorder=5, solid_capstyle="round")
    text(ax, first_damage + 0.1, top + 0.3, "gen-2 combat record: every event until "
         "the phase changes", size=8.7, weight="bold", ha="left", colour=STRUCT)
    # Gen-3 stops the record where the stack starts.
    ax.plot([12.55, 12.55], [ry - 0.3, top + 0.1], color=ABLATE, lw=1.6, zorder=6,
            ls=(0, (3, 2)))
    text(ax, 12.7, ry - 0.3, "gen-3 ends the record at the first\nability to go on the stack",
         size=7.9, colour=ABLATE, ha="left", va="top", linespacing=1.2, weight="bold")

    # Snapshots: gen-2 takes it at the first damage event; the gen-3 draft moves it.
    for y in (ry, fy):
        ax.plot([first_damage, first_damage], [y - 0.1, y + lane_h + 0.1], color=STRUCT,
                lw=2.0, zorder=6, solid_capstyle="round")
    text(ax, first_damage + 0.42, ry - 0.42, "gen-2 snapshot, at the\nstep's first damage event",
         size=7.9, colour=STRUCT, weight="bold", ha="left", va="top", linespacing=1.2)
    ax.plot([x0 + 0.06, x0 + 0.06], [ry - 0.1, ry + lane_h + 0.1], color=ABLATE, lw=1.6,
            zorder=6, ls=(0, (3, 2)))
    text(ax, x0 + 0.06, ry - 0.3, "step begins; the gen-3\ndraft's snapshot",
         size=7.9, colour=ABLATE, weight="bold", ha="left", va="top", linespacing=1.2)

    # ── the first-strike step: what reached a gen-2 record ──
    _hatch_box(ax, 6.85, fy + 0.12, 5.4, lane_h - 0.24, DEFECT)
    ftop = fy + lane_h + 0.32
    ax.plot([first_damage, first_damage, end_x, end_x],
            [ftop - 0.12, ftop, ftop, ftop - 0.12], color=DEFECT, lw=2.0, zorder=5,
            ls=(0, (4, 2)))
    text(ax, first_damage + 0.1, ftop + 0.3, "gen-2: a record only when a player was "
         "hit, holding that damage alone; the hatched parts reached no record", size=8.7,
         weight="bold", ha="left", colour=DEFECT)

    # ── defect badges, placed where each defect lives ──
    badge(ax, 9.55, fy + lane_h + 0.05, "1", DEFECT)
    badge(ax, 6.55, ry + lane_h - 0.05, "2", DEFECT)
    badge(ax, first_damage + 0.35, ry + 0.2, "3", DEFECT)
    badge(ax, 16.85, ry + lane_h - 0.05, "4", DEFECT)
    badge(ax, first_damage + 0.17, ry - 0.62, "5", DEFECT, r=0.22, size=8.5)

    # ── what the record holds ──
    cx, cw = 20.55, 3.65
    box(ax, cx, ry - 0.15, cw, fy + lane_h + 0.15 - ry + 0.15, fc=WHITE, ec=STRUCT, lw=1.2)
    text(ax, cx + 0.25, fy + lane_h - 0.25, "a combat record", size=10, weight="bold",
         ha="left")
    text(ax, cx + 0.25, fy + lane_h - 0.65, "one damage step", size=8.3, colour=MUTED,
         ha="left")
    items = [
        "the board snapshot",
        "attackers and blocks",
        "each creature's assigned\ndamage: targets, total",
        "the step: first strike\nor regular",
        "the step's events",
    ]
    yy = fy + lane_h - 1.2
    for item in items:
        ax.add_patch(Circle((cx + 0.35, yy), 0.07, fc=STRUCT, ec="none", zorder=5))
        text(ax, cx + 0.6, yy + 0.15, item, size=8.4, colour=SOFT, ha="left", va="top",
             linespacing=1.2)
        yy -= 0.85 if "\n" in item else 0.5
    text(ax, cx + 0.25, yy - 0.05, "No line acts, so [ACT]\nis empty. The model\n"
         "predicts what this step\nalone does to every\ncard and player.", size=8.3,
         colour=INK, ha="left", va="top", linespacing=1.25)

    # ── the defect list ──
    ly = 3.15
    text(ax, 0.3, ly + 0.35, "Collector defects in the gen-2 corpus", size=10,
         weight="bold", ha="left", colour=DEFECT)
    text(ax, 11.3, ly + 0.35, "what gen-3 collects instead", size=10, weight="bold",
         ha="left", colour=ABLATE)
    rows = [
        ("1", "Forge's first-strike step went unrecognised: creature damage, life "
         "changes and deaths reached no\nrecord, and the corpus repair dropped every "
         "blocked first-strike fight", "the first-strike step, whole"),
        ("2", "a double striker's second assignment is missing: the collector assumed "
         "one table per combat", "each step's own assignment, as the players made it"),
        ("3", "the assignment was read after prevention began rewriting it: a prevented "
         "hit could show less", "that assignment, copied before prevention rewrites it"),
        ("4", "draws, tokens, discards, mills and monarch changes from the window are "
         "filed into the combat record", "the record ends at the first ability on the stack"),
        ("5", "the snapshot is taken at the first damage event, so the creature hit "
         "first already shows it", "the gen-3 draft: a snapshot at the step's start"),
    ]
    yy = ly - 0.3
    for label, body, fix in rows:
        badge(ax, 0.6, yy - (0.17 if "\n" in body else 0), label, DEFECT, r=0.22,
              size=8.5)
        text(ax, 1.05, yy, body, size=8.3, colour=SOFT, ha="left", va="top" if "\n"
             in body else "center", linespacing=1.25)
        text(ax, 11.3, yy, fix, size=8.3, colour=ABLATE, ha="left", va="center")
        yy -= 0.95 if "\n" in body else 0.55

    save(fig, "diagram-combat-step")


# ── 4. what the encoder reads: gen-1 against gen-2 ──────────────────────

# Ajani's Pridemate, the gen-2 record's own example, exactly as Forge's script
# (cardsfolder/a/ajanis_pridemate.txt) and the two sidecars spell it. The
# converter renders a trait's parameters in key order, which is why the encoded
# texts differ in order from the script.
PRIDEMATE_SCRIPT = [
    "T:Mode$ LifeGained | ValidPlayer$ You | TriggerZones$ Battlefield |",
    "  Execute$ TrigPutCounter | TriggerDescription$ Whenever you gain life,",
    "  put a +1/+1 counter on CARDNAME.",
    "SVar:TrigPutCounter:DB$ PutCounter | Defined$ Self | CounterType$ P1P1 |",
    "  CounterNum$ 1",
]
PRIDEMATE_ROOT_GEN1 = [
    "Execute$ TrigPutCounter | Mode$ LifeGained |",
    "TriggerDescription$ Whenever you gain life, put",
    "a +1/+1 counter on CARDNAME. | TriggerZones$",
    "Battlefield | ValidPlayer$ You",
]
PRIDEMATE_ROOT_GEN2 = [
    "Execute$ SV1 | Mode$ LifeGained |",
    "TriggerDescription$ Whenever you gain life, put",
    "a +1/+1 counter on CARDNAME. | TriggerZones$",
    "Battlefield | ValidPlayer$ You",
]
PRIDEMATE_EFFECT_GEN2 = [
    "SV1: CounterNum$ 1 | CounterType$ P1P1 |",
    "DB$ PutCounter | Defined$ Self",
]
# Inspiring Overseer's trigger, from its gen-2 sidecar: a chain of three segments.
OVERSEER_GEN2 = [
    ["Destination$ Battlefield | Execute$ SV1 |", "Mode$ ChangesZone | TriggerDescription$",
     "When CARDNAME enters, you gain 1 life", "and draw a card. | ValidCard$ Card.Self"],
    ["SV1: DB$ GainLife | LifeAmount$ 1 |", "SubAbility$ SV2"],
    ["SV2: DB$ Draw"],
]
# Barbed Lightning, from its gen-2 sidecar: the root line and one line per mode.
CHARM_GEN2 = [
    ("spell", ["Choices$ SV1,SV2 | SP$ Charm"]),
    ("option 1", ["SV1: DB$ DealDamage | NumDmg$ 3 |", "SpellDescription$ CARDNAME deals 3",
                  "damage to target creature. |", "ValidTgts$ Creature"]),
    ("option 2", ["SV1: DB$ DealDamage | NumDmg$ 3 |", "SpellDescription$ CARDNAME deals 3",
                  "damage to target player or", "planeswalker. | TgtPrompt$ Select",
                  "target player or planeswalker |", "ValidTgts$ Player,Planeswalker"]),
]


class Mono:
    """Monospaced lines laid out on a canvas, with any substring highlightable."""
    def __init__(self, fig, ax, size: float):
        self.ax, self.size = ax, size
        probe = ax.text(0, 0, "M" * 40, fontsize=size, family="monospace")
        renderer = fig.canvas.get_renderer()
        extent = probe.get_window_extent(renderer).transformed(ax.transData.inverted())
        probe.remove()
        self.char = extent.width / 40
        self.line = extent.height * 1.32

    def block(self, x, y_top, lines, *, colour=INK, marks=()):
        """Draw ``lines`` from ``y_top`` down; ``marks`` are (substring, colour) pairs."""
        for i, line in enumerate(lines):
            y = y_top - (i + 0.5) * self.line
            for word, mark in marks:
                start = line.find(word)
                while start >= 0:
                    self.ax.add_patch(FancyBboxPatch(
                        (x + start * self.char - 0.03, y - self.line * 0.36),
                        len(word) * self.char + 0.06, self.line * 0.72,
                        boxstyle="round,pad=0,rounding_size=0.06",
                        fc=tint(mark, 0.2), ec="none", zorder=5))
                    start = line.find(word, start + 1)
            text(self.ax, x, y, line, size=self.size, colour=colour, ha="left", mono=True)
        return len(lines) * self.line

    def width(self, lines) -> float:
        return max(len(line) for line in lines) * self.char


def _segment(ax, mono, x, y_top, lines, *, fc, ec, marks=(), ls="-", pad=0.22):
    """A rounded box around a block of monospaced lines; returns its height."""
    h = len(lines) * mono.line + 2 * pad
    w = mono.width(lines) + 2 * pad
    box(ax, x, y_top - h, w, h, fc=fc, ec=ec, lw=1.1, ls=ls, radius=0.14)
    mono.block(x + pad, y_top - pad, lines, marks=marks)
    return w, h


def _seg_chip(ax, x, y):
    box(ax, x - 0.42, y - 0.17, 0.84, 0.34, fc=STRUCT, ec=STRUCT, radius=0.1, z=6)
    text(ax, x, y, "[SEG]", size=7.6, colour=WHITE, weight="bold", mono=True, z=7)


def encoder_reads() -> None:
    W, H = 24.0, 17.6
    fig, ax = canvas(W, H, scale=0.56, bottom=2.4)
    mono = Mono(fig, ax, 8.6)
    small = Mono(fig, ax, 7.9)
    root_fc, effect_fc = WHITE, tint(STRUCT, 0.07)
    label = ABLATE

    # ── the Forge script ──
    text(ax, 0.3, 17.1, "Ajani's Pridemate in Forge's script: a trigger line and the "
         "SVar its Execute$ names", size=10.5, weight="bold", ha="left")
    sw, sh = _segment(ax, mono, 0.3, 16.75, PRIDEMATE_SCRIPT, fc=PANEL, ec="none",
                      marks=[("TrigPutCounter", MUTED)], pad=0.3)
    text(ax, 0.3 + sw + 0.35, 16.75 - 0.3 - 1.4 * mono.line, "the trigger: when it "
         "fires, and\nthe label of the ability it executes", size=8.3, colour=SOFT,
         ha="left", va="center", linespacing=1.25)
    text(ax, 0.3 + sw + 0.35, 16.75 - 0.3 - 4.0 * mono.line, "the SVar that label "
         "names: the\neffect, which puts the counter", size=8.3, colour=SOFT, ha="left",
         va="center", linespacing=1.25)

    # ── gen-1 and gen-2, side by side ──
    top = 16.75 - sh - 0.75
    gx1, gx2 = 0.3, 12.35
    text(ax, gx1, top, "gen-1 encodes the root line only", size=10.3, weight="bold",
         ha="left")
    text(ax, gx2, top, "gen-2 encodes the whole chain, in resolution order", size=10.3,
         weight="bold", ha="left")
    y = top - 0.4
    w1, h1 = _segment(ax, mono, gx1, y, PRIDEMATE_ROOT_GEN1, fc=root_fc, ec=STRUCT,
                      marks=[("TrigPutCounter", MUTED)])
    ghost_y = y - h1 - 0.45
    w_ghost, h_ghost = _segment(ax, mono, gx1, ghost_y, ["DB$ PutCounter | Defined$ Self | "
                                "CounterType$ P1P1 | CounterNum$ 1"], fc=WHITE, ec=DEFECT,
                                ls=(0, (3, 2)))
    text(ax, gx1 + w_ghost + 0.25, ghost_y - h_ghost / 2, "not in the text: the counter\n"
         "type and count reach the\nmodel only as the English\nof the description", size=8.2,
         colour=DEFECT, ha="left", va="center", linespacing=1.25)
    ax.plot([gx1 + 0.2, gx1 + w_ghost - 0.2], [ghost_y - h_ghost / 2] * 2,
            color=tint(DEFECT, 0.6), lw=1.0, zorder=7)

    w2, h2 = _segment(ax, mono, gx2, y, PRIDEMATE_ROOT_GEN2, fc=root_fc, ec=STRUCT,
                      marks=[("SV1", label)])
    chip_y = y - h2 - 0.3
    _seg_chip(ax, gx2 + 0.75, chip_y)
    w3, h3 = _segment(ax, mono, gx2, chip_y - 0.3, PRIDEMATE_EFFECT_GEN2, fc=effect_fc,
                      ec=STRUCT, marks=[("SV1", label)])
    note_x = gx2 + max(w2, w3) + 0.3
    text(ax, note_x, y - 0.25, "root segment:\nthe trigger", size=8.2, colour=SOFT,
         ha="left", va="top", linespacing=1.25)
    text(ax, note_x, chip_y - 0.3 - h3 / 2, "the effect, opened\nby its label", size=8.2,
         colour=SOFT, ha="left", va="center", linespacing=1.25)
    text(ax, gx2, chip_y - 0.3 - h3 - 0.45, "TrigPutCounter is renamed SV1: chain "
         "labels are numbered SV1, SV2, …\nby first appearance, restarting on every "
         "line, so a reference and the segment\nit names still match while the "
         "author's name for it is dropped.", size=8.2, colour=label, ha="left",
         va="top", linespacing=1.3)

    # ── bottom insets: a sub-ability chain, and a charm's modes ──
    iy = min(ghost_y - h_ghost, chip_y - 0.3 - h3 - 1.35) - 0.8
    ax.plot([0.3, W - 0.3], [iy + 0.35, iy + 0.35], color=RULE, lw=1.0)
    text(ax, gx1, iy - 0.15, "A SubAbility$ adds a segment: Inspiring Overseer, gen-2",
         size=9.8, weight="bold", ha="left")
    yy = iy - 0.6
    for k, seg in enumerate(OVERSEER_GEN2):
        w, h = _segment(ax, small, gx1, yy, seg, fc=root_fc if k == 0 else effect_fc,
                        ec=STRUCT, marks=[("SV1", label), ("SV2", label)], pad=0.18)
        yy -= h
        if k < len(OVERSEER_GEN2) - 1:
            _seg_chip(ax, gx1 + 0.75, yy - 0.27)
            yy -= 0.54
    text(ax, gx1 + 5.3, iy - 1.55 - small.line * 4, "trigger, then the\nExecute$ effect, "
         "then\nits SubAbility$", size=8.2, colour=SOFT, ha="left", va="top",
         linespacing=1.25)

    text(ax, gx2, iy - 0.15, "A charm: each mode its own line, its own e — Barbed "
         "Lightning, gen-2", size=9.8, weight="bold", ha="left")
    yy = iy - 0.6
    for name, seg in CHARM_GEN2:
        w, h = _segment(ax, small, gx2 + 1.5, yy, seg, fc=root_fc if name == "spell"
                        else effect_fc, ec=STRUCT, marks=[("SV1", label), ("SV2", label)],
                        pad=0.18)
        text(ax, gx2 + 1.3, yy - h / 2, name, size=8.4, colour=SOFT, ha="right",
             weight="bold")
        ex_x = gx2 + 1.5 + w + 0.25
        arrow(ax, (ex_x, yy - h / 2), (ex_x + 0.6, yy - h / 2), colour=MUTED, lw=1.0,
              head=7, shrink=0)
        box(ax, ex_x + 0.65, yy - h / 2 - 0.25, 0.55, 0.5, fc=tint(STRUCT, 0.16),
            ec=STRUCT, lw=1.0, radius=0.1)
        text(ax, ex_x + 0.925, yy - h / 2, "e", size=10.5, style="italic",
             weight="bold")
        yy -= h + 0.25
    text(ax, gx2 + 1.5, yy - 0.05, "The root line keeps only its own segment, the choice. "
         "Each mode's line\nopens with SV1, because the numbering restarts on every "
         "line.",
         size=8.2, colour=label, ha="left", va="top", linespacing=1.3)

    save(fig, "diagram-encoder-reads")


# ── 5. the five shapes the training noise on e could take ───────────────

NOISE = "#5e6370"  # the training noise: a darker shade of TRAIN_ONLY, read on a white panel


def _cloud(n: int = 260, seed: int = 7):
    """A schematic e cloud: elongated, tilted 30° off both coordinate axes."""
    rng = np.random.default_rng(seed)
    angle = np.radians(30)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    scale = np.diag([3.0, 0.55])
    points = rng.standard_normal((n, 2)) @ scale @ rot.T
    return points, rot @ scale @ scale @ rot.T


def _ellipse(ax, centre, cov, k, **kw):
    """The ``k``-sigma ellipse of ``cov`` around ``centre``."""
    values, vectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    ax.add_patch(Ellipse(centre, 2 * k * np.sqrt(values[1]), 2 * k * np.sqrt(values[0]),
                         angle=angle, **kw))


def noise_shapes() -> None:
    """One panel per shape of the noise; the cloud is schematic, the noise exaggerated."""
    points, cov = _cloud()
    ratio = 0.35  # drawn far above gen-2's 0.1 so that every shape is visible
    std = np.sqrt(np.diag(cov))
    panels = [
        ("gen-1: a fixed noise", "0.05 per coordinate, whatever the\nspread. In gen-1's "
         "cache the noise\nis 0.40 long against vectors 53\nlong: under a hundredth",
         dict(kind="fixed")),
        ("each e normalized to\none length first", "the noise cannot be escaped, but\nlength "
         "is lost and every consumer\nof the cache reads a new e", dict(kind="sphere")),
        ("one level, sized to\nthe average spread", "rescaling e changes nothing, but\nthe "
         "long direction outruns the\nnoise and can carry a text key", dict(kind="iso")),
        ("each coordinate sized\nto its own spread", "widening one axis no longer helps,\nbut "
         "a diagonal direction still\noutruns the noise", dict(kind="axis")),
        ("the covariance of the\ne vectors (gen-2)", "the same noise-to-spread ratio\nalong "
         "every direction: no\nreshaping of e escapes it", dict(kind="cov")),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15.6, 4.9), dpi=150)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.78, bottom=0.27, wspace=0.08)
    anchor = np.array([2.2, 1.25])  # one text's e, the vector the noise is drawn around
    for ax, (title, body, spec) in zip(axes, panels):
        kind = spec["kind"]
        ax.set_aspect("equal")
        ax.set_xlim(-6.2, 6.2)
        ax.set_ylim(-4.6, 4.6)
        ax.axis("off")
        ax.axhline(0, color=RULE, lw=0.8, zorder=0)
        ax.axvline(0, color=RULE, lw=0.8, zorder=0)
        if kind == "sphere":
            directions = points / np.linalg.norm(points, axis=1, keepdims=True)
            shown, centre = 3.2 * directions, 3.2 * anchor / np.linalg.norm(anchor)
            ax.add_patch(Circle((0, 0), 3.2, fc="none", ec=RULE, lw=0.8, ls="--", zorder=1))
        else:
            shown, centre = points, anchor
            _ellipse(ax, (0, 0), cov, 2.0, fc="none", ec=MUTED, lw=0.9, ls="--", zorder=1)
        ax.scatter(shown[:, 0], shown[:, 1], s=5, color=tint(STRUCT, 0.4), lw=0, zorder=2)
        noise = dict(fc=tint(NOISE, 0.28), ec=NOISE, lw=1.6, zorder=4, alpha=0.9)
        if kind == "fixed":
            ax.add_patch(Circle(centre, 0.14, **noise))
            ax.annotate("noise", centre, xytext=(26, 18), textcoords="offset points",
                        fontsize=8.2, color=NOISE, arrowprops=dict(arrowstyle="-", color=NOISE,
                                                                  lw=0.8))
        elif kind == "sphere":
            ax.add_patch(Circle(centre, 0.6, **noise))
        elif kind == "iso":
            ax.add_patch(Circle(centre, ratio * 2 * np.sqrt(np.trace(cov) / 2), **noise))
        elif kind == "axis":
            ax.add_patch(Ellipse(centre, 2 * ratio * 2 * std[0], 2 * ratio * 2 * std[1],
                                 **noise))
        else:
            _ellipse(ax, centre, cov, ratio * 2, **noise)
        ax.plot(*centre, "o", ms=3.5, color=INK, zorder=5)
        if kind == "cov":
            ax.add_patch(FancyBboxPatch((-6.0, -4.4), 12.0, 8.8, boxstyle="round,pad=0,"
                                        "rounding_size=0.5", fc="none", ec=ABLATE, lw=1.6,
                                        zorder=0))
        ax.set_title(title, fontsize=10, color=INK, fontweight="bold", pad=6)
        ax.text(0.5, -0.03, body, transform=ax.transAxes, ha="center", va="top", fontsize=8.6,
                color=SOFT, linespacing=1.3)
    handles = [plt.Line2D([], [], marker="o", ls="none", ms=4, color=tint(STRUCT, 0.4),
                          label="e vectors (schematic cloud)"),
               plt.Line2D([], [], ls="--", color=MUTED, lw=0.9,
                          label="the cloud's spread, two standard deviations"),
               plt.Line2D([], [], marker="o", ls="none", ms=4, color=INK,
                          label="one text's e"),
               matplotlib.patches.Patch(fc=tint(NOISE, 0.28), ec=NOISE,
                                        label="the noise added to it in training, drawn at "
                                              "3.5 times gen-2's ratio of 0.1")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.4, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.01, 0.97, "Five shapes for the training noise on e, and how the encoder can "
             "still escape each", fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.905, "A length penalty in the loss, the sixth option, has no shape of "
             "its own: it keeps gen-1's fixed noise and lets the penalty's weight set its "
             "effective size.", fontsize=8.6, color=MUTED, va="top")
    save(fig, "diagram-noise-shapes")


# ── 6. where each of gen-2's design changes acts ────────────────────────

#: The pipeline's stages, left to right, each with its changes:
#: (text, acts at collection time, came during the runs).
DESIGN_STAGES = [
    ("convert", "Forge scripts → sidecars", [
        ("whole chain encoded: trigger,\nExecute$ and sub-abilities", True, False),
        ("SVar labels renumbered\nby position", True, False),
        ("each charm mode its own line", True, False),
        ("holdout chosen on the\nchained template", True, False),
    ]),
    ("text to tokens", "vocabulary, keyword expansion", [
        ("script tokenizer deleted;\ncamel-case compounds split", False, False),
        ("keywords left as tokens; found\nby display name; templates filled", False, False),
        ("one keyword withheld\nfor the zero-shot check", False, False),
        ("every arm deleted the withheld\nkeyword instead of expanding it", False, True),
    ]),
    ("collect", "Forge AI + connector", [
        ("a random seat in one match\nin four, at half its decisions", True, False),
        ("each charm mode recorded\nas its own ability", True, False),
        ("real decisions only, no what-if\nqueries; board in the dedup key", True, False),
        ("game-disjoint stratum drawn toward\nrare-keyword combats", True, False),
        ("corpus collected from scratch", True, False),
    ]),
    ("build-corpus", "curated training set", [
        ("rarity-weight ceiling set\nagainst the 99th percentile", False, False),
        ("rule families balanced;\noutcomes balanced within each", False, False),
        ("rare-keyword combats get\na budget of their own", False, False),
    ]),
    ("train", "encoder + effect head", [
        ("noise with the covariance\nof the e vectors", False, False),
        ("auxiliary value head reads\nthe script's values from e", False, False),
        ("a sweep over the width of e", False, False),
        ("lethality copies labelled\nby the damage rule", False, True),
        ("thermometer codes; \"at least k\"\nlabels for the value head", False, True),
    ]),
    ("evaluate", "gates + probes", [
        ("unseen texts reported per text\nand per rule family", False, False),
        ("no baselines: the knowledge\nprobes measure what they did", False, False),
        ("arms compared by hand", False, False),
    ]),
]


def design_changes() -> None:
    """Gen-2's pipeline, each design change pinned under the stage where it acts."""
    W = 6 * 4.3 + 0.4
    H = 13.6
    fig, ax = canvas(W, H, scale=0.62, bottom=0.0)
    collect_fc = tint(PROBE, 0.12)
    col_w, gap = 3.9, 0.4
    top = 11.6
    lowest = top
    for i, (name, sub, changes) in enumerate(DESIGN_STAGES):
        x = 0.3 + i * (col_w + gap)
        box(ax, x, top, col_w, 1.15, fc=STRUCT, ec=STRUCT, radius=0.22, z=3)
        text(ax, x + col_w / 2, top + 0.75, name, size=10.5, colour=WHITE, weight="bold")
        text(ax, x + col_w / 2, top + 0.35, sub, size=8.2, colour=tint(WHITE, 1.0))
        if i < len(DESIGN_STAGES) - 1:
            arrow(ax, (x + col_w, top + 0.575), (x + col_w + gap, top + 0.575),
                  colour=STRUCT, lw=1.6, shrink=0.5)
        y = top - 0.35
        for body, at_collection, in_run in changes:
            h = 0.5 + 0.42 * body.count("\n") + 0.42
            fc = collect_fc if at_collection else WHITE
            ec = PROBE if at_collection else RULE
            box(ax, x, y - h, col_w, h, fc=fc, ec=DEFECT if in_run else ec,
                lw=1.5 if in_run else 1.0, ls=(0, (3, 2)) if in_run else "-", radius=0.14)
            text(ax, x + 0.2, y - h / 2, body, size=9.2, ha="left", linespacing=1.25)
            if in_run:
                tag(ax, x + col_w - 0.12, y - 0.02, "during the runs", DEFECT, size=7.0,
                    ha="right")
            y -= h + 0.22
        lowest = min(lowest, y)
    text(ax, 0.3, H - 0.35, "Where each of gen-2's design changes acts", size=13,
         weight="bold", ha="left", va="top")
    ly = lowest - 0.55
    box(ax, 0.3, ly - 0.25, 0.7, 0.5, fc=collect_fc, ec=PROBE, radius=0.1)
    text(ax, 1.15, ly, "fixed when the corpus is collected: changing it means collecting again",
         size=8.8, ha="left", colour=SOFT)
    box(ax, 9.6, ly - 0.25, 0.7, 0.5, fc=WHITE, ec=RULE, radius=0.1)
    text(ax, 10.45, ly, "acts after collection", size=8.8, ha="left", colour=SOFT)
    box(ax, 14.0, ly - 0.25, 0.7, 0.5, fc=WHITE, ec=DEFECT, lw=1.5, ls=(0, (3, 2)),
        radius=0.1)
    text(ax, 14.85, ly, "came during the runs; the design sections do not describe it",
         size=8.8, ha="left", colour=SOFT)
    bottom = ly - 0.5
    ax.set_ylim(bottom, H)
    fig.set_size_inches(W * 0.62, (H - bottom) * 0.62)
    save(fig, "diagram-design-changes")


def main() -> None:
    """Draw every diagram, or only the ones named on the command line."""
    draw_requested([model_and_probes, validation_strata, combat_step, encoder_reads,
                    noise_shapes, design_changes])


if __name__ == "__main__":
    main()
