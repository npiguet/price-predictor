"""Chart the prefix-deck-value data. Two views, one renderer.

``--view by-pick`` (default) -- three panels over picks 1-45:

  1  level -- score of the best deck buildable from the first t picks
  2  step  -- level(t) - level(t-1), the same seat against itself
  3  gap   -- gen-4's lead over each reference at the same prefix length

``--view by-level`` -- the step plotted against the level it started from, over
picks 24-45. This separates two things the by-pick view confounds. Gen-4's
*observed* step is smaller than Forge's simply because gen-4 drafts at levels
where every agent gains less; at the same level the ordering reverses and gen-4
gains most. Three regimes, and the flanking two are shaded: left of -1 a bad
deck is easy to improve and every agent does it, right of +3 nothing is left to
add and none can, and only in between does evaluating cards well pay. Levels
below -3 are dropped, being the tail of the distribution where the lines cross
arbitrarily.

Corpus pairing is enforced. ``forge-full`` and ``gen1`` appear only in the
v-forge corpora, ``gen3`` only in v-gen3, and gen-4 in both; measuring the same
agent in a different corpus shifts these numbers. Every gap in panel 3 is taken
against the gen-4 seats of its own family. In panels 1 and 2 the gen-4 line is
the v-forge one, so gen-3's level sits beside it without being comparable to it.

Colours are the first four categorical slots of the reference palette, validated
on the adjacent pairlist in both modes (worst CVD dE 9.1 light / 8.4 dark,
normal-vision 22.9 / 19.8). Aqua and yellow fall below 3:1 on the light surface,
so every series is direct-labelled as well as being in the legend. A gap line
wears the hue of the opponent it measures, so a colour means one entity
throughout.

Usage
-----
    G=models/draft/agent/gen4

    python scripts/plot_prefix_deck_value.py --view by-pick \\
        --out chart.svg --png chart.png \\
        --raw $G/lr1e-5_t2all_decay0.3-prefix-deck-scores.jsonl \\
        --raw-gen3 $G/lr1e-5_t2all_decay0.3-vgen3-prefix-deck-scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

DECK = 23
# entity -> (light, dark, slot). Colour follows the entity everywhere, including
# the gap panel, where a line wears the hue of the opponent it measures.
COLOR = {"forge-full": ("#2a78d6", "#3987e5", 1),
         "gen1":       ("#eb6834", "#d95926", 2),
         "gen4":       ("#1baf7a", "#199e70", 3),
         "gen3":       ("#eda100", "#c98500", 4)}
ORDER = ("forge-full", "gen1", "gen3", "gen4")
INK, INK2, MUTED, RULE, BG = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"
W, ML, MR = 940, 62, 130
MIN_BIN = 300          # a level bin below this is too thin to mean anything
# Below -3 the bins are the tail of the distribution: they clear MIN_BIN but the
# lines cross each other arbitrarily there, so the region carries no reading.
LEVEL_FLOOR = -3.0
# Shaded regimes. Neutral tints, not hues: a coloured band would read as a
# category and compete with the series. One step off the surface in each mode.
REGION_FILL = ("#f2f1ee", "#232320")


def load(raw_paths, cap):
    rows, agent_of = defaultdict(dict), {}
    for p in raw_paths:
        with Path(p).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[(r["draft_id"], r["seat"])][r["n_picks"]] = r["deck_score"]
                agent_of[(r["draft_id"], r["seat"])] = r["agent"]
    return rows, agent_of


def curves(rows, agent_of, cap):
    level, delta = defaultdict(list), defaultdict(list)
    for key, by_t in rows.items():
        a = agent_of[key]
        for t, sc in by_t.items():
            if sc is None or t > cap:
                continue
            level[(a, t)].append(sc)
            prev = by_t.get(t - 1)
            if prev is not None:
                delta[(a, t)].append(sc - prev)
    return ({k: st.fmean(v) for k, v in level.items() if v},
            {k: st.fmean(v) for k, v in delta.items() if v})


def step_by_level(rows, agent_of, lo_pick, hi_pick, width=0.5):
    """Mean step, binned by the level the pick started from."""
    bins = defaultdict(list)
    for key, by_t in rows.items():
        a = agent_of[key]
        for t in range(lo_pick, hi_pick + 1):
            cur, prev = by_t.get(t), by_t.get(t - 1)
            if cur is None or prev is None:
                continue
            bins[(a, round(prev / width) * width)].append(cur - prev)
    return {k: st.fmean(v) for k, v in bins.items() if len(v) >= MIN_BIN}


def nice_ticks(lo, hi, n=5):
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10) * mag
    start = math.floor(lo / step) * step
    return [start + i * step for i in range(int((hi - lo) / step) + 3)
            if lo - step / 2 <= start + i * step <= hi + step / 2]


def build(panels, geom, xdom, xticks, xlabel, title, subtitle, marks,
          palette=None, legend=None, regions=()) -> str:
    """panels: (label, {(key, x): y}, [(key, cls, text)], zero_line, base0).

    palette maps a css class to (light, dark); legend is [(cls, text)].
    Defaults to the categorical agent palette used by the by-pick views.
    """
    if palette is None:
        palette = {f'a{n}': (l, d) for l, d, n in COLOR.values()}
    if legend is None:
        legend = [(f'a{COLOR[e][2]}', e) for e in ORDER]
    height_total = geom[-1][0] + geom[-1][1] + 78
    plot_w = W - ML - MR
    x_lo, x_hi = xdom
    xs = lambda v: ML + (v - x_lo) / (x_hi - x_lo) * plot_w              # noqa: E731
    out = []
    add = out.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height_total}" '
        f'width="{W}" height="{height_total}" font-family="ui-sans-serif,system-ui,'
        f'Segoe UI,Helvetica,Arial,sans-serif" role="img" aria-label="{title}">')
    # Literal colours are the base so the chart survives a consumer that strips
    # <style> or cannot resolve var(); CSS only overrides them for dark mode.
    dark = "".join(f".s{c}{{stroke:{v[1]}}}.d{c}{{fill:{v[1]}}}"
                   for c, v in palette.items())
    add("<style>"
        ".ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
        "@media (prefers-color-scheme:dark){"
        ".bg{fill:#1a1a19}.t{fill:#ffffff}.t2{fill:#c3c2b7}.rule{stroke:#383835}"
        ".reg{fill:" + REGION_FILL[1] + "}"
        + dark + "}</style>")
    add(f'<rect class="bg" width="{W}" height="{height_total}" fill="{BG}"/>')
    add(f'<text x="{ML}" y="28" class="t" fill="{INK}" font-size="15" '
        f'font-weight="600">{title}</text>')
    for i, line in enumerate(subtitle):
        add(f'<text x="{ML}" y="{48 + i * 15}" class="t2" fill="{INK2}" '
            f'font-size="11.5">{line}</text>')

    for pi, (label, data, series, zero, base0) in enumerate(panels):
        top, height = geom[pi]
        vals = [v for _k, v in data.items()]
        lo, hi = min(vals), max(vals)
        if base0:
            # A magnitude panel is anchored at zero: a floating baseline
            # exaggerates every difference drawn on it.
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        pad = (hi - lo) * 0.08
        lo = lo if (base0 and lo == 0.0) else lo - pad
        hi = hi if (base0 and hi == 0.0) else hi + pad
        ys = lambda v, top=top, height=height, lo=lo, hi=hi: (
            top + height - (v - lo) / (hi - lo) * height)

        for r_lo, r_hi, r_label in regions:
            x0 = xs(max(r_lo, x_lo)) if r_lo is not None else ML
            x1 = xs(min(r_hi, x_hi)) if r_hi is not None else ML + plot_w
            add(f'<rect class="reg" x="{x0:.1f}" y="{top}" width="{x1 - x0:.1f}" '
                f'height="{height}" fill="{REGION_FILL[0]}"/>')
            if r_label and pi == 0:
                # A narrow band cannot hold a long label on one line.
                rows_ = r_label if isinstance(r_label, (list, tuple)) else [r_label]
                for li, text_ in enumerate(rows_):
                    add(f'<text x="{(x0 + x1) / 2:.1f}" y="{top + 16 + li * 13}" '
                        f'fill="{MUTED}" font-size="10.5" text-anchor="middle">'
                        f'{text_}</text>')
        add(f'<text x="{ML}" y="{top - 30}" class="t2" fill="{INK2}" font-size="12" '
            f'font-weight="600">{label}</text>')
        for tick in nice_ticks(lo, hi):
            y = ys(tick)
            if not (top - 1 <= y <= top + height + 1):
                continue
            op = 0.9 if (zero and abs(tick) < 1e-9) else 0.35
            add(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + plot_w}" y2="{y:.1f}" '
                f'class="rule" stroke="{RULE}" stroke-width="1" opacity="{op}"/>')
            add(f'<text x="{ML - 8}" y="{y + 4:.1f}" fill="{MUTED}" font-size="11" '
                f'text-anchor="end">{tick:+.1f}</text>')
        for mx, name in marks:
            add(f'<line x1="{xs(mx):.1f}" y1="{top}" x2="{xs(mx):.1f}" '
                f'y2="{top + height}" stroke="{MUTED}" stroke-width="1" '
                f'stroke-dasharray="3 3" opacity="0.7"/>')
            if pi == 0:
                add(f'<text x="{xs(mx):.1f}" y="{top - 8}" fill="{MUTED}" '
                    f'font-size="10.5" text-anchor="middle">{name}</text>')

        ends = []
        for key, cls, text in series:
            pts = sorted((xs(x), ys(y)) for (k, x), y in data.items() if k == key)
            if not pts:
                continue
            light, slot = palette[cls][0], cls
            add(f'<path d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                + f'" class="ln s{slot}" stroke="{light}"/>')
            ends.append([pts[-1][1], pts[-1][0], slot, text, light, pts[-1][1]])
            for (k, x), y in data.items():
                if k != key:
                    continue
                add(f'<circle cx="{xs(x):.1f}" cy="{ys(y):.1f}" r="6" fill="transparent">'
                    f'<title>{text} at {x:g}: {y:+.3f}</title></circle>')

        if len(series) > 4:                    # past 4, direct labels are clutter
            ends = []
        ends.sort(key=lambda e: e[0])          # nudge colliding end labels apart
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < 15:
                ends[i][0] = ends[i - 1][0] + 15
        for ly, lx_, slot, text, light, ey in ends:
            # Series can end at different x. Park every label in the right
            # margin and run a hairline leader back to the line, so no label
            # lands on top of the plot.
            anchor = ML + plot_w
            if anchor - lx_ > 6 or abs(ly - ey) > 1:
                add(f'<path d="M{lx_:.1f},{ey:.1f} L{anchor:.1f},{ly:.1f}" fill="none" '
                    f'stroke="{light}" stroke-width="1" opacity="0.45"/>')
            add(f'<circle cx="{anchor + 10:.1f}" cy="{ly:.1f}" r="3.5" class="d{slot}" '
                f'fill="{light}"/>')
            add(f'<text x="{anchor + 18:.1f}" y="{ly + 4:.1f}" class="t" fill="{INK}" '
                f'font-size="11.5">{text}</text>')

    base = geom[-1][0] + geom[-1][1]
    add(f'<line x1="{ML}" y1="{base}" x2="{ML + plot_w}" y2="{base}" class="rule" '
        f'stroke="{RULE}" stroke-width="1"/>')
    for tv, tl in xticks:
        add(f'<text x="{xs(tv):.1f}" y="{base + 18}" fill="{MUTED}" font-size="11" '
            f'text-anchor="middle">{tl}</text>')
    add(f'<text x="{ML + plot_w / 2:.1f}" y="{base + 38}" class="t2" fill="{INK2}" '
        f'font-size="12" text-anchor="middle">{xlabel}</text>')

    lx = ML
    add(f'<g transform="translate(0,{height_total - 16})">')
    for cls, text in legend:
        add(f'<circle cx="{lx + 4}" cy="-4" r="4" class="d{cls}" fill="{palette[cls][0]}"/>')
        add(f'<text x="{lx + 14}" y="0" class="t2" fill="{INK2}" font-size="11.5">'
            f'{text}</text>')
        lx += 30 + len(text) * 6.4
    add("</g></svg>")
    return chr(10).join(out)


def view_by_pick(rows_f, ag_f, rows_g, ag_g, cap):
    lvl_f, step_f = curves(rows_f, ag_f, cap)
    lvl_g, step_g = curves(rows_g, ag_g, cap) if rows_g else ({}, {})
    level, step = dict(lvl_f), dict(step_f)
    level.update({k: v for k, v in lvl_g.items() if k[0] == "gen3"})
    step.update({k: v for k, v in step_g.items() if k[0] == "gen3"})
    gap = {}
    for t in range(1, cap + 1):
        for opp, src in (("forge-full", lvl_f), ("gen1", lvl_f), ("gen3", lvl_g)):
            if ("gen4", t) in src and (opp, t) in src:
                gap[(opp, t)] = src[("gen4", t)] - src[(opp, t)]
    four = [(a, f"a{COLOR[a][2]}", a) for a in ORDER]
    gaps = [(a, f"a{COLOR[a][2]}", f"vs {a}")
            for a in ("forge-full", "gen1", "gen3")]
    panels = [("deck score (level)", level, four, False, False),
              ("what the pick added", step, four, True, False),
              ("gen-4's lead, paired inside each corpus", gap, gaps, True, True)]
    geom = [(134, 246), (450, 140), (712, 150)]
    ticks = [(t, str(t)) for t in (1, 5, 10, 15, 20, 25, 30, 35, 40, 45) if t <= cap]
    n = len(rows_f) + len(rows_g)
    return (panels, geom, (1, cap), ticks, "picks taken",
            "The draft is decided in its first third",
            [f"Best deck buildable from the first t picks; {n:,} seats.",
             "Below 23 picks the pool is the deck, so levels there are not comparable across t.",
             "gen3 sits in the v-gen3 corpora and the rest in v-forge; only panel 3 is paired."],
            [(16, "pack 2"), (DECK, "deck size"), (31, "pack 3")])


def view_by_level(rows_f, ag_f, rows_g, ag_g, cap):
    """One panel: the mean step against the level the pick started from.

    Picks 24-45 pooled. Below deck size the pool is the deck, so a level bin
    there would mix seats holding different numbers of cards and the step would
    be dominated by set size rather than by the card.
    """
    rows = {**{("vf",) + k: v for k, v in rows_f.items()},
            **{("g3",) + k: v for k, v in rows_g.items()}}
    agents = {**{("vf",) + k: v for k, v in ag_f.items()},
              **{("g3",) + k: v for k, v in ag_g.items()}}
    data = {k: v for k, v in step_by_level(rows, agents, DECK + 1, cap).items()
            if k[1] >= LEVEL_FLOOR}
    lo_x = min(x for _a, x in data)
    hi_x = max(x for _a, x in data)
    series = [(a, f"a{COLOR[a][2]}", a) for a in ORDER]
    ticks = [(v / 2, f"{v / 2:+.0f}") for v in range(int(lo_x * 2), int(hi_x * 2) + 1)
             if v % 2 == 0]
    return ([("mean gain from one pick", data, series, True, True)],
            [(140, 396)], (lo_x - 0.25, hi_x + 0.25), ticks,
            "deck score before the pick",
            "Gen-4's edge is in the middle, where a better card is hard but findable",
            [f"Mean step over picks {DECK + 1}-{cap}, by the deck score before the pick. "
             f"0.5-wide bins; below {LEVEL_FLOOR:+.0f} the tail is too thin to read.",
             "Left of -1 a bad deck is easy to improve and every agent does it equally.",
             "Right of +3 nothing is left to add and no agent can find it. Between them "
             "gen-4 leads, and gen-3 by less."],
            [], None, None,
            [(None, -1.0, 'any card improves it'),
             (3.0, None, ('nothing left', 'to add'))])




def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, action="append", required=True)
    ap.add_argument("--raw-gen3", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--png", type=Path)
    ap.add_argument("--view", choices=("by-pick", "by-level"), default="by-pick")
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--max-picks", type=int, default=45)
    args = ap.parse_args()

    rows_f, ag_f = load(args.raw, args.max_picks)
    rows_g, ag_g = load(args.raw_gen3, args.max_picks) if args.raw_gen3 else ({}, {})
    fn = view_by_pick if args.view == "by-pick" else view_by_level
    spec = fn(rows_f, ag_f, rows_g, ag_g, args.max_picks)
    regions = spec[10] if len(spec) > 10 else ()
    args.out.write_text(build(*spec[:10], regions=regions), encoding="utf-8")
    print(f"wrote {args.out}  ({len(rows_f)} v-forge + {len(rows_g)} v-gen3 seats)")
    if args.png:
        import cairosvg
        # Rasterise on the light surface: a PNG carries no media query, so the
        # dark-mode rules in the SVG cannot apply to it.
        cairosvg.svg2png(url=str(args.out), write_to=str(args.png),
                         scale=args.scale, background_color=BG)
        print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
