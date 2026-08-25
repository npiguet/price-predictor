"""Render the prefix-deck-value curves as an SVG (and optionally a PNG).

Three stacked panels sharing the pick axis. Level and step are different
measures, and the gap is a third, so each gets its own panel rather than a
second y-axis:

  1  level -- the score of the best deck buildable from the first t picks
  2  step  -- level(t) - level(t-1), the same seat against itself
  3  gap   -- gen-4's lead over each reference at the same prefix length

Corpus pairing matters and the script enforces it. ``forge-full`` and ``gen1``
appear only in the v-forge corpora, ``gen3`` only in the v-gen3 corpora, and
gen-4 sits in both. Measuring the same agent in a different corpus shifts these
numbers, so every gap in panel 3 is taken against the gen-4 seats of the *same*
family: the first two from v-forge, the gen-3 gap from v-gen3. In panels 1 and 2
the gen-4 line is the v-forge one, which puts gen-3's level next to it on the
page without making the two directly comparable -- only panel 3 is paired.

Colours are the first four categorical slots of the reference palette, validated
on the adjacent pairlist for both modes (worst CVD dE 9.1 light / 8.4 dark,
normal-vision 22.9 / 19.8). Aqua and yellow fall below 3:1 on the light surface,
so every series carries a direct label as well as a legend entry. In panel 3 a
line wears the colour of the opponent it measures, so a hue means one entity
throughout.

Usage
-----
    G=models/draft/agent/gen4

    python scripts/plot_prefix_deck_value.py \\
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
# panel 3, where a gap line wears the hue of the opponent it measures.
COLOR = {"forge-full": ("#2a78d6", "#3987e5", 1),
         "gen1":       ("#eb6834", "#d95926", 2),
         "gen4":       ("#1baf7a", "#199e70", 3),
         "gen3":       ("#eda100", "#c98500", 4)}
INK, INK2, MUTED, RULE, BG = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"
W, H = 940, 958
ML, MR = 62, 130
PANELS = [(134, 246), (450, 140), (712, 150)]     # (top, height)
MARKS = [(16, "pack 2"), (23, "deck size"), (31, "pack 3")]


def load(raw_paths: list[Path], cap: int):
    rows: dict = defaultdict(dict)
    agent_of: dict = {}
    for p in raw_paths:
        with p.open(encoding="utf-8") as fh:
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
    level: dict = defaultdict(list)
    delta: dict = defaultdict(list)
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
            {k: st.fmean(v) for k, v in delta.items() if v},
            len(rows))


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10) * mag
    start = math.floor(lo / step) * step
    return [start + i * step for i in range(int((hi - lo) / step) + 3)
            if lo - step / 2 <= start + i * step <= hi + step / 2]


def build(panels, cap, n_seats) -> str:
    plot_w = W - ML - MR
    xs = lambda t: ML + (t - 1) / (cap - 1) * plot_w                       # noqa: E731
    out: list[str] = []
    add = out.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" font-family="ui-sans-serif,system-ui,Segoe UI,Helvetica,Arial,'
        f'sans-serif" role="img" aria-label="Deck score by number of picks for four '
        f'drafting agents, and gen-4 lead over each">')
    # Literal colours are the base so the chart survives a consumer that strips
    # <style> or cannot resolve var(); CSS only overrides them for dark mode.
    dark = "".join(f".s{n}{{stroke:{d}}}.d{n}{{fill:{d}}}" for _l, d, n in COLOR.values())
    add("<style>"
        ".ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
        "@media (prefers-color-scheme:dark){"
        ".bg{fill:#1a1a19}.t{fill:#ffffff}.t2{fill:#c3c2b7}.rule{stroke:#383835}"
        + dark + "}</style>")
    add(f'<rect class="bg" width="{W}" height="{H}" fill="{BG}"/>')

    add(f'<text x="{ML}" y="28" class="t" fill="{INK}" font-size="15" font-weight="600">'
        f'The draft is decided in its first third</text>')
    subtitle = [
        f"Best deck buildable from the first t picks; {n_seats:,} seats.",
        "Below 23 picks the pool is the deck, so levels there are not comparable across t.",
        "gen3 sits in the v-gen3 corpora and the rest in v-forge; only panel 3 is paired.",
    ]
    for i, line in enumerate(subtitle):
        add(f'<text x="{ML}" y="{48 + i * 15}" class="t2" fill="{INK2}" font-size="11.5">'
            f'{line}</text>')

    for pi, (label, data, series, zero) in enumerate(panels):
        top, height = PANELS[pi]
        vals = [v for (_s, t), v in data.items() if t <= cap]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        ys = lambda v, top=top, height=height, lo=lo, hi=hi: (
            top + height - (v - lo) / (hi - lo) * height)

        add(f'<text x="{ML}" y="{top - 30}" class="t2" fill="{INK2}" font-size="12" '
            f'font-weight="600">{label}</text>')
        for tick in nice_ticks(lo, hi):
            y = ys(tick)
            if not (top - 1 <= y <= top + height + 1):
                continue
            op = 0.9 if (zero and abs(tick) < 1e-9) else 0.35
            add(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + plot_w}" y2="{y:.1f}" class="rule" '
                f'stroke="{RULE}" stroke-width="1" opacity="{op}"/>')
            add(f'<text x="{ML - 8}" y="{y + 4:.1f}" fill="{MUTED}" font-size="11" '
                f'text-anchor="end">{tick:+.1f}</text>')
        for t, name in MARKS:
            x = xs(t)
            add(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" '
                f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="3 3" opacity="0.7"/>')
            if pi == 0:
                add(f'<text x="{x:.1f}" y="{top - 8}" fill="{MUTED}" font-size="10.5" '
                    f'text-anchor="middle">{name}</text>')

        ends = []
        for key, entity, text in series:
            pts = [(xs(t), ys(data[(key, t)])) for t in range(1, cap + 1)
                   if (key, t) in data]
            if not pts:
                continue
            light, _d, slot = COLOR[entity]
            path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            add(f'<path d="{path}" class="ln s{slot}" stroke="{light}"/>')
            ends.append([pts[-1][1], pts[-1][0], slot, text, light])
            for t in range(1, cap + 1):
                if (key, t) not in data:
                    continue
                add(f'<circle cx="{xs(t):.1f}" cy="{ys(data[(key, t)]):.1f}" r="6" '
                    f'fill="transparent"><title>{text} pick {t}: '
                    f'{data[(key, t)]:+.3f}</title></circle>')

        ends.sort(key=lambda e: e[0])            # nudge colliding end labels apart
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < 15:
                ends[i][0] = ends[i - 1][0] + 15
        for ly, lx_, slot, text, light in ends:
            add(f'<circle cx="{lx_ + 10:.1f}" cy="{ly:.1f}" r="3.5" class="d{slot}" '
                f'fill="{light}"/>')
            add(f'<text x="{lx_ + 18:.1f}" y="{ly + 4:.1f}" class="t" fill="{INK}" '
                f'font-size="11.5">{text}</text>')

    baseline = PANELS[-1][0] + PANELS[-1][1]
    add(f'<line x1="{ML}" y1="{baseline}" x2="{ML + plot_w}" y2="{baseline}" class="rule" '
        f'stroke="{RULE}" stroke-width="1"/>')
    for t in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
        if t > cap:
            continue
        add(f'<text x="{xs(t):.1f}" y="{baseline + 18}" fill="{MUTED}" font-size="11" '
            f'text-anchor="middle">{t}</text>')
    add(f'<text x="{ML + plot_w / 2:.1f}" y="{baseline + 38}" class="t2" fill="{INK2}" '
        f'font-size="12" text-anchor="middle">picks taken</text>')

    lx = ML
    add(f'<g transform="translate(0,{H - 16})">')
    for entity in ("forge-full", "gen1", "gen3", "gen4"):
        light, _d, slot = COLOR[entity]
        add(f'<circle cx="{lx + 4}" cy="-4" r="4" class="d{slot}" fill="{light}"/>')
        add(f'<text x="{lx + 14}" y="0" class="t2" fill="{INK2}" font-size="11.5">'
            f'{entity}</text>')
        lx += 30 + len(entity) * 6.4
    add("</g>")
    add("</svg>")
    return chr(10).join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, action="append", required=True,
                    help="v-forge prefix-score JSONL (forge-full, gen1, gen4)")
    ap.add_argument("--raw-gen3", type=Path, action="append", default=[],
                    help="v-gen3 prefix-score JSONL (gen3, gen4)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--png", type=Path, help="also rasterise here (needs cairosvg)")
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--max-picks", type=int, default=45)
    args = ap.parse_args()

    cap = args.max_picks
    lvl_f, step_f, n_f = load(args.raw, cap)
    lvl_g, step_g, n_g = (({}, {}, 0) if not args.raw_gen3
                          else load(args.raw_gen3, cap))

    level, step = dict(lvl_f), dict(step_f)
    level.update({k: v for k, v in lvl_g.items() if k[0] == "gen3"})
    step.update({k: v for k, v in step_g.items() if k[0] == "gen3"})

    gap: dict = {}
    for t in range(1, cap + 1):
        for opp, src in (("forge-full", lvl_f), ("gen1", lvl_f), ("gen3", lvl_g)):
            if ("gen4", t) in src and (opp, t) in src:
                gap[(opp, t)] = src[("gen4", t)] - src[(opp, t)]

    four = [(a, a, a) for a in ("forge-full", "gen1", "gen3", "gen4")]
    gaps = [(a, a, f"vs {a}") for a in ("forge-full", "gen1", "gen3")]
    panels = [("deck score (level)", level, four, False),
              ("what the pick added", step, four, True),
              ("gen-4's lead, paired inside each corpus", gap, gaps, True)]

    args.out.write_text(build(panels, cap, n_f + n_g), encoding="utf-8")
    print(f"wrote {args.out}  ({n_f} v-forge + {n_g} v-gen3 seats)")
    if args.png:
        import cairosvg
        # Rasterise on the light surface: a PNG carries no media query, so the
        # dark-mode rules in the SVG cannot apply to it.
        cairosvg.svg2png(url=str(args.out), write_to=str(args.png), scale=args.scale,
                         background_color=BG)
        print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
