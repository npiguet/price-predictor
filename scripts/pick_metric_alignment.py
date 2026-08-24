"""Which per-card win-rate axis does an agent's pick order actually follow?

``analyze_pick_quality.py`` scores picks on one quality scale, ``score_play``,
which is a net-winning-influence measure and the closest thing in
``cards-win-rates.txt`` to raw card power. The encoder (spec
2026-05-03-card-winnability-pretraining) is trained on five axes, and there is no
reason the drafter's preferences should line up with that one.

For every pick this measures where the taken card sat in the pack under each
axis, as a percentile among the cards that were still available:

  align  - mean percentile rank of the taken card (0.5 = indifferent to the
           axis, 1.0 = always takes the pack's maximum)
  top1   - share of decisions taking the axis's argmax outright

Axes compared, all shrunk forms:

  score_play   net winning influence on the play (the incumbent scale)
  score_draw   net winning influence on the draw
  played_rate  how often the card gets cast when it is in a deck
  cast_lift    how much casting it changes the result, net of deck quality
  color_lift   affinity for the colours this seat ended up in: the mean of
               color_lift_X over the seat's eventual top-2 colours

Metrics inter-correlate, so a high alignment on one axis does not mean the agent
reads that axis. The pairwise Spearman table printed at the end is what bounds
that interpretation.

Two filters restrict this to picks that were real choices, and both move the
numbers, so quote them alongside any figure this produces.

``--min-obs`` (default 20) drops cards with too few in-deck observations to
carry a meaningful label. In practice this is the basic lands: they fill a few
per cent of booster slots and are taken at mean pick 14.6 of 15, so scoring
those picks measures nothing about preference. ``cast_lift`` is unaffected,
having already excluded them by needing both played and not-played counts.

``--min-choices`` (default 5) skips a pick unless that many cards left in the
pack carry a label, dropping the tail where the choice is between two or three
leftovers.

Both cuts work the same way. A forced pick lands mid-pack whatever the agent
prefers, so leaving those picks in pulls every agent towards 0.5. Dropping both
lowers the reference levels by about 0.05 and shrinks the candidates' leads over
them by a fifth or more.

Usage
-----
    G=models/draft/agent/gen4

    python scripts/pick_metric_alignment.py \
        --drafts "t2all_decay0.3=$G/lr1e-5_t2all_decay0.3-yardstick-v-forge-drafts.jsonl" \
        --winrates <cards-win-rates.txt>

The write-up quotes each candidate's *paired* lead: its ``align`` minus the
``gen1`` align in the same corpus block, never against a gen-1 figure from
another corpus. The references are re-measured in every pod, and the pairing is
what separates candidates tied on the raw percentile.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from draft_corpus_common import (  # noqa: E402
    ColourResolver,
    parse_drafts_arg,
    read_records,
    seat_pool,
    top_two_colours,
)

AXES = ["score_play", "score_draw", "played_rate", "cast_lift"]
WUBRG = "WUBRG"


def load_winrates(path: Path, min_obs: int) -> dict:
    """card -> {axis: value}, keeping cards with enough in-deck observations."""
    out: dict = {}
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(";")
        idx = {name: i for i, name in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split(";")
            if len(f) != len(header):
                continue
            n = int(f[idx["wins_when_in_deck"]]) + int(f[idx["losses_when_in_deck"]])
            if n < min_obs:
                continue
            row = {}
            for axis in AXES:
                raw = f[idx[f"shrunk_{axis}"]]
                if raw:
                    row[axis] = float(raw)
            for c in WUBRG:
                raw = f[idx[f"shrunk_color_lift_{c}"]]
                if raw:
                    row[f"cl_{c}"] = float(raw)
            out[f[idx["card_name"]]] = row
    return out


def percentile(values: list, taken: float) -> float:
    """Midrank percentile of ``taken`` among ``values`` (which includes it)."""
    below = sum(1 for v in values if v < taken)
    equal = sum(1 for v in values if v == taken)
    return (below + (equal - 1) / 2.0) / (len(values) - 1)


def spearman(pairs: list) -> float:
    xs, ys = zip(*pairs)

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(list(xs)), ranks(list(ys))
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", action="append", required=True)
    ap.add_argument("--cards-path", default="output/cardsfolder-512")
    ap.add_argument("--winrates", required=True)
    ap.add_argument("--min-obs", type=int, default=20)
    ap.add_argument("--min-choices", type=int, default=5)
    args = ap.parse_args()

    wr = load_winrates(Path(args.winrates), args.min_obs)
    colours = ColourResolver(Path(args.cards_path))
    print(f"win-rate labels: {len(wr)} cards with >= {args.min_obs} in-deck observations")

    axes = AXES + ["color_lift"]
    for spec in args.drafts:
        label, path = parse_drafts_arg(spec)
        acc: dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0]))
        for record, geom in read_records(path):
            for seat_i, seat in enumerate(record.seats):
                pool = seat_pool(record, geom, seat_i)
                lane = sorted(top_two_colours(pool, colours))
                for pack in range(1, geom.packs + 1):
                    for pick in range(1, geom.pack_size + 1):
                        taken = geom.taken_card(record, seat_i, pack, pick)
                        avail = geom.legal_actions(record, seat_i, pack, pick)
                        for axis in axes:
                            if axis == "color_lift":
                                def val(card, lane=lane):
                                    row = wr.get(card)
                                    if not row:
                                        return None
                                    vs = [row[f"cl_{c}"] for c in lane
                                          if f"cl_{c}" in row]
                                    return sum(vs) / len(vs) if vs else None
                            else:
                                def val(card, axis=axis):
                                    return (wr.get(card) or {}).get(axis)
                            tv = val(taken)
                            if tv is None:
                                continue
                            vs = [v for v in (val(c) for c in avail) if v is not None]
                            if len(vs) < args.min_choices:
                                continue
                            cell = acc[seat.agent][axis]
                            cell[0] += percentile(vs, tv)
                            cell[1] += 1 if tv >= max(vs) else 0
                            cell[2] += 1

        print("=" * 78)
        print(label)
        print(f"     {'agent':14s} {'axis':12s} {'align':>7s} {'top1':>7s} {'picks':>9s}")
        for agent in sorted(acc):
            for axis in axes:
                s, top1, n = acc[agent][axis]
                if not n:
                    continue
                print(f"     {agent:14s} {axis:12s} {s/n:7.3f} {top1/n*100:6.1f}% {n:9d}")
        print()

    print("=" * 78)
    print("pairwise Spearman between axes, over the labelled card population")
    names = AXES + ["cl_W", "cl_U", "cl_B", "cl_R", "cl_G"]
    print(f"     {'':12s}" + "".join(f"{a[:11]:>12s}" for a in names))
    for a in names:
        row = f"     {a:12s}"
        for b in names:
            pairs = [(r[a], r[b]) for r in wr.values() if a in r and b in r]
            row += f"{spearman(pairs) if len(pairs) > 100 else float('nan'):12.2f}"
        print(row)


if __name__ == "__main__":
    main()
