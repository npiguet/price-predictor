"""When does a drafter leave the lane it has already committed to, and what does it take?

``analyze_pick_quality.py`` measures the *quality* of the off-lane picks an agent
made, anchored on the seat's **eventual** top-2 colours. This script anchors on
the **running** lane instead -- the top-2 of the cards picked so far -- because
only that explains a choice at the moment it was made. The eventual lane counts a
successful colour switch as on-lane in hindsight, which answers a different
question.

Reports, in the order the gen-4 write-up uses them:

1. ``scope``       -- what fraction of picks offer a real colour choice at all.
2. ``leave-lane``  -- P(leave the lane) by how much better the best off-lane card
                      is than the best on-lane one, per pack, with the
                      selectivity ratio.
3. ``quality``     -- mean score of the off-lane card taken, and how often it was
                      the pack's best, split by chosen vs forced.
4. ``conversion``  -- how often each kind of pick reaches the built deck, with
                      the mana-base control.
5. ``forced-deck`` -- pod-relative ``deck_score`` by how many forced picks the
                      seat ended up playing.

Definitions shared by every report
----------------------------------
*Running lane* -- the top-2 colours of the seat's picks so far. A pick is scored
only once the seat holds ``--min-pool`` coloured cards with ``--min-second`` in
its second colour, and is skipped while its 2nd and 3rd colours are level, since
the lane is then ambiguous. About a fifth of picks never qualify.

*Scored card* -- carries ``shrunk_score_play`` and has at least ``--min-obs``
in-deck observations. The threshold removes the basic lands, which fill booster
slots but are taken at mean pick 14.6 of 15 and are nobody's choice.

*Chosen vs forced* -- an off-lane pick is **chosen** when the pack still held an
on-lane scored card, and **forced** when it did not. Only chosen picks are
decisions, and only they enter reports 1 and 2.

*Colours* -- colourless cards are excluded throughout; a gold card is off-lane
unless the lane covers every colour in it.

Usage
-----
    G=models/draft/agent/gen4

    python scripts/analyze_lane_discipline.py \\
        --drafts $G/lr1e-5_t2all_decay0.3-yardstick-v-forge-drafts.jsonl \\
        --drafts $G/lr1e-5_t2all_nodecay-yardstick-v-forge-drafts.jsonl \\
        --winrates <cards-win-rates.txt>

Pass ``--report`` to print one section instead of all of them.
"""

from __future__ import annotations

import argparse
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from draft_corpus_common import (
    ColourResolver,
    mana_base_width,
    parse_drafts_arg,
    read_records,
    seat_pool,
)

DEFAULT_CARDS_PATH = Path("output/cardsfolder-512")
AGENTS = ("forge-full", "gen1", "gen4")
# Gap buckets on shrunk_score_play: best off-lane card minus best on-lane card.
BUCKETS = [(-9.0, -0.05, "much worse"), (-0.05, -0.01, "worse"),
           (-0.01, 0.01, "level"), (0.01, 0.05, "better"), (0.05, 9.0, "much better")]
GAP_PICK_CAP = 10   # the last third of a pack is leftovers; the gap means little there


def load_scores(path: Path, min_obs: int) -> dict[str, float]:
    """card -> ``shrunk_score_play``, for cards with enough in-deck observations."""
    out: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\r\n").split(";")
        need = ["card_name", "shrunk_score_play", "wins_when_in_deck", "losses_when_in_deck"]
        missing = [n for n in need if n not in header]
        if missing:
            raise SystemExit(f"{path} has no column(s): {', '.join(missing)}")
        ix = {n: header.index(n) for n in need}
        for line in handle:
            f = line.rstrip("\r\n").split(";")
            if len(f) != len(header) or not f[ix["shrunk_score_play"]]:
                continue
            try:
                seen = (int(f[ix["wins_when_in_deck"]]) + int(f[ix["losses_when_in_deck"]]))
                if seen < min_obs:
                    continue
                out[f[ix["card_name"]]] = float(f[ix["shrunk_score_play"]])
            except ValueError:
                continue
    return out


def running_lane(counts: Counter, min_pool: int, min_second: int) -> set[str] | None:
    """Top-2 colours of what the seat has picked so far, or None if not committed.

    Returns None while the pool is too small, while the second colour is too
    thin, and while the 2nd and 3rd colours are tied -- the lane is not a fact
    about the seat yet, so scoring a departure from it would be noise.
    """
    if sum(counts.values()) < min_pool:
        return None
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) < 2 or ranked[1][1] < min_second:
        return None
    if len(ranked) > 2 and ranked[2][1] == ranked[1][1]:
        return None
    return {ranked[0][0], ranked[1][0]}


class Walk:
    """One pass over the corpora, accumulating everything every report needs."""

    def __init__(self, colours, scores, args) -> None:
        self.colours, self.scores, self.args = colours, scores, args
        self.seats: Counter = Counter()
        self.scope: dict = defaultdict(Counter)
        self.gap: dict = defaultdict(lambda: [0, 0])        # (agent,pack,bucket)
        self.breaks: Counter = Counter()                    # break on a worse card
        self.break_conv: dict = defaultdict(lambda: [0, 0])  # (agent, worse/better)
        self.qual: dict = defaultdict(list)                 # (agent, chosen/forced)
        self.best: dict = defaultdict(lambda: [0, 0])
        self.kind_n: Counter = Counter()
        self.conv: dict = defaultdict(lambda: [0, 0])       # (agent, on/chosen/forced)
        self.conv2: dict = defaultdict(lambda: [0, 0])      # two-colour decks only
        self.width: dict = defaultdict(list)
        self.by_forced: dict = defaultdict(lambda: defaultdict(list))

    def run(self, paths) -> None:
        for path in paths:
            for record, geo in read_records(path):
                scores = [s.deck_score for s in record.seats]
                for i, seat in enumerate(record.seats):
                    self._seat(record, geo, i, seat, scores)

    def _seat(self, record, geo, i, seat, scores) -> None:
        a = seat.agent
        pool = seat_pool(record, geo, i)
        deck = Counter(seat.deck)
        counts: Counter = Counter()
        self.seats[a] += 1
        width = mana_base_width(seat.deck)
        self.width[a].append(width)
        played_forced = 0

        for t, card in enumerate(pool, 1):
            pack = (t - 1) // geo.pack_size + 1
            pick = (t - 1) % geo.pack_size + 1
            in_deck = deck[card] > 0
            if in_deck:
                deck[card] -= 1
            lane = running_lane(counts, self.args.min_pool, self.args.min_second)
            cs = self.colours(card)
            if lane is not None:
                self._pick(record, geo, i, a, pack, pick, card, cs, lane, in_deck)
                if cs and not cs <= lane:
                    legal = geo.legal_actions(record, i, pack, pick)
                    if not self._has_on_lane(legal[1:], lane):
                        played_forced += in_deck
                        if width == 2:
                            c = self.conv2[a]
                            c[0] += in_deck
                            c[1] += 1
            for c2 in cs:
                counts[c2] += 1

        others = [s for j, s in enumerate(scores) if j != i and s is not None]
        if seat.deck_score is not None and others:
            self.by_forced[a][min(played_forced, 2)].append(
                seat.deck_score - st.fmean(others))

    def _has_on_lane(self, cards, lane) -> bool:
        return any((c := self.colours(x)) and c <= lane and x in self.scores for x in cards)

    def _pick(self, record, geo, i, a, pack, pick, card, cs, lane, in_deck) -> None:
        legal = geo.legal_actions(record, i, pack, pick)
        on, off = [], []
        for x in legal:
            xc = self.colours(x)
            if not xc or x not in self.scores:
                continue
            (on if xc <= lane else off).append(x)

        # 1. scope: does this pick offer a colour choice at all?
        kind = ("both available" if on and off else
                "no on-lane card (forced)" if off else
                "no off-lane card" if on else "nothing scored")
        self.scope[a][kind] += 1

        broke = bool(cs) and not cs <= lane

        # 2. the gap reports need both kinds on offer, and a pack still worth choosing from
        if on and off and pick <= GAP_PICK_CAP:
            gap = max(self.scores[x] for x in off) - max(self.scores[x] for x in on)
            b = next(k for k, (lo, hi, _) in enumerate(BUCKETS) if lo <= gap < hi)
            cell = self.gap[(a, pack, b)]
            cell[0] += broke
            cell[1] += 1
            if broke:
                if gap < -0.01:
                    self.breaks[a] += 1
                    c = self.break_conv[(a, "worse")]
                    c[0] += in_deck
                    c[1] += 1
                elif gap > 0.01:
                    c = self.break_conv[(a, "better")]
                    c[0] += in_deck
                    c[1] += 1

        # 3-5. every pick with a lane, classified on-lane / chosen / forced
        if not cs:
            return
        if not broke:
            k = "on lane"
        else:
            k = "chosen" if self._has_on_lane(legal[1:], lane) else "forced"
            if card in self.scores:
                self.qual[(a, k)].append(self.scores[card])
                scored = [self.scores[x] for x in legal if x in self.scores]
                if len(scored) >= 2:
                    bc = self.best[(a, k)]
                    bc[0] += self.scores[card] >= max(scored)
                    bc[1] += 1
        self.kind_n[(a, k)] += 1
        c = self.conv[(a, k)]
        c[0] += in_deck
        c[1] += 1


def pct(cell) -> str:
    return f"{100 * cell[0] / cell[1]:.1f} %" if cell[1] else "-"


def report_scope(w) -> None:
    print("\n1. what fraction of picks offer a colour choice (picks with a running lane)\n")
    print(f"{'':<28}" + "".join(f"{a:>22}" for a in AGENTS))
    for k in ("both available", "no on-lane card (forced)", "no off-lane card", "nothing scored"):
        row = ""
        for a in AGENTS:
            tot = sum(w.scope[a].values())
            row += f"{100 * w.scope[a][k] / tot:>13.1f}%{w.scope[a][k] / w.seats[a]:>9.1f}"
        print(f"{k:<28}{row}")
    print("   (second figure in each pair = picks per draft)")


def report_leave_lane(w) -> None:
    print(f"\n2. P(leave the lane) by the gap, per pack   (picks 1-{GAP_PICK_CAP})\n")
    head = "".join(f"{n:>13}" for _, _, n in BUCKETS)
    print(f"{'agent and pack':<22}{head}{'ratio':>9}")
    for a in AGENTS:
        for pack in (1, 2, 3):
            cells = "".join(f"{pct(w.gap[(a, pack, k)]):>13}" for k in range(len(BUCKETS)))
            good = [sum(w.gap[(a, pack, k)][j] for k in (3, 4)) for j in (0, 1)]
            bad = [sum(w.gap[(a, pack, k)][j] for k in (0, 1)) for j in (0, 1)]
            ratio = ((good[0] / good[1]) / (bad[0] / bad[1])
                     if good[1] and bad[1] and bad[0] else float("nan"))
            print(f"{a + ', pack ' + str(pack):<22}{cells}{ratio:>8.1f}x")
    print("\n   breaks on a worse card per draft, and how often they reach the deck:")
    for a in AGENTS:
        conv_w, conv_b = w.break_conv[(a, "worse")], w.break_conv[(a, "better")]
        print(f"     {a:<12}{w.breaks[a] / w.seats[a]:>6.2f} per draft   "
              f"reaches deck {pct(conv_w):>7} (against {pct(conv_b)} for breaks on a better card)")


def report_quality(w) -> None:
    print("\n3. the off-lane card taken: mean score, and how often it was the pack's best\n")
    print(f"{'agent':<14}" + "".join(f"{k:>34}" for k in ("chosen", "forced")))
    print(f"{'':<14}" + "".join(f"{'score':>13}{'took best':>11}{'per draft':>10}" for _ in range(2)))
    for a in AGENTS:
        row = ""
        for k in ("chosen", "forced"):
            row += (f"{st.fmean(w.qual[(a, k)]):>+13.4f}{pct(w.best[(a, k)]):>11}"
                    f"{w.kind_n[(a, k)] / w.seats[a]:>10.1f}")
        print(f"{a:<14}{row}")


def report_conversion(w) -> None:
    print("\n4. share of picks that reach the built deck\n")
    print(f"{'agent':<14}" + "".join(f"{k:>18}" for k in ("on lane", "chosen", "forced"))
          + f"{'mana base':>12}{'forced, 2-colour decks':>24}")
    for a in AGENTS:
        row = "".join(f"{pct(w.conv[(a, k)]):>18}" for k in ("on lane", "chosen", "forced"))
        print(f"{a:<14}{row}{st.fmean(w.width[a]):>12.2f}{pct(w.conv2[a]):>24}")


def report_forced_deck(w) -> None:
    print("\n5. pod-relative deck_score by how many forced picks the seat played\n")
    print(f"{'agent':<14}" + "".join(f"{k:>16}" for k in ("played none", "played one", "two or more")))
    for a in AGENTS:
        print(f"{a:<14}" + "".join(
            f"{st.fmean(w.by_forced[a][k]):>+16.3f}" if w.by_forced[a][k] else f"{'-':>16}"
            for k in (0, 1, 2)))
    for a in AGENTS:
        print(f"     {a:<12} n = " + ", ".join(str(len(w.by_forced[a][k])) for k in (0, 1, 2)))


REPORTS = {"scope": report_scope, "leave-lane": report_leave_lane, "quality": report_quality,
           "conversion": report_conversion, "forced-deck": report_forced_deck}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drafts", action="append", required=True, metavar="PATH")
    ap.add_argument("--cards-path", type=Path, default=DEFAULT_CARDS_PATH)
    ap.add_argument("--winrates", type=Path, required=True)
    ap.add_argument("--min-obs", type=int, default=20)
    ap.add_argument("--min-pool", type=int, default=5,
                    help="coloured cards the seat must hold before it has a lane")
    ap.add_argument("--min-second", type=int, default=2,
                    help="cards required in the seat's second colour")
    ap.add_argument("--report", choices=sorted(REPORTS), action="append")
    args = ap.parse_args()

    scores = load_scores(args.winrates, args.min_obs)
    print(f"{len(scores)} cards scored with >= {args.min_obs} in-deck observations")
    colours = ColourResolver(args.cards_path)
    walk = Walk(colours, scores, args)
    walk.run([parse_drafts_arg(d)[1] for d in args.drafts])
    for name in (args.report or sorted(REPORTS, key=list(REPORTS).index)):
        REPORTS[name](walk)


if __name__ == "__main__":
    main()
