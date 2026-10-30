"""D6 — where in a pack did the reinforcement learning actually move the policy?

`deck_score` is computed on a *built* 40-card deck, so a card that does not make
the seat's best 23 spells contributes exactly zero to the reward. The gradient
therefore had nothing to say about the bottom of a pack. If the policy learned
the reward's shape, the gen-4 − gen-1 logit residual should be a threshold
function: large among cards that can plausibly make a deck, ≈ 0 among cards that
cannot. If instead the residual is flat across card quality, the update moved
everything uniformly — which is what a norm-clipped step carrying one trajectory-
wide advantage would do.

Both models are run on the **same** states (the gen-4 seats' recorded picks), and
every logit is centred within its state, because the policy head is invariant to
a per-state constant.

Cards are ranked inside their own pack by three graders, and the residual is read
against each:

- **human draft rank** — Forge's bundled pick-order file, exogenous to all three
  models and the only grader that is itself a pick order;
- **`shrunk_score_play`** — the encoder's own training label, reported because
  the prior studies use it, and flagged as circular;
- **the seat's eventual colours** — on-lane cards are the ones that could reach
  the deck at all.

Usage
-----
    python scripts/draft_probes/d6_buildfilter.py --limit-drafts 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.corpora import card_column
from _lib.jsonio import write_json
from _lib.markdown import num, print_table
from draft_corpus_common import ColourResolver, score_play
from draft_probes.probe_lib import (
    CARDS,
    CORPUS,
    HINTS,
    OUT,
    WIN_RATES,
    CardTable,
    deck_lane,
    iter_corpus_records,
    iter_corpus_states,
    iter_runners,
)
from draft_probes.probe_lib import centred as centred_logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-drafts", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    table = CardTable()
    colours = ColourResolver(CARDS)
    rank = card_column(HINTS, "draft_rank")  # Forge's human pick order, lower = earlier
    play = score_play(WIN_RATES) if WIN_RATES.exists() else {}
    print(f"graders: draft_rank {len(rank)} cards, shrunk_score_play {len(play)}")

    lanes: dict[tuple[str, int], frozenset] = {}
    for rec in iter_corpus_records(CORPUS, args.limit_drafts):
        for i, seat in enumerate(rec.seats):
            if seat.deck:
                lanes[(rec.draft_id, i)] = deck_lane(seat.deck, colours)

    samples = list(iter_corpus_states(
        CORPUS, table, labels=["gen4"], limit_drafts=args.limit_drafts))
    print(f"{len(samples)} states")

    centred = {gen: centred_logits(runner.logits(samples))
               for gen, runner in iter_runners(table, args.batch)}

    # Per (state, card) records: the residual and each grader's within-pack rank.
    rows = []
    for k, s in enumerate(samples):
        lane = lanes.get((s.draft_id, s.seat), frozenset())
        for grader, tbl in (("human", rank), ("play", play)):
            vals = np.array([tbl.get(nm, np.nan) for nm in s.pack_names], dtype=float)
            known = np.isfinite(vals)
            if known.sum() < 4:
                continue
            # Rank within the pack, 1.0 = best card present. Human draft_rank is
            # "lower is better", the win-rate label is "higher is better".
            order = rankdata(vals[known] if grader == "human" else -vals[known],
                             method="ordinal")
            pctl = np.full(len(vals), np.nan)
            pctl[known] = 1.0 - (order - 1) / max(1, known.sum() - 1)
            for j, nm in enumerate(s.pack_names):
                if np.isnan(pctl[j]):
                    continue
                ids = colours(nm)
                rows.append({
                    "grader": grader,
                    "pack": s.state.pack_number,
                    "pick": s.state.pick_number,
                    "pctl": float(pctl[j]),
                    "on_lane": bool(not ids or ids <= lane),
                    "d41": float(centred["gen4"][k][j] - centred["gen1"][k][j]),
                    "d43": float(centred["gen4"][k][j] - centred["gen3"][k][j]),
                    "d31": float(centred["gen3"][k][j] - centred["gen1"][k][j]),
                    "sib": float(centred["gen4"][k][j] - centred["gen4b"][k][j]),
                    "g4": float(centred["gen4"][k][j]),
                })

    print(f"{len(rows)} (state, card, grader) records")
    cards = pd.DataFrame(rows)
    for col in ("d41", "d31", "sib"):
        cards[f"abs_{col}"] = cards[col].abs()
    report = {}
    for grader in ("human", "play"):
        sub = cards[cards["grader"] == grader]
        if sub.empty:
            continue
        bins = pd.cut(sub["pctl"], [0.0, 0.2, 0.4, 0.6, 0.8, 1.01], right=False)
        quality = sub.groupby(bins, observed=True).agg(
            n=("pctl", "size"), abs_d41=("abs_d41", "mean"), abs_d31=("abs_d31", "mean"),
            abs_sib=("abs_sib", "mean"), mean_d41=("d41", "mean"),
            sd_g4=("g4", lambda v: float(np.std(v))))
        table_rows = [{"pctl_lo": iv.left, "pctl_hi": iv.right, **r}
                      for iv, r in zip(quality.index, quality.to_dict("records"))]
        lane = sub.groupby("on_lane").agg(n=("d41", "size"), abs_d41=("abs_d41", "mean"),
                                          mean_d41=("d41", "mean"))
        lane_rows = [{"on_lane": bool(on), **r}
                     for on, r in sorted(lane.to_dict("index").items(), reverse=True)]
        report[grader] = {"by_quality": table_rows, "by_lane": lane_rows}
        print(f"\n=== residual by within-pack quality ({grader} grader) ===")
        print_table(["pctl", "n", "|g4-g1|", "|g3-g1|", "|sibling|", "ratio", "g4-g1",
                     "sd(g4)"],
                    [[f"{r['pctl_lo']:.1f}-{r['pctl_hi']:.1f}", r["n"], num(r["abs_d41"], 3),
                      num(r["abs_d31"], 3), num(r["abs_sib"], 3),
                      num(r["abs_d41"] / r["abs_sib"] if r["abs_sib"] > 0 else float("nan"), 2),
                      num(r["mean_d41"], 3), num(r["sd_g4"], 3)] for r in table_rows],
                    "l" + "r" * 7)
        print("  by lane: " + ", ".join(
            f"{'on' if r['on_lane'] else 'off'}-lane abs={r['abs_d41']:.3f} "
            f"(mean {r['mean_d41']:+.3f}, n={r['n']})" for r in lane_rows))

    # The same residual by pick index — the leverage profile at card level.
    human = cards[cards["grader"] == "human"]
    by_pick = [
        {"pack": int(pn), "pick": int(pk), **r}
        for (pk, pn), r in human.groupby(["pick", "pack"]).agg(
            n=("d41", "size"), abs_d41=("abs_d41", "mean"), abs_d31=("abs_d31", "mean"),
            abs_sib=("abs_sib", "mean")).to_dict("index").items()
        if r["n"] >= 50]
    report["by_pick"] = by_pick
    print("\n=== residual size by position in the draft "
          "(gen4-gen1 over the sibling floor) ===")
    for pn in (1, 2, 3):
        vals = [r for r in by_pick if r["pack"] == pn]
        print(f"  pack {pn}: " + " ".join(
            f"{r['pick']}:{r['abs_d41'] / r['abs_sib']:.2f}"
            if r["abs_sib"] > 0 else f"{r['pick']}:-" for r in vals))

    write_json(OUT / "d6_buildfilter.json", {"n_states": len(samples), "report": report})
    print(f"\nwrote {OUT / 'd6_buildfilter.json'}")


if __name__ == "__main__":
    main()
