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
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.corpora import card_column
from draft_corpus_common import ColourResolver, score_play
from draft_probes.probe_lib import (
    CARDS,
    CORPUS,
    HINTS,
    MODELS,
    OUT,
    WIN_RATES,
    CardTable,
    PolicyRunner,
    deck_lane,
    iter_corpus_records,
    iter_corpus_states,
    load_agent,
)


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

    centred: dict[str, list[np.ndarray]] = {}
    for gen, ckpt in MODELS.items():
        model, _ = load_agent(ckpt)
        runner = PolicyRunner(model, table, batch_size=args.batch)
        lg = runner.logits(samples)
        centred[gen] = [x - x.mean() for x in lg]
        del model

    # Per (state, card) records: the residual and each grader's within-pack rank.
    rows = []
    for k, s in enumerate(samples):
        lane = lanes.get((s.draft_id, s.seat), frozenset())
        n = len(s.pack_names)
        for grader, tbl in (("human", rank), ("play", play)):
            vals = [tbl.get(nm) for nm in s.pack_names]
            known = [v for v in vals if v is not None]
            if len(known) < 4:
                continue
            # Rank within the pack, 1.0 = best card present. Human draft_rank is
            # "lower is better", the win-rate label is "higher is better".
            order = np.argsort([
                (v if grader == "human" else -v) if v is not None else 1e18
                for v in vals
            ])
            pctl = np.zeros(n)
            m = len(known)
            seen = 0
            for pos in order:
                if vals[pos] is None:
                    pctl[pos] = np.nan
                    continue
                pctl[pos] = 1.0 - (seen / max(1, m - 1))
                seen += 1
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
    report = {}
    for grader in ("human", "play"):
        sub = [r for r in rows if r["grader"] == grader]
        if not sub:
            continue
        p = np.array([r["pctl"] for r in sub])
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        table_rows = []
        for lo, hi in bins:
            m = (p >= lo) & (p < hi)
            if m.sum() == 0:
                continue
            idx = np.flatnonzero(m)
            table_rows.append({
                "pctl_lo": lo, "pctl_hi": hi, "n": int(m.sum()),
                "abs_d41": float(np.mean([abs(sub[i]["d41"]) for i in idx])),
                "abs_d31": float(np.mean([abs(sub[i]["d31"]) for i in idx])),
                "abs_sib": float(np.mean([abs(sub[i]["sib"]) for i in idx])),
                "mean_d41": float(np.mean([sub[i]["d41"] for i in idx])),
                "sd_g4": float(np.std([sub[i]["g4"] for i in idx])),
            })
        lane_rows = []
        for on in (True, False):
            m = np.array([r["on_lane"] == on for r in sub])
            if m.sum() == 0:
                continue
            lane_rows.append({
                "on_lane": on, "n": int(m.sum()),
                "abs_d41": float(np.mean([abs(sub[i]["d41"])
                                          for i in np.flatnonzero(m)])),
                "mean_d41": float(np.mean([sub[i]["d41"]
                                           for i in np.flatnonzero(m)])),
            })
        report[grader] = {"by_quality": table_rows, "by_lane": lane_rows}
        print(f"\n=== residual by within-pack quality ({grader} grader) ===")
        print(f"{'pctl':>10s} {'n':>7s} {'|g4-g1|':>9s} {'|g3-g1|':>9s} "
              f"{'|sibling|':>10s} {'ratio':>7s} {'g4-g1':>8s} {'sd(g4)':>8s}")
        for r in table_rows:
            ratio = r["abs_d41"] / r["abs_sib"] if r["abs_sib"] > 0 else float("nan")
            print(f"{r['pctl_lo']:.1f}-{r['pctl_hi']:.1f} {r['n']:9d} "
                  f"{r['abs_d41']:9.3f} {r['abs_d31']:9.3f} {r['abs_sib']:10.3f} "
                  f"{ratio:7.2f} {r['mean_d41']:8.3f} {r['sd_g4']:8.3f}")
        print("  by lane: " + ", ".join(
            f"{'on' if r['on_lane'] else 'off'}-lane abs={r['abs_d41']:.3f} "
            f"(mean {r['mean_d41']:+.3f}, n={r['n']})" for r in lane_rows))

    # The same residual by pick index — the leverage profile at card level.
    by_pick = []
    sub = [r for r in rows if r["grader"] == "human"]
    for pk in range(1, 16):
        for pn in (1, 2, 3):
            m = [r for r in sub if r["pick"] == pk and r["pack"] == pn]
            if len(m) < 50:
                continue
            by_pick.append({
                "pack": pn, "pick": pk, "n": len(m),
                "abs_d41": float(np.mean([abs(r["d41"]) for r in m])),
                "abs_d31": float(np.mean([abs(r["d31"]) for r in m])),
                "abs_sib": float(np.mean([abs(r["sib"]) for r in m])),
            })
    report["by_pick"] = by_pick
    print("\n=== residual size by position in the draft "
          "(gen4-gen1 over the sibling floor) ===")
    for pn in (1, 2, 3):
        vals = [r for r in by_pick if r["pack"] == pn]
        print(f"  pack {pn}: " + " ".join(
            f"{r['pick']}:{r['abs_d41'] / r['abs_sib']:.2f}"
            if r["abs_sib"] > 0 else f"{r['pick']}:-" for r in vals))

    (OUT / "d6_buildfilter.json").write_text(
        json.dumps({"n_states": len(samples), "report": report}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {OUT / 'd6_buildfilter.json'}")


if __name__ == "__main__":
    main()
