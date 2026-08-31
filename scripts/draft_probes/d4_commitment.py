"""D4 — is colour commitment a fixed rule or a clock-dependent one?

The causal version of "does its pool change its pick". A receiver state supplies
the pack and the clock; a donor seat drawn from a *different draft at the same
``(pack, pick)``* supplies the POOL block. Everything else is held fixed, so the
same physical card is scored against several different pools with its pack, its
recency and its pick number untouched.

Estimator. For one receiver state and one card in its pack, the D donor pools
give D observations of that card's centred logit against the donor pool's
fraction of the card's own colour. Demeaning within ``(state, card)`` and pooling
the slope gives a within-card fixed-effect estimate: card identity, pack
composition and clock all cancel, and what is left is the pool's causal pull.

The hypothesis is not that the pull exists — a two-colour drafter must have one,
and gen-1 inherited Forge's committed-colour bonus. It is that the pull **grows
with the clock**. A terminal, time-symmetric reward never states that an early
break is cheaper than a late one, so a slope that rises across the draft is
sequential policy rather than transmitted taste.

Usage
-----
    python scripts/draft_probes/d4_commitment.py --receivers 250 --donors 8
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draft_corpus_common import ColourResolver  # noqa: E402
from probe_lib import (  # noqa: E402
    TYPE_POOL,
    CardTable,
    PolicyRunner,
    iter_corpus_states,
    load_agent,
    transplant_pool,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "draft-probes"
G4 = REPO / "models/draft/agent/gen4"
CORPUS = G4 / "lr1e-5_t2all_decay0.3-yardstick-v-forge-drafts.jsonl"
CARDS = REPO / "output" / "cardsfolder-512"

MODELS = {
    "gen1": REPO / "models/draft/agent/gen1/l6_lr3e-4_decay_20260604_080249.pt",
    "gen3": REPO / "models/draft/agent/gen3/temperature-on-all-agents"
                   "/lr1e-5_t2_20260805_221050.pt",
    "gen4": G4 / "lr1e-5_t2all_decay0.3.pt",
    "gen4b": G4 / "lr1e-5_t2all_nodecay.pt",
}

# Absolute pick index across the draft -> (pack, pick), pack size 15.
CLOCKS = [3, 8, 16, 23, 31, 38, 43]


def clock_to_pp(t: int) -> tuple[int, int]:
    return (t - 1) // 15 + 1, (t - 1) % 15 + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-drafts", type=int, default=250)
    ap.add_argument("--receivers", type=int, default=250)
    ap.add_argument("--donors", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    table = CardTable()
    colours = ColourResolver(CARDS)

    wanted_pp = sorted({clock_to_pp(t) for t in CLOCKS})
    by_clock: dict[tuple[int, int], list] = defaultdict(list)
    for s in iter_corpus_states(CORPUS, table, clocks=wanted_pp,
                                limit_drafts=args.limit_drafts):
        by_clock[(s.state.pack_number, s.state.pick_number)].append(s)
    print("states per clock: "
          + ", ".join(f"{k}:{len(v)}" for k, v in sorted(by_clock.items())))

    mat = table.matrix()
    colour_of_row = [colours(n) for n in table.names]

    def pool_fraction(state) -> dict[str, float]:
        sel = state.type_idx == TYPE_POOL
        rows = state.card_idx[sel]
        if rows.size == 0:
            return {c: 0.0 for c in "WUBRG"}
        acc = {c: 0.0 for c in "WUBRG"}
        for r in rows:
            ids = colour_of_row[r]
            if not ids:
                continue
            for c in ids:
                acc[c] += 1.0 / len(ids)
        return {c: acc[c] / rows.size for c in "WUBRG"}

    # Build the (receiver, donor) work list once; every model sees the same one.
    work: list[tuple] = []          # (clock, receiver, donor_state, frac)
    for t in CLOCKS:
        pool = by_clock[clock_to_pp(t)]
        if len(pool) < args.donors + 1:
            continue
        receivers = pool if len(pool) <= args.receivers else rng.sample(
            pool, args.receivers)
        for recv in receivers:
            others = [d for d in pool if d.draft_id != recv.draft_id]
            if len(others) < args.donors:
                continue
            for donor in rng.sample(others, args.donors):
                work.append((t, recv, donor.state, pool_fraction(donor.state)))
    print(f"{len(work)} (receiver, donor) pairs")

    # One work item = one receiver state with one donor POOL spliced in. The
    # same receiver recurs across donors, so each item gets its own shallow copy
    # and the donor is looked up by object identity.
    samples = [copy.copy(w[1]) for w in work]
    donor_map = {id(s): w[2] for s, w in zip(samples, work)}

    def donor_of(sample):
        return donor_map[id(sample)]

    results = {}
    for gen, ckpt in MODELS.items():
        if not ckpt.exists():
            print(f"skip {gen}")
            continue
        model, _ = load_agent(ckpt)
        runner = PolicyRunner(model, table, batch_size=args.batch)
        logits = runner.logits(samples, transplant_pool(donor_of))

        # Per (state, card): the card's own-colour arm, and a placebo arm using
        # the pool share of a colour the card does not have. The placebo shares
        # every nuisance the real arm has — same donors, same pack, same clock —
        # and differs only in whether the colour is the card's, so a slope on it
        # would mean the estimator is reading pool quality rather than colour.
        obs: dict[int, dict[tuple, list[tuple[float, float]]]] = {
            t: defaultdict(list) for t in CLOCKS
        }
        plc: dict[int, dict[tuple, list[tuple[float, float]]]] = {
            t: defaultdict(list) for t in CLOCKS
        }
        share: dict[int, list[float]] = {t: [] for t in CLOCKS}
        for (t, recv, dstate, frac), lg in zip(work, logits):
            centred = lg - lg.mean()
            n_tok = int(dstate.type_idx.shape[0])
            n_pool = int((dstate.type_idx == TYPE_POOL).sum())
            share[t].append(n_pool / max(1, n_tok))
            for j, name in enumerate(recv.pack_names):
                ids = colours(name)
                if not ids:
                    continue  # colourless: no colour to match on
                key = (recv.draft_id, recv.seat, id(recv), j)
                f = sum(frac[c] for c in ids) / len(ids)
                obs[t][key].append((f, float(centred[j])))
                off = [c for c in "WUBRG" if c not in ids]
                if off:
                    fp = sum(frac[c] for c in off) / len(off)
                    plc[t][key].append((fp, float(centred[j])))

        def slope_of(cells) -> tuple[float, int, float]:
            num = den = 0.0
            n = 0
            spread = []
            for cell in cells.values():
                if len(cell) < 3:
                    continue
                f = np.array([a for a, _ in cell])
                y = np.array([b for _, b in cell])
                fd, yd = f - f.mean(), y - y.mean()
                v = float((fd * fd).sum())
                if v < 1e-9:
                    continue
                num += float((fd * yd).sum())
                den += v
                n += 1
                spread.append(float(f.max() - f.min()))
            if n == 0:
                return float("nan"), 0, float("nan")
            return num / den, n, float(np.mean(spread))

        rows = []
        for t in CLOCKS:
            s, n_cells, spread = slope_of(obs[t])
            sp, _, _ = slope_of(plc[t])
            if n_cells == 0:
                continue
            rows.append({
                "clock": t, "slope": s, "placebo_slope": sp,
                "cells": n_cells, "mean_frac_spread": spread,
                "pool_token_share": float(np.mean(share[t])),
            })
        results[gen] = rows
        print(f"\n=== {gen} — logit pull per unit of pool colour share ===")
        for r in rows:
            print(f"  pick {r['clock']:2d}  own-colour {r['slope']:+7.3f}"
                  f"  other-colour {r['placebo_slope']:+7.3f}"
                  f"  (cells {r['cells']}, spread {r['mean_frac_spread']:.3f},"
                  f" POOL share of tokens {r['pool_token_share']:.3f})")

    (OUT / "d4_commitment.json").write_text(
        json.dumps({"clocks": CLOCKS, "n_pairs": len(work),
                    "models": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'd4_commitment.json'}")


if __name__ == "__main__":
    main()
