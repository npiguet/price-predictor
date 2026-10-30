"""D8 — does a card already in the pool change what the same card is worth?

The scorer prices a second copy of a card exactly like the first, so no gradient
toward or against redundancy ever reached the policy. Whatever the draft agent
does about duplicates is therefore either inherited from Forge or an accident of
the architecture, and the size of the effect bounds how finely the policy reads
its own pool at all.

The probe is a two-arm causal edit on the same state. Two cards `X` and `Y` are
chosen from the pack, matched on colour identity and as close as possible in the
model's own base logit. One pool slot is then overwritten, in one arm with a copy
of `X` and in the other with a copy of `Y`. The arms differ only in which of two
near-equivalent cards the seat is now holding, so the pool's size, its recency
tags, its colour mix and the whole pack are identical between them.

Three quantities come out of the pair:

- the **duplicate effect** on `X`, its centred logit in the `X` arm minus its
  centred logit in the `Y` arm;
- the **mirror**, the same measured on `Y` with the arms swapped, which estimates
  the same quantity from the other side and has to agree in sign;
- a **placebo** on a third pack card `Z` that neither arm copied, which has to be
  near zero if the effect is about the copied card rather than about the edit.

A dose ladder repeats the edit over 1, 2 and 3 pool slots. Reading the effect by
position in the draft tests the one prediction the reward cannot state: late in a
draft a second copy displaces a card the seat already owns, so a policy that had
learned the deck-building consequence would price it *down*.

Usage
-----
    python scripts/draft_probes/d8_duplicates.py --limit-drafts 150
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.jsonio import write_json
from _lib.markdown import num, print_table
from _lib.stats import mean_se
from draft_corpus_common import ColourResolver
from draft_probes.probe_lib import (
    CARDS,
    CORPUS,
    MODELS,
    OUT,
    PACK_SIZE,
    TYPE_POOL,
    CardTable,
    absolute_pick,
    iter_corpus_states,
    iter_runners,
)


def write_slots(slots: list[int], row: int):
    """Overwrite the given POOL token positions with one card row."""
    def apply(sample):
        st = sample.state
        idx = st.card_idx.copy()
        idx[slots] = row
        return replace(st, card_idx=idx)
    return apply


def mean_se_n(values) -> dict:
    mean, se, n = mean_se(values)
    return {"mean": mean, "se": se, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-drafts", type=int, default=150)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-dose", type=int, default=3)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    table = CardTable()
    colours = ColourResolver(CARDS)

    base_samples = [
        s for s in iter_corpus_states(
            CORPUS, table, labels=["gen4"], limit_drafts=args.limit_drafts)
        if int((s.state.type_idx == TYPE_POOL).sum()) >= args.max_dose + 2
        and len(s.pack_names) >= 4
    ]
    print(f"{len(base_samples)} candidate states")

    # One reference model picks the X/Y pair, so every model is asked about the
    # same pair and the choice cannot favour any of them.
    _, ref = next(iter_runners(table, args.batch, {"gen4": MODELS["gen4"]}))
    base_logits = ref.logits(base_samples)
    del ref

    # Build the work list: per state, a colour-matched pair closest in base logit.
    mat = table.matrix()
    jobs = []
    for s, lg in zip(base_samples, base_logits):
        ids = [colours(n) for n in s.pack_names]
        best = None
        for i in range(len(s.pack_names)):
            if not ids[i]:
                continue  # colourless: no colour to match on
            for j in range(i + 1, len(s.pack_names)):
                if ids[j] != ids[i]:
                    continue
                gap = abs(float(lg[i] - lg[j]))
                if best is None or gap < best[0]:
                    best = (gap, i, j)
        if best is None:
            continue
        gap, i, j = best
        z = next((k for k in range(len(s.pack_names))
                  if k not in (i, j) and colours(s.pack_names[k])), None)
        if z is None:
            continue
        pool_slots = list(np.flatnonzero(s.state.type_idx == TYPE_POOL))
        rng.shuffle(pool_slots)
        xr = int(table.index(s.pack_names[i]))
        yr = int(table.index(s.pack_names[j]))
        vx, vy = mat[xr], mat[yr]
        cos = float(vx @ vy / (np.linalg.norm(vx) * np.linalg.norm(vy) + 1e-9))
        jobs.append({
            "sample": s, "x": i, "y": j, "z": z, "gap": gap, "cos": cos,
            "slots": [int(p) for p in pool_slots[:args.max_dose]],
            "x_row": xr, "y_row": yr,
            "pick": absolute_pick(s.state),
        })
    print(f"{len(jobs)} states with a colour-matched pack pair; "
          f"median base-logit gap {np.median([j['gap'] for j in jobs]):.3f}")

    results = {}
    for gen, runner in iter_runners(table, args.batch):
        per_dose = {}
        for dose in range(1, args.max_dose + 1):
            arm_x, arm_y = [], []
            for job in jobs:
                slots = job["slots"][:dose]
                sx = copy.copy(job["sample"])
                sy = copy.copy(job["sample"])
                sx.state = write_slots(slots, job["x_row"])(job["sample"])
                sy.state = write_slots(slots, job["y_row"])(job["sample"])
                arm_x.append(sx)
                arm_y.append(sy)
            lx = runner.logits(arm_x)
            ly = runner.logits(arm_y)
            eff_x, eff_y, plc, picks, sims = [], [], [], [], []
            for job, a, b in zip(jobs, lx, ly):
                ca, cb = a - a.mean(), b - b.mean()
                eff_x.append(float(ca[job["x"]] - cb[job["x"]]))
                eff_y.append(float(cb[job["y"]] - ca[job["y"]]))
                plc.append(float(ca[job["z"]] - cb[job["z"]]))
                picks.append(job["pick"])
                sims.append(job["cos"])
            eff = np.array(eff_x + eff_y)
            picks2 = np.array(picks + picks)
            sims2 = np.array(sims + sims)
            pack_of = (picks2 - 1) // PACK_SIZE + 1
            buckets = {f"pack {k}": mean_se_n(eff[pack_of == k]) for k in (1, 2, 3)
                       if (pack_of == k).sum() > 50}
            # Similarity control. Arm Y writes a colour- and logit-matched card,
            # not an embedding twin, so part of the rise on X could be "my pool
            # now looks more like X" rather than "I already own X". Splitting by
            # how close Y sits to X in embedding space separates the two: a pure
            # similarity effect has to shrink toward zero as Y approaches X.
            sim_rows = []
            qs = np.quantile(sims2, [0.0, 0.25, 0.5, 0.75, 1.0])
            for lo, hi in zip(qs[:-1], qs[1:]):
                m = (sims2 >= lo) & (sims2 <= hi)
                if m.sum() > 50:
                    sim_rows.append({"cos_lo": float(lo), "cos_hi": float(hi),
                                     **mean_se_n(eff[m])})
            effect, placebo = mean_se_n(eff), mean_se_n(plc)
            per_dose[dose] = {
                "effect_mean": effect["mean"],
                "effect_se": effect["se"],
                "effect_x": float(np.mean(eff_x)),
                "effect_y": float(np.mean(eff_y)),
                "placebo_mean": placebo["mean"],
                "placebo_se": placebo["se"],
                "n_pairs": len(jobs),
                "by_pack": buckets,
                "by_similarity": sim_rows,
            }
        results[gen] = per_dose
        print(f"\n=== {gen} — logit change on a card already in the pool ===")
        print_table(["copies", "effect", "se", "from X", "from Y", "placebo", "p1", "p2", "p3"],
                    [[dose, f"{r['effect_mean']:+.4f}", num(r["effect_se"]),
                      f"{r['effect_x']:+.4f}", f"{r['effect_y']:+.4f}",
                      f"{r['placebo_mean']:+.4f}"]
                     + [f"{r['by_pack'][k]['mean']:+.4f}" if k in r["by_pack"] else "-"
                        for k in ("pack 1", "pack 2", "pack 3")]
                     for dose, r in per_dose.items()], "r" * 9)
        sim = per_dose[1]["by_similarity"]
        if sim:
            print("  one copy, by how close the control card sits to it: "
                  + ", ".join(f"cos {r['cos_lo']:.2f}-{r['cos_hi']:.2f}: "
                              f"{r['mean']:+.4f}" for r in sim))

    write_json(OUT / "d8_duplicates.json", {"n_states": len(jobs), "models": results})
    print(f"\nwrote {OUT / 'd8_duplicates.json'}")


if __name__ == "__main__":
    main()
