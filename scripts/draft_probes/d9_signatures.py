"""D9 — which measured habits track strength, and which track training length?

Every gen-4 run drifts at a near-constant rate whether or not its rounds carried
signal (`d5_corpus.py`, analysis 3). So a habit that grows monotonically across
the generations might be what made the agent stronger, or might be what a
fixed-step walk away from the warm start produces on its own. The two readings
have the same shape and different consequences, and nothing in the study so far
separates them.

Six checkpoints carry two known axes: their yardstick margin over gen-1, and the
cumulative number of learner picks that produced them. Computing the same metric
on all six and reading it against both axes says which axis it follows.

The four gen-4 siblings alone cannot answer this — their yardstick order and
their training-length order are identical, so the two axes are collinear across
them. What breaks the tie is one pair. `t2all_decay0.3` trained on about two and
a half times the picks of `t2all_nodecay` and finished level with it on the
yardstick, well inside the error bars. A metric that moves between those two
is following training length; a metric that does not is following strength.

Metrics, all computed on identical inputs across the six models:

- the spread of the context-free pack-1-pick-1 card ranking;
- the white-and-green over blue-and-red colour lean in that ranking;
- its rank correlation with Forge's human pick order and with the scorer's card
  values;
- the share of all picks the context-free ranking already decides;
- the policy movement when the `POOL` and `TAKEN` card identities are blanked,
  which is architectural and should follow neither axis.

Usage
-----
    python scripts/draft_probes/d9_signatures.py --limit-drafts 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.corpora import card_column
from _lib.stats import spearman
from draft_corpus_common import ColourResolver
from draft_probes.d2_pickorder import card_scalars, p1p1_samples, static_agreement
from draft_probes.probe_lib import (
    CARD_VALUES,
    CARDS,
    CORPUS,
    G4,
    HINTS,
    MODELS,
    OUT,
    TYPE_POOL,
    TYPE_TAKEN,
    CardTable,
    argmaxes,
    centred,
    iter_corpus_states,
    iter_runners,
    mean_substitute,
)

# checkpoint -> (yardstick margin over gen-1, cumulative learner picks).
# Margins and per-generation pick counts are from
# experiments/2026-08-09-draft-agent-gen4-online-grpo.md; gen-3's ~110k picks are
# the shared warm start every gen-4 candidate inherits.
SUBJECTS = {
    "gen1": (MODELS["gen1"], 0.000, 0),
    "gen3": (MODELS["gen3"], 0.824, 110_000),
    "t3all": (G4 / "lr1e-5_t3all_decay0.3.pt", 1.152, 185_000),
    "t3learner": (G4 / "lr1e-5_t3learner_t2field_decay0.3.pt", 1.276, 200_000),
    "t2nodecay": (MODELS["gen4b"], 1.328, 205_000),
    "t2decay": (MODELS["gen4"], 1.380, 515_000),
}
WG, UR = frozenset("WG"), frozenset("UR")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-drafts", type=int, default=500)
    ap.add_argument("--state-drafts", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min-appearances", type=int, default=5)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    table = CardTable()
    colours = ColourResolver(CARDS)
    # Forge's draft rank is "lower is picked earlier", so flip it to make every
    # grader point the same way.
    human = card_column(HINTS, "draft_rank", -1.0)
    vswap = card_column(CARD_VALUES, "v_swap")
    print(f"graders: human {len(human)}, v_swap {len(vswap)}")

    p1p1 = p1p1_samples(table, args.limit_drafts)
    states = list(iter_corpus_states(CORPUS, table, labels=["gen4"],
                                     limit_drafts=args.state_drafts))
    print(f"{len(p1p1)} opening boosters, {len(states)} full states")

    mean_row = table.add_vector(table.matrix().mean(0), "<corpus-mean>")

    rows = []
    checkpoints = {name: ckpt for name, (ckpt, _, _) in SUBJECTS.items()}
    for name, runner in iter_runners(table, args.batch, checkpoints):
        _, margin, picks = SUBJECTS[name]
        # Context-free card scalar: centre each booster, then average per card.
        scalar, appear = card_scalars(p1p1, centred(runner.logits(p1p1)))
        scalar = {k: v for k, v in scalar.items() if appear[k] >= args.min_appearances}

        vals = np.array(list(scalar.values()))
        lean_wg = [v for k, v in scalar.items() if colours(k) and colours(k) <= WG]
        lean_ur = [v for k, v in scalar.items() if colours(k) and colours(k) <= UR]

        common_h = [k for k in scalar if k in human]
        common_v = [k for k in scalar if k in vswap]

        # How often the context-free ranking already decides the real pick,
        # over the picks that are a choice at all.
        base = runner.logits(states)
        base_arg = argmaxes(base)
        agree = static_agreement(states, scalar, base_arg)
        decided = agree[np.isfinite(agree) & (np.array([len(s.pack_names) for s in states]) >= 2)]

        def flip(interv) -> float:
            alt = runner.logits(states, interv)
            return float(np.mean([int(np.argmax(a) != b)
                                  for a, b in zip(alt, base_arg)]))

        rows.append({
            "model": name, "margin": margin, "picks": picks,
            "p1p1_sd": float(vals.std()),
            "colour_lean": float(np.mean(lean_wg) - np.mean(lean_ur)),
            "rho_human": spearman(np.array([scalar[k] for k in common_h]),
                                  np.array([human[k] for k in common_h])),
            "rho_vswap": spearman(np.array([scalar[k] for k in common_v]),
                                  np.array([vswap[k] for k in common_v])),
            "context_free": float(decided.mean()) if decided.size else float("nan"),
            "pool_flip": flip(mean_substitute(TYPE_POOL, mean_row=mean_row)),
            "taken_flip": flip(mean_substitute(TYPE_TAKEN, mean_row=mean_row)),
            "n_cards": len(scalar),
        })
        print(f"  {name}: {rows[-1]}")

    metrics = ["p1p1_sd", "colour_lean", "rho_human", "rho_vswap",
               "context_free", "pool_flip", "taken_flip"]
    margin = np.array([r["margin"] for r in rows])
    lpicks = np.log10(np.array([r["picks"] for r in rows], dtype=float) + 1.0)

    print("\n=== metric against each axis, over "
          f"{len(rows)} checkpoints ===")
    print(f"{'metric':>14s} {'r(strength)':>12s} {'r(training)':>12s} "
          f"{'t2nodecay':>10s} {'t2decay':>9s} {'gap/range':>10s}")
    summary = []
    by_name = {r["model"]: r for r in rows}
    for m in metrics:
        v = np.array([r[m] for r in rows])
        rng = float(v.max() - v.min())
        pair = float("nan")
        if "t2nodecay" in by_name and "t2decay" in by_name:
            pair = by_name["t2decay"][m] - by_name["t2nodecay"][m]
        rec = {
            "metric": m,
            "r_margin": float(np.corrcoef(v, margin)[0, 1]),
            "r_picks": float(np.corrcoef(v, lpicks)[0, 1]),
            "t2nodecay": by_name.get("t2nodecay", {}).get(m),
            "t2decay": by_name.get("t2decay", {}).get(m),
            "pair_gap_over_range": pair / rng if rng > 0 else float("nan"),
        }
        summary.append(rec)
        print(f"{m:>14s} {rec['r_margin']:12.3f} {rec['r_picks']:12.3f} "
              f"{rec['t2nodecay']:10.3f} {rec['t2decay']:9.3f} "
              f"{rec['pair_gap_over_range']:10.3f}")

    print(f"\nThe two axes correlate at "
          f"{np.corrcoef(margin, lpicks)[0, 1]:.3f} over these checkpoints, "
          "so only the last column identifies anything on its own.")

    (OUT / "d9_signatures.json").write_text(
        json.dumps({"checkpoints": rows, "summary": summary,
                    "axis_correlation": float(np.corrcoef(margin, lpicks)[0, 1])},
                   indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'd9_signatures.json'}")


if __name__ == "__main__":
    main()
