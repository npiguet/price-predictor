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
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draft_corpus_common import ColourResolver  # noqa: E402
from probe_lib import (  # noqa: E402
    TYPE_POOL,
    TYPE_TAKEN,
    CardTable,
    PolicyRunner,
    iter_corpus_states,
    load_agent,
    mean_substitute,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "draft-probes"
G4 = REPO / "models/draft/agent/gen4"
CORPUS = G4 / "lr1e-5_t2all_decay0.3-yardstick-v-forge-drafts.jsonl"
CARDS = REPO / "output" / "cardsfolder-512"
HINTS = REPO / "output" / "scorer-probes" / "forge_hints.csv"
VSWAP = REPO / "output" / "scorer-probes" / "t2_card_values.csv"

# checkpoint -> (yardstick margin over gen-1, cumulative learner picks).
# Margins and per-generation pick counts are from
# experiments/2026-08-09-draft-agent-gen4-online-grpo.md; gen-3's ~110k picks are
# the shared warm start every gen-4 candidate inherits.
SUBJECTS = {
    "gen1": (REPO / "models/draft/agent/gen1/l6_lr3e-4_decay_20260604_080249.pt",
             0.000, 0),
    "gen3": (REPO / "models/draft/agent/gen3/temperature-on-all-agents"
                    "/lr1e-5_t2_20260805_221050.pt", 0.824, 110_000),
    "t3all": (G4 / "lr1e-5_t3all_decay0.3.pt", 1.152, 185_000),
    "t3learner": (G4 / "lr1e-5_t3learner_t2field_decay0.3.pt", 1.276, 200_000),
    "t2nodecay": (G4 / "lr1e-5_t2all_nodecay.pt", 1.328, 205_000),
    "t2decay": (G4 / "lr1e-5_t2all_decay0.3.pt", 1.380, 515_000),
}
WG, UR = frozenset("WG"), frozenset("UR")


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    r = np.empty(x.size, dtype=float)
    r[order] = np.arange(x.size, dtype=float)
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def load_csv_column(path: Path, key: str, sign: float) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        rd = csv.DictReader(fh)
        name_col = next((c for c in rd.fieldnames or []
                         if c.lower() in ("name", "card_name", "card")), None)
        if name_col is None or key not in (rd.fieldnames or []):
            return out
        for row in rd:
            try:
                out[row[name_col]] = sign * float(row[key])
            except (TypeError, ValueError):
                continue
    return out


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
    human = load_csv_column(HINTS, "draft_rank", -1.0)
    vswap = load_csv_column(VSWAP, "v_swap", 1.0)
    print(f"graders: human {len(human)}, v_swap {len(vswap)}")

    p1p1 = list(iter_corpus_states(CORPUS, table, clocks=[(1, 1)],
                                   limit_drafts=args.limit_drafts))
    states = list(iter_corpus_states(CORPUS, table, labels=["gen4"],
                                     limit_drafts=args.state_drafts))
    print(f"{len(p1p1)} opening boosters, {len(states)} full states")

    mean_row = table.add_vector(table.matrix().mean(0), "<corpus-mean>")

    rows = []
    for name, (ckpt, margin, picks) in SUBJECTS.items():
        if not ckpt.exists():
            print(f"skip {name}: {ckpt} missing")
            continue
        model, _ = load_agent(ckpt)
        runner = PolicyRunner(model, table, batch_size=args.batch)

        # Context-free card scalar: centre each booster, then average per card.
        acc: dict[str, list[float]] = defaultdict(list)
        for s, lg in zip(p1p1, runner.logits(p1p1)):
            c = lg - lg.mean()
            for nm, v in zip(s.pack_names, c):
                acc[nm].append(float(v))
        scalar = {k: float(np.mean(v)) for k, v in acc.items()
                  if len(v) >= args.min_appearances}

        vals = np.array(list(scalar.values()))
        lean_wg = [v for k, v in scalar.items() if colours(k) and colours(k) <= WG]
        lean_ur = [v for k, v in scalar.items() if colours(k) and colours(k) <= UR]

        common_h = [k for k in scalar if k in human]
        common_v = [k for k in scalar if k in vswap]

        # How often the context-free ranking already decides the real pick.
        base = runner.logits(states)
        hit = tot = 0
        for s, lg in zip(states, base):
            sc = [scalar.get(nm) for nm in s.pack_names]
            if any(v is None for v in sc) or len(sc) < 2:
                continue
            hit += int(int(np.argmax(sc)) == int(np.argmax(lg)))
            tot += 1
        base_arg = np.array([int(np.argmax(l)) for l in base])

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
            "context_free": hit / tot if tot else float("nan"),
            "pool_flip": flip(mean_substitute(TYPE_POOL, mean_row=mean_row)),
            "taken_flip": flip(mean_substitute(TYPE_TAKEN, mean_row=mean_row)),
            "n_cards": len(scalar),
        })
        print(f"  {name}: {rows[-1]}")
        del model

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
