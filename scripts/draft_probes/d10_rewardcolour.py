"""D10 — does the reward the agent optimises already carry the colour lean?

`d2_pickorder.py` establishes that gen-4 opens the draft with a standing colour
preference gen-1 does not have, so the preference arrives with the reinforcement
learning. It does not say where in the reinforcement learning. The only signal
online GRPO carries is the scorer's deck score, made pod-relative by subtracting
the mean of the other seats, so if that reward pays differently for different
colours the policy would grow the lean without anything else teaching it.

The corpus answers this without running a model. Every seat of every recorded
draft carries both its finished deck and the scorer's score for it, which is the
reward term itself. Regressing the leave-one-out-centred score on the deck's
five colour shares prices each colour in the units the policy was trained in.

Two confounds have to be cleared before the coefficients mean anything.

Colour share is chosen, not assigned: a seat that ends up in red may be a seat
that was cut off and drafted a weak pool. The deck's own card quality is
therefore entered as a control, using two yardsticks that were built without
reference to each other — the win-rate label the encoder trains on
(``shrunk_score_play``) and the scorer's own causal swap value (``v_swap``, from
`output/scorer-probes/t2_card_values.csv`). A colour premium that survives both
is the scorer pricing the colour rather than the cards.

The stronger drafter also picks the colours it likes, so a premium measured over
the pooled corpus could be gen-4's skill showing up as its colours. The
regression is therefore repeated inside each drafting agent separately. Gen-1
and `forge-full` draft red-heavy decks and gen-4 drafts white-green ones, so a
premium of the same size in all three cannot be about who drafted the deck.

Usage
-----
    python scripts/draft_probes/d10_rewardcolour.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draft_corpus_common import ColourResolver  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "draft-probes"
G4 = REPO / "models/draft/agent/gen4"
CORPUS = G4 / "lr1e-5_t2all_decay0.3-yardstick-v-forge-drafts.jsonl"
CARDS = REPO / "output" / "cardsfolder-512"
VALUES = REPO / "output" / "scorer-probes" / "t2_card_values.csv"
P1P1 = OUT / "d2_p1p1_values.csv"

COLOURS = "WUBRG"


def card_column(path: Path, key: str, name_col: str = "name") -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row[name_col]] = float(row[key])
            except (TypeError, ValueError, KeyError):
                continue
    return out


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov = float(resid @ resid) / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), 1.0 - float(resid.var() / y.var())


def wg_ur(beta: np.ndarray) -> float:
    return (beta[0] + beta[4]) / 2 - (beta[1] + beta[3]) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-drafts", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    colours = ColourResolver(CARDS)
    quality = card_column(VALUES, "shrunk_score_play")
    swap = card_column(VALUES, "v_swap")
    print(f"card controls: win-rate label {len(quality)}, swap value {len(swap)}")

    pods: dict[str, list[dict]] = defaultdict(list)
    for line in CORPUS.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue                      # tolerated trailing partial line
        if args.limit_drafts and len(pods) >= args.limit_drafts:
            break
        for seat in rec["seats"]:
            if seat.get("deck_score") is None or not seat.get("deck"):
                continue                  # failed build: no deck, no reward
            acc: Counter[str] = Counter()
            n = nq = nv = 0
            q = v = 0.0
            for name in seat["deck"]:
                ids = colours(name)
                if ids:                   # basics and other colourless: no lane
                    n += 1
                    for c in ids:
                        acc[c] += 1.0 / len(ids)
                if name in quality:
                    q += quality[name]
                    nq += 1
                if name in swap:
                    v += swap[name]
                    nv += 1
            if min(n, nq, nv) < 10:
                continue
            pods[rec["draft_id"]].append({
                "agent": seat.get("agent"), "score": float(seat["deck_score"]),
                "frac": [acc[c] / n for c in COLOURS], "q": q / nq, "v": v / nv,
            })

    # The reward is pod-relative by leave-one-out, so centre each seat on the
    # mean of the *other* seats rather than on the pod mean.
    rows = []
    for seats in pods.values():
        if len(seats) < 2:
            continue
        total = sum(s["score"] for s in seats)
        k = len(seats)
        for s in seats:
            rows.append((s["agent"], s["score"] - (total - s["score"]) / (k - 1),
                         s["frac"], s["q"], s["v"]))
    print(f"{len(rows)} scored seats in {sum(1 for v in pods.values() if len(v) >= 2)}"
          f" pods, mean pod size {len(rows) / max(1, sum(1 for v in pods.values() if len(v) >= 2)):.1f}")

    y = np.array([r[1] for r in rows])
    C = np.stack([np.array(r[2]) for r in rows])
    q = np.array([r[3] for r in rows])
    v = np.array([r[4] for r in rows])
    q = (q - q.mean()) / q.std()
    v = (v - v.mean()) / v.std()
    print(f"reward sd {y.std():.3f}; mean colour share "
          + " ".join(f"{c}{C[:, i].mean():.3f}" for i, c in enumerate(COLOURS)))

    fits = {}
    print("\n=== reward paid per unit of deck colour share ===")
    for label, X, names in (
        ("colour only", C, list(COLOURS)),
        ("+ win-rate label", np.column_stack([C, q]), list(COLOURS) + ["q"]),
        ("+ swap value", np.column_stack([C, v]), list(COLOURS) + ["v"]),
        ("+ both", np.column_stack([C, q, v]), list(COLOURS) + ["q", "v"]),
    ):
        beta, se, r2 = ols(X, y)
        fits[label] = {"beta": beta.tolist(), "se": se.tolist(), "r2": r2,
                       "names": names, "wg_ur": wg_ur(beta)}
        print(f"  {label:18s} "
              + " ".join(f"{n}{beta[i]:+7.3f}" for i, n in enumerate(names))
              + f"   WG-UR {wg_ur(beta):+.3f}  R2 {r2:.3f}")
        print(f"  {'':18s} "
              + " ".join(f"{' ' * len(n)}({se[i]:.3f})"
                         for i, n in enumerate(names)))
    base = fits["colour only"]["wg_ur"]
    print(f"  premium retained: label {fits['+ win-rate label']['wg_ur'] / base:.0%},"
          f" swap {fits['+ swap value']['wg_ur'] / base:.0%},"
          f" both {fits['+ both']['wg_ur'] / base:.0%}")

    print("\n=== the same premium inside each drafting agent's own decks ===")
    per_agent = {}
    agents = sorted({r[0] for r in rows if r[0]},
                    key=lambda a: -sum(1 for r in rows if r[0] == a))
    for a in agents:
        sel = np.array([r[0] == a for r in rows])
        if sel.sum() < 200:
            continue
        beta, se, _ = ols(np.column_stack([C[sel], q[sel], v[sel]]), y[sel])
        per_agent[a] = {"n": int(sel.sum()), "beta": beta.tolist(),
                        "se": se.tolist(), "wg_ur": wg_ur(beta),
                        "share": C[sel].mean(0).tolist()}
        print(f"  {a:12s} n={int(sel.sum()):5d}  "
              + " ".join(f"{c}{beta[i]:+7.3f}" for i, c in enumerate(COLOURS))
              + f"   WG-UR {wg_ur(beta):+.3f}   drafted share "
              + " ".join(f"{c}{C[sel, i].mean():.2f}"
                         for i, c in enumerate(COLOURS)))

    # Does the reward's colour ordering match the policy's opening ranking? The
    # P1P1 scalars come from d2; gen-1 is the control that never saw the reward.
    ordering = {}
    if P1P1.exists():
        p1 = list(csv.DictReader(P1P1.open(encoding="utf-8")))
        beta = np.array(fits["colour only"]["beta"])
        print("\n=== colour order: the reward, then each policy's opening pick ===")
        print(f"  {'reward':12s} " + " > ".join(
            COLOURS[i] for i in np.argsort(-beta)))
        ordering["reward"] = [COLOURS[i] for i in np.argsort(-beta)]
        for gen in ("gen1", "gen3", "gen4", "gen4b"):
            key = f"p1p1_{gen}"
            if key not in (p1[0] if p1 else {}):
                continue
            per = []
            for c in COLOURS:
                vals = [float(r[key]) for r in p1
                        if r["colours"] and c in r["colours"] and r[key]]
                per.append(float(np.mean(vals)) if vals else float("nan"))
            per = np.array(per)
            ordering[gen] = [COLOURS[i] for i in np.argsort(-per)]
            print(f"  {gen:12s} " + " > ".join(ordering[gen])
                  + f"    (rank corr with reward "
                  f"{np.corrcoef(np.argsort(np.argsort(-per)), np.argsort(np.argsort(-beta)))[0, 1]:+.2f})")

    (OUT / "d10_rewardcolour.json").write_text(
        json.dumps({"n_seats": len(rows), "reward_sd": float(y.std()),
                    "fits": fits, "per_agent": per_agent,
                    "colour_order": ordering}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'd10_rewardcolour.json'}")


if __name__ == "__main__":
    main()
