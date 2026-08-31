"""D7 — what the CONTEXT token knows, and when it knows it.

An auxiliary head fitted to a token the deployed policy never reads. The trunk
puts one ``CONTEXT`` token in front of the cards; the policy head reads only the
``PACK`` positions, and gen-3/gen-4 carry the critic head untrained. So the
CONTEXT token's output is a summary the agent computes and then discards, which
makes it the honest place to ask what the model has worked out about the draft.

Three targets, fitted by ridge on a draft-disjoint split and read off pick by
pick:

- **the seat's eventual two colours** — five one-vs-rest logistic probes. Asks
  when commitment is represented, and whether it is represented before the pool
  itself reveals it.
- **the seat's final pod-relative `deck_score`** — the exact quantity GRPO
  optimised, and the one the untrained critic head would have carried. A trunk
  that predicts it has learned a value function without a value head.
- **the pool's current colour fractions** — the control. The model can read this
  straight off its own POOL tokens, so it bounds what "the probe can decode X"
  is worth.

Usage
-----
    python scripts/draft_probes/d7_contextprobe.py --limit-drafts 300
"""

from __future__ import annotations

import argparse
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
    CardTable,
    PolicyRunner,
    iter_corpus_records,
    iter_corpus_states,
    load_agent,
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
}
BASIC = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}
WUBRG = "WUBRG"


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 30.0) -> np.ndarray:
    X1 = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    A = X1.T @ X1
    A[np.diag_indices_from(A)] += lam
    return np.linalg.solve(A, X1.T @ y)


def ridge_pred(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)]) @ w


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = float(((y - p) ** 2).sum())
    tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss / tot if tot > 0 else float("nan")


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(order.size)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-drafts", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--labels", default="gen4")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    labels = args.labels.split(",")
    colours = ColourResolver(CARDS)

    # Per-seat targets: eventual colours, and the pod-relative reward.
    final_colours: dict[tuple[str, int], np.ndarray] = {}
    reward: dict[tuple[str, int], float] = {}
    for rec in iter_corpus_records(CORPUS, args.limit_drafts):
        scores = [s.deck_score for s in rec.seats]
        for i, seat in enumerate(rec.seats):
            if not seat.deck or seat.deck_score is None:
                continue
            acc: dict[str, float] = defaultdict(float)
            for name in seat.deck:
                if name in BASIC:
                    continue
                ids = colours(name)
                for c in ids:
                    acc[c] += 1.0 / len(ids)
            top = {c for c, _ in sorted(acc.items(), key=lambda kv: -kv[1])[:2]}
            final_colours[(rec.draft_id, i)] = np.array(
                [1.0 if c in top else 0.0 for c in WUBRG], dtype=np.float32)
            others = [s for j, s in enumerate(scores)
                      if j != i and s is not None]
            reward[(rec.draft_id, i)] = seat.deck_score - (
                sum(others) / len(others) if others else 0.0)

    table = CardTable()
    samples = [
        s for s in iter_corpus_states(
            CORPUS, table, labels=labels, limit_drafts=args.limit_drafts)
        if (s.draft_id, s.seat) in final_colours
    ]
    print(f"{len(samples)} states, {len(final_colours)} seats")

    mat = table.matrix()
    colour_of_row = [colours(n) for n in table.names]
    pool_frac = np.zeros((len(samples), 5), dtype=np.float32)
    for k, s in enumerate(samples):
        rows = s.state.card_idx[s.state.type_idx == TYPE_POOL]
        if rows.size == 0:
            continue
        for r in rows:
            ids = colour_of_row[r]
            if not ids:
                continue
            for c in ids:
                pool_frac[k, WUBRG.index(c)] += 1.0 / len(ids)
        pool_frac[k] /= rows.size

    y_col = np.stack([final_colours[(s.draft_id, s.seat)] for s in samples])
    y_rew = np.array([reward[(s.draft_id, s.seat)] for s in samples],
                     dtype=np.float32)
    pick_abs = np.array([(s.state.pack_number - 1) * 15 + s.state.pick_number
                         for s in samples])
    draft_ids = sorted({s.draft_id for s in samples})
    val_ids = set(draft_ids[::4])            # draft-disjoint 25 % validation
    is_val = np.array([s.draft_id in val_ids for s in samples])

    buckets = [(1, 5), (6, 10), (11, 15), (16, 22), (23, 30), (31, 38), (39, 45)]
    results = {}
    for gen, ckpt in MODELS.items():
        if not ckpt.exists():
            print(f"skip {gen}")
            continue
        model, _ = load_agent(ckpt)
        runner = PolicyRunner(model, table, batch_size=args.batch)
        ctx = runner.context_vectors(samples)
        ctx = (ctx - ctx.mean(0)) / (ctx.std(0) + 1e-6)
        rows = []
        for lo, hi in buckets:
            m = (pick_abs >= lo) & (pick_abs <= hi)
            tr, va = m & ~is_val, m & is_val
            if tr.sum() < 200 or va.sum() < 100:
                continue
            aucs = []
            for c in range(5):
                w = ridge_fit(ctx[tr], y_col[tr, c])
                aucs.append(auc(y_col[va, c], ridge_pred(ctx[va], w)))
            w = ridge_fit(ctx[tr], y_rew[tr])
            r2_rew = r2(y_rew[va], ridge_pred(ctx[va], w))
            r2_pool = float(np.mean([
                r2(pool_frac[va, c],
                   ridge_pred(ctx[va], ridge_fit(ctx[tr], pool_frac[tr, c])))
                for c in range(5)
            ]))
            rows.append({
                "pick_lo": lo, "pick_hi": hi, "n_train": int(tr.sum()),
                "n_val": int(va.sum()),
                "colour_auc": float(np.nanmean(aucs)),
                "reward_r2": r2_rew,
                "pool_colour_r2": r2_pool,
            })
        results[gen] = rows
        print(f"\n=== {gen} — probes on the CONTEXT token ===")
        print(f"{'picks':>8s} {'n_val':>7s} {'colour AUC':>11s} "
              f"{'reward R2':>10s} {'pool R2':>8s}")
        for r in rows:
            print(f"{r['pick_lo']:3d}-{r['pick_hi']:<4d} {r['n_val']:7d} "
                  f"{r['colour_auc']:11.3f} {r['reward_r2']:10.3f} "
                  f"{r['pool_colour_r2']:8.3f}")
        del model

    (OUT / "d7_contextprobe.json").write_text(
        json.dumps({"n_states": len(samples), "buckets": buckets,
                    "models": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'd7_contextprobe.json'}")


if __name__ == "__main__":
    main()
