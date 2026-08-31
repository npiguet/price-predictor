"""D1 — the information audit: what each generation's policy actually reads.

Every channel of the state is disabled one at a time and the policy is re-run on
the *same* states. Three design rules, all forced by the critique:

- **Count-preserving edits.** Deleting a token block also changes how many tokens
  the trunk averages over, which moves every logit on its own. Each block is
  instead blanked by substituting the corpus-mean card vector for its cards, so
  only *which cards they are* is destroyed.
- **A size-matched placebo.** Blanking a big block moves the policy more than
  blanking a small one whatever it contains. Random subsets of non-PACK tokens
  of every size give the magnitude law, and each block is reported against the
  placebo of its own size.
- **All three generations on identical states.** gen-1 is distilled Forge, so a
  channel gen-1 already reads is Forge's and not the RL's.

Retype arms ask a different question: relabelling POOL as TAKEN keeps every card
and every count and destroys only the ownership distinction.

Usage
-----
    python scripts/draft_probes/d1_channels.py --limit-drafts 150
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_lib import (  # noqa: E402
    TYPE_PACK,
    TYPE_PASSED,
    TYPE_POOL,
    TYPE_TAKEN,
    CardTable,
    PickSample,
    PolicyRunner,
    iter_corpus_states,
    load_agent,
    mean_substitute,
    retype,
    set_context,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "draft-probes"
G4 = REPO / "models/draft/agent/gen4"
CORPUS = G4 / "lr1e-5_t2all_decay0.3-yardstick-v-forge-drafts.jsonl"

MODELS = {
    "gen1": REPO / "models/draft/agent/gen1/l6_lr3e-4_decay_20260604_080249.pt",
    "gen3": REPO / "models/draft/agent/gen3/temperature-on-all-agents"
                   "/lr1e-5_t2_20260805_221050.pt",
    "gen4": G4 / "lr1e-5_t2all_decay0.3.pt",
    "gen4b": G4 / "lr1e-5_t2all_nodecay.pt",
}


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def kl(a, b):
        s = a > 0
        return float(np.sum(a[s] * np.log(a[s] / b[s])))
    return float(np.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))


def zero_recency_of(*types: int):
    """Zero recency on selected blocks only, leaving the rest intact."""
    sel_types = set(types)

    def apply(sample: PickSample):
        st = sample.state
        sel = np.isin(st.type_idx, list(sel_types))
        pa, pi = st.packs_ago.copy(), st.pick_ago.copy()
        pa[sel] = 0
        pi[sel] = 0
        return replace(st, packs_ago=pa, pick_ago=pi)

    return apply


def placebo(k: int, mean_row: int, seed: int):
    """Blank ``k`` uniformly-chosen non-PACK tokens — the magnitude control."""
    rng = np.random.default_rng(seed)

    def apply(sample: PickSample):
        st = sample.state
        cand = np.flatnonzero(st.type_idx != TYPE_PACK)
        if cand.size == 0:
            return st
        take = rng.choice(cand, size=min(k, cand.size), replace=False)
        idx = st.card_idx.copy()
        idx[take] = mean_row
        return replace(st, card_idx=idx)

    return apply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-drafts", type=int, default=150)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seat-label", default="gen4")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    table = CardTable()
    samples = list(iter_corpus_states(
        CORPUS, table, labels=[args.seat_label], limit_drafts=args.limit_drafts))
    mat = table.matrix()
    mean_row = table.add_vector(mat.mean(0), "<corpus-mean>")

    # One row per sample holding that sample's own POOL mean: the arm that keeps
    # the pool's average and destroys everything else about it.
    pool_row: dict[int, int] = {}
    for s in samples:
        sel = s.state.type_idx == TYPE_POOL
        vec = mat[s.state.card_idx[sel]].mean(0) if sel.any() else mat.mean(0)
        pool_row[id(s)] = table.add_vector(vec, "<pool-mean>")

    block_size = {
        "POOL": np.array([int((s.state.type_idx == TYPE_POOL).sum())
                          for s in samples]),
        "PASSED": np.array([int((s.state.type_idx == TYPE_PASSED).sum())
                            for s in samples]),
        "TAKEN": np.array([int((s.state.type_idx == TYPE_TAKEN).sum())
                           for s in samples]),
    }
    pack_no = np.array([s.state.pack_number for s in samples])
    pick_no = np.array([s.state.pick_number for s in samples])

    arms: list[tuple[str, object]] = [
        ("POOL identity", mean_substitute(TYPE_POOL, mean_row=mean_row)),
        ("POOL beyond its mean",
         mean_substitute(TYPE_POOL, mean_row=lambda s: pool_row[id(s)])),
        ("PASSED identity", mean_substitute(TYPE_PASSED, mean_row=mean_row)),
        ("TAKEN identity", mean_substitute(TYPE_TAKEN, mean_row=mean_row)),
        ("PASSED+TAKEN identity",
         mean_substitute(TYPE_PASSED, TYPE_TAKEN, mean_row=mean_row)),
        ("all context identity",
         mean_substitute(TYPE_POOL, TYPE_PASSED, TYPE_TAKEN, mean_row=mean_row)),
        ("ownership (POOL->TAKEN)", retype(TYPE_POOL, TYPE_TAKEN)),
        ("fate (TAKEN->PASSED)", retype(TYPE_TAKEN, TYPE_PASSED)),
        ("fate (PASSED->TAKEN)", retype(TYPE_PASSED, TYPE_TAKEN)),
        ("recency: PACK", zero_recency_of(TYPE_PACK)),
        ("recency: POOL", zero_recency_of(TYPE_POOL)),
        ("recency: PASSED+TAKEN", zero_recency_of(TYPE_PASSED, TYPE_TAKEN)),
        ("recency: all",
         zero_recency_of(TYPE_PACK, TYPE_POOL, TYPE_PASSED, TYPE_TAKEN)),
        ("pack_number -> 1", set_context(pack_number=1)),
        ("pick_number -> 1", set_context(pick_number=1)),
        ("pick_number -> 15", set_context(pick_number=15)),
    ]
    placebo_ks = [1, 2, 5, 10, 20, 40, 80, 160]
    for k in placebo_ks:
        arms.append((f"placebo k={k}", placebo(k, mean_row, seed=100 + k)))

    results = {}
    for gen, ckpt in MODELS.items():
        if not ckpt.exists():
            print(f"skip {gen}: {ckpt} missing")
            continue
        model, _ = load_agent(ckpt)
        runner = PolicyRunner(model, table, batch_size=args.batch)
        base = runner.logits(samples)
        base_p = [softmax(l) for l in base]
        base_arg = np.array([int(np.argmax(l)) for l in base])
        fidelity = float(np.mean([
            int(np.argmax(l) == s.target)
            for s, l in zip(samples, base) if s.target >= 0]))

        rows = []
        for name, interv in arms:
            alt = runner.logits(samples, interv)
            flip = np.array([int(np.argmax(a) != b)
                             for a, b in zip(alt, base_arg)])
            dist = np.array([js(p, softmax(a)) for p, a in zip(base_p, alt)])
            row = {
                "arm": name,
                "flip": float(flip.mean()),
                "js": float(dist.mean()),
                "flip_by_pack": {int(p): float(flip[pack_no == p].mean())
                                 for p in (1, 2, 3)},
                "flip_pick_le8": float(flip[pick_no <= 8].mean()),
                "flip_pick_ge9": float(flip[pick_no >= 9].mean()),
            }
            if name in ("POOL identity", "PASSED identity", "TAKEN identity"):
                row["mean_block_size"] = float(
                    block_size[name.split()[0]].mean())
            rows.append(row)
        results[gen] = {
            "checkpoint": str(ckpt),
            "n_states": len(samples),
            "replay_fidelity": fidelity,
            "arms": rows,
        }
        print(f"\n=== {gen}  n={len(samples)}  "
              f"self-replay fidelity {fidelity:.4f} ===")
        for r in rows:
            fb = r["flip_by_pack"]
            print(f"  {r['arm']:26s} flip {r['flip']:.4f}  JS {r['js']:.4f}"
                  f"  p1 {fb[1]:.3f} p2 {fb[2]:.3f} p3 {fb[3]:.3f}"
                  f"  <=8 {r['flip_pick_le8']:.3f} >=9 {r['flip_pick_ge9']:.3f}")

    meta = {
        "corpus": str(CORPUS),
        "seat_label": args.seat_label,
        "mean_block_size": {k: float(v.mean()) for k, v in block_size.items()},
        "placebo_ks": placebo_ks,
    }
    (OUT / "d1_channels.json").write_text(
        json.dumps({"meta": meta, "models": results}, indent=2), encoding="utf-8")
    print(f"\nmean block sizes: {meta['mean_block_size']}")
    print(f"wrote {OUT / 'd1_channels.json'}")


if __name__ == "__main__":
    main()
