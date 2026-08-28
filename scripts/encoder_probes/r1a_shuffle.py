"""R1a — the shuffle battery: bag-of-words vs composition.

Re-encodes the whole joined corpus under three text-degradation regimes,
refits honest ridge probes on the degraded embeddings, and reports how
much of the encoder's label knowledge survives.

Conditions (the shuffle unit is the *whitespace-separated word*, not the
token id — the tokenizer splits a mana cost like ``{3}{W}{W}`` into
per-symbol units and there is no detokenizer, so a word-level shuffle is
the closest faithful destruction of order that round-trips through the
encoder's own text interface):

* ``none``          — unshuffled control, re-encoded on the same batched
                      GPU path as the shuffles (removes the ~5e-7
                      batching offset from every comparison).
* ``full``          — every word of the name-stripped text shuffled.
* ``within_line``   — line order kept, words shuffled inside each line.
* ``line_order``    — lines permuted, word order inside lines intact.

Outputs ``output/encoder-probes/r1a_*.{csv,json}``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl  # noqa: E402

SEEDS = (11, 23)
CONDITIONS = ("full", "within_line", "line_order")
HEADS = ("score_play", "score_draw", "played_rate", "cast_lift",
         "color_lift_W", "color_lift_U", "color_lift_B", "color_lift_R",
         "color_lift_G")
BATCH = 96


def strip_name(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.startswith("name:"))


def degrade(text: str, condition: str, rng: np.random.Generator) -> str:
    """One degraded variant of a name-stripped card text."""
    lines = [l for l in text.splitlines() if l.strip()]
    if condition == "none":
        return "\n".join(lines)
    if condition == "full":
        words = " ".join(lines).split()
        order = rng.permutation(len(words))
        return " ".join(words[i] for i in order)
    if condition == "within_line":
        out = []
        for line in lines:
            words = line.split()
            order = rng.permutation(len(words))
            out.append(" ".join(words[i] for i in order))
        return "\n".join(out)
    if condition == "line_order":
        order = rng.permutation(len(lines))
        return "\n".join(lines[i] for i in order)
    raise ValueError(condition)


def cosine_shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return 1.0 - (an * bn).sum(axis=1)


def main() -> None:
    join = pl.build_join()
    join = join[join["is_primary"]].reset_index(drop=True)
    print(f"joined primary cards: {len(join)}", flush=True)

    texts = [strip_name(Path(p).read_text(encoding="utf-8", errors="replace"))
             for p in join["txt_path"]]
    base_emb = pl.load_embedding_matrix(list(join["name"]), join)

    fidelity = pl.load_probes("fidelity", True)
    base_pred = pl.predict_labels(base_emb, fidelity)

    runner = pl.EncoderRunner()
    print(f"encoder on {runner.device}, max_seq_len={runner.max_seq_len}", flush=True)

    _, corpus_ref = pl.corpus_embedding_matrix()
    sample_idx = np.random.default_rng(7).choice(len(join), 2000, replace=False)

    rows: list[dict] = []
    pred_rows: list[dict] = []

    plan = [("none", 0)] + [(c, s) for c in CONDITIONS for s in SEEDS]
    for condition, seed in plan:
        rng = np.random.default_rng(seed)
        variants = [degrade(t, condition, rng) for t in texts]
        emb = runner.encode_texts(variants, batch_size=BATCH)
        print(f"[{condition} seed={seed}] encoded", flush=True)

        # 1. how far the representation moved
        cos = cosine_shift(base_emb, emb)
        mdist, _ = pl.manifold_distance(emb[sample_idx], corpus_ref)

        # 2. how far the *prediction* moved through the unshuffled probe
        pred = pl.predict_labels(emb, fidelity)
        for head in HEADS:
            d = (pred[head] - base_pred[head]).to_numpy()
            pred_rows.append({
                "condition": condition, "seed": seed, "head": head,
                "mean_abs_delta": float(np.abs(d).mean()),
                "median_abs_delta": float(np.median(np.abs(d))),
                "mean_delta": float(d.mean()),
                "pearson_vs_base": float(np.corrcoef(pred[head], base_pred[head])[0, 1]),
            })

        # 3. refit honest probes on the degraded embeddings
        ps = pl.fit_probes(join, emb, mode="honest", weighted=True, heads=HEADS)
        for head in HEADS:
            m = ps.probes[head].metrics
            rows.append({
                "condition": condition, "seed": seed, "head": head,
                "alpha": ps.probes[head].alpha,
                "train_r2": m.get("train_r2"), "val_r2": m.get("val_r2"),
                "val_pearson": m.get("val_pearson"), "cv_r2": m.get("cv_r2"),
                "mean_cosine_shift": float(cos.mean()),
                "median_cosine_shift": float(np.median(cos)),
                "median_manifold_dist": float(np.median(mdist)),
                "frac_off_manifold": float((mdist > 0.35325).mean()),
            })
        print(f"[{condition} seed={seed}] "
              f"score_play val_r2={ps.probes['score_play'].metrics['val_r2']:.4f} "
              f"played_rate val_r2={ps.probes['played_rate'].metrics['val_r2']:.4f} "
              f"cos={cos.mean():.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(pl.SCRATCH / "r1a_shuffle_r2.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(pl.SCRATCH / "r1a_shuffle_pred_shift.csv", index=False)

    agg = (df.groupby(["condition", "head"])
             .agg(val_r2=("val_r2", "mean"), train_r2=("train_r2", "mean"),
                  val_pearson=("val_pearson", "mean"),
                  cos=("mean_cosine_shift", "mean"),
                  off_manifold=("frac_off_manifold", "mean"))
             .reset_index())
    agg.to_csv(pl.SCRATCH / "r1a_shuffle_r2_agg.csv", index=False)
    print(agg.to_string(index=False))

    with open(pl.SCRATCH / "r1a_meta.json", "w") as f:
        json.dump({"seeds": list(SEEDS), "conditions": ["none", *CONDITIONS],
                   "n_cards": int(len(join)), "batch": BATCH}, f, indent=2)


if __name__ == "__main__":
    main()
