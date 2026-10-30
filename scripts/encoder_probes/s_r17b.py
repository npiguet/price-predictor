"""R17 follow-up — the MLM-neighbourhood test, purged of its mechanical term.

Regressing (pred − y) on (nb_mean − y) is biased: both sides carry −y, so the
slope is positive even for a prediction that ignores the neighbourhood. The
honest question is whether ``nb_mean`` adds anything to ``pred`` **once the
card's own label is controlled for**, so that is what this fits.

``--neighbours embedding`` (the default) takes each card's k=10 neighbourhood
by cosine in the encoder's own embedding, as ``s_r17`` saved it, and also
reports the naive slope against its mechanical nulls; writes ``s_r17b.json``.
That metric risks reading the encoder's smoothing back off itself, so
``--neighbours bow`` rebuilds the neighbourhood from a plain bag-of-tokens
cosine over the converted text — a metric the encoder had no hand in — and
re-runs the controlled regression; writes ``s_r17c.json``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import probe_lib as pl
import s_common as S
from _lib.jsonio import write_json
from _lib.stats import wls

K = 10
BOW_CACHE = S.OUT / "s_r17c_bow_neighbours.npy"


def bow_neighbours(join) -> np.ndarray:
    """k nearest cards by cosine over log-scaled bag-of-token counts (cached)."""
    if BOW_CACHE.exists():
        return np.load(BOW_CACHE)
    tok = pl.load_tokenizer(pl.VOCAB_PATH)
    seqs = [tok.tokenize_to_ids(t) for t in join["text"]]
    vocab = sorted({t for s in seqs for t in s})
    index = {t: i for i, t in enumerate(vocab)}
    M = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for r, s in enumerate(seqs):
        for t, c in Counter(s).items():
            M[r, index[t]] = c
    M = np.log1p(M)
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    nb = S.top_k_cosine(M, K, chunk=512)
    np.save(BOW_CACHE, nb)
    print(f"bag-of-tokens vocab: {len(vocab)}", flush=True)
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--neighbours", choices=("embedding", "bow"), default="embedding")
    bow = ap.parse_args().neighbours == "bow"

    fr = S.ProbeFrame(text=bow)
    join, emb, val = fr.join, fr.emb, fr.val
    pred_f, pred_h = fr.pred_f, fr.pred_h
    n_in = fr.num("n_in_deck")
    nb_emb = np.load(S.OUT / "s_r17_neighbours.npy")
    nb_idx = bow_neighbours(join) if bow else nb_emb

    flag_cols = [c for c in join.columns if c.startswith(("kw_", "ph_"))]
    rare = {c: fr.num(c, 0) > 0 for c in flag_cols}
    rare_any = np.any(np.stack([v for v in rare.values() if 20 <= v.sum() < 200]), 0)

    if bow:
        overlap = np.mean([len(set(nb_idx[i]) & set(nb_emb[i])) / K
                           for i in range(len(nb_idx))])
        out: dict = {"k": K, "mean_overlap_with_embedding_neighbourhood": float(overlap)}
    else:
        Xn = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        nb_cos = np.array([float(Xn[i] @ Xn[nb_idx[i]].mean(0)) for i in range(len(Xn))])
        out = {"n_rare_cards": int(rare_any.sum()),
               "mean_cosine_to_neighbour_centroid": float(nb_cos.mean()),
               "p10_cosine": float(np.quantile(nb_cos, 0.1))}

    rows = []
    for head in ("score_play", "played_rate"):
        y, w, have = fr.label(head)
        nb_mean = S.neighbour_mean(y, nb_idx)
        for tag, p in (("fidelity/all", np.asarray(pred_f[head], float)),
                       ("honest/val", np.asarray(pred_h[head], float))):
            for sub_name, sub in (("all cards", np.ones(len(join), bool)),
                                  ("rare-mechanic cards", rare_any),
                                  ("n_in_deck < 200", n_in < 200),
                                  ("n_in_deck >= 2000" if bow else "n_in_deck ≥ 2000",
                                   n_in >= 2000)):
                m = have & np.isfinite(nb_mean) & sub
                if tag == "honest/val":
                    m = m & val
                if m.sum() < 60:
                    continue
                one = np.ones(int(m.sum()))
                XA = np.column_stack([one, y[m]])
                XB = np.column_stack([one, y[m], nb_mean[m]])
                fA, fB = wls(XA, p[m], w[m]), wls(XB, p[m], w[m])
                r2A = pl._r2(p[m], XA @ fA.beta, w[m])
                r2B = pl._r2(p[m], XB @ fB.beta, w[m])
                rows.append([head, tag, sub_name, int(m.sum()),
                             f"{fA.beta[1]:+.3f}", f"{fB.beta[1]:+.3f}", f"{fB.beta[2]:+.3f}",
                             f"{fB.se[2]:.3f}", f"{fB.t[2]:+.1f}", f"{r2A:.4f}", f"{r2B:.4f}"]
                            + ([] if bow else [f"{r2B - r2A:+.4f}"]))

    if bow:
        out["controlled"] = rows
        write_json(S.OUT / "s_r17c.json", out, indent=1)
        for r in rows:
            print(" | ".join(str(x) for x in r))
        print("overlap", out["mean_overlap_with_embedding_neighbourhood"])
        return

    out["neighbourhood_controlled"] = rows
    out["naive_slope_vs_null"] = naive_slopes(fr, nb_idx)
    write_json(S.OUT / "s_r17b.json", out, indent=1)
    print(json.dumps(out, indent=1, default=float))


def naive_slopes(fr: S.ProbeFrame, nb_idx: np.ndarray) -> list[list]:
    """The naive (biased) slope and its mechanical nulls, for the record."""
    rows = []
    for head in ("score_play", "played_rate"):
        y, w, have = fr.label(head)
        nb_mean = S.neighbour_mean(y, nb_idx)
        for tag, p in (("fidelity/all", np.asarray(fr.pred_f[head], float)),
                       ("honest/val", np.asarray(fr.pred_h[head], float))):
            m = have & np.isfinite(nb_mean) & (fr.val if tag == "honest/val"
                                               else np.ones(len(fr), bool))
            X = np.column_stack([np.ones(int(m.sum())), nb_mean[m] - y[m]])
            observed = wls(X, p[m] - y[m], w[m]).beta
            # null 1: the prediction replaced by the corpus mean
            mu = np.average(y[m], weights=w[m])
            to_mean = wls(X, mu - y[m], w[m]).beta
            # null 2: a label-preserving permutation of the prediction
            ps = p[m].copy()
            np.random.default_rng(0).shuffle(ps)
            permuted = wls(X, ps - y[m], w[m]).beta
            rows.append([head, tag, int(m.sum()), f"{observed[1]:+.3f}",
                         f"{to_mean[1]:+.3f}", f"{permuted[1]:+.3f}"])
    return rows


if __name__ == "__main__":
    main()
