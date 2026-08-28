"""R17 follow-up 2 — the neighbourhood test with an encoder-independent metric.

``s_r17b`` defines a card's neighbourhood by cosine distance in the encoder's
own embedding, which risks reading the encoder's smoothing back off itself.
This rebuilds the k=10 neighbourhood from a plain bag-of-tokens cosine over
the converted text — a metric the encoder had no hand in — and re-runs the
controlled regression.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import probe_lib as pl
import s_common as S

d = S.load_all()
join, emb = d["join"], d["emb"]
val = d["val"]
pf, ph = d["pf"], d["ph"]
pred_f = pl.predict_labels(emb, pf)
pred_h = pl.predict_labels(emb, ph)
n_in = S.num(join, "n_in_deck")

CACHE = S.OUT / "s_r17c_bow_neighbours.npy"
K = 10
if CACHE.exists():
    nb = np.load(CACHE)
else:
    tok = pl.load_tokenizer(pl.VOCAB_PATH)
    seqs = []
    for p in join["txt_path"]:
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        stripped = "\n".join(l for l in text.splitlines() if not l.startswith("name:"))
        seqs.append(tok.tokenize_to_ids(stripped))
    vocab = sorted({t for s in seqs for t in s})
    index = {t: i for i, t in enumerate(vocab)}
    M = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for r, s in enumerate(seqs):
        for t, c in Counter(s).items():
            M[r, index[t]] = c
    M = np.log1p(M)
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    nb = np.empty((len(M), K), dtype=np.int64)
    for lo in range(0, len(M), 512):
        sims = M[lo:lo + 512] @ M.T
        sims[np.arange(sims.shape[0]), np.arange(lo, min(lo + 512, len(M)))] = -np.inf
        part = np.argpartition(-sims, K, axis=1)[:, :K]
        ordr = np.take_along_axis(sims, part, 1).argsort(axis=1)[:, ::-1]
        nb[lo:lo + 512] = np.take_along_axis(part, ordr, 1)
    np.save(CACHE, nb)
    print(f"bag-of-tokens vocab: {len(vocab)}", flush=True)

nb_emb = np.load(S.OUT / "s_r17_neighbours.npy")
overlap = np.mean([len(set(nb[i]) & set(nb_emb[i])) / K for i in range(len(nb))])

flag_cols = [c for c in join.columns if c.startswith(("kw_", "ph_"))]
rare = {c: (np.nan_to_num(S.num(join, c)) > 0) for c in flag_cols}
rare = {c: v for c, v in rare.items() if 20 <= v.sum() < 200}
rare_any = np.any(np.stack(list(rare.values())), 0)

out: dict = {"k": K, "mean_overlap_with_embedding_neighbourhood": float(overlap)}
rows = []
for head in ("score_play", "played_rate"):
    y = S.num(join, f"shrunk_{head}")
    w = S.num(join, f"w_{head}")
    have = np.isfinite(y) & (w > 0)
    nb_mean = np.array([np.nanmean(y[nb[i]]) for i in range(len(y))])
    for tag, p in (("fidelity/all", np.asarray(pred_f[head], float)),
                   ("honest/val", np.asarray(pred_h[head], float))):
        for sub_name, sub in (("all cards", np.ones(len(join), bool)),
                              ("rare-mechanic cards", rare_any),
                              ("n_in_deck < 200", n_in < 200),
                              ("n_in_deck >= 2000", n_in >= 2000)):
            m = have & np.isfinite(nb_mean) & sub
            if tag == "honest/val":
                m = m & val
            if m.sum() < 60:
                continue
            one = np.ones(int(m.sum()))
            XA = np.column_stack([one, y[m]])
            XB = np.column_stack([one, y[m], nb_mean[m]])
            bA, _, _ = S.wls(XA, p[m], w[m])
            bB, seB, tB = S.wls(XB, p[m], w[m])
            rows.append([head, tag, sub_name, int(m.sum()),
                         f"{bA[1]:+.3f}", f"{bB[1]:+.3f}", f"{bB[2]:+.3f}",
                         f"{seB[2]:.3f}", f"{tB[2]:+.1f}",
                         f"{pl._r2(p[m], XA @ bA, w[m]):.4f}",
                         f"{pl._r2(p[m], XB @ bB, w[m]):.4f}"])
out["controlled"] = rows
with open(S.OUT / "s_r17c.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
for r in rows:
    print(" | ".join(str(x) for x in r))
print("overlap", overlap)
