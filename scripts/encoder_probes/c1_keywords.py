"""C1 (R10) — the keyword ladder by layout-matched counterfactual edit.

Design A (the ranked table): creatures carrying exactly one keyword
``static:`` line have that line's *body* substituted for every other
keyword in the set. Every arm has identical layout, so the line-position
artifact R1c measured (0.11-0.49 SD for an inert line move) cancels
exactly; what is left is a within-line token substitution, whose null is
mean-zero at ~0.09 SD.

The full ordered pairwise matrix is reduced to a one-dimensional additive
scale by weighted least squares on ``delta(K->K') = v(K') - v(K)``, with a
cluster bootstrap over base cards and an antisymmetry consistency check.

Design B (absolute premium): delete the keyword line outright. That
carries the line-deletion artifact, so it is reported but does not drive
the ranking.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402

KEYWORDS = [
    "flying", "lifelink", "deathtouch", "vigilance", "first strike",
    "double strike", "trample", "haste", "reach", "menace", "defender",
    "hexproof", "shroud", "indestructible", "flash", "ward {2}",
]
MAX_PER_KW = 400
SEED = 0


def keyword_statics(stripped: str) -> list[tuple[int, str]]:
    out = []
    for i, l in enumerate(cc.lines(stripped)):
        if l.startswith("static: "):
            body = l[len("static: "):].strip()
            if body in KEYWORDS:
                out.append((i, body))
    return out


def fit_scale(pairs: pd.DataFrame, keywords: list[str], col: str) -> dict:
    """Weighted least squares for v with delta(K->K') = v(K') - v(K)."""
    idx = {k: i for i, k in enumerate(keywords)}
    A = np.zeros((len(pairs), len(keywords)))
    for r, (f, t) in enumerate(zip(pairs["from"], pairs["to"])):
        A[r, idx[t]] += 1.0
        A[r, idx[f]] -= 1.0
    w = np.sqrt(pairs["n"].to_numpy(float))
    A2 = np.vstack([A * w[:, None], np.full((1, len(keywords)), 1e3)])
    y2 = np.concatenate([pairs[col].to_numpy(float) * w, [0.0]])
    v, *_ = np.linalg.lstsq(A2, y2, rcond=None)
    return {k: float(v[idx[k]]) for k in keywords}


def _scale_from_matrix(W: np.ndarray, groups: list[np.ndarray],
                       keywords: list[str]) -> np.ndarray:
    """Additive scale straight from the (n_base x n_kw) prediction matrix."""
    rows = []
    for gi, sel in enumerate(groups):
        if len(sel) == 0:
            continue
        means = W[sel].mean(axis=0)
        for gj in range(len(keywords)):
            if gj == gi:
                continue
            rows.append((gi, gj, len(sel), means[gj] - means[gi]))
    n = len(keywords)
    A = np.zeros((len(rows), n))
    y = np.empty(len(rows))
    w = np.empty(len(rows))
    for r, (gi, gj, cnt, d) in enumerate(rows):
        A[r, gj] += 1.0
        A[r, gi] -= 1.0
        y[r] = d
        w[r] = np.sqrt(cnt)
    A2 = np.vstack([A * w[:, None], np.full((1, n), 1e3)])
    y2 = np.concatenate([y * w, [0.0]])
    v, *_ = np.linalg.lstsq(A2, y2, rcond=None)
    return v


def main() -> None:
    rng = np.random.default_rng(SEED)
    j = cc.join_table()
    stripped_all = [cc.strip_name(t) for t in j["text"]]
    is_crea = j["is_creature"].fillna(False).astype(bool).to_numpy()

    hits = [keyword_statics(s) for s in stripped_all]
    n_kw = np.array([len(h) for h in hits])
    solo_idx = np.flatnonzero(is_crea & (n_kw == 1))
    solo_kw = np.array([hits[i][0][1] for i in solo_idx])

    base_rows: list[int] = []
    for k in KEYWORDS:
        pool = solo_idx[solo_kw == k]
        if len(pool) == 0:
            continue
        take = pool if len(pool) <= MAX_PER_KW else rng.choice(
            pool, MAX_PER_KW, replace=False)
        base_rows.extend(sorted(int(x) for x in take))
    base_rows = np.array(base_rows)
    own = np.array([hits[i][0][1] for i in base_rows])
    print(f"[A] base cards: {len(base_rows)}", flush=True)

    texts = []
    for r in base_rows:
        line_i = hits[r][0][0]
        for k in KEYWORDS:
            texts.append(cc.replace_in_line(stripped_all[r], line_i, f"static: {k}"))
    print(f"[A] encoding {len(texts)} texts", flush=True)
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    dist = cc.offmanifold(emb)

    nb, nk = len(base_rows), len(KEYWORDS)
    W_sp = pred["score_play"].to_numpy().reshape(nb, nk)
    W_pr = pred["played_rate"].to_numpy().reshape(nb, nk)
    W_d = dist.reshape(nb, nk)
    groups = [np.flatnonzero(own == k) for k in KEYWORDS]

    np.savez_compressed(cc.SCRATCH / "c1_pairwise_raw.npz",
                        rows=base_rows, own=own, keywords=np.array(KEYWORDS),
                        sp=W_sp, pr=W_pr, dist=W_d)

    pair_rows = []
    for gi, k in enumerate(KEYWORDS):
        sel = groups[gi]
        if len(sel) == 0:
            continue
        for gj, kp in enumerate(KEYWORDS):
            if gj == gi:
                continue
            d = W_sp[sel, gj] - W_sp[sel, gi]
            lo, hi = cc.bootstrap_ci(d, n_boot=1500)
            pair_rows.append({
                "from": k, "to": kp, "n": int(len(sel)),
                "delta_sp": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                "delta_pr": float((W_pr[sel, gj] - W_pr[sel, gi]).mean()),
                "frac_pos": float((d > 0).mean()),
                "off_manifold": float(((W_d[sel, gj] > cc.MANIFOLD_GATE) |
                                       (W_d[sel, gi] > cc.MANIFOLD_GATE)).mean()),
            })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(cc.SCRATCH / "c1_pairwise.csv", index=False)

    scale = fit_scale(pairs, KEYWORDS, "delta_sp")
    scale_pr = fit_scale(pairs, KEYWORDS, "delta_pr")

    # cluster bootstrap over base cards, within keyword group
    brng = np.random.default_rng(11)
    boot_sp = np.empty((600, nk))
    boot_pr = np.empty((600, nk))
    for b in range(600):
        res = [brng.choice(g, len(g), replace=True) if len(g) else g for g in groups]
        boot_sp[b] = _scale_from_matrix(W_sp, res, KEYWORDS)
        boot_pr[b] = _scale_from_matrix(W_pr, res, KEYWORDS)

    fitted = np.array([scale[t] - scale[f] for f, t in zip(pairs["from"], pairs["to"])])
    resid = pairs["delta_sp"].to_numpy() - fitted
    wn = pairs["n"].to_numpy(float)
    fit_r2 = 1.0 - float((wn * resid ** 2).sum() /
                         (wn * (pairs["delta_sp"] - pairs["delta_sp"].mean()) ** 2).sum())

    piv = pairs.pivot(index="from", columns="to", values="delta_sp")
    asym = [piv.loc[a, b] + piv.loc[b, a]
            for i, a in enumerate(KEYWORDS) for b in KEYWORDS[i + 1:]
            if a in piv.index and b in piv.index]
    asym = np.array([v for v in asym if np.isfinite(v)])

    scale_tab = pd.DataFrame({
        "keyword": KEYWORDS,
        "n_carriers": [len(g) for g in groups],
        "value_sp": [scale[k] for k in KEYWORDS],
        "ci_lo": np.quantile(boot_sp, 0.025, axis=0),
        "ci_hi": np.quantile(boot_sp, 0.975, axis=0),
        "value_pr": [scale_pr[k] for k in KEYWORDS],
        "pr_ci_lo": np.quantile(boot_pr, 0.025, axis=0),
        "pr_ci_hi": np.quantile(boot_pr, 0.975, axis=0),
    }).sort_values("value_sp", ascending=False)
    scale_tab.to_csv(cc.SCRATCH / "c1_scale.csv", index=False)
    print(scale_tab.to_string(index=False), flush=True)

    # ── Design B: line deletion ─────────────────────────────────────────
    dtexts = [cc.drop_line(stripped_all[r], hits[r][0][0]) for r in base_rows]
    demb = cc.encode(dtexts)
    dpred = cc.predict_sd(demb)
    ddist = cc.offmanifold(demb)
    own_gi = np.array([KEYWORDS.index(k) for k in own])
    base_sp = W_sp[np.arange(nb), own_gi]
    base_pr = W_pr[np.arange(nb), own_gi]
    del_sp = dpred["score_play"].to_numpy()
    del_pr = dpred["played_rate"].to_numpy()

    del_rows = []
    for gi, k in enumerate(KEYWORDS):
        sel = groups[gi]
        if len(sel) == 0:
            continue
        d = base_sp[sel] - del_sp[sel]
        lo, hi = cc.bootstrap_ci(d, n_boot=1500)
        del_rows.append({
            "keyword": k, "n": int(len(sel)), "premium_sp": float(d.mean()),
            "ci_lo": lo, "ci_hi": hi, "frac_pos": float((d > 0).mean()),
            "premium_pr": float((base_pr[sel] - del_pr[sel]).mean()),
            "off_manifold": float((ddist[sel] > cc.MANIFOLD_GATE).mean()),
        })
    dele = pd.DataFrame(del_rows).sort_values("premium_sp", ascending=False)
    dele.to_csv(cc.SCRATCH / "c1_delete.csv", index=False)
    print(dele.to_string(index=False), flush=True)

    rank_a = pd.Series([scale[k] for k in dele["keyword"]]).rank()
    summary = {
        "n_base_A": int(nb), "n_arms_A": int(len(texts)),
        "scale_fit_r2": fit_r2,
        "scale_fit_resid_sd": float(np.sqrt((wn * resid ** 2).sum() / wn.sum())),
        "antisymmetry_mean": float(asym.mean()),
        "antisymmetry_max_abs": float(np.abs(asym).max()),
        "off_manifold_A": float((dist > cc.MANIFOLD_GATE).mean()),
        "off_manifold_delete": float((ddist > cc.MANIFOLD_GATE).mean()),
        "spearman_A_vs_B": float(rank_a.corr(
            dele["premium_sp"].reset_index(drop=True).rank(), method="spearman")),
        "mean_delete_premium": float(dele["premium_sp"].mean()),
        "scale_sp": {k: float(scale[k]) for k in KEYWORDS},
        "scale_pr": {k: float(scale_pr[k]) for k in KEYWORDS},
    }
    (cc.SCRATCH / "c1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
