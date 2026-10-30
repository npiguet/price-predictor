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
import probe_lib as pl  # noqa: E402

KEYWORDS = cc.KEYWORDS
MAX_PER_KW = 400
SEED = 0


def keyword_statics(stripped: str) -> list[tuple[int, str]]:
    out = []
    for i, ln in enumerate(cc.lines(stripped)):
        if ln.startswith("static: "):
            body = ln[len("static: "):].strip()
            if body in KEYWORDS:
                out.append((i, body))
    return out


def fit_scale(pairs: pd.DataFrame, labels: list[str], col: str) -> dict:
    """Weighted least squares for v with delta(K->K') = v(K') - v(K).

    Rows are weighted by sqrt(n carriers); a heavy sum-to-zero row pins the
    otherwise free additive constant.
    """
    idx = {k: i for i, k in enumerate(labels)}
    A = np.zeros((len(pairs), len(labels)))
    for r, (f, t) in enumerate(zip(pairs["from"], pairs["to"])):
        A[r, idx[t]] += 1.0
        A[r, idx[f]] -= 1.0
    w = np.sqrt(pairs["n"].to_numpy(float))
    A2 = np.vstack([A * w[:, None], np.full((1, len(labels)), 1e3)])
    y2 = np.concatenate([pairs[col].to_numpy(float) * w, [0.0]])
    v, *_ = np.linalg.lstsq(A2, y2, rcond=None)
    return {k: float(v[idx[k]]) for k in labels}


def scale_from_matrix(W: np.ndarray, groups: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """:func:`fit_scale` straight from the (n_base x n_label) prediction matrix,
    each label's carriers being the rows in its group."""
    rows = []
    for gi, sel in enumerate(groups):
        if len(sel) == 0:
            continue
        means = W[sel].mean(axis=0)
        rows += [{"from": labels[gi], "to": labels[gj], "n": len(sel),
                  "delta": means[gj] - means[gi]}
                 for gj in range(len(labels)) if gj != gi]
    scale = fit_scale(pd.DataFrame(rows), labels, "delta")
    return np.array([scale[k] for k in labels])


def bootstrap_scales(Ws: list[np.ndarray], groups: list[np.ndarray], labels: list[str],
                     n_boot: int, seed: int) -> list[np.ndarray]:
    """Per matrix in ``Ws``, (n_boot x n_label) scales refitted on carriers
    resampled within each group; every matrix sees the same resamples."""
    rng = np.random.default_rng(seed)
    boots = [np.empty((n_boot, len(labels))) for _ in Ws]
    for b in range(n_boot):
        res = [rng.choice(g, len(g), replace=True) if len(g) else g for g in groups]
        for W, boot in zip(Ws, boots):
            boot[b] = scale_from_matrix(W, res, labels)
    return boots


def main() -> None:
    rng = np.random.default_rng(SEED)
    j = pl.load_frame(text=True)
    stripped_all = list(j["text"])
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

    texts = [cc.replace_in_line(stripped_all[r], hits[r][0][0], f"static: {k}")
             for r in base_rows for k in KEYWORDS]
    print(f"[A] encoding {len(texts)} texts", flush=True)
    nb, nk = len(base_rows), len(KEYWORDS)
    A = pl.score_texts(texts).reshape(nb, nk)
    groups = [np.flatnonzero(own == k) for k in KEYWORDS]

    np.savez_compressed(pl.SCRATCH / "c1_pairwise_raw.npz",
                        rows=base_rows, own=own, keywords=np.array(KEYWORDS),
                        sp=A.sp, pr=A.pr, dist=A.dist)

    pair_rows = []
    for gi, k in enumerate(KEYWORDS):
        sel = groups[gi]
        if len(sel) == 0:
            continue
        for gj, kp in enumerate(KEYWORDS):
            if gj == gi:
                continue
            d = A.sp[sel, gj] - A.sp[sel, gi]
            pair_rows.append({
                "from": k, "to": kp, "n": int(len(sel)),
                **pl.mean_ci(d, "delta_sp", n_boot=1500),
                "delta_pr": float((A.pr[sel, gj] - A.pr[sel, gi]).mean()),
                "frac_pos": float((d > 0).mean()),
                "off_manifold": float((A.off[sel, gj] | A.off[sel, gi]).mean()),
            })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(pl.SCRATCH / "c1_pairwise.csv", index=False)

    scale = fit_scale(pairs, KEYWORDS, "delta_sp")
    scale_pr = fit_scale(pairs, KEYWORDS, "delta_pr")

    boot_sp, boot_pr = bootstrap_scales([A.sp, A.pr], groups, KEYWORDS, 600, seed=11)

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
    scale_tab.to_csv(pl.SCRATCH / "c1_scale.csv", index=False)
    print(scale_tab.to_string(index=False), flush=True)

    # ── Design B: line deletion ─────────────────────────────────────────
    D = pl.score_texts([cc.drop_line(stripped_all[r], hits[r][0][0]) for r in base_rows])
    own_gi = np.array([KEYWORDS.index(k) for k in own])
    base_sp = A.sp[np.arange(nb), own_gi]
    base_pr = A.pr[np.arange(nb), own_gi]

    del_rows = []
    for gi, k in enumerate(KEYWORDS):
        sel = groups[gi]
        if len(sel) == 0:
            continue
        del_rows.append(pl.contrast_row(
            base_sp[sel] - D.sp[sel], base_pr[sel] - D.pr[sel], D.off[sel],
            key="premium_sp", pr_key="premium_pr", n_boot=1500, keyword=k))
    dele = pd.DataFrame(del_rows).sort_values("premium_sp", ascending=False)
    dele.to_csv(pl.SCRATCH / "c1_delete.csv", index=False)
    print(dele.to_string(index=False), flush=True)

    rank_a = pd.Series([scale[k] for k in dele["keyword"]]).rank()
    summary = {
        "n_base_A": int(nb), "n_arms_A": int(len(texts)),
        "scale_fit_r2": fit_r2,
        "scale_fit_resid_sd": float(np.sqrt((wn * resid ** 2).sum() / wn.sum())),
        "antisymmetry_mean": float(asym.mean()),
        "antisymmetry_max_abs": float(np.abs(asym).max()),
        "off_manifold_A": float(A.off.mean()),
        "off_manifold_delete": float(D.off.mean()),
        "spearman_A_vs_B": float(rank_a.corr(
            dele["premium_sp"].reset_index(drop=True).rank(), method="spearman")),
        "mean_delete_premium": float(dele["premium_sp"].mean()),
        "scale_sp": {k: float(scale[k]) for k in KEYWORDS},
        "scale_pr": {k: float(scale_pr[k]) for k in KEYWORDS},
    }
    (pl.SCRATCH / "c1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
