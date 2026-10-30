"""T5c: how heavily does the scorer weight each encoder label axis?

Two measurements, one associational and one causal, written to
``t5c_results.json`` (consumed by ``make_figures.py``).

Associational: regress the T2 swap-in card values (``t2_card_values.csv``) on
the label axes jointly — standardized coefficients and unique R² per axis.
Winnability is the mean of ``score_play`` and ``score_draw`` (r = 0.83).

Causal: for each label axis, the step is the average text-space displacement
that accompanies a one-standard-deviation increase of the label — the
regression of the embedding on the standardized label, the same construction
as a one-sd step along a principal component. Perturb one card at a time in
real held-out decks by ±that step (central difference), score through the
production model, and average. A ridge probe's R² per label is reported as a
readability check but plays no part in the step. Perturbed vectors are
synthetic but represent a typical card-to-card displacement.
"""

import json
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.jsonio import write_json
from _lib.stats import ols
from scorer_probes import probe_lib as pl

N_DECKS = 300
OUT_JSON = pl.SCRATCH / "t5c_results.json"
MATCH_FILE = pl.YDATA / "matches-b07" / "match-outcomes-gen5-vs-gen4-forge.txt"
LABELS = ["shrunk_score_play", "shrunk_score_draw", "shrunk_played_rate", "shrunk_cast_lift"]
RIDGE_LAMBDA = 10.0


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean(0)) / v.std(0)


def assoc_regression():
    df = pd.read_csv(pl.SCRATCH / "t2_card_values.csv")
    d = df.dropna(subset=LABELS + ["v_swap"]).copy()
    d["winnability"] = (d.shrunk_score_play + d.shrunk_score_draw) / 2
    names = ["winnability", "shrunk_played_rate", "shrunk_cast_lift"]
    Xs = zscore(d[names].values)
    y = (d.v_swap.values - d.v_swap.mean()) / d.v_swap.std()

    def fit(cols: list[int]):
        return ols(np.column_stack([np.ones(len(y)), Xs[:, cols]]), y)

    full = fit(list(range(len(names))))
    uniq = {nm: float(full.r2 - fit([j for j in range(len(names)) if j != i]).r2)
            for i, nm in enumerate(names)}
    out = {"n_cards": int(len(d)), "r2": float(full.r2),
           "beta": dict(zip(names, map(float, full.beta[1:]))), "unique_r2": uniq}
    print("associational:", json.dumps(out, indent=1))
    return out


def label_steps(probe: pl.Probe, text_dim: int) -> tuple[dict, dict]:
    """``(steps, probe_r2)``: per label axis and leading PC, the mean raw
    text-space displacement per +1 sd, and the ridge probe's readability R²."""
    X_rows, Y_rows = [], []
    for name, rec in pl.load_win_rates().items():
        vals = [rec.get(k) for k in LABELS]
        if any(v is None for v in vals):
            continue
        e = probe.locator.load_embedding(name)
        if e is None:
            continue
        X_rows.append(e[:text_dim])
        Y_rows.append(vals)
    X = np.array(X_rows, dtype=np.float64)
    Y = np.array(Y_rows)
    Y = np.column_stack([(Y[:, 0] + Y[:, 1]) / 2, Y[:, 2], Y[:, 3]])
    print(f"{len(X)} labeled cards for probes", flush=True)

    mu = X.mean(0)
    Xs = zscore(X)
    steps, probe_r2 = {}, {}
    G = Xs.T @ Xs + RIDGE_LAMBDA * np.eye(text_dim)
    for j, nm in enumerate(["winnability", "played_rate", "cast_lift"]):
        y = zscore(Y[:, j])
        pred = Xs @ np.linalg.solve(G, Xs.T @ y)
        probe_r2[nm] = float(1 - (y - pred).var() / y.var())
        # step = E[(x − μ) · y_std]: the mean raw-space displacement per +1
        # label sd, matching the PC-step construction below
        steps[nm] = ((X - mu) * y[:, None]).mean(0)
        print(f"probe {nm}: readability R2 = {probe_r2[nm]:.3f}, "
              f"step norm = {np.linalg.norm(steps[nm]):.3f}", flush=True)

    pca = np.load(str(pl.SCRATCH / "text_pca_512.npz"))
    comps = pca["components"]
    P = (X - pca["mean"]) @ comps.T
    for pc in (0, 1):
        steps[f"PC{pc + 1}"] = comps[pc] * P[:, pc].std()
        probe_r2[f"PC{pc + 1}"] = None
    return steps, probe_r2


def deck_sensitivity(probe: pl.Probe, decks: list[np.ndarray], delta: np.ndarray,
                     text_dim: int) -> np.ndarray:
    """Per deck, the mean over its cards of the central difference
    ``(s(card + delta) - s(card - delta)) / 2``, one card perturbed at a time."""
    deck_means = []
    for lo in range(0, len(decks), 50):
        mats, index = [], []
        for di, m in enumerate(decks[lo:lo + 50]):
            for ci in range(m.shape[0]):
                for sign in (+1.0, -1.0):
                    mm = m.copy()
                    mm[ci, :text_dim] += sign * delta
                    mats.append(mm.astype(np.float32))
                    index.append((di, ci, sign))
        acc = {}
        for (di, ci, sign), v in zip(index, probe.score_matrices(mats)):
            acc.setdefault((di, ci), {})[sign] = v
        per_deck = {}
        for (di, ci), pair in acc.items():
            per_deck.setdefault(di, []).append((pair[1.0] - pair[-1.0]) / 2)
        deck_means.extend(float(np.mean(v)) for v in per_deck.values())
    return np.array(deck_means)


def main():
    probe = pl.Probe()
    text_dim = probe.d_model - pl.FC
    assoc = assoc_regression()
    steps, probe_r2 = label_steps(probe, text_dim)

    decks = list(islice(pl.sample_match_decks(
        [MATCH_FILE], sides="A", make=lambda d: probe.deck_matrix_or_none(d.cards)), N_DECKS))
    print(f"{len(decks)} context decks", flush=True)

    results = {}
    for nm, delta in steps.items():
        per_deck = deck_sensitivity(probe, decks, delta, text_dim)
        results[nm] = {"mean_dscore_per_sd": float(per_deck.mean()),
                       "se": float(per_deck.std(ddof=1) / np.sqrt(len(per_deck))),
                       "n_decks": int(len(per_deck)),
                       "probe_r2": probe_r2[nm]}
        print(f"{nm:12s} dScore/+1sd = {per_deck.mean():+.4f} "
              f"+/- {results[nm]['se']:.4f}", flush=True)

    write_json(OUT_JSON, {"assoc": assoc, "causal": results}, indent=1)
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
