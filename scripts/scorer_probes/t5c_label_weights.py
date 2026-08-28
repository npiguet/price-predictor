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
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl  # noqa: E402

TEXT_DIM = 512
N_DECKS = 300
OUT_JSON = pl.SCRATCH / "t5c_results.json"
MATCH_FILE = pl.YDATA / "matches-b07" / "match-outcomes-gen5-vs-gen4-forge.txt"


def assoc_regression():
    df = pd.read_csv(pl.SCRATCH / "t2_card_values.csv")
    cols = ["shrunk_score_play", "shrunk_score_draw",
            "shrunk_played_rate", "shrunk_cast_lift"]
    d = df.dropna(subset=cols + ["v_swap"]).copy()
    d["winnability"] = (d.shrunk_score_play + d.shrunk_score_draw) / 2
    names = ["winnability", "shrunk_played_rate", "shrunk_cast_lift"]
    X = d[names].values
    Xs = (X - X.mean(0)) / X.std(0)
    y = (d.v_swap.values - d.v_swap.mean()) / d.v_swap.std()
    A = np.column_stack([np.ones(len(y)), Xs])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - (y - A @ b).var()
    uniq = {}
    for i, nm in enumerate(names):
        rest = [j for j in range(len(names)) if j != i]
        Ar = np.column_stack([np.ones(len(y)), Xs[:, rest]])
        br, *_ = np.linalg.lstsq(Ar, y, rcond=None)
        uniq[nm] = float(r2 - (1 - (y - Ar @ br).var()))
    out = {"n_cards": int(len(d)), "r2": float(r2),
           "beta": dict(zip(names, map(float, b[1:]))), "unique_r2": uniq}
    print("associational:", json.dumps(out, indent=1))
    return out


def main():
    probe = pl.Probe()
    assoc = assoc_regression()

    # --- labeled text vectors ---
    wr = pl.load_win_rates()
    loc = probe.locator
    X_rows, Y_rows = [], []
    LBL = ["shrunk_score_play", "shrunk_score_draw",
           "shrunk_played_rate", "shrunk_cast_lift"]
    for name, rec in wr.items():
        vals = [rec.get(k) for k in LBL]
        if any(v is None for v in vals):
            continue
        e = loc.load_embedding(name)
        if e is None:
            continue
        X_rows.append(e[:TEXT_DIM])
        Y_rows.append(vals)
    X = np.array(X_rows, dtype=np.float64)
    Y = np.array(Y_rows)
    Y = np.column_stack([(Y[:, 0] + Y[:, 1]) / 2, Y[:, 2], Y[:, 3]])
    axis_names = ["winnability", "played_rate", "cast_lift"]
    print(f"{len(X)} labeled cards for probes", flush=True)

    # --- linear probes -> unit-label-sd steps in raw text space ---
    mu, sd = X.mean(0), X.std(0)
    Xs = (X - mu) / sd
    steps, probe_r2 = {}, {}
    lam = 10.0
    G = Xs.T @ Xs + lam * np.eye(TEXT_DIM)
    for j, nm in enumerate(axis_names):
        y = (Y[:, j] - Y[:, j].mean()) / Y[:, j].std()
        w = np.linalg.solve(G, Xs.T @ y)
        pred = Xs @ w
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

    # --- held-out context decks ---
    decks, seen = [], set()
    with open(MATCH_FILE, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split(";")
            if len(p) < 10:
                continue
            names = [c for c in p[5].split("|")
                     if c.lower() not in pl.BASIC_LAND_NAMES]
            key = "|".join(sorted(names))
            if key in seen:
                continue
            try:
                m = probe.deck_matrix(names)
            except KeyError:
                continue
            seen.add(key)
            decks.append(m)
            if len(decks) >= N_DECKS:
                break
    print(f"{len(decks)} context decks", flush=True)

    # --- central-difference sensitivity ---
    results = {}
    for nm, delta in steps.items():
        deck_means = []
        for lo in range(0, len(decks), 50):
            mats, index = [], []
            for di, m in enumerate(decks[lo:lo + 50]):
                for ci in range(m.shape[0]):
                    for sign in (+1.0, -1.0):
                        mm = m.copy()
                        mm[ci, :TEXT_DIM] += sign * delta
                        mats.append(mm.astype(np.float32))
                        index.append((di, ci, sign))
            s = probe.score_matrices(mats)
            acc = {}
            for (di, ci, sign), v in zip(index, s):
                acc.setdefault((di, ci), {})[sign] = v
            per_deck = {}
            for (di, ci), pair in acc.items():
                per_deck.setdefault(di, []).append((pair[1.0] - pair[-1.0]) / 2)
            deck_means.extend(float(np.mean(v)) for v in per_deck.values())
        per_deck = np.array(deck_means)
        results[nm] = {"mean_dscore_per_sd": float(per_deck.mean()),
                       "se": float(per_deck.std(ddof=1) / np.sqrt(len(per_deck))),
                       "n_decks": int(len(per_deck)),
                       "probe_r2": probe_r2[nm]}
        print(f"{nm:12s} dScore/+1sd = {per_deck.mean():+.4f} "
              f"+/- {results[nm]['se']:.4f}", flush=True)

    OUT_JSON.write_text(json.dumps({"assoc": assoc, "causal": results},
                                   indent=1), encoding="utf-8")
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
