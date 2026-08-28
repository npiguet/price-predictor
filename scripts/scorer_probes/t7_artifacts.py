"""T7: training-data artifacts of the gen-4 sealed scorer.

Four probes, each asking whether a distinct property of the *training corpus*
(not of Magic) is baked into the scorer's opinions.

A -- Builder fingerprint (H-fingerprint), CPU.
    The corpus mixes Forge-built decks with model-built decks, and the two
    families have different macro signatures (most visibly: the gen builders
    always emit exactly 23 spells, Forge's builders emit ~22 and vary). A
    logistic regression on six macro features recovers which family built a
    deck; if the scorer's score still tracks that recovered family probability
    after controlling for the deck's mean card quality, the scorer is partly
    scoring the *builder's signature* rather than the cards.

B -- Land-count opinion (H-landcount), GPU.
    The scorer only ever sees nonbasic cards, so a deck's basic-land count is
    invisible to it and "how many spells does it want?" IS its land-count
    opinion. For real gen-4 pool/deck pairs we build a 22-spell, the actual
    23-spell, and a 24-spell version of each deck and ask which it prefers,
    overall and as a function of the deck's curve.

C -- Per-set blind spots + score->win-prob calibration (H-perset), GPU.
    Held-out gen5-vs-gen4/forge matches: pairwise accuracy sliced by set, and
    a decile calibration table mapping score margin to empirical win rate.

D -- Sibling-checkpoint agreement (H-swap-resolution), GPU, plus two
    card-level joins against Forge's own annotations (forge_hints.csv):
    "borrowed human taste" (draft rankings vs win-rate labels) and "blacklist
    inheritance" (AI:RemoveDeck cards vs matched controls).
    Three checkpoints trained on the same data with different bodies/weighting
    are asked to rank the same 20 single-card swaps per deck. High agreement on
    whole decks but low agreement on swaps means the greedy builder's
    single-swap decisions are model noise, not signal.

Read-only w.r.t. the repo and the Y: drive. `--smoke` runs everything tiny on
CPU. Results land in t7_results.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import probe_lib as pl

SEED = 42

T0_CSV = pl.SCRATCH / "t0_decks.csv"
HINTS_CSV = pl.SCRATCH / "forge_hints.csv"
POOLS = pl.YDATA / "pools" / "pools-gen4-256.txt"
GEN_DECKS = pl.YDATA / "decks" / "generated-decks-gen4-256.txt"
MATCH_FILE = pl.YDATA / "matches-b07" / "match-outcomes-gen5-vs-gen4-forge.txt"
EXP_DIR = Path(r"Y:\Nicolas\mtg\mtg-models-data\sealed\trained-models"
               r"\gen4\scorer\experiments")
SIBLINGS = {
    "prod_mwlog_ff2176": pl.SCORER_CKPT,
    "unw_small_ff1088": EXP_DIR / "512-best_l6_h4_s4_ff1088_mlp256_lr1e-05.pt",
    "unw_big_ff2176": EXP_DIR / "512-best_l6_h4_s4_ff2176_mlp512_lr1e-05.pt",
}
DEFAULT_OUT = pl.SCRATCH / "t7_results.json"

FORGE_FAMILY = ("forge-best", "forge-3sub", "forge-8sub")
GEN_FAMILY = ("gen3-128", "gen3-256", "gen4-256", "gen4-512", "gen5")
FP_FEATURES = ["n_cards", "n_colors", "n_creatures", "avg_mv",
               "primary_pip_share", "n_nonbasic_lands"]


# ============================================================ small numerics

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def rank_avg(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties averaged."""
    order = np.argsort(x, kind="mergesort")
    s = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc_rank(y: np.ndarray, score: np.ndarray) -> float:
    """AUC as the Mann-Whitney rank statistic (ties count 0.5)."""
    y = np.asarray(y, dtype=float)
    n1, n0 = float(y.sum()), float((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rank_avg(np.asarray(score, dtype=float))
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def logistic_irls(X: np.ndarray, y: np.ndarray, lam: float = 1e-3,
                  iters: int = 60) -> np.ndarray:
    """Ridge-penalized logistic regression by Newton/IRLS.

    ``X`` must already carry an intercept column at index 0; the intercept is
    left unpenalized. Penalty is ``0.5 * lam * ||w[1:]||^2`` on the summed
    negative log-likelihood.
    """
    n, d = X.shape
    w = np.zeros(d)
    pen = np.full(d, lam)
    pen[0] = 0.0
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (y - p) - pen * w
        wgt = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * wgt[:, None]) + np.diag(pen) + 1e-9 * np.eye(d)
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def resid(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """OLS residual of y on [1, controls...]."""
    A = np.column_stack([np.ones(len(y))] + [c for c in controls.T]) \
        if controls.ndim == 2 else np.column_stack([np.ones(len(y)), controls])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def jsonify(o):
    if isinstance(o, dict):
        return {str(k): jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonify(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if f != f else f
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsonify(o.tolist())
    return o


def fmt(x, nd: int = 4) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if x != x else f"{x:.{nd}f}"


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ====================================================== shared deck plumbing

def ensure_cards(probe: pl.Probe, cache: dict, names) -> bool:
    """Populate ``cache`` with float32 embeddings; False if any is missing."""
    for n in names:
        if n in cache:
            if cache[n] is None:
                return False
            continue
        e = probe.embedding(n)
        cache[n] = None if e is None else e.astype(np.float32)
        if cache[n] is None:
            return False
    return True


def score_name_lists(probe: pl.Probe, cache: dict, lists: list[list[str]],
                     chunk: int = 256) -> np.ndarray:
    """Score decks given as name lists, building matrices chunk by chunk."""
    out = np.empty(len(lists), dtype=np.float64)
    for lo in range(0, len(lists), chunk):
        sub = lists[lo:lo + chunk]
        mats = [np.stack([cache[n] for n in names]) for names in sub]
        out[lo:lo + len(sub)] = probe.score_matrices(mats, batch_size=len(sub))
    return out


def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def load_aligned_pairs(probe: pl.Probe, cache: dict, n_decks: int,
                       label_of, seed: int = SEED) -> tuple[list[dict], Counter]:
    """Sample line-aligned (pool, deck) pairs from the gen4-256 corpus.

    Returns one record per usable pair with the deck split into nonbasic lands
    (never touched) and spells, plus the unused-pool spells that are legal
    additions. Records carry the labels used to pick the best/worst card.
    """
    pools, decks = read_lines(POOLS), read_lines(GEN_DECKS)
    n = min(len(pools), len(decks))
    stats: Counter = Counter(pool_lines=len(pools), deck_lines=len(decks))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    out: list[dict] = []
    for idx in order:
        if len(out) >= n_decks:
            break
        prow, drow = pools[idx], decks[idx]
        if ";" not in prow:
            stats["bad_pool_line"] += 1
            continue
        pset, pcards = prow.split(";", 1)
        dparts = drow.split(";")
        if len(dparts) != 3:
            stats["bad_deck_line"] += 1
            continue
        _, dset, dcards = dparts
        if pset != dset:
            stats["set_mismatch"] += 1
            continue
        pool = pcards.split("|")
        deck = dcards.split("|")

        nonbasic = [c for c in deck if c.lower() not in pl.BASIC_LAND_NAMES]
        pool_cnt = Counter(pool)
        deck_cnt = Counter(nonbasic)
        if any(deck_cnt[k] > pool_cnt.get(k, 0) for k in deck_cnt):
            stats["deck_not_in_pool"] += 1
            continue
        if not ensure_cards(probe, cache, nonbasic):
            stats["deck_card_no_embedding"] += 1
            continue

        lands, spells = [], []
        for c in nonbasic:
            det = pl.det_features(cache[c])
            (lands if det["is_land"] > 0.5 else spells).append(c)
        if len(spells) < 4:
            stats["too_few_spells"] += 1
            continue

        unused = []
        for card, k in pool_cnt.items():
            extra = k - deck_cnt.get(card, 0)
            if extra <= 0 or card.lower() in pl.BASIC_LAND_NAMES:
                continue
            if not ensure_cards(probe, cache, [card]):
                stats["pool_card_no_embedding"] += 1
                continue
            if pl.det_features(cache[card])["is_land"] > 0.5:
                continue                       # only nonland cards may move
            unused.extend([card] * extra)
        if not unused:
            stats["no_unused_spell"] += 1
            continue

        mvs = [float(pl.det_features(cache[c])["mv"]) for c in spells]
        out.append({
            "line": int(idx),
            "set": pset,
            "lands": lands,
            "spells": spells,
            "unused": unused,
            "avg_spell_mv": float(np.mean(mvs)),
            "spell_labels": [label_of(c) for c in spells],
            "unused_labels": [label_of(c) for c in unused],
        })
    stats["pairs_used"] = len(out)
    return out, stats


# ================================================================= probe A

def load_t0(limit: int | None = None) -> list[dict]:
    """t0_decks.csv rows; a seeded random subsample when limited.

    The file is ordered by source match file, so the first N rows are all one
    era's builders -- limiting has to sample, not truncate.
    """
    with open(T0_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit and limit < len(rows):
        rng = np.random.default_rng([SEED, 0])
        pick = rng.choice(len(rows), size=limit, replace=False)
        rows = [rows[i] for i in sorted(pick)]
    return rows


def probe_a(rows: list[dict]) -> dict:
    header("A -- BUILDER FINGERPRINT (H-fingerprint)")

    # ---- deck-size signature, all methods
    per_method: dict[str, dict] = {}
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    print("\n### Nonbasic card / spell counts by builder method\n")
    print("| method | n | n_cards mean | n_cards mode | share n_cards<=22 | "
          "n_spells mean | n_spells mode | share n_spells<=22 |")
    print("|---|---|---|---|---|---|---|---|")
    for m in sorted(by_method):
        sub = by_method[m]
        nc = np.array([int(r["n_cards"]) for r in sub])
        ns = np.array([int(r["n_spells"]) for r in sub])
        rec = {
            "n": len(sub),
            "n_cards_mean": float(nc.mean()),
            "n_cards_mode": int(Counter(nc.tolist()).most_common(1)[0][0]),
            "share_cards_le22": float((nc <= 22).mean()),
            "n_spells_mean": float(ns.mean()),
            "n_spells_mode": int(Counter(ns.tolist()).most_common(1)[0][0]),
            "share_spells_le22": float((ns <= 22).mean()),
            "n_spells_hist": {int(k): int(v) for k, v in
                              sorted(Counter(ns.tolist()).items())[:40]},
        }
        per_method[m] = rec
        print(f"| {m} | {rec['n']} | {rec['n_cards_mean']:.2f} | "
              f"{rec['n_cards_mode']} | {100 * rec['share_cards_le22']:.1f}% | "
              f"{rec['n_spells_mean']:.2f} | {rec['n_spells_mode']} | "
              f"{100 * rec['share_spells_le22']:.1f}% |")
    print("\n(n_cards counts NONBASIC cards only, so basics are excluded from "
          "both columns; n_spells strips nonbasic lands too.)")

    # ---- fingerprint classifier on the two families
    used = [r for r in rows
            if r["method"] in FORGE_FAMILY or r["method"] in GEN_FAMILY]
    cols = FP_FEATURES + ["score", "mean_score_play"]
    keep = []
    for r in used:
        try:
            vals = [float(r[c]) for c in cols]
        except (ValueError, KeyError):
            continue
        if any(v != v for v in vals):
            continue
        keep.append(r)
    if len(keep) < 50:
        print("\n(too few complete rows for the fingerprint fit)")
        return {"per_method": per_method, "n_rows": len(keep)}

    X_raw = np.array([[float(r[c]) for c in FP_FEATURES] for r in keep])
    y = np.array([1.0 if r["method"] in GEN_FAMILY else 0.0 for r in keep])
    score = np.array([float(r["score"]) for r in keep])
    msp = np.array([float(r["mean_score_play"]) for r in keep])
    n_creat = np.array([float(r["n_creatures"]) for r in keep])
    n_col = np.array([float(r["n_colors"]) for r in keep])

    C2 = np.column_stack([msp, n_creat, n_col])
    r_score_msp = pearson(score, msp)

    def fit(feats: list[str], tag: str) -> dict:
        Xr = np.array([[float(r[c]) for c in feats] for r in keep])
        mu, sd = Xr.mean(0), Xr.std(0)
        sd[sd == 0] = 1.0
        X = np.column_stack([np.ones(len(Xr)), (Xr - mu) / sd])
        w = logistic_irls(X, y, lam=1e-3)
        p_gen = sigmoid(X @ w)
        auc = auc_rank(y, p_gen)
        sep = float(np.mean((p_gen > 0.01) & (p_gen < 0.99)))

        print(f"\n### Fingerprint classifier [{tag}]  (y=1 is the gen family)\n")
        print(f"rows: {len(keep)}  gen={int(y.sum())}  "
              f"forge={int((1 - y).sum())}")
        print(f"features: {', '.join(feats)} (standardized, L2 lambda=1e-3)")
        print(f"AUC(P_hat(gen) vs family) = {fmt(auc)}   "
              f"accuracy@0.5 = {fmt(((p_gen > 0.5) == (y > 0.5)).mean())}   "
              f"share of P_hat strictly inside (0.01, 0.99) = {fmt(sep, 3)}")
        print("\n| feature | standardized coef |")
        print("|---|---|")
        print(f"| (intercept) | {w[0]:+.4f} |")
        for f_, c_ in zip(feats, w[1:]):
            print(f"| {f_} | {c_:+.4f} |")

        r_raw = pearson(score, p_gen)
        rs_raw = spearman(score, p_gen)
        r1 = pearson(resid(score, msp), resid(p_gen, msp))
        r2 = pearson(resid(score, C2), resid(p_gen, C2))
        print("\n| relation | pearson |")
        print("|---|---|")
        print(f"| score vs P_hat(gen), raw | {fmt(r_raw)} |")
        print(f"| score vs P_hat(gen), spearman | {fmt(rs_raw)} |")
        print(f"| score vs P_hat(gen), partial \\| mean_score_play | {fmt(r1)} |")
        print("| score vs P_hat(gen), partial \\| mean_score_play + "
              f"n_creatures + n_colors | {fmt(r2)} |")
        return {
            "features": feats, "auc_gen": auc,
            "acc_at_half": float(((p_gen > 0.5) == (y > 0.5)).mean()),
            "share_unsaturated": sep,
            "coef_standardized": dict(zip(["intercept"] + feats, w.tolist())),
            "feature_mean": dict(zip(feats, mu.tolist())),
            "feature_std": dict(zip(feats, sd.tolist())),
            "pearson_score_phat": r_raw, "spearman_score_phat": rs_raw,
            "partial_corr_ctrl_msp": r1,
            "partial_corr_ctrl_msp_creat_colors": r2,
        }

    print("\n### Score vs recovered builder identity")
    print(f"\nreference: pearson(score, mean_score_play) = {fmt(r_score_msp)}")
    main_fit = fit(FP_FEATURES, "specified 6 features")

    # The two size features encode n_spells (= n_cards - n_nonbasic_lands),
    # which is exactly 23 for every gen deck and never 23 for a Forge deck, so
    # the specified fit is linearly separable and P_hat saturates at 0/1. The
    # size-free refit asks whether the families are still distinguishable --
    # and whether the score still tracks them -- once that giveaway is gone.
    shape_feats = [f for f in FP_FEATURES
                   if f not in ("n_cards", "n_nonbasic_lands")]
    shape_fit = fit(shape_feats, "size-free control")

    # Direct reference: correlation with the true family label.
    r_lab = pearson(score, y)
    r_lab1 = pearson(resid(score, msp), resid(y, msp))
    r_lab2 = pearson(resid(score, C2), resid(y, C2))
    print("\n| relation (true family label, not P_hat) | pearson |")
    print("|---|---|")
    print(f"| score vs is_gen_family, raw | {fmt(r_lab)} |")
    print(f"| score vs is_gen_family, partial \\| mean_score_play | {fmt(r_lab1)} |")
    print("| score vs is_gen_family, partial \\| mean_score_play + "
          f"n_creatures + n_colors | {fmt(r_lab2)} |")

    if main_fit["share_unsaturated"] < 0.05:
        print("\nNOTE: the specified feature set separates the two families "
              "perfectly (AUC "
              f"{fmt(main_fit['auc_gen'], 3)}), because n_cards minus "
              "n_nonbasic_lands is the spell count and every gen deck has "
              "exactly 23 spells\nwhile no Forge deck does. P_hat is therefore "
              "a saturated copy of the family label,\nand its partial "
              "correlation with score equals the label's. The size-free "
              "control\nfit is the informative one: it measures how much of "
              "the builder signature survives\nin deck SHAPE alone.")
    print("\nA large partial correlation means the scorer separates the two "
          "builder families\nby something other than the average empirical "
          "quality of the cards they picked --\ni.e. it has learned the "
          "builders' signatures, which are corpus artifacts.")

    return {
        "per_method": per_method,
        "n_rows": len(keep),
        "n_gen": int(y.sum()),
        "n_forge": int((1 - y).sum()),
        "pearson_score_mean_score_play": r_score_msp,
        "fit_specified": main_fit,
        "fit_size_free": shape_fit,
        "true_label": {"pearson_raw": r_lab,
                       "partial_ctrl_msp": r_lab1,
                       "partial_ctrl_msp_creat_colors": r_lab2},
    }


# ================================================================= probe B

def probe_b(probe: pl.Probe, cache: dict, pairs: list[dict],
            chunk: int) -> dict:
    header("B -- LAND-COUNT OPINION (H-landcount)")
    if not pairs:
        print("no usable pairs")
        return {}

    rng = np.random.default_rng([SEED, 1])
    variants = ["v22", "v23", "v24", "v22_rand", "v24_rand"]
    lists: list[list[str]] = []
    meta: list[tuple[int, str]] = []
    for i, rec in enumerate(pairs):
        spells, lands = rec["spells"], rec["lands"]
        lab = np.array(rec["spell_labels"], dtype=float)
        ulab = np.array(rec["unused_labels"], dtype=float)
        worst = int(np.argmin(lab))
        best = int(np.argmax(ulab))
        r_drop = int(rng.integers(len(spells)))
        r_add = int(rng.integers(len(rec["unused"])))
        built = {
            "v22": lands + [c for k, c in enumerate(spells) if k != worst],
            "v23": lands + spells,
            "v24": lands + spells + [rec["unused"][best]],
            "v22_rand": lands + [c for k, c in enumerate(spells) if k != r_drop],
            "v24_rand": lands + spells + [rec["unused"][r_add]],
        }
        rec["dropped_worst"] = spells[worst]
        rec["added_best"] = rec["unused"][best]
        for v in variants:
            lists.append(built[v])
            meta.append((i, v))

    print(f"scoring {len(lists)} variants over {len(pairs)} decks "
          f"({len(variants)} per deck) ...")
    s = score_name_lists(probe, cache, lists, chunk)
    S = {v: np.array([s[k] for k, (_, vv) in enumerate(meta) if vv == v])
         for v in variants}

    d22 = S["v22"] - S["v23"]
    d24 = S["v24"] - S["v23"]
    d22r = S["v22_rand"] - S["v23"]
    d24r = S["v24_rand"] - S["v23"]
    stack = np.column_stack([S["v22"], S["v23"], S["v24"]])
    pref = np.argmax(stack, axis=1)              # 0->22, 1->23, 2->24
    labels = ["22 spells", "23 spells (as built)", "24 spells"]

    print("\n### Which spell count does the scorer prefer?\n")
    print("| variant | share preferred | n |")
    print("|---|---|---|")
    for k, lb in enumerate(labels):
        print(f"| {lb} | {100 * (pref == k).mean():.1f}% | "
              f"{int((pref == k).sum())} |")

    print("\n### Score deltas vs the as-built 23-spell deck\n")
    print("| change | mean dScore | median | std | share > 0 |")
    print("|---|---|---|---|---|")
    rows = [("drop worst-label spell -> 22", d22),
            ("add best-label spell   -> 24", d24),
            ("drop RANDOM spell      -> 22 (control)", d22r),
            ("add RANDOM spell       -> 24 (control)", d24r)]
    for name, d in rows:
        print(f"| {name} | {d.mean():+.4f} | {np.median(d):+.4f} | "
              f"{d.std():.4f} | {100 * (d > 0).mean():.1f}% |")

    # ---- curve conditioning
    mv = np.array([r["avg_spell_mv"] for r in pairs])
    nb = min(5, max(2, len(pairs) // 8))
    edges = np.unique(np.quantile(mv, np.linspace(0, 1, nb + 1)))
    bin_idx = np.clip(np.digitize(mv, edges[1:-1]), 0, len(edges) - 2)
    bins = []
    print("\n### Delta by deck curve (avg nonland MV bins)\n")
    print("| MV bin | n | mean MV | mean d(24-23) | mean d(22-23) | "
          "mean d(24r-23) | pref 22/23/24 |")
    print("|---|---|---|---|---|---|---|")
    for b in range(len(edges) - 1):
        sel = bin_idx == b
        if not sel.any():
            continue
        p = [float((pref[sel] == k).mean()) for k in range(3)]
        rec = {
            "lo": float(edges[b]), "hi": float(edges[b + 1]),
            "n": int(sel.sum()), "mean_mv": float(mv[sel].mean()),
            "mean_d24": float(d24[sel].mean()),
            "mean_d22": float(d22[sel].mean()),
            "mean_d24_rand": float(d24r[sel].mean()),
            "mean_d22_rand": float(d22r[sel].mean()),
            "pref_shares": p,
        }
        bins.append(rec)
        print(f"| [{edges[b]:.2f}, {edges[b + 1]:.2f}] | {rec['n']} | "
              f"{rec['mean_mv']:.2f} | {rec['mean_d24']:+.4f} | "
              f"{rec['mean_d22']:+.4f} | {rec['mean_d24_rand']:+.4f} | "
              f"{100 * p[0]:.0f}/{100 * p[1]:.0f}/{100 * p[2]:.0f} |")

    print("\n### Count effect vs card-quality effect\n")
    print("The random-card variants change the spell COUNT without choosing a "
          "good/bad card,\nso they isolate the scorer's raw count preference; "
          "the residual is what picking\nthe worst/best card by empirical "
          "label buys.\n")
    print("| step | count effect (random card) | quality effect (greedy - random) |")
    print("|---|---|---|")
    print(f"| 23 -> 22 | {d22r.mean():+.4f} | {(d22 - d22r).mean():+.4f} |")
    print(f"| 23 -> 24 | {d24r.mean():+.4f} | {(d24 - d24r).mean():+.4f} |")

    slope = pearson(mv, d24)
    print(f"\npearson(avg nonland MV, d(24-23)) = {fmt(slope)}  "
          "(negative = high-curve decks want fewer spells / more lands)")
    if (d24 > 0).mean() > 0.95:
        print("The scorer wants a 24th spell in essentially every deck: its "
              "land-count opinion\nis unconditional, not curve-aware -- adding "
              "any playable body beats the 17th land.")
    elif (d24 > 0).mean() < 0.05:
        print("The scorer never wants a 24th spell: it is unconditionally "
              "pro-land at this size.")

    return {
        "n_decks": len(pairs),
        "pref_shares": {labels[k]: float((pref == k).mean()) for k in range(3)},
        "pref_counts": {labels[k]: int((pref == k).sum()) for k in range(3)},
        "deltas": {
            name: {"mean": float(d.mean()), "median": float(np.median(d)),
                   "std": float(d.std()), "share_gt0": float((d > 0).mean())}
            for name, d in [("d22", d22), ("d24", d24),
                            ("d22_rand", d22r), ("d24_rand", d24r)]},
        "score_means": {v: float(S[v].mean()) for v in variants},
        "mv_bins": bins,
        "pearson_mv_d24": slope,
        "pearson_mv_d22": pearson(mv, d22),
        "count_effect": {"d22_rand": float(d22r.mean()),
                         "d24_rand": float(d24r.mean())},
        "quality_effect": {"d22_minus_rand": float((d22 - d22r).mean()),
                           "d24_minus_rand": float((d24 - d24r).mean())},
    }


# ================================================================= probe C

def parse_matches(path: Path, limit: int | None = None):
    """Yield (set, method_A, deck_A, method_B, deck_B, winner) per match line.

    Winner = the side with more game wins in field 7 (a string like "ABAAA").
    """
    rows, undecided, malformed = [], 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split(";")
            if len(p) < 10:
                malformed += 1
                continue
            a, b = p[7].count("A"), p[7].count("B")
            if a == b:
                undecided += 1
                continue
            rows.append((p[2], p[3], p[5].split("|"), p[4], p[6].split("|"),
                         "A" if a > b else "B"))
            if limit and len(rows) >= limit:
                break
    return rows, undecided, malformed


def probe_c(probe: pl.Probe, cache: dict, limit: int | None,
            chunk: int, min_n: int = 30) -> dict:
    header("C -- PER-SET BLIND SPOTS + SCORE CALIBRATION (H-perset)")
    raw, undecided, malformed = parse_matches(MATCH_FILE, limit)
    print(f"parsed {len(raw)} decided matches "
          f"({undecided} undecided, {malformed} malformed) from "
          f"{MATCH_FILE.name}")

    rng = np.random.default_rng([SEED, 2])
    index: dict[tuple, int] = {}
    decks: list[list[str]] = []
    recs, dropped, flipped = [], 0, 0

    def add(deck: list[str]) -> int | None:
        nb = tuple(sorted(c for c in deck
                          if c.lower() not in pl.BASIC_LAND_NAMES))
        if not nb:
            return None
        hit = index.get(nb)
        if hit is not None:
            return hit
        if not ensure_cards(probe, cache, nb):
            return None
        index[nb] = len(decks)
        decks.append(list(nb))
        return len(decks) - 1

    for set_code, m_a, deck_a, m_b, deck_b, winner in raw:
        # Side A is systematically the newer builder in this corpus; a seeded
        # coin flip makes the calibration table symmetric around delta=0.
        if rng.random() < 0.5:
            m_a, deck_a, m_b, deck_b = m_b, deck_b, m_a, deck_a
            winner = "B" if winner == "A" else "A"
            flipped += 1
        ia, ib = add(deck_a), add(deck_b)
        if ia is None or ib is None:
            dropped += 1
            continue
        recs.append((set_code, " vs ".join(sorted((m_a, m_b))), ia, ib,
                     1.0 if winner == "A" else 0.0))
    print(f"kept {len(recs)} matches ({dropped} dropped for missing "
          f"embeddings, {flipped} orientation-flipped); "
          f"{len(decks)} distinct decks, {len(cache)} cards cached")
    if not recs:
        return {}

    print(f"scoring {len(decks)} distinct decks ...")
    s = score_name_lists(probe, cache, decks, chunk)
    delta = np.array([s[ia] - s[ib] for _, _, ia, ib, _ in recs])
    y = np.array([w for _, _, _, _, w in recs])
    sets = [r[0] for r in recs]
    pairs = [r[1] for r in recs]
    correct = np.where(delta > 0, y, np.where(delta < 0, 1 - y, 0.5))

    print(f"\noverall pairwise accuracy = {fmt(correct.mean())} "
          f"(n={len(correct)}, ties {100 * (delta == 0).mean():.1f}%)")

    # ---- per set
    by_set: dict[str, list[int]] = defaultdict(list)
    for i, sc in enumerate(sets):
        by_set[sc].append(i)
    per_set = {sc: {"n": len(ix), "acc": float(correct[ix].mean()),
                    "mean_abs_delta": float(np.abs(delta[ix]).mean())}
               for sc, ix in by_set.items()}
    elig = {k: v for k, v in per_set.items() if v["n"] >= min_n}
    order = sorted(elig, key=lambda k: elig[k]["acc"])
    print(f"\n### Accuracy by set (n >= {min_n}; {len(elig)} of "
          f"{len(per_set)} sets qualify)\n")
    print("| rank | worst set | n | acc | | best set | n | acc |")
    print("|---|---|---|---|---|---|---|---|")
    for i in range(min(8, len(order))):
        lo = order[i]
        hi = order[-(i + 1)] if len(order) > i else None
        left = f"| {i + 1} | {lo} | {elig[lo]['n']} | {fmt(elig[lo]['acc'], 3)} |"
        right = (f" | {hi} | {elig[hi]['n']} | {fmt(elig[hi]['acc'], 3)} |"
                 if hi else " | | | |")
        print(left + right)
    spread = {}
    if elig:
        accs = np.array([v["acc"] for v in elig.values()])
        ns = np.array([v["n"] for v in elig.values()], dtype=float)
        p_bar = float((accs * ns).sum() / ns.sum())
        sd_null = float(np.sqrt(p_bar * (1 - p_bar) * np.mean(1.0 / ns)))
        sd_obs = float(accs.std())
        excess = float(np.sqrt(max(sd_obs ** 2 - sd_null ** 2, 0.0)))
        spread = {"n_sets": len(accs), "min": float(accs.min()),
                  "median": float(np.median(accs)), "max": float(accs.max()),
                  "sd_observed": sd_obs, "sd_under_null": sd_null,
                  "sd_excess": excess, "weighted_acc": p_bar,
                  "se_at_min_n": float(np.sqrt(0.25 / min_n))}
        print(f"\nspread across qualifying sets: min {accs.min():.3f}  "
              f"median {np.median(accs):.3f}  max {accs.max():.3f}  "
              f"sd {sd_obs:.3f}")
        print(f"sd expected from sampling noise alone = {sd_null:.3f} "
              f"(binomial at these n) -> excess sd {excess:.3f}")
        print(f"a single set at n={min_n} carries +/-"
              f"{2 * np.sqrt(0.25 / min_n):.3f} at 2 SE, so the worst/best "
              "lists above are\nmostly noise unless the excess sd is "
              "comparable to the observed sd.")

    # ---- method-pair x set cells
    by_cell: dict[tuple, list[int]] = defaultdict(list)
    for i, (sc, pr) in enumerate(zip(sets, pairs)):
        by_cell[(pr, sc)].append(i)
    cells = sorted(by_cell.items(), key=lambda kv: -len(kv[1]))[:12]
    per_cell = []
    print("\n### Largest method-pair x set cells\n")
    print("| method pair | set | n | acc |")
    print("|---|---|---|---|")
    for (pr, sc), ix in cells:
        acc = float(correct[ix].mean())
        per_cell.append({"pair": pr, "set": sc, "n": len(ix), "acc": acc})
        print(f"| {pr} | {sc} | {len(ix)} | {fmt(acc, 3)} |")

    per_pair = {}
    for pr in sorted(set(pairs)):
        ix = [i for i, p in enumerate(pairs) if p == pr]
        per_pair[pr] = {"n": len(ix), "acc": float(correct[ix].mean())}

    # ---- calibration
    a, b = logistic_irls(np.column_stack([np.ones(len(delta)), delta]), y,
                         lam=1e-6)
    p_fit = sigmoid(a + b * delta)
    p_raw = sigmoid(delta)
    nb = min(10, max(2, len(delta) // 10))
    edges = np.unique(np.quantile(delta, np.linspace(0, 1, nb + 1)))
    bidx = np.clip(np.digitize(delta, edges[1:-1]), 0, len(edges) - 2)
    calib = []
    print("\n### Score margin -> win probability (deciles of "
          "delta = s(A) - s(B))\n")
    print("| bin | delta range | n | mean delta | empirical P(A wins) | "
          "fitted sigmoid(a+b*delta) | raw sigmoid(delta) |")
    print("|---|---|---|---|---|---|---|")
    for k in range(len(edges) - 1):
        sel = bidx == k
        if not sel.any():
            continue
        rec = {"bin": k + 1, "lo": float(edges[k]), "hi": float(edges[k + 1]),
               "n": int(sel.sum()), "mean_delta": float(delta[sel].mean()),
               "emp_p": float(y[sel].mean()),
               "fit_p": float(p_fit[sel].mean()),
               "raw_sigmoid_p": float(p_raw[sel].mean())}
        calib.append(rec)
        print(f"| {k + 1} | [{edges[k]:+.2f}, {edges[k + 1]:+.2f}] | "
              f"{rec['n']} | {rec['mean_delta']:+.3f} | {rec['emp_p']:.3f} | "
              f"{rec['fit_p']:.3f} | {rec['raw_sigmoid_p']:.3f} |")

    eps = 1e-12
    ll = float(np.mean(y * np.log(p_fit + eps) + (1 - y) * np.log(1 - p_fit + eps)))
    ll_raw = float(np.mean(y * np.log(p_raw + eps) + (1 - y) * np.log(1 - p_raw + eps)))
    print(f"\nfitted link: P(A wins) = sigmoid({a:+.4f} + {b:.4f} * delta)  "
          f"-> 1 score point = {100 * (sigmoid(b) - 0.5):+.1f}pp at the "
          "midpoint")
    print(f"mean log-loss  fitted {-ll:.4f}   raw sigmoid(delta) {-ll_raw:.4f} "
          f"  coin flip {0.6931:.4f}")
    print(f"brier          fitted {np.mean((p_fit - y) ** 2):.4f}   "
          f"raw {np.mean((p_raw - y) ** 2):.4f}")

    return {
        "match_file": str(MATCH_FILE),
        "n_matches": len(recs), "n_dropped": dropped,
        "n_undecided": undecided, "n_distinct_decks": len(decks),
        "overall_acc": float(correct.mean()),
        "tie_rate": float((delta == 0).mean()),
        "min_n_per_set": min_n,
        "per_set": per_set,
        "set_spread": spread,
        "worst_sets": [{"set": k, **elig[k]} for k in order[:8]],
        "best_sets": [{"set": k, **elig[k]} for k in order[::-1][:8]],
        "per_pair": per_pair,
        "biggest_cells": per_cell,
        "calibration": calib,
        "platt": {"intercept": float(a), "slope": float(b),
                  "logloss_fitted": -ll, "logloss_raw_sigmoid": -ll_raw,
                  "brier_fitted": float(np.mean((p_fit - y) ** 2)),
                  "brier_raw": float(np.mean((p_raw - y) ** 2))},
        "delta_std": float(delta.std()),
    }


# ================================================================= probe D

def probe_d_swaps(probes: dict[str, pl.Probe], cache: dict, pairs: list[dict],
                  n_swaps: int, chunk: int) -> dict:
    header("D1 -- SIBLING-CHECKPOINT AGREEMENT (H-swap-resolution)")
    if not pairs:
        print("no usable decks")
        return {}

    lists: list[list[str]] = []
    per_deck: list[dict] = []
    for rec in pairs:
        rng = np.random.default_rng([SEED, 3, rec["line"]])
        base = rec["lands"] + rec["spells"]
        base_i = len(lists)
        lists.append(base)
        idxs = []
        for _ in range(n_swaps):
            out_k = int(rng.integers(len(rec["spells"])))
            in_k = int(rng.integers(len(rec["unused"])))
            spells = list(rec["spells"])
            spells[out_k] = rec["unused"][in_k]
            idxs.append(len(lists))
            lists.append(rec["lands"] + spells)
        per_deck.append({"base": base_i, "swaps": idxs})

    scores: dict[str, np.ndarray] = {}
    for name, p in probes.items():
        print(f"scoring {len(lists)} decks with {name} ...")
        scores[name] = score_name_lists(p, cache, lists, chunk)

    names = list(probes)
    base_scores = {n: np.array([s[d["base"]] for d in per_deck])
                   for n, s in scores.items()}
    deltas = {n: np.array([[s[i] - s[d["base"]] for i in d["swaps"]]
                           for d in per_deck]) for n, s in scores.items()}

    print("\n### Per-model swap-delta scale\n")
    print("| model | base score mean | base sd | mean |dScore| per swap | "
          "share of swaps improving |")
    print("|---|---|---|---|---|")
    for n in names:
        d = deltas[n]
        print(f"| {n} | {base_scores[n].mean():+.3f} | "
              f"{base_scores[n].std():.3f} | {np.abs(d).mean():.4f} | "
              f"{100 * (d > 0).mean():.1f}% |")

    out_pairs = []
    print("\n### Between-model agreement\n")
    print("| model pair | deck-level spearman (base scores) | "
          "mean per-deck swap spearman | median | share rho<0 | "
          "delta sign agreement | top-1 swap agreement |")
    print("|---|---|---|---|---|---|---|")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            deck_rho = spearman(base_scores[a], base_scores[b])
            rhos = np.array([spearman(deltas[a][k], deltas[b][k])
                             for k in range(len(per_deck))], dtype=float)
            ok = rhos[~np.isnan(rhos)]
            sign = float(np.mean(np.sign(deltas[a]) == np.sign(deltas[b])))
            top1 = float(np.mean([np.argmax(deltas[a][k]) == np.argmax(deltas[b][k])
                                  for k in range(len(per_deck))]))
            rec = {"pair": f"{a} vs {b}", "deck_spearman": deck_rho,
                   "swap_spearman_mean": float(ok.mean()) if len(ok) else float("nan"),
                   "swap_spearman_median": float(np.median(ok)) if len(ok) else float("nan"),
                   "swap_spearman_share_negative": float((ok < 0).mean()) if len(ok) else float("nan"),
                   "n_decks_with_rho": int(len(ok)),
                   "delta_sign_agreement": sign,
                   "top1_swap_agreement": top1}
            out_pairs.append(rec)
            print(f"| {a} vs {b} | {fmt(deck_rho, 3)} | "
                  f"{fmt(rec['swap_spearman_mean'], 3)} | "
                  f"{fmt(rec['swap_spearman_median'], 3)} | "
                  f"{100 * rec['swap_spearman_share_negative']:.0f}% | "
                  f"{fmt(sign, 3)} | {fmt(top1, 3)} |")
    print(f"\nchance level for top-1 swap agreement = "
          f"{1.0 / max(n_swaps, 1):.3f}; for sign agreement ~0.5 when the "
          "improve-rate is balanced.")
    print("Deck-level agreement >> swap-level agreement means the checkpoints "
          "rank whole decks\nthe same way but disagree about which single card "
          "swap is best -- so a greedy builder's\nindividual picks are riding "
          "on checkpoint noise.")

    return {
        "n_decks": len(per_deck), "n_swaps_per_deck": n_swaps,
        "models": {n: str(SIBLINGS[n]) for n in names},
        "per_model": {n: {"base_mean": float(base_scores[n].mean()),
                          "base_sd": float(base_scores[n].std()),
                          "mean_abs_delta": float(np.abs(deltas[n]).mean()),
                          "share_improving": float((deltas[n] > 0).mean())}
                      for n in names},
        "pairs": out_pairs,
    }


def load_hints() -> dict[str, dict]:
    if not HINTS_CSV.exists():
        return {}
    with open(HINTS_CSV, encoding="utf-8") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def probe_d_joins(wr: dict[str, dict]) -> dict:
    header("D2 -- FORGE ANNOTATIONS vs WIN-RATE LABELS")
    hints = load_hints()
    if not hints:
        print(f"missing {HINTS_CSV} -- run forge_hints.py first")
        return {}
    print(f"forge_hints rows {len(hints)}; win-rate rows {len(wr)}; "
          f"name overlap {len(set(hints) & set(wr))}")

    out: dict = {"n_hints": len(hints), "n_win_rates": len(wr),
                 "n_overlap": len(set(hints) & set(wr))}

    # ---- borrowed human taste: draft rank vs empirical label
    print("\n### Borrowed human taste: Forge draft rank vs win-rate labels\n")
    print("| label | n | spearman(draft_rank, label) | spearman(best rank, label) |")
    print("|---|---|---|---|")
    taste = {}
    for lab in ("shrunk_score_play", "shrunk_played_rate", "shrunk_cast_lift"):
        xs, xb, ys = [], [], []
        for name, h in hints.items():
            r = wr.get(name)
            if r is None or h["draft_rank"] == "":
                continue
            v = r.get(lab)
            if v is None:
                continue
            xs.append(float(h["draft_rank"]))
            xb.append(float(h["draft_rank_best"]))
            ys.append(float(v))
        rho = spearman(xs, ys)
        rho_b = spearman(xb, ys)
        taste[lab] = {"n": len(xs), "spearman_mean_rank": rho,
                      "spearman_best_rank": rho_b}
        print(f"| {lab} | {len(xs)} | {fmt(rho)} | {fmt(rho_b)} |")
    print("\ndraft_rank is normalized position in a set's human pick order, so "
          "LOWER is better;\na negative spearman therefore means human taste "
          "and the empirical labels agree.")
    out["draft_rank_vs_labels"] = taste

    # ---- blacklist inheritance: AI:RemoveDeck cards vs matched controls
    print("\n### Blacklist inheritance: AI:RemoveDeck cards vs matched "
          "controls\n")
    black = {}
    for lab in ("shrunk_score_play", "shrunk_played_rate"):
        rows = []
        for name, h in hints.items():
            r = wr.get(name)
            if r is None or h["mv"] == "":
                continue
            v = r.get(lab)
            if v is None:
                continue
            mv = float(h["mv"])
            rows.append((int(h["ai_remove_deck"]),
                         (min(int(mv), 7), int(h["is_creature"])),
                         float(v)))
        if not rows:
            continue
        rem = [r for r in rows if r[0]]
        oth = [r for r in rows if not r[0]]
        cells_o: dict = defaultdict(list)
        cells_r: dict = defaultdict(list)
        for _, c, v in oth:
            cells_o[c].append(v)
        for _, c, v in rem:
            cells_r[c].append(v)
        # direct standardization: control mean reweighted to the removed
        # cards' (MV bucket x is_creature) composition
        num = den = 0.0
        matched_cells = []
        for c, vs in sorted(cells_r.items()):
            if c not in cells_o:
                continue
            w_ = len(vs)
            num += w_ * float(np.mean(cells_o[c]))
            den += w_
            matched_cells.append({"mv_bucket": c[0], "is_creature": c[1],
                                  "n_removed": len(vs),
                                  "mean_removed": float(np.mean(vs)),
                                  "n_control": len(cells_o[c]),
                                  "mean_control": float(np.mean(cells_o[c]))})
        m_rem = float(np.mean([v for _, _, v in rem])) if rem else float("nan")
        m_oth = float(np.mean([v for _, _, v in oth])) if oth else float("nan")
        m_match = num / den if den else float("nan")
        black[lab] = {"n_removed": len(rem), "n_control": len(oth),
                      "mean_removed": m_rem, "mean_control_raw": m_oth,
                      "mean_control_matched": m_match,
                      "diff_raw": m_rem - m_oth,
                      "diff_matched": m_rem - m_match,
                      "cells": matched_cells}
        print(f"{lab}: removed n={len(rem)} mean={m_rem:+.4f} | "
              f"control n={len(oth)} raw mean={m_oth:+.4f} "
              f"matched mean={m_match:+.4f} | "
              f"diff raw {m_rem - m_oth:+.4f}, matched {m_rem - m_match:+.4f}")

    if black:
        lab = "shrunk_played_rate"
        if lab in black:
            print("\n| MV bucket | creature | n removed | mean removed | "
                  "n control | mean control |")
            print("|---|---|---|---|---|---|")
            for c in black[lab]["cells"]:
                print(f"| {c['mv_bucket']} | {c['is_creature']} | "
                      f"{c['n_removed']} | {c['mean_removed']:+.4f} | "
                      f"{c['n_control']} | {c['mean_control']:+.4f} |")
        print("\nA negative matched diff on shrunk_played_rate is the direct "
              "footprint of Forge's\nhand-written blacklist in the corpus: "
              "those cards were rarely put in decks, so the\nlabels -- and any "
              "model trained on them -- inherit the blacklist.")
    out["ai_remove_deck"] = black
    return out


# ==================================================================== main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="CPU, tiny samples in every section")
    ap.add_argument("--only", default="ABCD",
                    help="subset of sections to run, e.g. 'AC'")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--b-decks", type=int, default=None)
    ap.add_argument("--c-limit", type=int, default=None,
                    help="max match lines for probe C (default: all)")
    ap.add_argument("--d-decks", type=int, default=None)
    ap.add_argument("--d-swaps", type=int, default=None)
    ap.add_argument("--a-rows", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    smoke = args.smoke
    b_decks = args.b_decks if args.b_decks is not None else (12 if smoke else 800)
    d_decks = args.d_decks if args.d_decks is not None else (8 if smoke else 300)
    d_swaps = args.d_swaps if args.d_swaps is not None else (4 if smoke else 20)
    c_limit = args.c_limit if args.c_limit is not None else (300 if smoke else None)
    a_rows = args.a_rows if args.a_rows is not None else (4000 if smoke else None)
    sections = set(args.only.upper())
    device = "cpu" if smoke else None
    chunk = 32 if smoke else args.chunk

    print(f"t7_artifacts: sections={''.join(sorted(sections))} "
          f"smoke={smoke} seed={SEED}")
    results: dict = {"meta": {
        "seed": SEED, "smoke": smoke, "sections": sorted(sections),
        "b_decks": b_decks, "c_limit": c_limit,
        "d_decks": d_decks, "d_swaps": d_swaps,
        "t0_csv": str(T0_CSV), "hints_csv": str(HINTS_CSV),
        "pools": str(POOLS), "decks": str(GEN_DECKS),
        "checkpoints": {k: str(v) for k, v in SIBLINGS.items()},
    }}

    if "A" in sections:
        results["A_builder_fingerprint"] = probe_a(load_t0(a_rows))

    need_probe = bool({"B", "C", "D"} & sections)
    probe = None
    cache: dict = {}
    if need_probe:
        print(f"\nloading production scorer (device={device or 'auto'}) ...")
        probe = pl.Probe(device=device)
        results["meta"]["device"] = probe.device
        print(f"device={probe.device}  d_model={probe.d_model}")

    wr = pl.load_win_rates() if {"B", "D"} & sections else {}
    med = float("nan")
    if wr:
        vals = [r["shrunk_score_play"] for r in wr.values()
                if r.get("shrunk_score_play") is not None]
        med = float(np.median(vals)) if vals else 0.0
        print(f"win-rate labels: {len(wr)} cards, "
              f"median shrunk_score_play {med:+.4f} (fallback for unlabeled)")

    def label_of(name: str) -> float:
        r = wr.get(name)
        v = r.get("shrunk_score_play") if r else None
        return med if v is None else float(v)

    pairs: list[dict] = []
    if {"B", "D"} & sections:
        want = max(b_decks if "B" in sections else 0,
                   d_decks if "D" in sections else 0)
        pairs, pstats = load_aligned_pairs(probe, cache, want, label_of)
        print(f"aligned gen4-256 pairs: kept {len(pairs)} of {want} requested; "
              f"skips {dict((k, v) for k, v in pstats.items() if 'line' not in k and k != 'pairs_used')}")
        results["meta"]["pair_stats"] = dict(pstats)

    if "B" in sections:
        results["B_land_count"] = probe_b(probe, cache, pairs[:b_decks], chunk)

    if "C" in sections:
        results["C_per_set_calibration"] = probe_c(
            probe, cache, c_limit, chunk, min_n=3 if smoke else 30)

    if "D" in sections:
        probes = {"prod_mwlog_ff2176": probe}
        for name, ck in SIBLINGS.items():
            if name in probes:
                continue
            print(f"\nloading sibling scorer {name} ...")
            probes[name] = pl.Probe(device=device, checkpoint=ck)
        results["D_sibling_agreement"] = probe_d_swaps(
            probes, cache, pairs[:d_decks], d_swaps, chunk)
        results["D_forge_annotation_joins"] = probe_d_joins(wr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(jsonify(results), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
