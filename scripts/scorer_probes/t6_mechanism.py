"""T6: mechanism probes of the gen-4 sealed Set-Transformer scorer.

T5 showed *which input block* the scorer reads. T6 asks *how the network turns
that input into a scalar* -- how many text directions it actually uses, whether
the SAB stack collapses card identities, whether magnitude survives pooling,
and where the model's scale is anchored (mean-pooling vs sum-pooling).

Five probes, all inference-time, all read-only w.r.t. the repo and Y::

    P1  PC-truncation of the text block. Project every card's text dims
        [0:512] onto the top-k principal components of the text corpus
        (precomputed in ``text_pca_512.npz``), reconstruct, leave the trailing
        32 deterministic dims intact, and rescore. k=0 replaces the text block
        with the PCA mean (no per-card text at all); k=512 is the identity.
        Reported both as score fidelity vs the full model (400 real decks) and
        as *held-out* match-prediction accuracy (first N matches of a file that
        postdates the scorer's training cutoff).
        => how many text directions the scorer actually reads.

    P2  SAB over-smoothing. Forward hooks on the 6 SAB layers: per deck, mean
        pairwise cosine similarity between the (non-padding) card
        representations at each depth. Rising toward 1.0 = the stack washes
        out per-card identity. Plus PMA pooling diagnostics: per-seed/per-head
        attention entropy (normalized by log n_cards; 1.0 = uniform = pure
        mean pooling) and per-seed attention mass on lands vs spells.

    P3  LayerNorm magnitude stripping. Scale the SAB-stack output (the input to
        PMA) by alpha and rescore. Near-invariance means the representation's
        magnitude is destroyed at the pooling boundary (PMA's LayerNorm) and
        only direction survives.

    P4  Replication invariance. Score D vs D-repeated-k-times, and k identical
        copies of one card for k in {1,5,10,23}. A pure-mean pooler is exactly
        invariant to replication; any dependence on k means the pooling reads
        something size-like.

    P5  OOD envelope. Where do degenerate decks (truncated, all-land,
        23-copies-of-a-bomb) sit relative to the percentile band of 400 real
        tournament decks?

Deck sample: deck_A of the first 700 lines of
``matches-b07/match-outcomes-gen5-vs-gen4-forge.txt``, deduped by sorted
nonbasic content, first 400 with every nonbasic embedding resolvable. Basics
are dropped throughout (they carry no ``.npz``), exactly as ``probe_lib``
does.

Usage::

    python t6_mechanism.py --smoke          # CPU, 40 decks, k subset, ~2 min
    python t6_mechanism.py                  # full run
    python t6_mechanism.py --probes p1,p3   # subset of probes
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.jsonio import write_json
from _lib.markdown import banner, print_table
from _lib.markdown import num as fmt
from _lib.stats import coin_flip_band, pairwise_correct
from scorer_probes import probe_lib as pl
from scorer_probes.probe_lib import DeckBank, Preset

from sealed.domain import card_embedding_layout as layout

MATCH_FILE = pl.YDATA / "matches-b07" / "match-outcomes-gen5-vs-gen4-forge.txt"
PCA_PATH = pl.SCRATCH / "text_pca_512.npz"
DEFAULT_OUT = pl.SCRATCH / "t6_results.json"

SEED = 42
SAMPLE_LINES = 700          # deck_A of the first N match lines feeds the sample
K_GRID = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
K_GRID_SMOKE = [0, 1, 8, 64, 512]
ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0]
REPLICATION_KS = [2, 3]
COPY_KS = [1, 5, 10, 23]
TRUNC_SIZES = [5, 10, 15, 20]
MID_SCORE_RANGE = (0.05, 0.15)   # shrunk_score_play window for the k-copies probe
PERCENTILES = [1, 5, 25, 50, 75, 95, 99]
ALL_PROBES = ["p1", "p2", "p3", "p4", "p5"]


def corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(pearson r, spearman rho); nan when either side is constant."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])


def pct_rank(ref: np.ndarray, x: float) -> float:
    """Percentile of ``x`` within the reference distribution (mid-rank ties)."""
    return 100.0 * float((ref < x).mean() + 0.5 * (ref == x).mean())


class Ctx:
    """What the probes share: the scorer, the bank of distinct decks, batching
    and the rng."""

    def __init__(self, args):
        self.probe = pl.Probe(device=args.device)
        self.model = self.probe.model
        self.device = self.probe.device
        self.d_model = self.probe.d_model
        self.text_dim = self.d_model - pl.FC
        self.batch = args.batch
        self.rng = np.random.default_rng(SEED)
        self.bank = DeckBank(self.probe.embedding)

    def mats(self, which: list[int], transform=None) -> list[np.ndarray]:
        """Raw matrices of bank decks ``which``, each passed through ``transform``."""
        return [self.bank.matrix(i) if transform is None else transform(self.bank.matrix(i))
                for i in which]

    def score(self, mats: list[np.ndarray], per_batch=None) -> np.ndarray:
        """Score raw matrices; ``per_batch(cards, mask)`` runs after each batch
        with its padded inputs, so hook consumers can align their captures."""
        return self.probe.score_matrices(mats, self.batch, on_batch=per_batch)


def build_sample(ctx, n_decks: int) -> tuple[list[int], int, int]:
    """First ``n_decks`` distinct, fully-embeddable deck_A of the first 700 lines.

    Returns (deck indices into ``ctx.bank``, match lines read, decks dropped).
    """
    dropped = 0

    def make(d: pl.MatchDeck):
        nonlocal dropped
        i = ctx.bank.add(d.cards)
        dropped += i is None
        return None if i is None else (d.line, i)

    picked = list(islice(pl.sample_match_decks([MATCH_FILE], sides="A", lines=SAMPLE_LINES,
                                               make=make), n_decks))
    lines_read = picked[-1][0] + 1 if len(picked) == n_decks else SAMPLE_LINES
    return [i for _, i in picked], lines_read, dropped


def truncate_text(emb: np.ndarray, k: int, mean: np.ndarray,
                  comps: np.ndarray, text_dim: int) -> np.ndarray:
    """Copy of ``emb`` with text dims projected onto the top-k PCs.

    k=0 -> every card's text block becomes the corpus PCA mean.
    k=512 -> exact reconstruction (the components are a full orthonormal basis).
    The trailing deterministic block is left untouched.
    """
    out = emb.copy()
    if k >= comps.shape[0]:
        return out
    text = emb[:, :text_dim]
    if k == 0:
        out[:, :text_dim] = mean
    else:
        V = comps[:k]
        out[:, :text_dim] = mean + (text - mean) @ V.T @ V
    return out


def probe1(ctx, sample: list[int], full_scores: np.ndarray,
           matches: list[tuple[int, int]], ks: list[int]) -> dict:
    banner("## P1 -- PC-truncation of the text block")
    pca = np.load(PCA_PATH)
    mean = pca["mean"].astype(np.float32)
    comps = pca["components"].astype(np.float32)
    explained = pca["explained"].astype(np.float64)
    cum = np.cumsum(explained)
    print(f"PCA: {PCA_PATH.name}  components={comps.shape}  "
          f"orthonormality err={np.abs(comps @ comps.T - np.eye(comps.shape[0])).max():.2e}")

    win = np.array([w for w, _ in matches], dtype=np.int64)
    lose = np.array([ln for _, ln in matches], dtype=np.int64)
    match_decks = sorted({int(i) for i in np.concatenate([win, lose])}) if matches else []
    pos = {d: j for j, d in enumerate(match_decks)}
    print(f"score fidelity on {len(sample)} real decks; "
          f"held-out task on {len(matches)} matches / {len(match_decks)} distinct decks")

    rows = []
    for k in ks:
        t0 = time.time()

        def trunc(mat, k=k):
            return truncate_text(mat, k, mean, comps, ctx.text_dim)

        s = ctx.score(ctx.mats(sample, trunc))
        r, rho = corr(s, full_scores)
        row = {
            "k": k,
            "explained_var": float(cum[k - 1]) if k > 0 else 0.0,
            "pearson_vs_full": r,
            "spearman_vs_full": rho,
            "mean_abs_dscore": float(np.abs(s - full_scores).mean()),
            "max_abs_dscore": float(np.abs(s - full_scores).max()),
            "score_mean": float(s.mean()),
            "score_std": float(s.std()),
        }
        if match_decks:
            ms = ctx.score(ctx.mats(match_decks, trunc))
            diff = ms[[pos[int(i)] for i in win]] - ms[[pos[int(i)] for i in lose]]
            row["match_acc"] = float(pairwise_correct(diff).mean())
            row["match_tie_rate"] = float((diff == 0).mean())
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"  k={k:<4d} rho={fmt(rho)} r={fmt(r)} "
              f"|dS|={fmt(row['mean_abs_dscore'], 3)} "
              f"acc={fmt(row.get('match_acc'), 4)} ({row['seconds']}s)")

    def smallest_k(thresh: float) -> int | None:
        for row in sorted(rows, key=lambda r: r["k"]):
            if row["spearman_vs_full"] >= thresh:
                return row["k"]
        return None

    k95, k99 = smallest_k(0.95), smallest_k(0.99)
    full_row = max(rows, key=lambda r: r["k"]) if rows else {}
    full_acc = full_row.get("match_acc")

    print("\n### P1a -- score fidelity vs full model\n")
    print_table(["k", "cum. explained var", "pearson r", "spearman rho", "mean abs dScore",
                 "max abs dScore"],
                [[row["k"], fmt(row["explained_var"]), fmt(row["pearson_vs_full"]),
                  fmt(row["spearman_vs_full"]), fmt(row["mean_abs_dscore"]),
                  fmt(row["max_abs_dscore"])] for row in rows])
    print(f"\n(full-model score std over the sample = {fmt(full_scores.std(), 4)})")
    print(f"smallest k with spearman >= 0.95 : {k95}")
    print(f"smallest k with spearman >= 0.99 : {k99}")

    if match_decks:
        print("\n### P1b -- held-out match prediction (score(winner) > score(loser))\n")
        print_table(["k", "accuracy", "tie rate", "delta vs full"], [
            [row["k"], fmt(row["match_acc"]), f"{100 * row['match_tie_rate']:.1f}%",
             fmt(row["match_acc"] - full_acc if full_acc is not None else float("nan"))]
            for row in rows])
        n = len(matches)
        band = coin_flip_band(n)
        print(f"\n(noise band +/-{band:.4f} = 2 SE at n={n} matches)")
        keep = [row["k"] for row in rows
                if full_acc is not None and abs(row["match_acc"] - full_acc) <= band]
        if keep:
            print(f"smallest k whose held-out accuracy is within noise of the full "
                  f"model: k={min(keep)}")

    print("\n**Reading**: k is the number of text directions the scorer is allowed "
          "to see. k=0 erases per-card text entirely (deterministic block only); "
          "the k where rho saturates is the effective text rank the scorer reads.")
    return {"ks": ks, "rows": rows, "smallest_k_rho95": k95, "smallest_k_rho99": k99,
            "full_match_acc": full_acc, "n_matches": len(matches),
            "n_match_decks": len(match_decks),
            "explained_var_cumulative": {str(k): (float(cum[k - 1]) if k > 0 else 0.0)
                                         for k in ks}}


def mean_pairwise_cos(x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Per-deck mean off-diagonal cosine similarity over non-padding rows."""
    xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sim = xn @ xn.transpose(1, 2)
    m = mask.to(x.dtype)
    pair = m.unsqueeze(2) * m.unsqueeze(1)
    n = m.sum(1)
    total = (sim * pair).sum(dim=(1, 2)) - n           # drop the unit diagonal
    denom = (n * (n - 1)).clamp_min(1.0)
    out = (total / denom).float().cpu().numpy()
    return np.where(n.cpu().numpy() >= 2, out, np.nan)


def mean_norm(x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    m = mask.to(x.dtype)
    nrm = (x.norm(dim=-1) * m).sum(1) / m.sum(1).clamp_min(1.0)
    return nrm.float().cpu().numpy()


class PmaWeightCapture:
    """Context manager: make ``pma.attn`` hand back its attention weights.

    ``PMA.forward`` calls ``self.attn(...)`` and discards the second return
    value, so a plain forward hook cannot see the weights. Instead we shadow
    the module's bound ``forward`` with a closure that forces
    ``need_weights=True, average_attn_weights=False`` and stashes the result.
    Instance-attribute assignment shadows the class method for
    ``nn.Module.__call__`` (which resolves ``self.forward``); ``__exit__``
    deletes it, restoring the original.

    Note: ``need_weights=True`` forces torch's non-fused attention path, so
    scores from a captured forward can differ from the fused path in the last
    couple of float digits. Only the weights are used from these runs.
    """

    def __init__(self, pma):
        self.pma = pma
        self.orig = None
        self.weights: torch.Tensor | None = None

    def __enter__(self):
        attn = self.pma.attn
        self.orig = attn.forward
        cap = self

        def wrapped(query, key, value, key_padding_mask=None, need_weights=True,
                    attn_mask=None, average_attn_weights=True, is_causal=False):
            out, w = cap.orig(query, key, value, key_padding_mask=key_padding_mask,
                              need_weights=True, attn_mask=attn_mask,
                              average_attn_weights=False, is_causal=is_causal)
            cap.weights = w.detach()
            return out, w

        attn.forward = wrapped
        return self

    def __exit__(self, *exc):
        try:
            del self.pma.attn.forward
        except AttributeError:
            pass
        return False


def probe2(ctx, sample: list[int]) -> dict:
    banner("## P2 -- SAB over-smoothing and PMA pooling")
    model, n_layers = ctx.model, len(ctx.model.sab_layers)
    labels = ["input(norm)"] + [f"sab{i + 1}" for i in range(n_layers)]

    cos_acc: list[list[np.ndarray]] = [[] for _ in labels]
    nrm_acc: list[list[np.ndarray]] = [[] for _ in labels]
    ent_acc, cv_acc, maxr_acc, land_acc, landfrac_acc = [], [], [], [], []

    captured: dict[int, torch.Tensor] = {}

    def make_hook(i):
        def hook(_module, _args, output):
            captured[i] = output.detach()
        return hook

    handles = [sab.register_forward_hook(make_hook(i))
               for i, sab in enumerate(model.sab_layers)]

    def per_batch(cards, mask):
        with torch.no_grad():
            reps = [model.normalize_features(cards)] + \
                   [captured[i] for i in range(n_layers)]
            for j, rep in enumerate(reps):
                cos_acc[j].append(mean_pairwise_cos(rep, mask))
                nrm_acc[j].append(mean_norm(rep, mask))
            w = cap.weights                                  # (B, H, seeds, L)
            if w is None:
                raise RuntimeError("PMA attention weights were not captured")
            m = mask.to(w.dtype)
            p = w * m[:, None, None, :]
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
            ent = -(torch.where(p > 0, p * p.log(), torch.zeros_like(p))).sum(-1)
            n_cards = m.sum(1).clamp_min(2.0)
            ent_acc.append((ent / n_cards.log()[:, None, None]).float().cpu().numpy())
            # Normalized entropy saturates at ~1.0 for near-uniform attention
            # (H/log n = 1 - O(cv^2)), so also record two linear dispersion
            # measures against the uniform weight 1/n.
            uni = (1.0 / n_cards)[:, None, None]              # (B,1,1)
            var = (((p - uni[..., None]) ** 2) * m[:, None, None, :]).sum(-1) \
                / n_cards[:, None, None]
            cv_acc.append((var.sqrt() / uni).float().cpu().numpy())
            maxr_acc.append((p.max(-1).values / uni).float().cpu().numpy())
            # per-seed (head-averaged) attention mass on lands
            land = pl.land_mask(cards).to(w.dtype) * m
            ps = p.mean(1)                                   # (B, seeds, L)
            land_acc.append((ps * land[:, None, :]).sum(-1).float().cpu().numpy())
            landfrac_acc.append((land.sum(1) / m.sum(1)).float().cpu().numpy())
        captured.clear()

    try:
        with PmaWeightCapture(model.pma) as cap:
            ctx.score(ctx.mats(sample), per_batch=per_batch)
    finally:
        for h in handles:
            h.remove()

    cos = [np.concatenate(a) for a in cos_acc]
    nrm = [np.concatenate(a) for a in nrm_acc]
    ent = np.concatenate(ent_acc)          # (decks, heads, seeds)
    cv = np.concatenate(cv_acc)
    maxr = np.concatenate(maxr_acc)
    land = np.concatenate(land_acc)        # (decks, seeds)
    landfrac = np.concatenate(landfrac_acc)

    print("\n### P2a -- mean pairwise cosine between card representations\n")
    layer_rows = [{"layer": lab, "cos_mean": float(np.nanmean(c)),
                   "cos_std": float(np.nanstd(c)), "cos_min": float(np.nanmin(c)),
                   "cos_max": float(np.nanmax(c)), "norm_mean": float(nn_.mean())}
                  for lab, c, nn_ in zip(labels, cos, nrm)]
    print_table(["depth", "mean cos", "std", "min deck", "max deck", "mean L2 norm"],
                [[r["layer"], fmt(r["cos_mean"]), fmt(r["cos_std"]), fmt(r["cos_min"]),
                  fmt(r["cos_max"]), fmt(r["norm_mean"], 3)] for r in layer_rows])
    print("\n(cos -> 1.0 with depth = over-smoothing: every card ends up pointing "
          "the same way, so the pooled vector can only read the deck-level average. "
          f"Every SAB output is LayerNormed, so its norm is pinned near "
          f"sqrt(d_model) = {math.sqrt(ctx.d_model):.2f}.)")

    print("\n### P2b -- PMA attention entropy (normalized by log n_cards)\n")
    n_heads, n_seeds = ent.shape[1], ent.shape[2]
    print_table(["seed \\ head"] + [f"h{h}" for h in range(n_heads)] + ["seed mean"],
                [[f"seed{s}"] + [fmt(ent[:, h, s].mean(), 6) for h in range(n_heads)]
                 + [fmt(ent[:, :, s].mean(), 6)] for s in range(n_seeds)])
    print(f"\noverall mean normalized entropy = {fmt(ent.mean(), 6)} "
          f"(1.0 = uniform attention = plain mean pooling; "
          f"std over decks = {fmt(ent.mean(axis=(1, 2)).std(), 6)})")

    print("\nNormalized entropy is quadratically flat near uniform "
          "(H/log n = 1 - O(cv^2)), so two linear dispersion measures:\n")
    disp_rows = [{"seed": s, "one_minus_entropy": float(1.0 - ent[:, :, s].mean()),
                  "cv": float(cv[:, :, s].mean()),
                  "max_over_uniform": float(maxr[:, :, s].mean())} for s in range(n_seeds)]
    print_table(["seed", "1 - norm. entropy", "attn CV (std / uniform weight)",
                 "max weight / uniform"],
                [[f"seed{r['seed']}", f"{r['one_minus_entropy']:.2e}", fmt(r["cv"]),
                  fmt(r["max_over_uniform"])] for r in disp_rows])
    print("\n(CV = 0 and max/uniform = 1 would be exactly-uniform attention; the "
          "PMA is then literally a mean-pooler over card representations.)")

    print("\n### P2c -- per-seed attention mass on lands vs spells\n")
    with_land = landfrac > 0
    print(f"deck land fraction (nonbasic lands / nonbasic cards): "
          f"mean {fmt(landfrac.mean())}, max {fmt(landfrac.max())}; "
          f"{int(with_land.sum())}/{len(landfrac)} decks contain a nonbasic land")
    seed_rows = []
    for s in range(n_seeds):
        m_l = float(land[:, s].mean())
        lift = (float(np.mean(land[with_land, s] / landfrac[with_land]))
                if with_land.any() else float("nan"))
        seed_rows.append({"seed": s, "land_mass": m_l, "spell_mass": 1.0 - m_l,
                          "land_lift": lift,
                          "land_mass_land_decks": (float(land[with_land, s].mean())
                                                   if with_land.any() else float("nan"))})
    print()
    print_table(["seed", "mass on lands", "mass on spells",
                 "land lift (mass/frac, land-holding decks)"],
                [[f"seed{r['seed']}", fmt(r["land_mass"]), fmt(r["spell_mass"]),
                  fmt(r["land_lift"], 3)] for r in seed_rows])
    print("\n(lift = 1 means the seed spreads attention over lands exactly in "
          "proportion to their share of the deck; >> 1 would be a 'land seed'. "
          "Decks with no nonbasic land are excluded from the lift, where the "
          "ratio is 0/0.)")

    return {"layers": layer_rows,
            "entropy_mean": float(ent.mean()),
            "entropy_by_seed_head": ent.mean(axis=0).tolist(),
            "entropy_per_seed": ent.mean(axis=(0, 1)).tolist(),
            "attn_cv_mean": float(cv.mean()),
            "attn_max_over_uniform_mean": float(maxr.mean()),
            "dispersion_per_seed": disp_rows,
            "n_heads": n_heads, "n_seeds": n_seeds,
            "seed_land_mass": seed_rows,
            "deck_land_fraction_mean": float(landfrac.mean()),
            "n_decks_with_nonbasic_land": int(with_land.sum()),
            "n_decks": len(sample)}


def probe3(ctx, sample: list[int], full_scores: np.ndarray, n_decks: int) -> dict:
    banner("## P3 -- magnitude stripping at the pooling boundary")
    which = sample[:n_decks]
    ref = full_scores[:n_decks]
    mats = ctx.mats(which)
    last = ctx.model.sab_layers[-1]
    print(f"scaling the SAB-stack output (input to PMA) on {len(which)} decks")

    rows = []
    for alpha in ALPHAS:
        def hook(_m, _a, output, a=alpha):
            return output * a
        handle = last.register_forward_hook(hook)
        try:
            s = ctx.score(mats)
        finally:
            handle.remove()
        r, rho = corr(s, ref)
        rows.append({"alpha": alpha, "pearson_vs_full": r, "spearman_vs_full": rho,
                     "mean_abs_dscore": float(np.abs(s - ref).mean()),
                     "max_abs_dscore": float(np.abs(s - ref).max()),
                     "score_mean": float(s.mean()), "score_std": float(s.std())})

    print()
    print_table(["alpha", "pearson vs full", "spearman vs full", "mean abs dScore",
                 "max abs dScore", "score mean", "score std"],
                [[row["alpha"], fmt(row["pearson_vs_full"]), fmt(row["spearman_vs_full"]),
                  fmt(row["mean_abs_dscore"]), fmt(row["max_abs_dscore"]),
                  fmt(row["score_mean"], 3), fmt(row["score_std"], 3)] for row in rows])
    print(f"\n(unscaled score std on these decks = {fmt(ref.std(), 4)})")
    print("Scaling x also scales PMA's attention logits (q.k grows with |k|), so "
          "invariance here is a strong claim: neither the value magnitudes nor "
          "the attention sharpness reach the MLP -- PMA's LayerNorm removes both.")
    return {"n_decks": len(which), "alphas": ALPHAS, "rows": rows,
            "ref_score_std": float(ref.std())}


def probe4(ctx, sample: list[int], full_scores: np.ndarray,
           n_decks: int, n_cards: int) -> dict:
    banner("## P4 -- replication invariance (mean vs sum pooling)")
    which = sample[:n_decks]
    ref = full_scores[:n_decks]
    base = ctx.mats(which)

    print(f"\n### P4a -- deck replication D vs D(x)k on {len(which)} decks\n")
    rep_rows = []
    for k in REPLICATION_KS:
        s = ctx.score([np.repeat(m, k, axis=0) for m in base])
        rep_rows.append({"k": k, "mean_abs_dscore": float(np.abs(s - ref).mean()),
                         "max_abs_dscore": float(np.abs(s - ref).max()),
                         "mean_signed_dscore": float((s - ref).mean()),
                         "spearman_vs_base": corr(s, ref)[1]})
    print_table(["k", "mean abs dScore", "max abs dScore", "spearman vs D",
                 "mean dScore (signed)"],
                [[r["k"], fmt(r["mean_abs_dscore"], 5), fmt(r["max_abs_dscore"], 5),
                  fmt(r["spearman_vs_base"]), fmt(r["mean_signed_dscore"], 5)]
                 for r in rep_rows])
    print(f"\n(score std over these decks = {fmt(ref.std(), 4)}; a pure-mean "
          "pooler is exactly replication-invariant, so dScore ~ 0 means the "
          "network cannot count copies.)")

    lo, hi = MID_SCORE_RANGE
    wr = pl.load_win_rates()
    cands = sorted((n for n, r in wr.items()
                    if r.get("shrunk_score_play") is not None
                    and lo <= r["shrunk_score_play"] <= hi),
                   key=lambda n: -(wr[n].get("wins_when_in_deck") or 0.0))
    picked = list(islice((n for n in cands if ctx.bank.card_ok(n)), n_cards))

    print(f"\n### P4b -- decks of k identical copies "
          f"({len(picked)} cards with shrunk_score_play in [{lo}, {hi}])\n")
    copy_scores, copy_mean = {}, []
    if picked:
        mats = [np.repeat(ctx.bank.emb[n][None, :], k, axis=0)
                for n in picked for k in COPY_KS]
        cs = ctx.score(mats).reshape(len(picked), len(COPY_KS))
        copy_scores = {n: dict(zip(map(str, COPY_KS), row.tolist()))
                       for n, row in zip(picked, cs)}
        copy_mean = cs.mean(axis=0).tolist()
        ks = [f"k={k}" for k in COPY_KS]
        print_table(["card", "shrunk_score_play"] + ks + ["spread"],
                    [[n, fmt(wr[n]["shrunk_score_play"], 3)] + [fmt(v, 3) for v in row]
                     + [fmt(row.max() - row.min())] for n, row in zip(picked, cs)])
        print()
        print_table(["stat"] + ks,
                    [["mean over cards"] + [fmt(v, 3) for v in copy_mean],
                     ["std over cards"] + [fmt(v, 3) for v in cs.std(axis=0)]])
        print("\nmean |s(k) - s(1)| over cards: " +
              ", ".join(f"k={k}: {fmt(np.abs(cs[:, j] - cs[:, 0]).mean(), 4)}"
                        for j, k in enumerate(COPY_KS)))
        print("(flat rows = pure mean pooling: the model sees 23 copies of a card "
              "exactly as it sees one copy.)")
    else:
        print("_no mid-quality cards resolved_")

    return {"replication": rep_rows, "n_decks": len(which),
            "copy_ks": COPY_KS, "copy_cards": picked,
            "copy_scores": copy_scores, "copy_mean_by_k": copy_mean,
            "mid_score_range": list(MID_SCORE_RANGE)}


def probe5(ctx, sample: list[int], full_scores: np.ndarray,
           n_trunc: int, n_land_samples: int) -> dict:
    banner("## P5 -- OOD envelope")
    wr = pl.load_win_rates()
    scored = [(n, r["shrunk_score_play"]) for n, r in wr.items()
              if r.get("shrunk_score_play") is not None]
    thresh = float(np.percentile([v for _, v in scored], 90))
    top = sorted((n for n, v in scored if v >= thresh),
                 key=lambda n: -(wr[n].get("wins_when_in_deck") or 0.0))
    bombs = list(islice((n for n in dict.fromkeys(["Shivan Dragon"] + top)
                         if ctx.bank.card_ok(n)), 3))

    emb = ctx.bank.emb
    ref = full_scores
    pcts = {str(p): float(np.percentile(ref, p)) for p in PERCENTILES}
    print(f"\n### P5a -- reference band ({len(ref)} real decks)\n")
    print_table([f"p{p}" for p in PERCENTILES] + ["mean", "std"],
                [[fmt(pcts[str(p)], 3) for p in PERCENTILES]
                 + [fmt(ref.mean(), 3), fmt(ref.std(), 3)]])

    families: dict[str, np.ndarray] = {}

    # (b) truncated real decks
    which = sample[:n_trunc]
    for n in TRUNC_SIZES:
        mats = []
        for mat in ctx.mats(which):
            if len(mat) < n:
                continue
            mats.append(mat[ctx.rng.choice(len(mat), size=n, replace=False)])
        if mats:
            families[f"truncated_{n}_cards"] = ctx.score(mats)

    # (c) all-nonbasic-land decks. The land pool is drawn from the cards of the
    # deck sample only, so it does not depend on which other probes ran (P1
    # registers thousands of extra match decks into the same bank). Cards stay
    # in the order the bank first met them, which the seeded draw depends on.
    first_met = {n: i for i, n in enumerate(emb)}
    sample_cards = sorted({c for i in sample for c in ctx.bank.keys[i]}, key=first_met.get)
    lands = [c for c in sample_cards if layout.is_land_embedding(emb[c])]
    replace = len(lands) < 23
    if lands:
        mats = [np.stack([emb[c] for c in ctx.rng.choice(lands, size=23, replace=replace)])
                for _ in range(n_land_samples)]
        families["23_nonbasic_lands"] = ctx.score(mats)

    # (d) 23 copies of a bomb
    for name in bombs:
        families[f"23x_{name}"] = ctx.score([np.repeat(emb[name][None, :], 23, axis=0)])

    print("\n### P5b-d -- degenerate families vs the real-deck band\n")
    print(f"nonbasic-land pool: {len(lands)} distinct lands among the "
          f"{len(sample_cards)} distinct cards of the deck sample"
          f"{' (sampled WITH replacement: pool < 23)' if replace else ''}; "
          f"bombs: {', '.join(bombs) if bombs else '(none resolved)'} "
          f"(top-decile shrunk_score_play >= {fmt(thresh, 4)})")
    fam_rows = []
    for name, s in families.items():
        med = float(np.median(s))
        fam_rows.append({"family": name, "n": int(len(s)), "mean": float(s.mean()),
                         "median": med, "min": float(s.min()), "max": float(s.max()),
                         "pct_rank_of_median": pct_rank(ref, med),
                         "frac_above_p99": float((s > pcts["99"]).mean()),
                         "frac_below_p1": float((s < pcts["1"]).mean())})
    print()
    print_table(["family", "n", "mean", "median", "min", "max",
                 "pct-rank of median in real band"],
                [[r["family"], r["n"], fmt(r["mean"], 3), fmt(r["median"], 3),
                  fmt(r["min"], 3), fmt(r["max"], 3), f"{r['pct_rank_of_median']:.1f}"]
                 for r in fam_rows])
    print("\n(pct-rank 100 = the degenerate family scores above every real deck; "
          "a well-calibrated scorer should put all of these at the bottom.)")

    return {"real_percentiles": pcts, "real_mean": float(ref.mean()),
            "real_std": float(ref.std()), "n_real": int(len(ref)),
            "families": fam_rows, "bombs": bombs,
            "n_nonbasic_lands_in_pool": len(lands),
            "lands_sampled_with_replacement": bool(replace),
            "top_decile_threshold": thresh,
            "family_scores": {k: v.tolist() for k, v in families.items()}}


SIZES = {
    "device": Preset(None, "cpu"),
    "decks": Preset(400, 40),
    "matches": Preset(2000, 200),
    "batch": Preset(256, 64),
    "ks": Preset(K_GRID, K_GRID_SMOKE),
    "p3_decks": Preset(200, 20),
    "p4_decks": Preset(100, 15),
    "p4_cards": Preset(30, 8),
    "p5_trunc_decks": Preset(50, 10),
    "p5_land_samples": Preset(20, 5),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="CPU, 40 decks, k subset, 200 matches")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--decks", type=int, default=None, help="deck sample size")
    ap.add_argument("--matches", type=int, default=None,
                    help="held-out matches for P1b")
    ap.add_argument("--batch", type=int, default=None, help="decks per forward")
    ap.add_argument("--probes", default="all",
                    help="comma-separated subset of p1,p2,p3,p4,p5")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON output path")
    args = ap.parse_args()

    pl.apply_presets(args, SIZES)

    want = ALL_PROBES if args.probes == "all" else \
        [p.strip().lower() for p in args.probes.split(",") if p.strip()]
    bad = [p for p in want if p not in ALL_PROBES]
    if bad:
        raise SystemExit(f"unknown probe(s): {bad}; choose from {ALL_PROBES}")

    torch.manual_seed(SEED)
    t_start = time.time()
    banner("# T6 -- mechanism probes of the gen-4 sealed scorer")
    print(f"loading scorer (device={args.device or 'auto'}) ...")
    ctx = Ctx(args)
    print(f"checkpoint : {pl.SCORER_CKPT.name}")
    print(f"d_model={ctx.d_model}  text[0:{ctx.text_dim}]  "
          f"det[{ctx.text_dim}:{ctx.d_model}]  layers={len(ctx.model.sab_layers)}  "
          f"seeds={ctx.model.pma.seeds.shape[1]}  device={ctx.device}  "
          f"batch={ctx.batch}  smoke={args.smoke}")

    sample, lines_read, dropped = build_sample(ctx, args.decks)
    print(f"\ndeck sample: {len(sample)} distinct deck_A from the first "
          f"{lines_read} match lines ({dropped} dropped for unembeddable cards); "
          f"{len(ctx.bank.emb)} distinct cards; "
          f"{len(ctx.bank.missing)} card names unembeddable")
    if not sample:
        raise SystemExit("no usable decks")
    sizes = np.array([len(ctx.bank.keys[i]) for i in sample])
    print(f"nonbasic deck sizes: min {sizes.min()}, median {np.median(sizes):.0f}, "
          f"max {sizes.max()}")

    full_scores = ctx.score(ctx.mats(sample))
    print(f"full-model scores: mean {full_scores.mean():.4f}  "
          f"std {full_scores.std():.4f}  "
          f"range [{full_scores.min():.4f}, {full_scores.max():.4f}]")

    matches: list[tuple[int, int]] = []
    m_meta = {}
    if "p1" in want:
        rows, undecided, malformed = pl.decided_matches(MATCH_FILE, args.matches)
        m_dropped = 0
        for m, winner in rows:
            ia, ib = ctx.bank.add(m.deck_a.cards), ctx.bank.add(m.deck_b.cards)
            if ia is None or ib is None:
                m_dropped += 1
                continue
            matches.append((ia, ib) if winner == "A" else (ib, ia))
        m_meta = {"parsed": len(rows), "undecided": undecided,
                  "malformed": malformed, "dropped_missing": m_dropped,
                  "kept": len(matches)}
        print(f"held-out matches: parsed {len(rows)}, kept {len(matches)} "
              f"({m_dropped} dropped for missing embeddings, "
              f"{undecided} undecided)")

    res = {"meta": {
        "checkpoint": str(pl.SCORER_CKPT), "match_file": str(MATCH_FILE),
        "pca_file": str(PCA_PATH), "device": ctx.device, "seed": SEED,
        "smoke": bool(args.smoke), "batch": ctx.batch,
        "d_model": ctx.d_model, "text_dim": ctx.text_dim,
        "det_dim": pl.FC,
        "n_sab_layers": len(ctx.model.sab_layers),
        "n_seeds": int(ctx.model.pma.seeds.shape[1]),
        "sample_lines": lines_read, "n_decks": len(sample),
        "decks_dropped": dropped, "n_distinct_cards": len(ctx.bank.emb),
        "n_unembeddable": len(ctx.bank.missing),
        "unembeddable_sample": sorted(ctx.bank.missing)[:20],
        "deck_size_min": int(sizes.min()), "deck_size_max": int(sizes.max()),
        "probes_run": want, "match_corpus": m_meta,
    }, "full_scores": {
        "mean": float(full_scores.mean()), "std": float(full_scores.std()),
        "min": float(full_scores.min()), "max": float(full_scores.max()),
        "values": full_scores.tolist(),
    }}

    if "p1" in want:
        res["p1_pc_truncation"] = probe1(ctx, sample, full_scores, matches, args.ks)
    if "p2" in want:
        res["p2_oversmoothing"] = probe2(ctx, sample)
    if "p3" in want:
        res["p3_magnitude"] = probe3(ctx, sample, full_scores, args.p3_decks)
    if "p4" in want:
        res["p4_replication"] = probe4(ctx, sample, full_scores,
                                       args.p4_decks, args.p4_cards)
    if "p5" in want:
        res["p5_ood"] = probe5(ctx, sample, full_scores, args.p5_trunc_decks,
                               args.p5_land_samples)

    res["meta"]["seconds"] = round(time.time() - t_start, 1)
    write_json(args.out, res)
    print(f"\n\nwrote {args.out}  ({res['meta']['seconds']}s total)")


if __name__ == "__main__":
    main()
