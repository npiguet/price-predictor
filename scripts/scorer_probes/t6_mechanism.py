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
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

import probe_lib as pl

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


# ------------------------------------------------------------------ formatting

def fmt(x, nd: int = 4) -> str:
    if x is None:
        return "-"
    x = float(x)
    return "nan" if x != x else f"{x:.{nd}f}"


def h1(text: str) -> None:
    print("\n\n" + "=" * 78)
    print(text)
    print("=" * 78)


def jsonable(o):
    """Recursively convert numpy scalars/arrays to plain Python for json.dump."""
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(pearson r, spearman rho); nan when either side is constant."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])


def pct_rank(ref: np.ndarray, x: float) -> float:
    """Percentile of ``x`` within the reference distribution (mid-rank ties)."""
    return 100.0 * float((ref < x).mean() + 0.5 * (ref == x).mean())


# --------------------------------------------------------------- card / decks

class CardTable:
    """Name -> row index into a dense (N, d_model) raw-embedding array."""

    def __init__(self, probe: pl.Probe):
        self.probe = probe
        self.index: dict[str, int] = {}
        self.names: list[str] = []
        self.missing: set[str] = set()
        self._rows: list[np.ndarray] = []
        self._arr: np.ndarray | None = None

    def row(self, name: str) -> int | None:
        hit = self.index.get(name)
        if hit is not None:
            return hit
        if name in self.missing:
            return None
        emb = self.probe.embedding(name)
        if emb is None:
            self.missing.add(name)
            return None
        idx = len(self._rows)
        self.index[name] = idx
        self.names.append(name)
        self._rows.append(emb.astype(np.float32))
        self._arr = None
        return idx

    @property
    def emb(self) -> np.ndarray:
        if self._arr is None:
            self._arr = np.stack(self._rows) if self._rows else np.zeros((0, 1), np.float32)
        return self._arr

    def deck_rows(self, deck) -> np.ndarray | None:
        """Row indices of a deck's nonbasic cards; None if any is unembeddable."""
        idx = []
        for name in deck:
            if name.lower() in pl.BASIC_LAND_NAMES:
                continue
            r = self.row(name)
            if r is None:
                return None
            idx.append(r)
        return np.asarray(idx, dtype=np.int64) if idx else None


class DeckPool:
    """Distinct decks (by sorted nonbasic multiset) as row-index arrays."""

    def __init__(self, table: CardTable):
        self.table = table
        self.index: dict[tuple[str, ...], int] = {}
        self.rows: list[np.ndarray] = []
        self.keys: list[tuple[str, ...]] = []

    def add(self, deck) -> int | None:
        key = tuple(sorted(c for c in deck if c.lower() not in pl.BASIC_LAND_NAMES))
        if not key:
            return None
        hit = self.index.get(key)
        if hit is not None:
            return hit
        rows = self.table.deck_rows(key)
        if rows is None:
            return None
        i = len(self.rows)
        self.index[key] = i
        self.rows.append(rows)
        self.keys.append(key)
        return i

    def mats(self, emb: np.ndarray, which: list[int]) -> list[np.ndarray]:
        return [emb[self.rows[i]] for i in which]


def parse_matches(path: Path, limit: int):
    """(deck_A, deck_B, winner) for the first ``limit`` decidable match lines.

    Winner = the side with more of its letters in field 7, the per-game winner
    string (e.g. "AABBABA" -> A wins 4-3). Same convention as t5_ablation.
    """
    rows, undecided, malformed = [], 0, 0
    if limit <= 0:
        return rows, undecided, malformed
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
            rows.append((p[5].split("|"), p[6].split("|"), "A" if a > b else "B"))
            if len(rows) >= limit:
                break
    return rows, undecided, malformed


# ------------------------------------------------------------------- batching

def pad_batch(mats: list[np.ndarray], d_model: int):
    """(cards, mask) padded tensors for a list of (n_cards, d_model) matrices."""
    n = len(mats)
    mx = max(m.shape[0] for m in mats)
    cards = torch.zeros(n, mx, d_model)
    mask = torch.zeros(n, mx, dtype=torch.bool)
    for i, m in enumerate(mats):
        cards[i, : m.shape[0]] = torch.from_numpy(np.ascontiguousarray(m))
        mask[i, : m.shape[0]] = True
    return cards, mask


@torch.no_grad()
def score_mats(ctx, mats: list[np.ndarray], per_batch=None) -> np.ndarray:
    """Score decks given as raw (n_cards, d_model) matrices.

    Replicates ``probe_lib.Probe.score_matrices`` batching, but calls the model
    directly so forward hooks / patches installed by the caller see every batch.
    ``per_batch(cards, mask)`` is invoked after each batch with the *raw*
    (pre-normalization) padded inputs, so hook consumers can align their
    captures with the batch.
    """
    if not mats:
        return np.zeros(0, dtype=np.float64)
    out = np.empty(len(mats), dtype=np.float64)
    for lo in range(0, len(mats), ctx.batch):
        chunk = mats[lo:lo + ctx.batch]
        cards, mask = pad_batch(chunk, ctx.d_model)
        cards, mask = cards.to(ctx.device), mask.to(ctx.device)
        s = ctx.model(cards, mask)
        out[lo:lo + len(chunk)] = s.squeeze(-1).float().cpu().numpy()
        if per_batch is not None:
            per_batch(cards, mask)
    return out


class Ctx:
    """Everything the probes share: model handle, batching, sample, rng."""

    def __init__(self, args):
        self.args = args
        self.probe = pl.Probe(device=args.device)
        self.model = self.probe.model
        self.device = self.probe.device
        self.d_model = self.probe.d_model
        self.text_dim = self.d_model - pl.layout.FEATURE_COUNT
        self.batch = args.batch
        self.rng = np.random.default_rng(SEED)
        self.table = CardTable(self.probe)
        self.pool = DeckPool(self.table)
        self.results: dict = {}


# ----------------------------------------------------------------- deck sample

def build_sample(ctx, n_decks: int) -> tuple[list[int], int, int]:
    """First ``n_decks`` distinct, fully-embeddable deck_A of the first 700 lines.

    Returns (deck indices into ``ctx.pool``, match lines read, decks dropped).
    """
    sample, seen_lines, dropped = [], 0, 0
    for _set, _ma, deck_a, _mb, _db in pl.read_match_decks(MATCH_FILE, limit=SAMPLE_LINES):
        seen_lines += 1
        before = len(ctx.pool.rows)
        idx = ctx.pool.add(deck_a)
        if idx is None:
            dropped += 1
            continue
        if len(ctx.pool.rows) > before:      # newly distinct deck
            sample.append(idx)
            if len(sample) >= n_decks:
                break
    return sample, seen_lines, dropped


# ------------------------------------------------------- P1: PC truncation

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
    h1("## P1 -- PC-truncation of the text block")
    pca = np.load(PCA_PATH)
    mean = pca["mean"].astype(np.float32)
    comps = pca["components"].astype(np.float32)
    explained = pca["explained"].astype(np.float64)
    cum = np.cumsum(explained)
    print(f"PCA: {PCA_PATH.name}  components={comps.shape}  "
          f"orthonormality err={np.abs(comps @ comps.T - np.eye(comps.shape[0])).max():.2e}")

    win = np.array([w for w, _ in matches], dtype=np.int64)
    lose = np.array([l for _, l in matches], dtype=np.int64)
    match_decks = sorted({int(i) for i in np.concatenate([win, lose])}) if matches else []
    pos = {d: j for j, d in enumerate(match_decks)}
    print(f"score fidelity on {len(sample)} real decks; "
          f"held-out task on {len(matches)} matches / {len(match_decks)} distinct decks")

    rows, base = [], ctx.table.emb
    for k in ks:
        t0 = time.time()
        emb_k = truncate_text(base, k, mean, comps, ctx.text_dim)
        s = score_mats(ctx, ctx.pool.mats(emb_k, sample))
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
            ms = score_mats(ctx, ctx.pool.mats(emb_k, match_decks))
            diff = ms[[pos[int(i)] for i in win]] - ms[[pos[int(i)] for i in lose]]
            row["match_acc"] = float(np.where(diff > 0, 1.0,
                                              np.where(diff < 0, 0.0, 0.5)).mean())
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
    print("| k | cum. explained var | pearson r | spearman rho | mean abs dScore | max abs dScore |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['k']} | {fmt(row['explained_var'], 4)} | "
              f"{fmt(row['pearson_vs_full'])} | {fmt(row['spearman_vs_full'])} | "
              f"{fmt(row['mean_abs_dscore'], 4)} | {fmt(row['max_abs_dscore'], 4)} |")
    print(f"\n(full-model score std over the sample = {fmt(full_scores.std(), 4)})")
    print(f"smallest k with spearman >= 0.95 : {k95}")
    print(f"smallest k with spearman >= 0.99 : {k99}")

    if match_decks:
        print("\n### P1b -- held-out match prediction (score(winner) > score(loser))\n")
        print("| k | accuracy | tie rate | delta vs full |")
        print("|---|---|---|---|")
        for row in rows:
            d = row["match_acc"] - full_acc if full_acc is not None else float("nan")
            print(f"| {row['k']} | {fmt(row['match_acc'])} | "
                  f"{100 * row['match_tie_rate']:.1f}% | {fmt(d)} |")
        n = len(matches)
        band = 2 * (0.25 / max(n, 1)) ** 0.5
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


# ------------------------------------------- P2: over-smoothing / PMA pooling

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
    h1("## P2 -- SAB over-smoothing and PMA pooling")
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
            land = ((cards[..., -pl.layout.FEATURE_COUNT + pl.layout.IS_LAND] > 0.5)
                    .to(w.dtype) * m)
            ps = p.mean(1)                                   # (B, seeds, L)
            land_acc.append((ps * land[:, None, :]).sum(-1).float().cpu().numpy())
            landfrac_acc.append((land.sum(1) / m.sum(1)).float().cpu().numpy())
        captured.clear()

    try:
        with PmaWeightCapture(model.pma) as cap:
            score_mats(ctx, ctx.pool.mats(ctx.table.emb, sample), per_batch=per_batch)
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
    print("| depth | mean cos | std | min deck | max deck | mean L2 norm |")
    print("|---|---|---|---|---|---|")
    layer_rows = []
    for lab, c, nn_ in zip(labels, cos, nrm):
        rec = {"layer": lab, "cos_mean": float(np.nanmean(c)),
               "cos_std": float(np.nanstd(c)), "cos_min": float(np.nanmin(c)),
               "cos_max": float(np.nanmax(c)), "norm_mean": float(nn_.mean())}
        layer_rows.append(rec)
        print(f"| {lab} | {fmt(rec['cos_mean'])} | {fmt(rec['cos_std'])} | "
              f"{fmt(rec['cos_min'])} | {fmt(rec['cos_max'])} | "
              f"{fmt(rec['norm_mean'], 3)} |")
    print("\n(cos -> 1.0 with depth = over-smoothing: every card ends up pointing "
          "the same way, so the pooled vector can only read the deck-level average. "
          f"Every SAB output is LayerNormed, so its norm is pinned near "
          f"sqrt(d_model) = {math.sqrt(ctx.d_model):.2f}.)")

    print("\n### P2b -- PMA attention entropy (normalized by log n_cards)\n")
    n_heads, n_seeds = ent.shape[1], ent.shape[2]
    print("| seed \\ head | " + " | ".join(f"h{h}" for h in range(n_heads)) + " | seed mean |")
    print("|---|" + "---|" * (n_heads + 1))
    for s in range(n_seeds):
        cells = " | ".join(fmt(ent[:, h, s].mean(), 6) for h in range(n_heads))
        print(f"| seed{s} | {cells} | {fmt(ent[:, :, s].mean(), 6)} |")
    print(f"\noverall mean normalized entropy = {fmt(ent.mean(), 6)} "
          f"(1.0 = uniform attention = plain mean pooling; "
          f"std over decks = {fmt(ent.mean(axis=(1, 2)).std(), 6)})")

    print("\nNormalized entropy is quadratically flat near uniform "
          "(H/log n = 1 - O(cv^2)), so two linear dispersion measures:\n")
    print("| seed | 1 - norm. entropy | attn CV (std / uniform weight) | "
          "max weight / uniform |")
    print("|---|---|---|---|")
    disp_rows = []
    for s in range(n_seeds):
        rec = {"seed": s, "one_minus_entropy": float(1.0 - ent[:, :, s].mean()),
               "cv": float(cv[:, :, s].mean()), "max_over_uniform": float(maxr[:, :, s].mean())}
        disp_rows.append(rec)
        print(f"| seed{s} | {rec['one_minus_entropy']:.2e} | {fmt(rec['cv'], 4)} | "
              f"{fmt(rec['max_over_uniform'], 4)} |")
    print("\n(CV = 0 and max/uniform = 1 would be exactly-uniform attention; the "
          "PMA is then literally a mean-pooler over card representations.)")

    print("\n### P2c -- per-seed attention mass on lands vs spells\n")
    with_land = landfrac > 0
    print(f"deck land fraction (nonbasic lands / nonbasic cards): "
          f"mean {fmt(landfrac.mean())}, max {fmt(landfrac.max())}; "
          f"{int(with_land.sum())}/{len(landfrac)} decks contain a nonbasic land")
    print("\n| seed | mass on lands | mass on spells | land lift (mass/frac, "
          "land-holding decks) |")
    print("|---|---|---|---|")
    seed_rows = []
    for s in range(n_seeds):
        m_l = float(land[:, s].mean())
        lift = (float(np.mean(land[with_land, s] / landfrac[with_land]))
                if with_land.any() else float("nan"))
        seed_rows.append({"seed": s, "land_mass": m_l, "spell_mass": 1.0 - m_l,
                          "land_lift": lift,
                          "land_mass_land_decks": (float(land[with_land, s].mean())
                                                   if with_land.any() else float("nan"))})
        print(f"| seed{s} | {fmt(m_l)} | {fmt(1.0 - m_l)} | {fmt(lift, 3)} |")
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


# --------------------------------------------- P3: magnitude at the pool edge

def probe3(ctx, sample: list[int], full_scores: np.ndarray, n_decks: int) -> dict:
    h1("## P3 -- magnitude stripping at the pooling boundary")
    which = sample[:n_decks]
    ref = full_scores[:n_decks]
    mats = ctx.pool.mats(ctx.table.emb, which)
    last = ctx.model.sab_layers[-1]
    print(f"scaling the SAB-stack output (input to PMA) on {len(which)} decks")

    rows = []
    for alpha in ALPHAS:
        def hook(_m, _a, output, a=alpha):
            return output * a
        handle = last.register_forward_hook(hook)
        try:
            s = score_mats(ctx, mats)
        finally:
            handle.remove()
        r, rho = corr(s, ref)
        rows.append({"alpha": alpha, "pearson_vs_full": r, "spearman_vs_full": rho,
                     "mean_abs_dscore": float(np.abs(s - ref).mean()),
                     "max_abs_dscore": float(np.abs(s - ref).max()),
                     "score_mean": float(s.mean()), "score_std": float(s.std())})

    print("\n| alpha | pearson vs full | spearman vs full | mean abs dScore | "
          "max abs dScore | score mean | score std |")
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['alpha']} | {fmt(row['pearson_vs_full'])} | "
              f"{fmt(row['spearman_vs_full'])} | {fmt(row['mean_abs_dscore'], 4)} | "
              f"{fmt(row['max_abs_dscore'], 4)} | {fmt(row['score_mean'], 3)} | "
              f"{fmt(row['score_std'], 3)} |")
    print(f"\n(unscaled score std on these decks = {fmt(ref.std(), 4)})")
    print("Scaling x also scales PMA's attention logits (q.k grows with |k|), so "
          "invariance here is a strong claim: neither the value magnitudes nor "
          "the attention sharpness reach the MLP -- PMA's LayerNorm removes both.")
    return {"n_decks": len(which), "alphas": ALPHAS, "rows": rows,
            "ref_score_std": float(ref.std())}


# ------------------------------------------------ P4: replication invariance

def probe4(ctx, sample: list[int], full_scores: np.ndarray,
           n_decks: int, n_cards: int) -> dict:
    h1("## P4 -- replication invariance (mean vs sum pooling)")
    which = sample[:n_decks]
    ref = full_scores[:n_decks]
    base = ctx.pool.mats(ctx.table.emb, which)

    print(f"\n### P4a -- deck replication D vs D(x)k on {len(which)} decks\n")
    print("| k | mean abs dScore | max abs dScore | spearman vs D | mean dScore (signed) |")
    print("|---|---|---|---|---|")
    rep_rows = []
    for k in REPLICATION_KS:
        s = score_mats(ctx, [np.repeat(m, k, axis=0) for m in base])
        _, rho = corr(s, ref)
        rec = {"k": k, "mean_abs_dscore": float(np.abs(s - ref).mean()),
               "max_abs_dscore": float(np.abs(s - ref).max()),
               "mean_signed_dscore": float((s - ref).mean()),
               "spearman_vs_base": rho}
        rep_rows.append(rec)
        print(f"| {k} | {fmt(rec['mean_abs_dscore'], 5)} | "
              f"{fmt(rec['max_abs_dscore'], 5)} | {fmt(rho)} | "
              f"{fmt(rec['mean_signed_dscore'], 5)} |")
    print(f"\n(score std over these decks = {fmt(ref.std(), 4)}; a pure-mean "
          "pooler is exactly replication-invariant, so dScore ~ 0 means the "
          "network cannot count copies.)")

    # ---- k identical copies of one mid-quality card
    lo, hi = MID_SCORE_RANGE
    wr = pl.load_win_rates()
    cands = sorted((n for n, r in wr.items()
                    if r.get("shrunk_score_play") is not None
                    and lo <= r["shrunk_score_play"] <= hi
                    and n.lower() not in pl.BASIC_LAND_NAMES),
                   key=lambda n: -(wr[n].get("wins_when_in_deck") or 0.0))
    picked, rows_idx = [], []
    for name in cands:
        r = ctx.table.row(name)
        if r is None:
            continue
        picked.append(name)
        rows_idx.append(r)
        if len(picked) >= n_cards:
            break

    print(f"\n### P4b -- decks of k identical copies "
          f"({len(picked)} cards with shrunk_score_play in [{lo}, {hi}])\n")
    copy_scores, copy_mean = {}, []
    if picked:
        emb = ctx.table.emb          # re-fetch: picking may have added rows
        mats = [np.repeat(emb[r][None, :], k, axis=0)
                for r in rows_idx for k in COPY_KS]
        cs = score_mats(ctx, mats).reshape(len(picked), len(COPY_KS))
        copy_scores = {n: dict(zip(map(str, COPY_KS), row.tolist()))
                       for n, row in zip(picked, cs)}
        copy_mean = cs.mean(axis=0).tolist()
        print("| card | shrunk_score_play | " +
              " | ".join(f"k={k}" for k in COPY_KS) + " | spread |")
        print("|---|---|" + "---|" * (len(COPY_KS) + 1))
        for n, row in zip(picked, cs):
            print(f"| {n} | {fmt(wr[n]['shrunk_score_play'], 3)} | " +
                  " | ".join(fmt(v, 3) for v in row) +
                  f" | {fmt(row.max() - row.min(), 4)} |")
        print("\n| stat | " + " | ".join(f"k={k}" for k in COPY_KS) + " |")
        print("|---|" + "---|" * len(COPY_KS))
        print("| mean over cards | " + " | ".join(fmt(v, 3) for v in copy_mean) + " |")
        print("| std over cards | " + " | ".join(fmt(v, 3) for v in cs.std(axis=0)) + " |")
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


# ----------------------------------------------------------- P5: OOD envelope

def probe5(ctx, sample: list[int], full_scores: np.ndarray,
           n_trunc: int, n_land_samples: int) -> dict:
    h1("## P5 -- OOD envelope")
    # Resolve the bomb cards first: ``table.row`` can append new rows, which
    # would invalidate an ``emb`` array captured earlier.
    wr = pl.load_win_rates()
    scored = [(n, r["shrunk_score_play"]) for n, r in wr.items()
              if r.get("shrunk_score_play") is not None]
    thresh = float(np.percentile([v for _, v in scored], 90))
    top = sorted((n for n, v in scored if v >= thresh
                  and n.lower() not in pl.BASIC_LAND_NAMES),
                 key=lambda n: -(wr[n].get("wins_when_in_deck") or 0.0))
    bombs: list[str] = []
    for name in ["Shivan Dragon"] + top:
        if name in bombs or len(bombs) >= 3:
            continue
        if ctx.table.row(name) is not None:
            bombs.append(name)

    emb = ctx.table.emb
    ref = full_scores
    pcts = {str(p): float(np.percentile(ref, p)) for p in PERCENTILES}
    print(f"\n### P5a -- reference band ({len(ref)} real decks)\n")
    print("| " + " | ".join(f"p{p}" for p in PERCENTILES) + " | mean | std |")
    print("|" + "---|" * (len(PERCENTILES) + 2))
    print("| " + " | ".join(fmt(pcts[str(p)], 3) for p in PERCENTILES) +
          f" | {fmt(ref.mean(), 3)} | {fmt(ref.std(), 3)} |")

    families: dict[str, np.ndarray] = {}

    # (b) truncated real decks
    which = sample[:n_trunc]
    for n in TRUNC_SIZES:
        mats = []
        for i in which:
            rows = ctx.pool.rows[i]
            if len(rows) < n:
                continue
            sel = ctx.rng.choice(len(rows), size=n, replace=False)
            mats.append(emb[rows[sel]])
        if mats:
            families[f"truncated_{n}_cards"] = score_mats(ctx, mats)

    # (c) all-nonbasic-land decks. The land pool is drawn from the cards of the
    # deck sample only, so it does not depend on which other probes ran (P1
    # registers thousands of extra match decks into the same table).
    sample_rows = np.unique(np.concatenate([ctx.pool.rows[i] for i in sample]))
    land_rows = [int(r) for r in sample_rows if pl.layout.is_land_embedding(emb[r])]
    replace = len(land_rows) < 23
    if land_rows:
        mats = [emb[ctx.rng.choice(land_rows, size=23, replace=replace)]
                for _ in range(n_land_samples)]
        families["23_nonbasic_lands"] = score_mats(ctx, mats)

    # (d) 23 copies of a bomb
    for name in bombs:
        r = ctx.table.row(name)
        families[f"23x_{name}"] = score_mats(ctx, [np.repeat(emb[r][None, :], 23, axis=0)])

    print(f"\n### P5b-d -- degenerate families vs the real-deck band\n")
    print(f"nonbasic-land pool: {len(land_rows)} distinct lands among the "
          f"{len(sample_rows)} distinct cards of the deck sample"
          f"{' (sampled WITH replacement: pool < 23)' if replace else ''}; "
          f"bombs: {', '.join(bombs) if bombs else '(none resolved)'} "
          f"(top-decile shrunk_score_play >= {fmt(thresh, 4)})")
    print("\n| family | n | mean | median | min | max | pct-rank of median in real band |")
    print("|---|---|---|---|---|---|---|")
    fam_rows = []
    for name, s in families.items():
        med = float(np.median(s))
        rec = {"family": name, "n": int(len(s)), "mean": float(s.mean()),
               "median": med, "min": float(s.min()), "max": float(s.max()),
               "pct_rank_of_median": pct_rank(ref, med),
               "frac_above_p99": float((s > pcts["99"]).mean()),
               "frac_below_p1": float((s < pcts["1"]).mean())}
        fam_rows.append(rec)
        print(f"| {name} | {rec['n']} | {fmt(rec['mean'], 3)} | {fmt(med, 3)} | "
              f"{fmt(rec['min'], 3)} | {fmt(rec['max'], 3)} | "
              f"{rec['pct_rank_of_median']:.1f} |")
    print("\n(pct-rank 100 = the degenerate family scores above every real deck; "
          "a well-calibrated scorer should put all of these at the bottom.)")

    return {"real_percentiles": pcts, "real_mean": float(ref.mean()),
            "real_std": float(ref.std()), "n_real": int(len(ref)),
            "families": fam_rows, "bombs": bombs,
            "n_nonbasic_lands_in_pool": len(land_rows),
            "lands_sampled_with_replacement": bool(replace),
            "top_decile_threshold": thresh,
            "family_scores": {k: v.tolist() for k, v in families.items()}}


# ------------------------------------------------------------------------ main

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

    if args.smoke:
        args.device = args.device or "cpu"
        args.decks = args.decks or 40
        args.matches = args.matches or 200
        args.batch = args.batch or 64
        ks = K_GRID_SMOKE
        n_p3, n_p4_decks, n_p4_cards, n_trunc, n_lands = 20, 15, 8, 10, 5
    else:
        args.decks = args.decks or 400
        args.matches = args.matches or 2000
        args.batch = args.batch or 256
        ks = K_GRID
        n_p3, n_p4_decks, n_p4_cards, n_trunc, n_lands = 200, 100, 30, 50, 20

    want = ALL_PROBES if args.probes == "all" else \
        [p.strip().lower() for p in args.probes.split(",") if p.strip()]
    bad = [p for p in want if p not in ALL_PROBES]
    if bad:
        raise SystemExit(f"unknown probe(s): {bad}; choose from {ALL_PROBES}")

    torch.manual_seed(SEED)
    t_start = time.time()
    h1("# T6 -- mechanism probes of the gen-4 sealed scorer")
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
          f"{len(ctx.table.names)} distinct cards; "
          f"{len(ctx.table.missing)} card names unembeddable")
    if not sample:
        raise SystemExit("no usable decks")
    sizes = np.array([len(ctx.pool.rows[i]) for i in sample])
    print(f"nonbasic deck sizes: min {sizes.min()}, median {np.median(sizes):.0f}, "
          f"max {sizes.max()}")

    full_scores = score_mats(ctx, ctx.pool.mats(ctx.table.emb, sample))
    print(f"full-model scores: mean {full_scores.mean():.4f}  "
          f"std {full_scores.std():.4f}  "
          f"range [{full_scores.min():.4f}, {full_scores.max():.4f}]")

    matches: list[tuple[int, int]] = []
    m_meta = {}
    if "p1" in want:
        rows, undecided, malformed = parse_matches(MATCH_FILE, args.matches)
        m_dropped = 0
        for deck_a, deck_b, winner in rows:
            ia, ib = ctx.pool.add(deck_a), ctx.pool.add(deck_b)
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
        "det_dim": pl.layout.FEATURE_COUNT,
        "n_sab_layers": len(ctx.model.sab_layers),
        "n_seeds": int(ctx.model.pma.seeds.shape[1]),
        "sample_lines": lines_read, "n_decks": len(sample),
        "decks_dropped": dropped, "n_distinct_cards": len(ctx.table.names),
        "n_unembeddable": len(ctx.table.missing),
        "unembeddable_sample": sorted(ctx.table.missing)[:20],
        "deck_size_min": int(sizes.min()), "deck_size_max": int(sizes.max()),
        "probes_run": want, "match_corpus": m_meta,
    }, "full_scores": {
        "mean": float(full_scores.mean()), "std": float(full_scores.std()),
        "min": float(full_scores.min()), "max": float(full_scores.max()),
        "values": full_scores.tolist(),
    }}

    if "p1" in want:
        res["p1_pc_truncation"] = probe1(ctx, sample, full_scores, matches, ks)
    if "p2" in want:
        res["p2_oversmoothing"] = probe2(ctx, sample)
    if "p3" in want:
        res["p3_magnitude"] = probe3(ctx, sample, full_scores, n_p3)
    if "p4" in want:
        res["p4_replication"] = probe4(ctx, sample, full_scores,
                                       n_p4_decks, n_p4_cards)
    if "p5" in want:
        res["p5_ood"] = probe5(ctx, sample, full_scores, n_trunc, n_lands)

    res["meta"]["seconds"] = round(time.time() - t_start, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(jsonable(res), f, indent=2)
    print(f"\n\nwrote {args.out}  ({res['meta']['seconds']}s total)")


if __name__ == "__main__":
    main()
