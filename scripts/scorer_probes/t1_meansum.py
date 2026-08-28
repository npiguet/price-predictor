"""T1: mean-vs-sum mechanism + land-add deltas (H46 / H5 / H53).

Takes (pool, deck) pairs from the gen-4 corpus, and for every card still left
in the pool asks the scorer what *adding* that card to the deck does:

    delta_add(c) = score(D + c) - score(D)

A **sum-like** scorer gives delta_add ~ f(q(c)) alone: good cards always help.
A **mean-like** scorer gives delta_add ~ (q(c) - qbar(D)) / (n+1): the same card
helps a weak deck and hurts a strong one. The regression, the strong/weak
context table, and the sign-disagreement fractions below separate the two.

The land subset answers H5: would the greedy builder ever add a dual land, i.e.
is delta_add of an on-color land ever positive (and how does it compare to the
best spells still in the pool)?

Writes t1_add_deltas.csv (one row per context x candidate) and prints the
analysis. Read-only w.r.t. the repo and the Y: drive.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter

import numpy as np
from scipy import stats

import probe_lib as pl

OUT = pl.SCRATCH / "t1_add_deltas.csv"
POOLS = pl.YDATA / "pools" / "pools-gen4-256.txt"
DECKS = pl.YDATA / "decks" / "generated-decks-gen4-256.txt"

WUBRG = "WUBRG"
FC = pl.layout.FEATURE_COUNT
SEED = 42


# --------------------------------------------------------------------------
# shared card-fact section (duplicated in t3_ladders.py by design: the task
# forbids adding a third module to the scratchpad)
# --------------------------------------------------------------------------

class CardBook:
    """Memoized per-name card facts: embedding + classification + label."""

    def __init__(self, probe: pl.Probe, win_rates: dict):
        self.probe = probe
        self.wr = win_rates
        self._cache: dict[str, dict | None] = {}

    def get(self, name: str) -> dict | None:
        """None means 'unusable' (basic land, missing .npz, or missing .txt)."""
        if name in self._cache:
            return self._cache[name]
        info = self._build(name)
        self._cache[name] = info
        return info

    def _build(self, name: str) -> dict | None:
        if name.lower() in pl.BASIC_LAND_NAMES:
            return None
        emb = self.probe.embedding(name)
        if emb is None:
            return None
        text = self.probe.locator.load_text(name)
        if text is None:
            return None
        d = emb[-FC:]
        mana_cost = text.mana_cost_line()
        feats = pl.card_features(text.text)
        return {
            "name": name,
            "emb": np.ascontiguousarray(emb, dtype=np.float32),
            "is_land": float(d[pl.layout.IS_LAND]) > 0.5,
            "mv": float(d[pl.layout.MANA_VALUE]),
            "pips": np.asarray(d[pl.layout.COLOR_PIPS], dtype=np.float64),
            "colors": frozenset(
                c for c, f in zip(WUBRG, d[pl.layout.COLOR_FLAGS]) if float(f) > 0.5
            ),
            "produces": frozenset(
                c for c, f in zip(WUBRG, d[pl.layout.PRODUCES_COLORS]) if float(f) > 0.5
            ),
            # '/' in the mana cost line == hybrid or phyrexian; those contaminate
            # pip counting, so every probe excludes them as swap candidates.
            "hybrid": bool(mana_cost and "/" in mana_cost),
            "is_creature": bool(feats["is_creature"]),
            # split / adventure / transform / meld: the converted text marks the
            # second face with a bare 'ALTERNATE' line ('alternate cost:' is a
            # different thing -- madness and friends -- and must not match).
            "is_multiface": " // " in name or any(
                ln.strip() == "ALTERNATE" for ln in text.text.splitlines()
            ),
            "q": (self.wr.get(name) or {}).get("shrunk_score_play"),
        }


def main_colors(book: CardBook, deck: list[str], k: int = 2) -> tuple[str, ...]:
    """The deck's top-k pip colors (only colors actually present)."""
    pips = np.zeros(5)
    for n in deck:
        info = book.get(n)
        if info is not None:
            pips += info["pips"]
    order = np.argsort(-pips)
    return tuple(WUBRG[i] for i in order[:k] if pips[i] > 0)


def score_streaming(probe: pl.Probe, n: int, build, batch_size: int = 1024,
                    label: str = "") -> np.ndarray:
    """Score n decks whose matrices are materialized one batch at a time.

    Deviation from 'build all matrices first': 400 x 41 stacked 27x544 float32
    matrices is ~1 GB resident. Materializing per batch is numerically identical
    and keeps the footprint at a few MB.
    """
    out = np.empty(n, dtype=np.float64)
    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        out[lo:hi] = probe.score_matrices([build(i) for i in range(lo, hi)],
                                          batch_size=batch_size)
        print(f"  {label} scored {hi}/{n}", flush=True)
    return out


# --------------------------------------------------------------------------
# context sampling
# --------------------------------------------------------------------------

def sample_contexts(book: CardBook, n_ctx: int, rng: random.Random) -> list[dict]:
    """(pool, deck) pairs spread over distinct sets, fully resolvable."""
    pools = list(pl.read_pools(POOLS))
    decks = list(pl.read_generated_decks(DECKS))
    if len(pools) != len(decks):
        raise SystemExit(f"corpus misalignment: {len(pools)} pools vs {len(decks)} decks")

    by_set: dict[str, list[int]] = {}
    for i, ((set_code, _pool), (_lab, deck_set, _deck)) in enumerate(zip(pools, decks)):
        if set_code != deck_set:
            continue
        by_set.setdefault(set_code, []).append(i)
    for v in by_set.values():
        rng.shuffle(v)
    set_order = sorted(by_set)
    rng.shuffle(set_order)

    # round-robin over sets so the sample spreads instead of clumping
    queue: list[int] = []
    depth = 0
    while True:
        added = False
        for s in set_order:
            if depth < len(by_set[s]):
                queue.append(by_set[s][depth])
                added = True
        if not added:
            break
        depth += 1

    out: list[dict] = []
    skipped = 0
    for idx in queue:
        if len(out) >= n_ctx:
            break
        set_code, pool = pools[idx]
        deck = decks[idx][2]
        nonbasic = [c for c in deck if c.lower() not in pl.BASIC_LAND_NAMES]
        if any(book.get(n) is None for n in pool) or any(book.get(n) is None for n in nonbasic):
            skipped += 1
            continue
        avail = Counter(pool) - Counter(nonbasic)
        if not avail:
            skipped += 1
            continue
        out.append({
            "cid": idx,
            "set_code": set_code,
            "deck": nonbasic,
            "avail": avail,
            "main": main_colors(book, nonbasic),
        })
    print(f"sampled {len(out)} contexts over {len({c['set_code'] for c in out})} sets "
          f"({skipped} skipped as unresolvable)", flush=True)
    return out


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------

def mean_se(x) -> tuple[float, float, int]:
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    se = a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else float("nan")
    return float(a.mean()), float(se), int(a.size)


def report_fit(tag: str, x, y) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        print(f"  {tag:<34} n={x.size} (too few)")
        return
    r, p = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    print(f"  {tag:<34} n={x.size:>6}  r={r:+.4f} (p={p:.2e})  "
          f"slope={slope:+.5f}  intercept={intercept:+.6f}  resid_sd={resid.std():.5f}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_rows(probe: pl.Probe, book: CardBook, contexts: list[dict],
               max_cand: int, rng: random.Random) -> list[dict]:
    """One row per (context, candidate); scores base + all adds in one pass."""
    specs: list[tuple[int, str | None]] = []   # (ctx position, candidate or None=base)
    base_mats: list[np.ndarray] = []
    cand_lists: list[list[str]] = []

    for ci, ctx in enumerate(contexts):
        base_mats.append(np.stack([book.get(n)["emb"] for n in ctx["deck"]]))
        cands = sorted(ctx["avail"])           # distinct names; dupes score identically
        if len(cands) > max_cand:
            cands = sorted(rng.sample(cands, max_cand))
        cand_lists.append(cands)
        specs.append((ci, None))
        specs.extend((ci, c) for c in cands)

    def build(i: int) -> np.ndarray:
        ci, cand = specs[i]
        base = base_mats[ci]
        if cand is None:
            return base
        return np.concatenate([base, book.get(cand)["emb"][None, :]], axis=0)

    print(f"scoring {len(specs)} decks ({len(contexts)} bases + adds)...", flush=True)
    scores = score_streaming(probe, len(specs), build, label="t1")

    base_score = {}
    for i, (ci, cand) in enumerate(specs):
        if cand is None:
            base_score[ci] = scores[i]

    rows: list[dict] = []
    for i, (ci, cand) in enumerate(specs):
        if cand is None:
            continue
        ctx = contexts[ci]
        deck = ctx["deck"]
        main = set(ctx["main"])
        qs = [book.get(n)["q"] for n in deck if not book.get(n)["is_land"]]
        qs = [q for q in qs if q is not None]
        qbar = float(np.mean(qs)) if qs else float("nan")
        info = book.get(cand)
        s0 = base_score[ci]

        land_class = ""
        if info["is_land"]:
            if info["produces"] & main:
                land_class = "on_color_land"
            elif info["produces"]:
                land_class = "off_color_land"
            else:
                land_class = "colorless_land"

        rows.append({
            "context_id": ctx["cid"],
            "set_code": ctx["set_code"],
            "n": len(deck),
            "s0": round(float(s0), 6),
            "qbar": round(qbar, 6) if np.isfinite(qbar) else "",
            "card": cand,                      # candidate card NAME
            "q": "" if info["q"] is None else round(float(info["q"]), 6),
            "is_land": int(info["is_land"]),
            "is_creature": int(info["is_creature"]),
            "hybrid": int(info["hybrid"]),
            "is_multiface": int(info["is_multiface"]),
            "on_color": int(info["colors"] <= main),
            "mv": info["mv"],
            "land_class": land_class,
            "main_colors": "".join(sorted(main)),
            "score": round(float(scores[i]), 6),
            "delta_add": round(float(scores[i] - s0), 6),
        })

    # Adding a 24th card carries a large constant size penalty that swamps the
    # per-card quality signal. delta_rel removes each context's own mean delta,
    # isolating "which card is relatively best to add here".
    by_ctx: dict[int, list[float]] = {}
    for r in rows:
        by_ctx.setdefault(r["context_id"], []).append(r["delta_add"])
    ctx_mean = {c: float(np.mean(v)) for c, v in by_ctx.items()}
    for r in rows:
        r["delta_rel"] = round(r["delta_add"] - ctx_mean[r["context_id"]], 6)
    return rows


def analyse(rows: list[dict]) -> None:
    d = np.array([r["delta_add"] for r in rows], dtype=np.float64)
    q = np.array([r["q"] if r["q"] != "" else np.nan for r in rows], dtype=np.float64)
    qbar = np.array([r["qbar"] if r["qbar"] != "" else np.nan for r in rows], dtype=np.float64)
    s0 = np.array([r["s0"] for r in rows], dtype=np.float64)
    is_land = np.array([r["is_land"] for r in rows], dtype=bool)
    n_cards = np.array([r["n"] for r in rows], dtype=np.float64)
    dr = np.array([r["delta_rel"] for r in rows], dtype=np.float64)
    mv = np.array([r["mv"] for r in rows], dtype=np.float64)

    print()
    print("=" * 78)
    print(f"T1  ADD-DELTA ANALYSIS   rows={len(rows)}  "
          f"contexts={len({r['context_id'] for r in rows})}")
    print("=" * 78)

    m, se, n = mean_se(d)
    print(f"\ndelta_add overall: mean={m:+.5f} +/- {se:.5f} (n={n})  "
          f"frac>0={float((d > 0).mean()):.3f}  "
          f"median={np.median(d):+.5f}  sd={d.std():.5f}")
    print(f"deck size n: {int(n_cards.min())}..{int(n_cards.max())}   "
          f"base score s0: {s0.min():.3f}..{s0.max():.3f}")

    print("\n-- 1. MECHANISM: raw quality vs mean-centered quality --------------")
    report_fit("delta_add ~ q(c)", q, d)
    report_fit("delta_add ~ (q(c) - qbar(D))", q - qbar, d)
    report_fit("delta_add ~ (q-qbar)/(n+1)", (q - qbar) / (n_cards + 1.0), d)
    print("  [mean-like => centered fits better and intercept ~ 0;"
          " sum-like => raw q fits and deltas are mostly > 0 for q>0]")
    print("  -- same fits on delta_rel (context's own mean delta removed, so the")
    print("     constant 'one more card' size penalty cannot mask the q signal):")
    report_fit("delta_rel ~ q(c)", q, dr)
    report_fit("delta_rel ~ (q(c) - qbar(D))", q - qbar, dr)
    report_fit("delta_rel ~ mv(c)", mv, dr)
    report_fit("delta_rel ~ is_land(c)", is_land.astype(float), dr)

    print("\n-- 2. KEY TABLE: same candidate quality, strong vs weak context ----")
    ctx_s0 = {}
    for r in rows:
        ctx_s0[r["context_id"]] = r["s0"]
    vals = np.array(sorted(ctx_s0.values()))
    lo_cut, hi_cut = np.percentile(vals, [33.3333, 66.6667])
    ctx_bucket = {c: ("weak" if v <= lo_cut else "strong" if v >= hi_cut else "mid")
                  for c, v in ctx_s0.items()}
    bucket = np.array([ctx_bucket[r["context_id"]] for r in rows])
    print(f"  context s0 terciles: weak <= {lo_cut:.3f} < mid < {hi_cut:.3f} <= strong")
    q_buckets = [("decent  q in [.05,.15]", 0.05, 0.15),
                 ("good    q in [.02,.05)", 0.02, 0.05),
                 ("filler  q in [-.02,.02)", -0.02, 0.02),
                 ("bad     q < -.02", -9.0, -0.02)]
    for metric, arr in (("delta_add", d), ("delta_rel", dr)):
        print(f"  [{metric}]")
        print(f"  {'q(c) bucket':<24} {'strong ctx':>22} {'weak ctx':>22} {'strong-weak':>13}")
        for label, lo, hi in q_buckets:
            if label.startswith("decent"):
                sel = np.isfinite(q) & (q >= lo) & (q <= hi)
            else:
                sel = np.isfinite(q) & (q >= lo) & (q < hi)
            ms, ss, ns = mean_se(arr[sel & (bucket == "strong")])
            mw, sw, nw = mean_se(arr[sel & (bucket == "weak")])
            print(f"  {label:<24} {ms:+.5f}+/-{ss:.5f} n={ns:<5} "
                  f"{mw:+.5f}+/-{sw:.5f} n={nw:<5} {ms - mw:+.5f}")
    print("  [sign flip across the strong/weak columns at fixed q = mean-like]")

    print("\n-- 3. SIGN DISAGREEMENT vs the deck mean ---------------------------")
    ok = np.isfinite(q) & np.isfinite(qbar)
    above = ok & (q > qbar)
    below = ok & (q < qbar)
    if above.sum():
        print(f"  q(c) > qbar(D) but delta_add < 0 : "
              f"{float((d[above] < 0).mean()):.3f}  (n={int(above.sum())})")
    if below.sum():
        print(f"  q(c) < qbar(D) but delta_add > 0 : "
              f"{float((d[below] > 0).mean()):.3f}  (n={int(below.sum())})")
    if ok.sum():
        agree = ((q[ok] > qbar[ok]) == (d[ok] > 0)).mean()
        print(f"  sign(q-qbar) == sign(delta_add)   : {float(agree):.3f}")

    print("\n-- 4. LAND SUBSET (H5: would the greedy ever add a dual?) ----------")
    lc = np.array([r["land_class"] for r in rows])
    for cls in ("on_color_land", "off_color_land", "colorless_land"):
        sel = lc == cls
        m2, se2, n2 = mean_se(d[sel])
        frac = float((d[sel] > 0).mean()) if sel.sum() else float("nan")
        print(f"  {cls:<16} mean={m2:+.5f} +/- {se2:.5f}  n={n2:<5} frac(delta>0)={frac:.3f}")
    spell = ~is_land & np.isfinite(q)
    if spell.sum() > 10:
        cut = np.percentile(q[spell], 90)
        best = spell & (q >= cut)
        mb, seb, nb = mean_se(d[best])
        mo, seo, no = mean_se(d[lc == "on_color_land"])
        print(f"  best-10% spells (q >= {cut:.4f}) mean={mb:+.5f} +/- {seb:.5f} n={nb}")
        print(f"  on-color land  - best-10% spell gap = {mo - mb:+.5f}")
    has_land: dict[int, bool] = {}
    land_pos: dict[int, bool] = {}
    any_pos_by_ctx: dict[int, bool] = {}
    for r in rows:
        cid = r["context_id"]
        pos = r["delta_add"] > 0
        any_pos_by_ctx[cid] = any_pos_by_ctx.get(cid, False) or pos
        if r["is_land"]:
            has_land[cid] = True
            land_pos[cid] = land_pos.get(cid, False) or pos
    n_ctx = len(any_pos_by_ctx)
    n_ctx_with_land = len(has_land)
    n_ctx_land_pos = sum(land_pos.values())
    print(f"  contexts with >=1 land candidate at delta_add > 0: "
          f"{n_ctx_land_pos}/{n_ctx_with_land} (of {n_ctx} contexts; "
          f"{n_ctx - n_ctx_with_land} had no land candidate)")
    print(f"  contexts with >=1 candidate (any type) at delta_add > 0: "
          f"{sum(any_pos_by_ctx.values())}/{n_ctx}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="20 contexts on CPU")
    ap.add_argument("--contexts", type=int, default=400)
    ap.add_argument("--max-cand", type=int, default=40)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    n_ctx = 20 if args.smoke else args.contexts
    device = args.device or ("cpu" if args.smoke else None)

    rng = random.Random(SEED)
    probe = pl.Probe(device=device)
    print(f"device={probe.device} contexts={n_ctx} max_cand={args.max_cand}", flush=True)
    book = CardBook(probe, pl.load_win_rates())

    contexts = sample_contexts(book, n_ctx, rng)
    rows = build_rows(probe, book, contexts, args.max_cand, rng)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)", flush=True)

    analyse(rows)


if __name__ == "__main__":
    main()
