"""T2: per-card marginal value under the production sealed scorer.

Two complementary measurements, both written to one per-card CSV:

Stage 1 -- leave-one-out (LOO) in *realistic* contexts.
    For each collected deck D (non-basic cards only) score D and every D\\{c_i}.
    delta_i = s(D) - s(D\\{c_i}) is the card's raw marginal contribution; the
    within-deck centred value drel_i = delta_i - mean_j(delta_j) removes the
    deck-size effect exactly (every variant has the same card count, so the
    "one card fewer" component is common to all i and cancels).

Stage 2 -- standardized swap-in value on fixed contexts.
    40 fixed two-colour forge-best context decks (10 colour pairs x 4 sets).
    In each, the "median-LOO-delta" spell slot is the replaceable slot. For a
    card c eligible in that context (its colour flags fit the pair),
    v_swap(c, D) = s(D[slot -> c]) - s(D). v_swap(c) averages over contexts.

Read-only w.r.t. the repo and Y:. GPU-friendly; --smoke runs tiny on CPU.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import probe_lib as pl

OUT_DEFAULT = pl.SCRATCH / "t2_card_values.csv"

MATCH_FILES = [
    "matches-b07/match-outcomes-gen0.txt.gz",
    "matches-b07/match-outcomes-gen1-vs-0.txt.gz",
    "matches-b07/match-outcomes-gen2-vs-0-1.txt.gz",
    "matches-b07/match-outcomes-gen3-vs-0-2.txt",
    "matches-b07/match-outcomes-gen3-vs-forge-best.txt",
    "matches-b07/match-outcomes-gen4-vs-forge-best-gen3.txt",
    "matches-b07/match-outcomes-gen5-vs-gen4-forge.txt",
]

KEEP_METHODS = ("forge-best", "forge-3sub", "forge-8sub", "gen3-256", "gen4-512", "gen5")
CONTEXT_METHOD = "forge-best"

WUBRG = ("W", "U", "B", "R", "G")
PAIRS = [
    ("W", "U"), ("W", "B"), ("W", "R"), ("W", "G"),
    ("U", "B"), ("U", "R"), ("U", "G"),
    ("B", "R"), ("B", "G"), ("R", "G"),
]

MIN_DECK_CARDS = 15         # skip degenerate / truncated decks (~3% of forge-* builds)
MAX_DECK_CARDS = 60
CONTEXT_MIN_CARDS = 21      # contexts must sit in the training-distribution shape
CONTEXT_MAX_CARDS = 26
LOO_GROUP = 128             # decks per scoring group (memory-bounded)
SWAP_CHUNK = 2048           # swapped matrices built per scoring call


# --------------------------------------------------------------------------
# corpus collection (same pattern as t0_landscape.collect_decks)
# --------------------------------------------------------------------------

def iter_match_lines(path: Path, max_lines: int | None = None):
    op = gzip.open if path.suffix == ".gz" else open
    n = 0
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(";")
            if len(parts) >= 10:
                yield parts
            n += 1
            if max_lines and n >= max_lines:
                return


def collect_decks(per_method_cap: int, max_lines: int | None = None):
    """Dedup by content, cap per method. Returns {method: [(set_code, deck)]}."""
    seen: set[str] = set()
    per_method: dict[str, list] = {m: [] for m in KEEP_METHODS}
    for rel in MATCH_FILES:
        path = pl.YDATA / rel
        if not path.exists():
            print("missing:", path, flush=True)
            continue
        if all(len(v) >= per_method_cap for v in per_method.values()):
            break
        for parts in iter_match_lines(path, max_lines):
            set_code = parts[2]
            for m_idx, d_idx in ((3, 5), (4, 6)):
                method = parts[m_idx]
                bucket = per_method.get(method)
                if bucket is None or len(bucket) >= per_method_cap:
                    continue
                deck = parts[d_idx].split("|")
                key = hashlib.md5(("|".join(sorted(deck)) + set_code).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                bucket.append((set_code, deck))
    print("collected:", {m: len(v) for m, v in per_method.items()}, flush=True)
    return per_method


def round_robin(per_method: dict[str, list], total_cap: int):
    """Interleave methods so the LOO sample is balanced. -> [(method, set, deck)]"""
    out = []
    i = 0
    while len(out) < total_cap:
        added = False
        for m in KEEP_METHODS:
            bucket = per_method.get(m, [])
            if i < len(bucket) and len(out) < total_cap:
                set_code, deck = bucket[i]
                out.append((m, set_code, deck))
                added = True
        if not added:
            break
        i += 1
    return out


# --------------------------------------------------------------------------
# helpers on deck matrices
# --------------------------------------------------------------------------

def det_block(mat: np.ndarray) -> np.ndarray:
    return mat[:, -pl.layout.FEATURE_COUNT:]


def land_mask(mat: np.ndarray) -> np.ndarray:
    return det_block(mat)[:, pl.layout.IS_LAND] > 0.5


def deck_color_identity(mat: np.ndarray) -> frozenset:
    """Union of pip colours over the deck's spells (non-land rows)."""
    det = det_block(mat)
    spells = ~(det[:, pl.layout.IS_LAND] > 0.5)
    if not spells.any():
        return frozenset()
    pips = det[spells, pl.layout.COLOR_PIPS]
    present = pips.sum(axis=0) > 0
    return frozenset(c for c, f in zip(WUBRG, present) if f)


def card_colors(emb: np.ndarray) -> frozenset:
    flags = emb[-pl.layout.FEATURE_COUNT:][pl.layout.COLOR_FLAGS]
    return frozenset(c for c, f in zip(WUBRG, flags) if float(f) > 0.5)


def build_deck_matrix(probe: pl.Probe, deck: list[str]):
    """(names, matrix) for the non-basic part of a deck, or None if unresolvable."""
    names = [c for c in deck if c.lower() not in pl.BASIC_LAND_NAMES]
    if not (MIN_DECK_CARDS <= len(names) <= MAX_DECK_CARDS):
        return None
    rows = []
    for n in names:
        e = probe.embedding(n)
        if e is None:
            return None
        rows.append(e)
    return names, np.stack(rows).astype(np.float32)


def loo_deltas(probe: pl.Probe, mat: np.ndarray, rows: np.ndarray | None = None,
               batch: int = 1024):
    """(base_score, deltas) where deltas[k] = s(D) - s(D minus rows[k])."""
    idx = np.arange(mat.shape[0]) if rows is None else np.asarray(rows)
    mats = [mat] + [np.delete(mat, i, axis=0) for i in idx]
    s = probe.score_matrices(mats, batch_size=batch)
    return float(s[0]), s[0] - s[1:]


# --------------------------------------------------------------------------
# stage 1: leave-one-out over realistic decks
# --------------------------------------------------------------------------

def stage1_loo(probe: pl.Probe, decks, batch: int):
    """-> {name: [sum_drel, sum_delta, count, sum_deck_score]}"""
    acc: dict[str, list] = defaultdict(lambda: [0.0, 0.0, 0, 0.0])
    t0 = time.time()
    n_ok = n_skip = n_fwd = 0

    for start in range(0, len(decks), LOO_GROUP):
        group = decks[start:start + LOO_GROUP]
        prepared, mats = [], []
        for _method, _set_code, deck in group:
            built = build_deck_matrix(probe, deck)
            if built is None:
                n_skip += 1
                continue
            names, mat = built
            prepared.append((names, mat.shape[0]))
            mats.append(mat)
            for i in range(mat.shape[0]):
                mats.append(np.delete(mat, i, axis=0))
        if not mats:
            continue
        scores = probe.score_matrices(mats, batch_size=batch)
        n_fwd += len(mats)

        pos = 0
        for names, n in prepared:
            base = float(scores[pos])
            deltas = base - scores[pos + 1: pos + 1 + n]
            pos += 1 + n
            drel = deltas - deltas.mean()
            for name, d, dr in zip(names, deltas, drel):
                a = acc[name]
                a[0] += float(dr)
                a[1] += float(d)
                a[2] += 1
                a[3] += base
            n_ok += 1

        el = time.time() - t0
        done = start + len(group)
        print(f"  LOO {done}/{len(decks)} decks  ok={n_ok} skip={n_skip} "
              f"fwd={n_fwd}  {el:.0f}s  ({n_fwd / max(el, 1e-6):.0f} fwd/s)", flush=True)

    print(f"stage 1 done: {n_ok} decks, {n_skip} skipped, {n_fwd} forwards, "
          f"{len(acc)} distinct cards, {time.time() - t0:.0f}s", flush=True)
    return acc


# --------------------------------------------------------------------------
# stage 2: fixed contexts + swap-in values
# --------------------------------------------------------------------------

def pick_contexts(probe: pl.Probe, forge_best, per_pair: int, batch: int):
    """Choose per_pair context decks per colour pair, each from a different set."""
    buckets: dict[tuple, list] = {p: [] for p in PAIRS}
    used_sets: dict[tuple, set] = {p: set() for p in PAIRS}
    n_seen = 0
    for set_code, deck in forge_best:
        if all(len(v) >= per_pair for v in buckets.values()):
            break
        built = build_deck_matrix(probe, deck)
        n_seen += 1
        if built is None:
            continue
        names, mat = built
        if not (CONTEXT_MIN_CARDS <= mat.shape[0] <= CONTEXT_MAX_CARDS):
            continue
        ident = deck_color_identity(mat)
        if len(ident) != 2:
            continue
        pair = tuple(c for c in WUBRG if c in ident)
        b = buckets.get(pair)
        if b is None or len(b) >= per_pair or set_code in used_sets[pair]:
            continue
        used_sets[pair].add(set_code)
        b.append((set_code, names, mat))

    contexts = []
    for pair in PAIRS:
        for set_code, names, mat in buckets[pair]:
            spells = np.flatnonzero(~land_mask(mat))
            if len(spells) < 4:
                continue
            base, deltas = loo_deltas(probe, mat, spells, batch)
            order = np.argsort(deltas, kind="stable")
            slot = int(spells[order[len(order) // 2]])
            contexts.append({
                "pair": "".join(pair),
                "pair_set": frozenset(pair),
                "set_code": set_code,
                "matrix": mat,
                "slot": slot,
                "slot_card": names[slot],
                "base": base,
                "median_delta": float(deltas[order[len(order) // 2]]),
            })
            print(f"  context {contexts[-1]['pair']} {set_code} n={mat.shape[0]} "
                  f"base={base:+.4f} slot={slot} ({names[slot]}) "
                  f"med_delta={contexts[-1]['median_delta']:+.4f}", flush=True)
    print(f"picked {len(contexts)} contexts from {n_seen} forge-best decks", flush=True)
    return contexts


def build_universe(probe: pl.Probe, wr: dict, min_obs: int, limit: int | None):
    """Non-land cards with enough empirical observations and a resolvable embedding."""
    scored = []
    for name, rec in wr.items():
        w = rec.get("wins_when_in_deck") or 0.0
        l = rec.get("losses_when_in_deck") or 0.0
        n_obs = w + l
        if n_obs >= min_obs:
            scored.append((-n_obs, name))
    scored.sort()

    names, embs, colors = [], [], []
    n_missing = n_land = 0
    for _neg, name in scored:
        e = probe.embedding(name)
        if e is None:
            n_missing += 1
            continue
        if float(e[-pl.layout.FEATURE_COUNT:][pl.layout.IS_LAND]) > 0.5:
            n_land += 1
            continue
        names.append(name)
        embs.append(e.astype(np.float32))
        colors.append(card_colors(e))
        if limit and len(names) >= limit:
            break
    print(f"universe: {len(names)} cards (min_obs={min_obs}; "
          f"{n_missing} unresolvable, {n_land} lands dropped)", flush=True)
    return names, embs, colors


def stage2_swap(probe: pl.Probe, contexts, names, embs, colors, swap_cap: int, batch: int):
    """-> (sum_v, count) arrays aligned with `names`."""
    n = len(names)
    v_sum = np.zeros(n)
    v_cnt = np.zeros(n, dtype=np.int32)
    if not contexts:
        print("no contexts available -- skipping stage 2", flush=True)
        return v_sum, v_cnt

    # eligibility: context pair must cover the card's colour flags
    per_context: list[list[int]] = [[] for _ in contexts]
    n_ineligible = 0
    for ci in range(n):
        cc = colors[ci]
        used = 0
        for k, ctx in enumerate(contexts):
            if cc <= ctx["pair_set"]:
                per_context[k].append(ci)
                used += 1
                if used >= swap_cap:
                    break
        if used == 0:
            n_ineligible += 1
    print(f"eligibility: {n_ineligible} cards have no eligible context "
          f"(3+ colours); total swaps = {sum(len(x) for x in per_context)}", flush=True)

    t0 = time.time()
    done = 0
    total = sum(len(x) for x in per_context)
    for k, ctx in enumerate(contexts):
        cand = per_context[k]
        if not cand:
            continue
        mat, slot, base = ctx["matrix"], ctx["slot"], ctx["base"]
        for lo in range(0, len(cand), SWAP_CHUNK):
            chunk = cand[lo:lo + SWAP_CHUNK]
            mats = []
            for ci in chunk:
                m = mat.copy()
                m[slot] = embs[ci]
                mats.append(m)
            s = probe.score_matrices(mats, batch_size=batch)
            v_sum[chunk] += s - base
            v_cnt[chunk] += 1
            done += len(chunk)
        el = time.time() - t0
        print(f"  swap ctx {k + 1}/{len(contexts)} {ctx['pair']}/{ctx['set_code']} "
              f"cards={len(cand)}  {done}/{total}  {el:.0f}s "
              f"({done / max(el, 1e-6):.0f} fwd/s)", flush=True)
    return v_sum, v_cnt


# --------------------------------------------------------------------------
# metadata + output
# --------------------------------------------------------------------------

DET_KEYS = ["is_land", "pips_w", "pips_u", "pips_b", "pips_r", "pips_g",
            "generic", "mv", "colors", "power", "toughness", "produces_any"]
DET_RENAME = {"is_land": "det_is_land", "power": "det_power", "toughness": "det_toughness"}

FEAT_KEYS = ["is_creature", "is_land", "is_instant", "is_sorcery", "is_artifact",
             "is_enchantment", "is_planeswalker", "is_aura", "is_equipment",
             "power", "toughness", "has_evasion", "has_flying", "combat_kw_count",
             "is_removal", "draws_cards", "vanilla"]

LABEL_KEYS = ["shrunk_score_play", "shrunk_score_draw", "shrunk_played_rate",
              "shrunk_cast_lift"]

COLUMNS = (
    ["name", "n_loo_contexts", "v_loo_rel", "v_loo_raw", "mean_deck_score",
     "n_swap_contexts", "v_swap"]
    + [DET_RENAME.get(k, k) for k in DET_KEYS]
    + FEAT_KEYS
    + LABEL_KEYS
    + ["n_obs", "rarity", "release_year", "printings_count"]
)


def load_rarity_map():
    try:
        from price_predictor.infrastructure.mtgjson_loader import build_metadata_map
        metadata, _prices = build_metadata_map(
            pl.REPO / "resources/AllPrintings.json",
            pl.REPO / "resources/AllPricesToday.json",
        )
        print(f"rarity metadata: {len(metadata)} names", flush=True)
        return metadata
    except Exception as exc:  # missing files, schema change, OOM -- blanks are fine
        print(f"rarity metadata unavailable ({type(exc).__name__}: {exc})", flush=True)
        return {}


def num(v):
    """CSV-safe scalar: bools -> 0/1, floats rounded, None -> ''."""
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not np.isfinite(f):
        return ""
    return round(f, 6)


def write_rows(out: Path, probe: pl.Probe, loo_acc, names, v_sum, v_cnt, wr, rarity):
    swap_idx = {n: i for i, n in enumerate(names)}
    all_names = sorted(set(loo_acc) | set(names))

    rows = []
    n_no_text = n_no_emb = 0
    for i, name in enumerate(all_names):
        if i and i % 5000 == 0:
            print(f"  metadata {i}/{len(all_names)}", flush=True)
        row = {c: "" for c in COLUMNS}
        row["name"] = name

        a = loo_acc.get(name)
        if a and a[2] > 0:
            row["n_loo_contexts"] = a[2]
            row["v_loo_rel"] = num(a[0] / a[2])
            row["v_loo_raw"] = num(a[1] / a[2])
            row["mean_deck_score"] = num(a[3] / a[2])
        else:
            row["n_loo_contexts"] = 0

        i = swap_idx.get(name)
        if i is not None and v_cnt[i] > 0:
            row["n_swap_contexts"] = int(v_cnt[i])
            row["v_swap"] = num(v_sum[i] / v_cnt[i])
        else:
            row["n_swap_contexts"] = 0

        emb = probe.embedding(name)
        if emb is None:
            n_no_emb += 1
        else:
            det = pl.det_features(emb)
            for k in DET_KEYS:
                row[DET_RENAME.get(k, k)] = num(det[k])

        text = probe.locator.load_text(name)
        if text is None:
            n_no_text += 1
        else:
            feats = pl.card_features(text.text)
            for k in FEAT_KEYS:
                row[k] = num(feats[k])

        rec = wr.get(name)
        if rec:
            for k in LABEL_KEYS:
                row[k] = num(rec.get(k))
            w = rec.get("wins_when_in_deck") or 0.0
            l = rec.get("losses_when_in_deck") or 0.0
            row["n_obs"] = int(w + l)

        pdata = rarity.get(name)
        if pdata is not None:
            row["rarity"] = getattr(pdata, "rarity", "")
            row["release_year"] = getattr(pdata, "release_year", "")
            row["printings_count"] = getattr(pdata, "printings_count", "")

        rows.append(row)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} -- {len(rows)} rows "
          f"({n_no_emb} without embedding, {n_no_text} without card text)", flush=True)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loo-decks", type=int, default=3500)
    ap.add_argument("--swap-cap", type=int, default=8)
    ap.add_argument("--min-obs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--contexts-per-pair", type=int, default=4)
    ap.add_argument("--universe-limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU run: 30 decks, 200 cards, 1 context per pair")
    args = ap.parse_args()

    max_lines = None
    if args.smoke:
        args.loo_decks = 30
        args.universe_limit = 200
        args.contexts_per_pair = 1
        args.device = args.device or "cpu"
        args.batch = min(args.batch, 256)
        max_lines = 4000
        if args.out == OUT_DEFAULT:
            args.out = pl.SCRATCH / "t2_card_values_smoke.csv"

    t_start = time.time()
    print(f"t2_marginal_values: loo_decks={args.loo_decks} swap_cap={args.swap_cap} "
          f"min_obs={args.min_obs} smoke={args.smoke}", flush=True)

    probe = pl.Probe(device=args.device)
    print(f"scorer on {probe.device}, d_model={probe.d_model}", flush=True)
    wr = pl.load_win_rates()
    print(f"win-rate labels: {len(wr)} cards", flush=True)

    per_method_cap = max(args.loo_decks, 400)
    per_method = collect_decks(per_method_cap, max_lines)
    loo_decks = round_robin(per_method, args.loo_decks)
    print(f"LOO sample: {len(loo_decks)} decks", flush=True)

    print("--- stage 1: leave-one-out ---", flush=True)
    loo_acc = stage1_loo(probe, loo_decks, args.batch)

    print("--- stage 2: fixed contexts ---", flush=True)
    contexts = pick_contexts(probe, per_method.get(CONTEXT_METHOD, []),
                             args.contexts_per_pair, args.batch)
    names, embs, colors = build_universe(probe, wr, args.min_obs, args.universe_limit)
    v_sum, v_cnt = stage2_swap(probe, contexts, names, embs, colors,
                               args.swap_cap, args.batch)
    covered = int((v_cnt > 0).sum())
    if covered:
        vals = v_sum[v_cnt > 0] / v_cnt[v_cnt > 0]
        print(f"v_swap: {covered} cards, mean={vals.mean():+.4f} sd={vals.std():.4f} "
              f"range=[{vals.min():+.4f}, {vals.max():+.4f}]", flush=True)

    print("--- metadata + output ---", flush=True)
    rarity = load_rarity_map()
    write_rows(args.out, probe, loo_acc, names, v_sum, v_cnt, wr, rarity)
    print(f"total {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
