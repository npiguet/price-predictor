"""T3: controlled shape ladders (H1 / H2 / H3 / H4 / H27 / H7).

Each ladder takes a real (pool, deck) context and walks the deck along ONE
shape axis by swapping cards drawn from that pool's own leftovers, holding
everything else as fixed as the pool allows:

    A  color count      k = 0..4 on-color spells -> mono-colored spells of a new color
    B  creatures        k = -4..+4 non-creature <-> creature (MV-matched)
    C  curve shift      k = -4..+4 spells -> same-color spells >=2 MV cheaper / pricier
    D  mean-preserving  {3,3} -> {2,4} spread (and the reverse) vs a same-MV control
    E  splash threshold 1..3 single-pip spells of a 3rd color added to a 2-color deck
    F  fixing x splash  2x2: base / +splash spell / +fixer land / both

Every rung is scored fresh; deltas are paired against that context's own rung 0,
so the per-rung t-stat IS mean/SE of the paired difference. Hybrid and phyrexian
cards ('/' in the mana cost line) are excluded everywhere -- they would corrupt
the pip counts the realized-color and splash measures depend on.

Writes t3_ladders.csv (ladder, context_id, rung, realized_x, score, delta) and
prints one summary table per ladder. Read-only w.r.t. the repo and Y:.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter

import numpy as np

import probe_lib as pl

OUT = pl.SCRATCH / "t3_ladders.csv"
POOLS = pl.YDATA / "pools" / "pools-gen4-256.txt"
DECKS = pl.YDATA / "decks" / "generated-decks-gen4-256.txt"

WUBRG = "WUBRG"
FC = pl.layout.FEATURE_COUNT
SEED = 42


# --------------------------------------------------------------------------
# shared card-fact section (duplicated from t1_meansum.py by design: the task
# forbids adding a third module to the scratchpad)
# --------------------------------------------------------------------------

class CardBook:
    """Memoized per-name card facts: embedding + classification + label."""

    def __init__(self, probe: pl.Probe, win_rates: dict):
        self.probe = probe
        self.wr = win_rates
        self._cache: dict[str, dict | None] = {}

    def get(self, name: str) -> dict | None:
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
        q = (self.wr.get(name) or {}).get("shrunk_score_play")
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
            "hybrid": bool(mana_cost and "/" in mana_cost),
            "is_creature": bool(feats["is_creature"]),
            # split / adventure / transform / meld: the converted text marks the
            # second face with a bare 'ALTERNATE' line ('alternate cost:' is a
            # different thing -- madness and friends -- and must not match).
            "is_multiface": " // " in name or any(
                ln.strip() == "ALTERNATE" for ln in text.text.splitlines()
            ),
            "q": q,
            "qv": 0.0 if q is None else float(q),   # ordering fallback
        }


def deck_pips(book: CardBook, deck: list[str]) -> np.ndarray:
    pips = np.zeros(5)
    for n in deck:
        info = book.get(n)
        if info is not None:
            pips += info["pips"]
    return pips


def main_colors(book: CardBook, deck: list[str], k: int = 2) -> tuple[str, ...]:
    pips = deck_pips(book, deck)
    order = np.argsort(-pips)
    return tuple(WUBRG[i] for i in order[:k] if pips[i] > 0)


def score_streaming(probe: pl.Probe, n: int, build, batch_size: int = 1024,
                    label: str = "") -> np.ndarray:
    out = np.empty(n, dtype=np.float64)
    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        out[lo:hi] = probe.score_matrices([build(i) for i in range(lo, hi)],
                                          batch_size=batch_size)
        print(f"  {label} scored {hi}/{n}", flush=True)
    return out


def mean_se(x) -> tuple[float, float, int]:
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    se = a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else float("nan")
    return float(a.mean()), float(se), int(a.size)


# --------------------------------------------------------------------------
# context sampling
# --------------------------------------------------------------------------

def ef_eligible(ctx: dict, book: CardBook) -> tuple[bool, bool]:
    """(ladder-E eligible, ladder-F eligible) for this context."""
    if ctx["n_colors"] != 2:
        return False, False
    return (_splash_setup(ctx, book) is not None,
            _splash_setup(ctx, book, require_land=True) is not None)


def sample_contexts(book: CardBook, n_ctx: int, n_ef: int, scan_cap: int,
                    rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Return (base contexts for ladders A-D, extra contexts for E/F only).

    Ladders E and F are conditioned on the deck being exactly 2-colour (and, for
    F, on a 3rd-colour fixer land surviving in the pool) -- only ~23% and ~10%
    of contexts qualify. Rather than reporting n=25 for F, the sampler keeps
    scanning past the base block for eligible contexts until each ladder hits
    its target or `scan_cap` contexts have been examined.
    """
    pools = list(pl.read_pools(POOLS))
    decks = list(pl.read_generated_decks(DECKS))
    if len(pools) != len(decks):
        raise SystemExit(f"corpus misalignment: {len(pools)} pools vs {len(decks)} decks")

    by_set: dict[str, list[int]] = {}
    for i, ((set_code, _pool), (_lab, deck_set, _deck)) in enumerate(zip(pools, decks)):
        if set_code == deck_set:
            by_set.setdefault(set_code, []).append(i)
    for v in by_set.values():
        rng.shuffle(v)
    set_order = sorted(by_set)
    rng.shuffle(set_order)

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

    base: list[dict] = []
    extra: list[dict] = []
    need_e = need_f = n_ef
    scanned = skipped = 0
    for idx in queue:
        if scanned >= scan_cap:
            break
        if len(base) >= n_ctx and need_e <= 0 and need_f <= 0:
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
        scanned += 1
        ctx = {
            "cid": idx,
            "set_code": set_code,
            "deck": nonbasic,
            "avail": avail,
            "main": main_colors(book, nonbasic),
            "n_colors": int((deck_pips(book, nonbasic) > 0).sum()),
        }
        e_ok, f_ok = ef_eligible(ctx, book)
        ctx["e_ok"], ctx["f_ok"] = e_ok, f_ok
        if len(base) < n_ctx:
            base.append(ctx)
        elif (e_ok and need_e > 0) or (f_ok and need_f > 0):
            extra.append(ctx)
        else:
            continue
        need_e -= int(e_ok)
        need_f -= int(f_ok)
    print(f"sampled {len(base)} base contexts (+{len(extra)} extra for E/F) over "
          f"{len({c['set_code'] for c in base + extra})} sets; scanned {scanned}, "
          f"{skipped} unresolvable", flush=True)
    return base, extra


# --------------------------------------------------------------------------
# swap machinery
# --------------------------------------------------------------------------

def pool_cards(ctx: dict, book: CardBook, pred) -> list[str]:
    """Remaining-pool names WITH multiplicity, non-hybrid, matching pred."""
    out: list[str] = []
    for name, cnt in sorted(ctx["avail"].items()):
        info = book.get(name)
        if info is None or info["hybrid"]:
            continue
        if pred(info):
            out.extend([name] * cnt)
    return out


def deck_spell_idx(ctx: dict, book: CardBook) -> list[int]:
    """Indices of non-land, non-hybrid deck cards (legal swap-out slots)."""
    return [i for i, n in enumerate(ctx["deck"])
            if not book.get(n)["is_land"] and not book.get(n)["hybrid"]]


def mv_tier(out_i: dict, cand: dict, tight: float = 1.0, loose: float = 2.0):
    """0 = |dMV| <= tight (preferred), 1 = <= loose (relaxed), None = reject."""
    d = abs(cand["mv"] - out_i["mv"])
    if d <= tight:
        return 0
    if d <= loose:
        return 1
    return None


def color_tier(out_i: dict, cand: dict, main: set):
    """0 = identical color set, 1 = merely on-color, None = reject."""
    if cand["colors"] == out_i["colors"]:
        return 0
    if cand["colors"] <= main:
        return 1
    return None


def build_pairs(deck: list[str], outs: list[int], cands: list[str], book: CardBook,
                kmax: int, pair_key) -> list[tuple[int, str]]:
    """Greedily match swap-out slots to pool candidates.

    `pair_key(out_info, cand_info)` returns a sort key (lower = better) or None
    to reject the pairing. Out slots are consumed in the order given; an out
    slot with no legal candidate is skipped (the next one is tried) rather than
    aborting the ladder.
    """
    used: set[int] = set()
    pairs: list[tuple[int, str]] = []
    for i in outs:
        if len(pairs) >= kmax:
            break
        oi = book.get(deck[i])
        best = None
        for j, name in enumerate(cands):
            if j in used:
                continue
            key = pair_key(oi, book.get(name))
            if key is None:
                continue
            if best is None or key < best[0]:
                best = (key, j)
        if best is None:
            continue
        used.add(best[1])
        pairs.append((i, cands[best[1]]))
    return pairs


def apply_pairs(deck: list[str], pairs: list[tuple[int, str]]) -> list[str]:
    d = list(deck)
    for i, name in pairs:
        d[i] = name
    return d


# realized-shape measures ---------------------------------------------------

def rx_colors(book, deck):
    return float((deck_pips(book, deck) > 0).sum())


def rx_creatures(book, deck):
    return float(sum(1 for n in deck
                     if book.get(n)["is_creature"] and not book.get(n)["is_land"]))


def rx_mean_mv(book, deck):
    mvs = [book.get(n)["mv"] for n in deck if not book.get(n)["is_land"]]
    return float(np.mean(mvs)) if mvs else float("nan")


def rx_var_mv(book, deck):
    mvs = [book.get(n)["mv"] for n in deck if not book.get(n)["is_land"]]
    return float(np.var(mvs)) if mvs else float("nan")


def rx_pips_of(book, deck, color):
    idx = WUBRG.index(color)
    return float(deck_pips(book, deck)[idx])


# --------------------------------------------------------------------------
# ladders: each returns [(rung, realized_x, deck_names), ...] or [] to drop
# --------------------------------------------------------------------------

def ladder_A(ctx, book, rng):
    """H1: add a whole new color, k spells at a time."""
    deck, main = ctx["deck"], set(ctx["main"])
    outs = [i for i in deck_spell_idx(ctx, book)
            if book.get(deck[i])["colors"] and book.get(deck[i])["colors"] <= main]
    outs.sort(key=lambda i: book.get(deck[i])["qv"])
    if not outs:
        return []

    best_color, best_cands = None, []
    for X in WUBRG:
        if X in main:
            continue
        cands = pool_cards(ctx, book, lambda i, X=X: (not i["is_land"])
                           and i["colors"] == frozenset(X))
        cands.sort(key=lambda n: -book.get(n)["qv"])
        if len(cands) > len(best_cands):
            best_color, best_cands = X, cands
    if best_color is None or not best_cands:
        return []

    kmax = min(4, len(outs), len(best_cands))
    variants = []
    for k in range(kmax + 1):
        names = apply_pairs(deck, list(zip(outs[:k], best_cands[:k])))
        variants.append((k, rx_colors(book, names), names))
    return variants


def ladder_B(ctx, book, rng):
    """H2: trade non-creature spells for creatures and back, MV-matched."""
    deck, main = ctx["deck"], set(ctx["main"])
    spells = deck_spell_idx(ctx, book)
    noncre = sorted([i for i in spells if not book.get(deck[i])["is_creature"]],
                    key=lambda i: book.get(deck[i])["qv"])
    creat = sorted([i for i in spells if book.get(deck[i])["is_creature"]],
                   key=lambda i: book.get(deck[i])["qv"])
    cre_pool = pool_cards(ctx, book, lambda i: (not i["is_land"]) and i["is_creature"]
                          and i["colors"] <= main)
    non_pool = pool_cards(ctx, book, lambda i: (not i["is_land"]) and (not i["is_creature"])
                          and i["colors"] <= main)
    cre_pool.sort(key=lambda n: -book.get(n)["qv"])
    non_pool.sort(key=lambda n: -book.get(n)["qv"])

    def key(oi, ci):
        t = mv_tier(oi, ci)
        return None if t is None else (t, -ci["qv"])

    up = build_pairs(deck, noncre, cre_pool, book, 4, key)     # k > 0: more creatures
    down = build_pairs(deck, creat, non_pool, book, 4, key)    # k < 0: fewer creatures
    if not up and not down:
        return []

    variants = [(0, rx_creatures(book, deck), list(deck))]
    for k in range(1, len(up) + 1):
        names = apply_pairs(deck, up[:k])
        variants.append((k, rx_creatures(book, names), names))
    for k in range(1, len(down) + 1):
        names = apply_pairs(deck, down[:k])
        variants.append((-k, rx_creatures(book, names), names))
    return variants


def ladder_C(ctx, book, rng):
    """H3: shift the curve down / up by >=2 MV per swap, q-matched."""
    deck, main = ctx["deck"], set(ctx["main"])
    spells = deck_spell_idx(ctx, book)
    pool = pool_cards(ctx, book, lambda i: not i["is_land"])

    def make_key(direction):
        def key(oi, ci):
            if direction < 0 and not ci["mv"] <= oi["mv"] - 2:
                return None
            if direction > 0 and not ci["mv"] >= oi["mv"] + 2:
                return None
            ct = color_tier(oi, ci, main)
            if ct is None:
                return None
            return (ct, 0 if ci["is_creature"] == oi["is_creature"] else 1,
                    abs(ci["qv"] - oi["qv"]))
        return key

    # cheapen the most expensive slots first; inflate the cheapest first
    cheap_outs = sorted(spells, key=lambda i: -book.get(deck[i])["mv"])
    pricey_outs = sorted(spells, key=lambda i: book.get(deck[i])["mv"])
    down = build_pairs(deck, cheap_outs, pool, book, 4, make_key(-1))
    up = build_pairs(deck, pricey_outs, pool, book, 4, make_key(+1))
    if not down and not up:
        return []

    variants = [(0, rx_mean_mv(book, deck), list(deck))]
    for k in range(1, len(down) + 1):
        names = apply_pairs(deck, down[:k])
        variants.append((-k, rx_mean_mv(book, names), names))
    for k in range(1, len(up) + 1):
        names = apply_pairs(deck, up[:k])
        variants.append((k, rx_mean_mv(book, names), names))
    return variants


def _pick(oi, cands, used, book, key):
    best = None
    for j, name in enumerate(cands):
        if j in used:
            continue
        k = key(oi, book.get(name))
        if k is None:
            continue
        if best is None or k < best[0]:
            best = (k, j)
    return best


def ladder_D(ctx, book, rng):
    """H4: mean-preserving spread {3,3} -> {2,4}, its reverse, and a control.

    rung 0 = base, +1 = spread, -1 = compress, +2 = same-MV q-matched control.
    """
    deck, main = ctx["deck"], set(ctx["main"])
    spells = deck_spell_idx(ctx, book)
    pool = pool_cards(ctx, book, lambda i: not i["is_land"])

    def key(oi, ci):
        ct = color_tier(oi, ci, main)
        if ct is None:
            return None
        return (ct, 0 if ci["is_creature"] == oi["is_creature"] else 1,
                abs(ci["qv"] - oi["qv"]))

    def by_mv(m):
        return [j for j, n in enumerate(pool) if book.get(n)["mv"] == m]

    variants = [(0, rx_var_mv(book, deck), list(deck))]

    # --- spread: two 3-MV deck spells -> a 2-MV and a 4-MV pool spell
    threes = sorted([i for i in spells if book.get(deck[i])["mv"] == 3],
                    key=lambda i: book.get(deck[i])["qv"])
    if len(threes) >= 2:
        for a, b in ((threes[0], threes[1]), (threes[1], threes[0])):
            c2 = _pick(book.get(deck[a]), pool, set(), book,
                       lambda oi, ci: key(oi, ci) if ci["mv"] == 2 else None)
            if c2 is None:
                continue
            c4 = _pick(book.get(deck[b]), pool, {c2[1]}, book,
                       lambda oi, ci: key(oi, ci) if ci["mv"] == 4 else None)
            if c4 is None:
                continue
            names = apply_pairs(deck, [(a, pool[c2[1]]), (b, pool[c4[1]])])
            variants.append((1, rx_var_mv(book, names), names))
            break

    # --- compress: a 2-MV and a 4-MV deck spell -> two 3-MV pool spells
    twos = sorted([i for i in spells if book.get(deck[i])["mv"] == 2],
                  key=lambda i: book.get(deck[i])["qv"])
    fours = sorted([i for i in spells if book.get(deck[i])["mv"] == 4],
                   key=lambda i: book.get(deck[i])["qv"])
    if twos and fours and len(by_mv(3)) >= 2:
        a, b = twos[0], fours[0]
        c3a = _pick(book.get(deck[a]), pool, set(), book,
                    lambda oi, ci: key(oi, ci) if ci["mv"] == 3 else None)
        if c3a is not None:
            c3b = _pick(book.get(deck[b]), pool, {c3a[1]}, book,
                        lambda oi, ci: key(oi, ci) if ci["mv"] == 3 else None)
            if c3b is not None:
                names = apply_pairs(deck, [(a, pool[c3a[1]]), (b, pool[c3b[1]])])
                variants.append((-1, rx_var_mv(book, names), names))

    # --- control: two random deck spells -> pool spells of the SAME MV
    shuffled = list(spells)
    rng.shuffle(shuffled)
    used: set[int] = set()
    ctrl: list[tuple[int, str]] = []
    for i in shuffled:
        if len(ctrl) >= 2:
            break
        oi = book.get(deck[i])
        hit = _pick(oi, pool, used, book,
                    lambda o, ci: key(o, ci) if ci["mv"] == o["mv"] else None)
        if hit is None:
            continue
        used.add(hit[1])
        ctrl.append((i, pool[hit[1]]))
    if len(ctrl) == 2:
        names = apply_pairs(deck, ctrl)
        variants.append((2, rx_var_mv(book, names), names))

    return variants if len(variants) > 1 else []


def _splash_setup(ctx, book, require_land: bool = False):
    """Best 3rd color X for a 2-color deck -> (X, single-pip spells, fixer lands).

    `require_land` restricts the choice to colors that ALSO have a producing
    land left in the pool -- ladder F needs both halves of its 2x2 to exist for
    the same colour, so the colour must be picked jointly rather than by spell
    count alone (picking it by spells first strands F on colours with no fixer).
    """
    main = set(ctx["main"])
    best = None
    for X in WUBRG:
        if X in main:
            continue
        cands = pool_cards(ctx, book, lambda i, X=X: (not i["is_land"])
                           and i["colors"] == frozenset(X) and i["pips"].sum() == 1)
        if not cands:
            continue
        lands = pool_cards(ctx, book,
                           lambda i, X=X: i["is_land"] and X in i["produces"])
        if require_land and not lands:
            continue
        cands.sort(key=lambda n: -book.get(n)["qv"])
        if best is None or len(cands) > len(best[1]):
            best = (X, cands, lands)
    return best


def ladder_E(ctx, book, rng):
    """H27: 1..3 single-pip splash cards into an exactly-2-color deck."""
    if ctx["n_colors"] != 2:
        return []
    deck, main = ctx["deck"], set(ctx["main"])
    setup = _splash_setup(ctx, book)
    if setup is None:
        return []
    X, cands, _lands = setup
    outs = [i for i in deck_spell_idx(ctx, book)
            if book.get(deck[i])["colors"] and book.get(deck[i])["colors"] <= main]
    outs.sort(key=lambda i: book.get(deck[i])["qv"])
    kmax = min(3, len(outs), len(cands))
    if kmax < 1:
        return []
    variants = []
    for k in range(kmax + 1):
        names = apply_pairs(deck, list(zip(outs[:k], cands[:k])))
        variants.append((k, rx_pips_of(book, names, X), names))
    return variants


def ladder_F(ctx, book, rng):
    """H7: 2x2 of one splash spell x one fixer land (the land is ADDED)."""
    if ctx["n_colors"] != 2:
        return []
    deck, main = ctx["deck"], set(ctx["main"])
    setup = _splash_setup(ctx, book, require_land=True)
    if setup is None:
        return []
    X, cands, lands = setup
    # prefer a fixer that also produces one of the deck's own colors (a real dual)
    lands.sort(key=lambda n: (-len(book.get(n)["produces"] & main), n))
    fixer, splash = lands[0], cands[0]

    outs = [i for i in deck_spell_idx(ctx, book)
            if book.get(deck[i])["colors"] and book.get(deck[i])["colors"] <= main]
    if not outs:
        return []
    out_i = min(outs, key=lambda i: book.get(deck[i])["qv"])

    base = list(deck)
    with_splash = apply_pairs(deck, [(out_i, splash)])
    with_fixer = base + [fixer]
    both = with_splash + [fixer]
    return [(0, rx_colors(book, base), base),
            (1, rx_colors(book, with_splash), with_splash),
            (2, rx_colors(book, with_fixer), with_fixer),
            (3, rx_colors(book, both), both)]


LADDERS = [
    ("A_color", ladder_A, "realized color count"),
    ("B_creature", ladder_B, "realized creature count"),
    ("C_curve", ladder_C, "realized mean spell MV"),
    ("D_spread", ladder_D, "realized spell-MV variance"),
    ("E_splash", ladder_E, "splash pips of the new color"),
    ("F_fixing", ladder_F, "realized color count"),
]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def rung_table(rows: list[dict], rx_label: str) -> None:
    rungs = sorted({r["rung"] for r in rows})
    n_ctx = len({r["context_id"] for r in rows})
    print(f"  contexts on ladder: {n_ctx}")
    print(f"  {'rung':>5} {'n':>5} {'mean d':>10} {'SE':>9} {'t vs 0':>8} "
          f"{'mean ' + rx_label:>28}")
    for k in rungs:
        sel = [r for r in rows if r["rung"] == k]
        m, se, n = mean_se([r["delta"] for r in sel])
        rx = float(np.mean([r["realized_x"] for r in sel]))
        t = m / se if se and np.isfinite(se) and se > 0 else float("nan")
        print(f"  {k:>5} {n:>5} {m:>+10.4f} {se:>9.4f} {t:>8.2f} {rx:>28.3f}")


def by_realized(rows: list[dict], label: str, binw: float | None = None) -> None:
    print(f"  -- delta by {label}" + (f" (bin width {binw})" if binw else ""))
    keyed: dict[float, list[float]] = {}
    for r in rows:
        x = r["realized_x"]
        if not np.isfinite(x):
            continue
        k = np.floor(x / binw) * binw if binw else x
        keyed.setdefault(float(k), []).append(r["delta"])
    for k in sorted(keyed):
        m, se, n = mean_se(keyed[k])
        print(f"     {label}={k:>8.2f}  n={n:>5}  mean d={m:>+9.4f} +/- {se:.4f}")


def report(all_rows: list[dict], attempted: dict, kept: dict) -> None:
    by_ladder: dict[str, list[dict]] = {}
    for r in all_rows:
        by_ladder.setdefault(r["ladder"], []).append(r)

    for name, _fn, rx_label in LADDERS:
        rows = by_ladder.get(name, [])
        print()
        print("=" * 78)
        print(f"LADDER {name}   (attempted {attempted.get(name, 0)} contexts, "
              f"kept {kept.get(name, 0)}, attrition "
              f"{1 - kept.get(name, 0) / max(1, attempted.get(name, 0)):.1%})")
        print("=" * 78)
        if not rows:
            print("  no usable contexts")
            continue
        rung_table(rows, rx_label)

        nz = [r for r in rows if r["rung"] != 0]
        if name == "A_color":
            by_realized(nz, "n_colors")
        elif name == "B_creature":
            by_realized(rows, "n_creatures")
        elif name == "C_curve":
            by_realized(nz, "mean_mv", binw=0.25)
        elif name == "D_spread":
            print("  -- cell means (1=spread {3,3}->{2,4}, -1=compress, 2=same-MV control)")
            for k, lab in ((1, "spread "), (-1, "compress"), (2, "control ")):
                sel = [r for r in rows if r["rung"] == k]
                if not sel:
                    continue
                m, se, n = mean_se([r["delta"] for r in sel])
                am, ase, _ = mean_se([abs(r["delta"]) for r in sel])
                print(f"     {lab} n={n:>4}  mean d={m:>+8.4f} +/- {se:.4f}   "
                      f"mean |d|={am:>7.4f} +/- {ase:.4f}")
            # paired spread-vs-control on the contexts that have both
            ctrl = {r["context_id"]: r["delta"] for r in rows if r["rung"] == 2}
            for k, lab in ((1, "spread"), (-1, "compress")):
                pair = [abs(r["delta"]) - abs(ctrl[r["context_id"]])
                        for r in rows if r["rung"] == k and r["context_id"] in ctrl]
                if pair:
                    m, se, n = mean_se(pair)
                    print(f"     |{lab}| - |control| paired: n={n:>4} {m:>+8.4f} "
                          f"+/- {se:.4f}  t={m / se if se else float('nan'):+.2f}")
        elif name == "E_splash":
            print("  -- marginal delta of the k-th splash card (paired within context)")
            per_ctx: dict[int, dict[int, float]] = {}
            for r in rows:
                per_ctx.setdefault(r["context_id"], {})[r["rung"]] = r["delta"]
            for k in (1, 2, 3):
                marg = [d[k] - d[k - 1] for d in per_ctx.values() if k in d and k - 1 in d]
                if marg:
                    m, se, n = mean_se(marg)
                    print(f"     card {k}: n={n:>4}  marginal={m:>+8.4f} +/- {se:.4f}  "
                          f"t={m / se if se else float('nan'):+.2f}")
            print("     [first card much worse than later ones => threshold;"
                  " roughly equal => linear]")
        elif name == "F_fixing":
            per_ctx = {}
            for r in rows:
                per_ctx.setdefault(r["context_id"], {})[r["rung"]] = r["score"]
            inter = [c[3] - c[2] - c[1] + c[0] for c in per_ctx.values() if len(c) == 4]
            m, se, n = mean_se(inter)
            print(f"  -- interaction (both - fixer - splash + base): n={n}  "
                  f"{m:>+.4f} +/- {se:.4f}  t={m / se if se else float('nan'):+.2f}")
            print("     [>0 = the fixer land makes the splash less bad (they cooperate)]")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="20 contexts on CPU")
    ap.add_argument("--contexts", type=int, default=250)
    ap.add_argument("--scan", type=int, default=0,
                    help="max contexts to examine while filling E/F (default 12x)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    n_ctx = 20 if args.smoke else args.contexts
    device = args.device or ("cpu" if args.smoke else None)

    rng = random.Random(SEED)
    probe = pl.Probe(device=device)
    print(f"device={probe.device} contexts={n_ctx}", flush=True)
    book = CardBook(probe, pl.load_win_rates())
    base, extra = sample_contexts(book, n_ctx, n_ctx, args.scan or 12 * n_ctx, rng)

    variants: list[dict] = []
    attempted: dict[str, int] = {}
    kept: dict[str, int] = {}
    for name, fn, _lab in LADDERS:
        n_try = n_keep = 0
        # E and F are conditioned on 2-colour decks (F additionally on a fixer
        # land), so they draw on the extra scan and are pre-filtered to eligible
        # contexts -- otherwise the attempt cap fills up with contexts the
        # ladder can never use, and 'attrition' reports eligibility instead.
        if name == "E_splash":
            pool_ctx = [c for c in base + extra if c["e_ok"]]
        elif name == "F_fixing":
            pool_ctx = [c for c in base + extra if c["f_ok"]]
        else:
            pool_ctx = base
        for ctx in pool_ctx:
            if n_try >= n_ctx:
                break
            n_try += 1
            try:
                vs = fn(ctx, book, rng)
            except Exception as exc:                      # noqa: BLE001
                print(f"  !! {name} ctx {ctx['cid']}: {exc}", flush=True)
                vs = []
            if len(vs) < 2:
                continue
            n_keep += 1
            for rung, rx, names in vs:
                variants.append({"ladder": name, "context_id": ctx["cid"], "rung": rung,
                                 "realized_x": float(rx), "names": tuple(sorted(names))})
        attempted[name], kept[name] = n_try, n_keep
        print(f"built {name}: {n_keep}/{n_try} contexts, "
              f"{sum(1 for v in variants if v['ladder'] == name)} variants", flush=True)

    # one forward per DISTINCT deck (rung 0 is shared across ladders); decks are
    # canonicalised to sorted name order so equal multisets collapse exactly.
    uniq: dict[tuple, int] = {}
    for v in variants:
        uniq.setdefault(v["names"], len(uniq))
    keys = [None] * len(uniq)
    for names, i in uniq.items():
        keys[i] = names
    print(f"{len(variants)} variants -> {len(uniq)} distinct decks", flush=True)

    def build(i: int) -> np.ndarray:
        return np.stack([book.get(n)["emb"] for n in keys[i]])

    scores = score_streaming(probe, len(uniq), build, label="t3")

    for v in variants:
        v["score"] = float(scores[uniq[v["names"]]])
    base = {(v["ladder"], v["context_id"]): v["score"]
            for v in variants if v["rung"] == 0}
    rows = []
    for v in variants:
        b = base.get((v["ladder"], v["context_id"]))
        if b is None:
            continue
        rows.append({"ladder": v["ladder"], "context_id": v["context_id"],
                     "rung": v["rung"], "realized_x": round(v["realized_x"], 4),
                     "score": round(v["score"], 6), "delta": round(v["score"] - b, 6)})

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ladder", "context_id", "rung",
                                          "realized_x", "score", "delta"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)", flush=True)

    report(rows, attempted, kept)


if __name__ == "__main__":
    main()
