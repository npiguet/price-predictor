"""Post-hoc slices behind the prose numbers of 2026-08-27-scorer-preferences.md
that no t*-script prints directly. Run after t0/t1/t2 have produced their CSVs.

Sections (``--section all`` default):
  t0        score by builder; within-set/method Spearman; per-set score offsets;
            dynamic-range split; duplicates-within-method
  decks     nonbasic lands per deck by builder file; gen4-512 color-count
            distribution and per-set extremes; snow-basics/Wastes count
  t2class   card-class slices of t2_card_values.csv (tricks, counterspells,
            removal split, planeswalkers, vehicles, token-makers, X-cost,
            hybrid) + text-length partial regression
  t1color   on/off-color decomposition of t1_add_deltas.csv + multi-face slice
  addrobust GPU: add-a-card robustness on forge-best (non-scorer-built) decks
"""

import argparse
import gzip
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import probe_lib as pl

OUT = pl.SCRATCH


def sec_t0():
    df = pd.read_csv(OUT / "t0_decks.csv")
    print("=== score by method ===")
    print(df.groupby("method")["score"].agg(["mean", "std", "count"]).round(3))

    feats = ["mean_score_play", "n_colors", "n_creatures", "n_removal", "n_flying",
             "avg_mv", "n_instant", "n_sorcery", "n_carddraw", "n_dupes",
             "n_nonbasic_lands", "sum_power", "n_cards", "mean_played_rate",
             "splash_pips"]
    for m in ["forge-best", "random", "gen4-512"]:
        sub = df[df.method == m].groupby("set_code").filter(lambda g: len(g) >= 30).copy()
        sub["z"] = sub.groupby("set_code")["score"].transform(
            lambda s: (s.rank() - s.rank().mean()) / max(s.rank().std(), 1e-9))
        c = sub[feats + ["z"]].dropna().corr(method="spearman")["z"].drop("z")
        print(f"\n=== within-set Spearman, method={m} (n={len(sub)}) ===")
        print(c.sort_values(key=abs, ascending=False).round(3))

    s = df[df.method == "forge-best"].groupby("set_code")["score"].agg(["mean", "count"])
    s = s[s["count"] >= 20].sort_values("mean")
    print("\n=== forge-best mean score by set (n>=20): bottom/top 5 ===")
    print(s.head(5).round(2)); print(s.tail(5).round(2))
    print("spread of set means:", round(s["mean"].max() - s["mean"].min(), 2),
          "| within-set std:",
          round(df[df.method == "forge-best"].groupby("set_code")["score"].std().mean(), 2))

    m = df.groupby("method")["score"].mean()
    print("\n=== dynamic-range split ===")
    print("span random->gen5:", round(m["gen5"] - m["random"], 2),
          "| random->forge-best:", round(m["forge-best"] - m["random"], 2),
          "| forge-best->gen5:", round(m["gen5"] - m["forge-best"], 2))

    print("\n=== duplicates vs score, within set ===")
    for meth in ["forge-best", "gen4-512", "random"]:
        sub = df[df.method == meth].groupby("set_code").filter(lambda g: len(g) >= 30).copy()
        sub["z"] = sub.groupby("set_code")["score"].transform(lambda s: s.rank(pct=True))
        print(meth, round(sub[["n_dupes", "z"]].corr(method="spearman").iloc[0, 1], 3),
              "| mean dupes:", round(sub.n_dupes.mean(), 2))


def _colors_of_deck(cards, loc, cache):
    pres = set()
    for c in cards:
        if c.lower() in pl.BASIC_LAND_NAMES:
            continue
        if c not in cache:
            e = loc.load_embedding(c)
            cache[c] = None if e is None else (e[-32:][1:6] > 0)
        v = cache[c]
        if v is not None:
            for i, l in enumerate("WUBRG"):
                if v[i]:
                    pres.add(l)
    return len(pres)


def sec_decks():
    loc = pl.ConvertedCardLocator(pl.CARDS_PATH)
    typecache: dict = {}

    def is_nonbasic_land(name):
        if name not in typecache:
            t = loc.load_text(name)
            types = ""
            if t is not None:
                for ln in t.text.splitlines():
                    if ln.startswith("types:"):
                        types = ln
                        break
            typecache[name] = "land" in types
        return typecache[name]

    for fname in ["generated-decks-gen4-512.txt", "generated-decks-gen4-256.txt",
                  "generated-decks-gen5.txt"]:
        counts: Counter = Counter(); tot = 0
        for _lbl, _sc, cards in pl.read_generated_decks(pl.YDATA / "decks" / fname, limit=3000):
            nb = sum(1 for c in cards
                     if c.lower() not in pl.BASIC_LAND_NAMES and is_nonbasic_land(c))
            counts[nb] += 1; tot += 1
        dist = " ".join(f"{k}:{100 * v / tot:.0f}%" for k, v in sorted(counts.items()))
        print(f"{fname}: nonbasic lands/deck -> {dist}")

    pipcache: dict = {}
    cnt: Counter = Counter(); snow = wastes = tot = 0
    per_set = defaultdict(list)
    for _lbl, sc, cards in pl.read_generated_decks(pl.YDATA / "decks" / "generated-decks-gen4-512.txt"):
        tot += 1
        if any("snow-covered" in c.lower() for c in cards):
            snow += 1
        if any(c.lower() == "wastes" for c in cards):
            wastes += 1
        if tot <= 5000:
            n = _colors_of_deck(cards, loc, pipcache)
            cnt[n] += 1
            per_set[sc].append(n)
    print(f"gen4-512: {tot} decks, snow-basic decks {snow}, wastes decks {wastes}")
    n5 = sum(cnt.values())
    print("color-count distribution (first 5000):",
          {k: f"{100 * v / n5:.0f}%" for k, v in sorted(cnt.items())})
    rows = [(s, np.mean(v)) for s, v in per_set.items() if len(v) >= 15]
    rows.sort(key=lambda r: r[1])
    print("mean colors by set: lowest", [(s, round(m, 2)) for s, m in rows[:5]],
          "highest", [(s, round(m, 2)) for s, m in rows[-5:]])


def sec_t2class():
    df = pd.read_csv(OUT / "t2_card_values.csv")
    df = df[df.v_swap.notna()].copy()
    loc = pl.ConvertedCardLocator(pl.CARDS_PATH)
    cols = {"is_trick": [], "is_counter": [], "is_pw": [], "is_vehicle": [],
            "makes_tokens": [], "x_cost": [], "hyb": [], "tok_len": []}
    for name in df.name:
        t = loc.load_text(name)
        txt = t.text.lower() if t else ""
        types = re.search(r"types:.*", txt)
        types = types.group(0) if types else ""
        mc = re.search(r"mana cost:.*", txt)
        mc = mc.group(0) if mc else ""
        cols["is_trick"].append(bool("instant" in types
                                     and re.search(r"gets? \+\d+/\+\d+ until end of turn", txt)))
        cols["is_counter"].append("counter target" in txt and "spell" in txt)
        cols["is_pw"].append("planeswalker" in types)
        cols["is_vehicle"].append("vehicle" in types)
        cols["makes_tokens"].append(bool(re.search(r"create.{0,60}token", txt)))
        cols["x_cost"].append("{x}" in mc)
        cols["hyb"].append("/" in mc)
        cols["tok_len"].append(len(txt.split()))
    for k, v in cols.items():
        df[k] = v
    base = df[df.det_is_land == 0]

    def grp(mask, label):
        g = base[mask]
        print(f"{label:34s} n={len(g):5d}  v_swap={g.v_swap.mean():+.3f}"
              f"  score_play={g.shrunk_score_play.mean():+.3f}  mv={g.mv.mean():.1f}")

    grp(base.is_trick, "combat trick (pump instant)")
    grp(base.is_counter, "counterspell")
    grp((base.is_instant == True) & (~base.is_trick) & (~base.is_counter)  # noqa: E712
        & (base.is_removal == True), "instant removal")  # noqa: E712
    grp((base.is_sorcery == True) & (base.is_removal == True), "sorcery removal")  # noqa: E712
    grp(base.is_pw, "planeswalker")
    grp(base.is_vehicle, "vehicle")
    grp(base.makes_tokens & (base.is_creature == False), "noncreature token-maker")  # noqa: E712
    grp(base.x_cost, "X-cost spell")
    grp(base.hyb, "hybrid-mana card")
    grp(base.is_creature == True, "ALL creatures")  # noqa: E712
    grp(base.is_creature == False, "ALL noncreature spells")  # noqa: E712

    pw = base[base.is_pw]
    for lo, hi in [(3, 5), (5, 9)]:
        p = pw[(pw.mv >= lo) & (pw.mv < hi)]
        c = base[(base.is_creature == True) & (base.mv >= lo) & (base.mv < hi)]  # noqa: E712
        if len(p) > 4:
            print(f"  PW mv[{lo},{hi}): n={len(p)} v={p.v_swap.mean():+.3f}"
                  f"  vs creatures {c.v_swap.mean():+.3f}")

    sub = base[base.shrunk_score_play.notna()].copy()
    X = np.column_stack([np.ones(len(sub)), sub.shrunk_score_play, sub.mv,
                         sub.is_creature.astype(float), sub.tok_len])
    Xs = X.copy()
    Xs[:, 1:] = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0)
    b, *_ = np.linalg.lstsq(Xs, sub.v_swap.values, rcond=None)
    resid = sub.v_swap.values - Xs @ b
    se = np.sqrt(np.sum(resid ** 2) / (len(sub) - 5)
                 * np.linalg.inv(Xs.T @ Xs).diagonal())
    print(f"text-length std beta={b[4]:+.4f} (t={b[4] / se[4]:+.1f})"
          f" controlling score_play/mv/is_creature (n={len(sub)})")


def sec_t1color():
    df = pd.read_csv(OUT / "t1_add_deltas.csv")
    sp = df[df.is_land == False]  # noqa: E712
    for oc in [True, False]:
        s = sp[sp.on_color == oc]
        hq = s[s.q >= 0.05]
        print(f"spell adds on_color={oc}: n={len(s)} mean={s.delta_add.mean():+.3f}"
              f" frac>0={100 * (s.delta_add > 0).mean():.1f}%"
              f" | high-q: mean={hq.delta_add.mean():+.3f}"
              f" frac>0={100 * (hq.delta_add > 0).mean():.1f}%")
    s = sp[(sp.on_color == True) & sp.q.notna()]  # noqa: E712
    print("on-color spells corr(delta_add, q) =",
          round(np.corrcoef(s.delta_add, s.q)[0, 1], 3))
    onc = df[(df.is_land == False) & (df.on_color == True)]  # noqa: E712
    for mf in [False, True]:
        g = onc[onc.is_multiface == mf]
        print(f"on-color spell adds, multiface={mf}: n={len(g)}"
              f" mean={g.delta_add.mean():+.3f} mean q={g.q.mean():+.4f}")


def sec_addrobust():
    import random
    random.seed(42)
    probe = pl.Probe()
    wr = pl.load_win_rates()
    decks_by_set: dict = {}
    with gzip.open(pl.YDATA / "matches-b07" / "match-outcomes-gen0.txt.gz",
                   "rt", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split(";")
            if len(p) < 10:
                continue
            for mi, di in ((3, 5), (4, 6)):
                if p[mi] == "forge-best":
                    decks_by_set.setdefault(p[2], []).append(p[di].split("|"))
    base_mats, adds = [], []
    for s in [s for s, v in decks_by_set.items() if len(v) >= 6][:50]:
        v = decks_by_set[s]
        deck = [c for c in v[0] if c.lower() not in pl.BASIC_LAND_NAMES]
        try:
            bm = probe.deck_matrix(deck)
        except KeyError:
            continue
        cands = []
        for other in v[1:6]:
            for c in other:
                if c.lower() in pl.BASIC_LAND_NAMES or c in deck:
                    continue
                e = probe.embedding(c)
                if e is None or e[-32:][0] > 0.5:
                    continue
                q = (wr.get(c) or {}).get("shrunk_score_play")
                if q is not None:
                    cands.append((q, c))
        cands = sorted(set(cands), reverse=True)[:10]
        if len(cands) < 5:
            continue
        base_mats.append(bm)
        adds.append(cands)
        if len(base_mats) >= 40:
            break
    mats, deltas_meta = [], []
    for i, bm in enumerate(base_mats):
        mats.append(bm)
        for q, c in adds[i]:
            mats.append(np.vstack([bm, probe.embedding(c)[None, :]]))
            deltas_meta.append((i, q))
    scores = probe.score_matrices(mats)
    idx = 0
    d, qv = [], []
    for i, bm in enumerate(base_mats):
        base = scores[idx]
        for j, (q, _c) in enumerate(adds[i]):
            d.append(scores[idx + 1 + j] - base)
            qv.append(q)
        idx += 1 + len(adds[i])
    d = np.array(d); qv = np.array(qv)
    print(f"forge-best contexts: {len(base_mats)}, adds: {len(d)}")
    print(f"delta_add mean={d.mean():+.3f} sd={d.std():.3f}"
          f" frac>0={100 * (d > 0).mean():.1f}%")
    hi = qv >= 0.05
    print(f"top-quality adds (q>=0.05): mean={d[hi].mean():+.3f}"
          f" n={hi.sum()} frac>0={100 * (d[hi] > 0).mean():.1f}%")


def sec_colors():
    """Splash economics in the gen4-512 builds: off-color vs main-color card
    labels by deck color count, exact color-count census, and the 5-color
    decks' set concentration and multicolor share."""
    wr = pl.load_win_rates()
    loc = pl.ConvertedCardLocator(pl.CARDS_PATH)
    pip_cache: dict = {}

    def pips(name):
        if name not in pip_cache:
            e = loc.load_embedding(name)
            pip_cache[name] = None if e is None else e[-32:][1:6]
        return pip_cache[name]

    census: Counter = Counter()
    sets5: Counter = Counter()
    sets4: Counter = Counter()
    gold5 = []
    grp = defaultdict(lambda: {"main": [], "splash": [], "n_splash": []})
    for _lbl, sc, cards in pl.read_generated_decks(
            pl.YDATA / "decks" / "generated-decks-gen4-512.txt"):
        nonbasic = [c for c in cards if c.lower() not in pl.BASIC_LAND_NAMES]
        tot = np.zeros(5)
        percard = {}
        ok = True
        for c in nonbasic:
            p = pips(c)
            if p is None:
                ok = False
                break
            percard[c] = p
            tot += p
        if not ok:
            continue
        n = int((tot > 0).sum())
        census[n] += 1
        if n == 5:
            sets5[sc] += 1
            gold5.append(np.mean([(p > 0).sum() >= 2 for p in percard.values()]))
        elif n == 4:
            sets4[sc] += 1
        if n < 2:
            continue
        primary = set(np.argsort(tot)[::-1][:2])
        key = min(n, 4)
        n_splash = 0
        for c, p in percard.items():
            cols = set(np.nonzero(p > 0)[0])
            if not cols:
                continue
            q = (wr.get(c) or {}).get("shrunk_score_play")
            if q is None:
                continue
            if cols <= primary:
                grp[key]["main"].append(q)
            else:
                grp[key]["splash"].append(q)
                n_splash += 1
        grp[key]["n_splash"].append(n_splash)

    print("color-count census:", dict(sorted(census.items())))
    rows = {}
    print(f"{'deck colors':12s} {'decks':>6s} {'main mean q':>12s} "
          f"{'off-color mean q':>17s} {'off-color cards/deck':>21s}")
    for k in sorted(grp):
        s = grp[k]
        sp = np.mean(s["splash"]) if s["splash"] else float("nan")
        lbl = f"{k}" + ("+" if k == 4 else "")
        rows[lbl] = {"decks": len(s["n_splash"]),
                     "main_q": float(np.mean(s["main"])),
                     "splash_q": (None if not s["splash"] else float(sp)),
                     "splash_cards": float(np.mean(s["n_splash"]))}
        print(f"{lbl:12s} {len(s['n_splash']):6d} {np.mean(s['main']):12.4f} "
              f"{sp:17.4f} {np.mean(s['n_splash']):21.1f}")
    print("5-color decks by set:", dict(sets5.most_common()))
    if gold5:
        print(f"mean multicolor-card share in 5-color decks: "
              f"{100 * np.mean(gold5):.0f}%")
    print("4-color decks, top sets:", dict(sets4.most_common(8)))
    import json
    (OUT / "post_hoc_colors.json").write_text(json.dumps(
        {"census": dict(sorted(census.items())), "by_colors": rows,
         "sets5": dict(sets5.most_common()),
         "gold_share_5c": (float(np.mean(gold5)) if gold5 else None),
         "sets4_top": dict(sets4.most_common(8))}, indent=1), encoding="utf-8")
    print("saved", OUT / "post_hoc_colors.json")


SECTIONS = {"t0": sec_t0, "decks": sec_decks, "t2class": sec_t2class,
            "t1color": sec_t1color, "colors": sec_colors,
            "addrobust": sec_addrobust}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all", choices=["all", *SECTIONS])
    args = ap.parse_args()
    for name, fn in SECTIONS.items():
        if args.section in ("all", name):
            print(f"\n########## {name} ##########")
            fn()
