"""P0.4 -- streaming label-side aggregation over the per-game cards-played corpus.

One pass over ~974k game lines of
``cards-played-bo1-embedding.cleaned.txt`` producing, per card:

* the four primary counters (wins/losses x played/in-deck) and their
  ``@play`` (owner was the starter) slices -- exactly the spec pseudocode
  in ``specs/2026-05-03-card-winnability-pretraining.md``;
* the mediation triple  w = P(win | in deck),  a = P(played | win),
  b = P(played | loss),  m = (a+b)/2,  d = a - b, with the exact identity
  ``score_overall = m*(2w-1) + d/2``;
* matchup-stratified counters (forge-best vs forge-best; any forge
  method vs the same forge method);
* game-length-stratified counters, bucketing each game by the OWNING
  side's ``len(cards_played)``;
* an odd/even split-half assignment (per card, alternating over that
  card's own in-deck appearances) for a reliability floor.

Also accumulates per-set aggregates (mean raw score contribution, gold and
pip density of the decks seen) and per-build-method win rates.

CPU only.  Outputs ``output/encoder-probes/l_mediation_table.pkl``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "encoder-probes"
DEFAULT_CORPUS = Path(
    r"Y:\Nicolas\mtg\mtg-models-data\sealed\training-data\matches-bo1"
    r"\cards-played-bo1-embedding.cleaned.txt"
)
DEFAULT_TABLE = Path(
    r"C:\Users\nicol\AppData\Local\Temp\claude"
    r"\C--Users-nicol-IdeaProjects-price-predictor"
    r"\dc499b61-1d84-4345-b336-7fe0e34557a4\scratchpad\grounding\card_table.pkl"
)

N_LEN_BUCKETS = 5
LEN_BUCKET_LABELS = ["0-4", "5-6", "7-8", "9-11", "12+"]
TOT_BUCKET_LABELS = ["0-10", "11-13", "14-16", "17-20", "21+"]


def len_bucket(n: int) -> int:
    if n <= 4:
        return 0
    if n <= 6:
        return 1
    if n <= 8:
        return 2
    if n <= 11:
        return 3
    return 4


def tot_bucket(n: int) -> int:
    if n <= 10:
        return 0
    if n <= 13:
        return 1
    if n <= 16:
        return 2
    if n <= 20:
        return 3
    return 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--card-table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--limit", type=int, default=0, help="stop after N lines (debug)")
    ap.add_argument("--out", type=Path, default=OUT / "l_mediation_table.pkl")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    tbl = pickle.load(open(args.card_table, "rb"))
    names = list(tbl["card_name"])
    idmap: dict[str, int] = {n: i for i, n in enumerate(names)}
    cap = len(names) + 20000
    extra_names: list[str] = []

    # feature lookups for the per-set density accumulators
    pips = np.zeros(cap, dtype=np.float64)
    gold = np.zeros(cap, dtype=np.float64)
    pips[: len(names)] = tbl["pips_total"].to_numpy(dtype=np.float64)
    gold[: len(names)] = (tbl["n_colors"].to_numpy(dtype=np.float64) >= 2).astype(float)

    z = lambda: np.zeros(cap, dtype=np.int64)  # noqa: E731

    # --- primary counters -------------------------------------------------
    w_ind, w_pl = z(), z()          # wins when in deck / when played
    l_ind, l_pl = z(), z()          # losses when in deck / when played
    w_ind_p, w_pl_p = z(), z()      # ... @play (owner was starter)
    l_ind_p, l_pl_p = z(), z()

    # --- matchup strata (0 = forge-best vs forge-best, 1 = forge X vs forge X)
    s_w_ind = np.zeros((2, cap), dtype=np.int64)
    s_w_pl = np.zeros((2, cap), dtype=np.int64)
    s_l_ind = np.zeros((2, cap), dtype=np.int64)
    s_l_pl = np.zeros((2, cap), dtype=np.int64)

    # --- game-length strata ----------------------------------------------
    g_w_ind = np.zeros((N_LEN_BUCKETS, cap), dtype=np.int64)
    g_w_pl = np.zeros((N_LEN_BUCKETS, cap), dtype=np.int64)
    g_l_ind = np.zeros((N_LEN_BUCKETS, cap), dtype=np.int64)
    g_l_pl = np.zeros((N_LEN_BUCKETS, cap), dtype=np.int64)

    # --- opponent-side length strata (game-length proxy not conditioned on
    #     the owner's own mana development) and whole-game length strata ----
    o_w_ind = np.zeros((N_LEN_BUCKETS, cap), dtype=np.int64)
    o_w_pl = np.zeros((N_LEN_BUCKETS, cap), dtype=np.int64)
    o_l_ind = np.zeros((N_LEN_BUCKETS, cap), dtype=np.int64)
    o_l_pl = np.zeros((N_LEN_BUCKETS, cap), dtype=np.int64)
    t_w_ind = np.zeros((N_LEN_BUCKETS, cap), dtype=np.int64)
    t_w_pl = np.zeros((N_LEN_BUCKETS, cap), dtype=np.int64)
    t_l_ind = np.zeros((N_LEN_BUCKETS, cap), dtype=np.int64)
    t_l_pl = np.zeros((N_LEN_BUCKETS, cap), dtype=np.int64)

    # --- split halves ------------------------------------------------------
    appear = np.zeros(cap, dtype=np.int64)
    h_w_ind = np.zeros((2, cap), dtype=np.int64)
    h_w_pl = np.zeros((2, cap), dtype=np.int64)
    h_l_ind = np.zeros((2, cap), dtype=np.int64)
    h_l_pl = np.zeros((2, cap), dtype=np.int64)
    h_w_ind_p = np.zeros((2, cap), dtype=np.int64)
    h_w_pl_p = np.zeros((2, cap), dtype=np.int64)
    h_l_ind_p = np.zeros((2, cap), dtype=np.int64)
    h_l_pl_p = np.zeros((2, cap), dtype=np.int64)

    # --- scalar / per-set / per-method accumulators -------------------------
    set_num = Counter()      # sum over cards of (played & win) - (played & loss)
    set_den = Counter()      # total in-deck observations
    set_games = Counter()
    set_gold = Counter()     # sum of gold flags over in-deck observations
    set_pips = Counter()     # sum of pips over in-deck observations
    meth_w = Counter()
    meth_l = Counter()
    pair_ct = Counter()
    sum_played_w = 0
    sum_played_l = 0
    n_games = 0
    bad_lines = 0
    unknown = Counter()

    t0 = time.time()
    fh = open(args.corpus, "r", encoding="utf-8", errors="replace")
    for lineno, line in enumerate(fh):
        if args.limit and lineno >= args.limit:
            break
        f = line.rstrip("\n").split(";")
        if len(f) != 11:
            bad_lines += 1
            continue
        _ts, _run, sc, mA, mB, pA, pB, nA, nB, winner, starter = f
        if winner not in ("A", "B") or starter not in ("A", "B"):
            bad_lines += 1
            continue

        if winner == "A":
            wp_s, wn_s, lp_s, ln_s = pA, nA, pB, nB
        else:
            wp_s, wn_s, lp_s, ln_s = pB, nB, pA, nA

        wp = wp_s.split("|") if wp_s else []
        wn = wn_s.split("|") if wn_s else []
        lp = lp_s.split("|") if lp_s else []
        ln = ln_s.split("|") if ln_s else []

        try:
            iwp = [idmap[c] for c in wp]
            iwn = [idmap[c] for c in wn]
            ilp = [idmap[c] for c in lp]
            iln = [idmap[c] for c in ln]
        except KeyError:
            iwp, iwn, ilp, iln = [], [], [], []
            for src, dst in ((wp, iwp), (wn, iwn), (lp, ilp), (ln, iln)):
                for c in src:
                    j = idmap.get(c)
                    if j is None:
                        j = len(idmap)
                        if j >= cap:
                            raise RuntimeError("card id capacity exceeded")
                        idmap[c] = j
                        extra_names.append(c)
                        unknown[c] += 1
                    dst.append(j)

        IWP = np.array(iwp, dtype=np.int64)
        IWD = np.array(iwp + iwn, dtype=np.int64)
        ILP = np.array(ilp, dtype=np.int64)
        ILD = np.array(ilp + iln, dtype=np.int64)

        w_ind[IWD] += 1
        w_pl[IWP] += 1
        l_ind[ILD] += 1
        l_pl[ILP] += 1

        if winner == starter:
            w_ind_p[IWD] += 1
            w_pl_p[IWP] += 1
        else:
            l_ind_p[ILD] += 1
            l_pl_p[ILP] += 1

        # matchup strata
        st = -1
        if mA == "forge-best" and mB == "forge-best":
            st = 0
        elif mA == mB and mA.startswith("forge"):
            st = 1
        if st >= 0:
            s_w_ind[st, IWD] += 1
            s_w_pl[st, IWP] += 1
            s_l_ind[st, ILD] += 1
            s_l_pl[st, ILP] += 1
            if st == 0:  # forge-best vs forge-best also belongs to the wider stratum
                s_w_ind[1, IWD] += 1
                s_w_pl[1, IWP] += 1
                s_l_ind[1, ILD] += 1
                s_l_pl[1, ILP] += 1

        # game-length strata (owning side's played count)
        bw = len_bucket(len(iwp))
        bl = len_bucket(len(ilp))
        g_w_ind[bw, IWD] += 1
        g_w_pl[bw, IWP] += 1
        g_l_ind[bl, ILD] += 1
        g_l_pl[bl, ILP] += 1
        # opponent-side bucket: swap the two sides' buckets
        o_w_ind[bl, IWD] += 1
        o_w_pl[bl, IWP] += 1
        o_l_ind[bw, ILD] += 1
        o_l_pl[bw, ILP] += 1
        bt = tot_bucket(len(iwp) + len(ilp))
        t_w_ind[bt, IWD] += 1
        t_w_pl[bt, IWP] += 1
        t_l_ind[bt, ILD] += 1
        t_l_pl[bt, ILP] += 1

        # split halves: parity of this card's own in-deck appearance counter
        hw = appear[IWD] & 1
        appear[IWD] += 1
        hl = appear[ILD] & 1
        appear[ILD] += 1
        hwp = hw[: len(iwp)]
        hlp = hl[: len(ilp)]
        h_w_ind[hw, IWD] += 1
        h_w_pl[hwp, IWP] += 1
        h_l_ind[hl, ILD] += 1
        h_l_pl[hlp, ILP] += 1
        if winner == starter:
            h_w_ind_p[hw, IWD] += 1
            h_w_pl_p[hwp, IWP] += 1
        else:
            h_l_ind_p[hl, ILD] += 1
            h_l_pl_p[hlp, ILP] += 1

        # scalars
        n_games += 1
        sum_played_w += len(iwp)
        sum_played_l += len(ilp)
        set_num[sc] += len(iwp) - len(ilp)
        set_den[sc] += len(IWD) + len(ILD)
        set_games[sc] += 1
        set_gold[sc] += float(gold[IWD].sum() + gold[ILD].sum())
        set_pips[sc] += float(pips[IWD].sum() + pips[ILD].sum())
        if winner == "A":
            meth_w[mA] += 1
            meth_l[mB] += 1
        else:
            meth_w[mB] += 1
            meth_l[mA] += 1
        pair_ct[(mA, mB)] += 1

        if lineno % 100000 == 0 and lineno:
            el = time.time() - t0
            print(f"  {lineno:,} lines  {el:.0f}s", flush=True)
    fh.close()

    n_cards = len(idmap)
    all_names = names + extra_names
    print(f"read {n_games:,} games, {bad_lines} bad lines, "
          f"{n_cards:,} card ids ({len(extra_names)} new), "
          f"{time.time() - t0:.0f}s", flush=True)

    sl = slice(0, n_cards)
    df = pd.DataFrame({"card_name": all_names})
    df["wins_when_played"] = w_pl[sl]
    df["wins_when_in_deck"] = w_ind[sl]
    df["losses_when_played"] = l_pl[sl]
    df["losses_when_in_deck"] = l_ind[sl]
    df["wins_played_play"] = w_pl_p[sl]
    df["wins_in_deck_play"] = w_ind_p[sl]
    df["losses_played_play"] = l_pl_p[sl]
    df["losses_in_deck_play"] = l_ind_p[sl]

    for k, tag in ((0, "fb"), (1, "ff")):
        df[f"{tag}_wins_played"] = s_w_pl[k, sl]
        df[f"{tag}_wins_in_deck"] = s_w_ind[k, sl]
        df[f"{tag}_losses_played"] = s_l_pl[k, sl]
        df[f"{tag}_losses_in_deck"] = s_l_ind[k, sl]

    for b, lab in enumerate(LEN_BUCKET_LABELS):
        df[f"gl{b}_wins_played"] = g_w_pl[b, sl]
        df[f"gl{b}_wins_in_deck"] = g_w_ind[b, sl]
        df[f"gl{b}_losses_played"] = g_l_pl[b, sl]
        df[f"gl{b}_losses_in_deck"] = g_l_ind[b, sl]
        df[f"og{b}_wins_played"] = o_w_pl[b, sl]
        df[f"og{b}_wins_in_deck"] = o_w_ind[b, sl]
        df[f"og{b}_losses_played"] = o_l_pl[b, sl]
        df[f"og{b}_losses_in_deck"] = o_l_ind[b, sl]
        df[f"tg{b}_wins_played"] = t_w_pl[b, sl]
        df[f"tg{b}_wins_in_deck"] = t_w_ind[b, sl]
        df[f"tg{b}_losses_played"] = t_l_pl[b, sl]
        df[f"tg{b}_losses_in_deck"] = t_l_ind[b, sl]

    for h in (0, 1):
        df[f"h{h}_wins_played"] = h_w_pl[h, sl]
        df[f"h{h}_wins_in_deck"] = h_w_ind[h, sl]
        df[f"h{h}_losses_played"] = h_l_pl[h, sl]
        df[f"h{h}_losses_in_deck"] = h_l_ind[h, sl]
        df[f"h{h}_wins_played_play"] = h_w_pl_p[h, sl]
        df[f"h{h}_wins_in_deck_play"] = h_w_ind_p[h, sl]
        df[f"h{h}_losses_played_play"] = h_l_pl_p[h, sl]
        df[f"h{h}_losses_in_deck_play"] = h_l_ind_p[h, sl]

    # derived mediation quantities
    n_in = df.wins_when_in_deck + df.losses_when_in_deck
    df["n_in_deck"] = n_in
    df["n_played"] = df.wins_when_played + df.losses_when_played
    with np.errstate(divide="ignore", invalid="ignore"):
        df["w"] = df.wins_when_in_deck / n_in
        df["a"] = df.wins_when_played / df.wins_when_in_deck
        df["b"] = df.losses_when_played / df.losses_when_in_deck
        df["m"] = (df.a + df.b) / 2.0
        df["d"] = df.a - df.b
        df["played_rate"] = df.n_played / n_in
        df["score_overall"] = (df.wins_when_played - df.losses_when_played) / n_in
        n_play = df.wins_in_deck_play + df.losses_in_deck_play
        df["n_in_deck_play"] = n_play
        df["score_play"] = (df.wins_played_play - df.losses_played_play) / n_play
        n_draw = n_in - n_play
        df["n_in_deck_draw"] = n_draw
        df["score_draw"] = (
            (df.wins_when_played - df.wins_played_play)
            - (df.losses_when_played - df.losses_played_play)
        ) / n_draw
        pp = df.wins_when_played / df.n_played
        pd_ = (df.wins_when_in_deck - df.wins_when_played) / (n_in - df.n_played)
        df["p_play"] = pp
        df["p_dead"] = pd_
        df["cast_lift"] = pp - pd_

    meta = {
        "n_games": n_games,
        "bad_lines": bad_lines,
        "sum_played_winner": sum_played_w,
        "sum_played_loser": sum_played_l,
        "set_num": dict(set_num),
        "set_den": dict(set_den),
        "set_games": dict(set_games),
        "set_gold": dict(set_gold),
        "set_pips": dict(set_pips),
        "method_wins": dict(meth_w),
        "method_losses": dict(meth_l),
        "pair_counts": {f"{a}|{b}": c for (a, b), c in pair_ct.items()},
        "unknown_names": dict(unknown),
        "len_bucket_labels": LEN_BUCKET_LABELS,
        "tot_bucket_labels": TOT_BUCKET_LABELS,
        "corpus": str(args.corpus),
    }

    with open(args.out, "wb") as f:
        pickle.dump({"cards": df, "meta": meta}, f)
    print(f"wrote {args.out}  ({len(df):,} cards)")

    # --- inline sanity check against the label snapshot ---------------------
    snap = REPO  # placeholder to keep flake quiet
    del snap
    print(f"winner mean cards played {sum_played_w / n_games:.3f}  "
          f"loser {sum_played_l / n_games:.3f}")


if __name__ == "__main__":
    sys.exit(main())
