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
import bisect
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import OUTPUT, SEALED_TRAINING_DATA  # noqa: E402

OUT = OUTPUT / "encoder-probes"
DEFAULT_CORPUS = (SEALED_TRAINING_DATA / "matches-bo1"
                  / "cards-played-bo1-embedding.cleaned.txt")
DEFAULT_TABLE = OUT / "card_table.pkl"  # written by p0b_card_table.py

N_LEN_BUCKETS = 5
LEN_BUCKET_LABELS = ["0-4", "5-6", "7-8", "9-11", "12+"]
TOT_BUCKET_LABELS = ["0-10", "11-13", "14-16", "17-20", "21+"]
# The lowest count of every bucket after the first, for one side and for both.
LEN_BUCKET_STARTS = [5, 7, 9, 12]
TOT_BUCKET_STARTS = [11, 14, 17, 21]

WIN, LOSS = 0, 1
IN_DECK, PLAYED = 0, 1
# Strata: all games; games the side started; the forge-best mirror; any forge
# method against itself; a game-length bucket by the owning side's (GL), the
# opponent's (OG) and both sides' (TG) distinct cards played; split half H,
# and split half restricted to games the side started (HP).
ALL, PLAY, FB, FF = 0, 1, 2, 3
GL = 4
OG = GL + N_LEN_BUCKETS
TG = OG + N_LEN_BUCKETS
H = TG + N_LEN_BUCKETS
HP = H + 2
N_STRATA = HP + 2


def column(outcome: str, kind: str, prefix: str = "", suffix: str = "", when: str = "") -> str:
    """A counter column of the table: ``outcome`` wins/losses, ``kind``
    played/in_deck, in the stratum ``prefix``/``suffix`` name."""
    return f"{prefix}{outcome}_{when}{kind}{suffix}"


def stratum_scores(df: pd.DataFrame, prefix: str = "", suffix: str = "", when: str = ""):
    """``(n_in_deck, score, played_rate)`` of one stratum's four counters, where
    score = (played and won − played and lost) / in-deck observations."""
    def col(outcome: str, kind: str) -> pd.Series:
        return df[column(outcome, kind, prefix, suffix, when)]

    n = col("wins", "in_deck") + col("losses", "in_deck")
    return (n, (col("wins", "played") - col("losses", "played")) / n,
            (col("wins", "played") + col("losses", "played")) / n)


def draw_score(df: pd.DataFrame, prefix: str = "", when: str = "") -> pd.Series:
    """The score in games the side did not start: a stratum's all-games
    counters minus its on-the-play ones."""
    def drawn(outcome: str, kind: str) -> pd.Series:
        return (df[column(outcome, kind, prefix, when=when)]
                - df[column(outcome, kind, prefix, "_play")])

    return ((drawn("wins", "played") - drawn("losses", "played"))
            / (drawn("wins", "in_deck") + drawn("losses", "in_deck")))


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

    # One counter array, [stratum, outcome, kind, card]. A game adds 1 to the
    # in-deck (and played) counters of every card on the winning side under
    # WIN, and of every card on the losing side under LOSS, in each stratum
    # the game belongs to for that side.
    C = np.zeros((N_STRATA, 2, 2, cap), dtype=np.int64)
    appear = np.zeros(cap, dtype=np.int64)

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

        def tally(outcome, s_deck, s_played=None):
            deck, played = (IWD, IWP) if outcome == WIN else (ILD, ILP)
            C[s_deck, outcome, IN_DECK, deck] += 1
            C[s_deck if s_played is None else s_played, outcome, PLAYED, played] += 1

        starter_won = winner == starter
        tally(WIN, ALL)
        tally(LOSS, ALL)
        tally(WIN if starter_won else LOSS, PLAY)

        if mA == mB and mA.startswith("forge"):
            # forge-best vs forge-best also belongs to the wider same-method stratum
            for st in ((FB, FF) if mA == "forge-best" else (FF,)):
                tally(WIN, st)
                tally(LOSS, st)

        # game-length strata: the owning side's played count (gl), the
        # opponent's (og) and both sides' together (tg)
        bw = bisect.bisect_right(LEN_BUCKET_STARTS, len(iwp))
        bl = bisect.bisect_right(LEN_BUCKET_STARTS, len(ilp))
        bt = bisect.bisect_right(TOT_BUCKET_STARTS, len(iwp) + len(ilp))
        tally(WIN, GL + bw)
        tally(LOSS, GL + bl)
        tally(WIN, OG + bl)
        tally(LOSS, OG + bw)
        tally(WIN, TG + bt)
        tally(LOSS, TG + bt)

        # split halves: parity of this card's own in-deck appearance counter
        hw = appear[IWD] & 1
        appear[IWD] += 1
        hl = appear[ILD] & 1
        appear[ILD] += 1
        tally(WIN, H + hw, H + hw[: len(iwp)])
        tally(LOSS, H + hl, H + hl[: len(ilp)])
        if starter_won:
            tally(WIN, HP + hw, HP + hw[: len(iwp)])
        else:
            tally(LOSS, HP + hl, HP + hl[: len(ilp)])

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

    def counts(s: int, prefix: str = "", suffix: str = "", when: str = "") -> dict:
        n = slice(0, n_cards)
        return {column(outcome, kind, prefix, suffix, when): C[s, o, k, n]
                for outcome, o in (("wins", WIN), ("losses", LOSS))
                for kind, k in (("played", PLAYED), ("in_deck", IN_DECK))}

    cols = {"card_name": all_names, **counts(ALL, when="when_"),
            **counts(PLAY, suffix="_play"), **counts(FB, "fb_"), **counts(FF, "ff_")}
    for b in range(N_LEN_BUCKETS):
        cols |= counts(GL + b, f"gl{b}_") | counts(OG + b, f"og{b}_") | counts(TG + b, f"tg{b}_")
    for h in (0, 1):
        cols |= counts(H + h, f"h{h}_") | counts(HP + h, f"h{h}_", "_play")
    df = pd.DataFrame(cols)

    # derived mediation quantities
    n_in, score_overall, played_rate = stratum_scores(df, when="when_")
    n_play, score_play, _ = stratum_scores(df, suffix="_play")
    df["n_in_deck"] = n_in
    df["n_played"] = df.wins_when_played + df.losses_when_played
    with np.errstate(divide="ignore", invalid="ignore"):
        df["w"] = df.wins_when_in_deck / n_in
        df["a"] = df.wins_when_played / df.wins_when_in_deck
        df["b"] = df.losses_when_played / df.losses_when_in_deck
        df["m"] = (df.a + df.b) / 2.0
        df["d"] = df.a - df.b
        df["played_rate"] = played_rate
        df["score_overall"] = score_overall
        df["n_in_deck_play"] = n_play
        df["score_play"] = score_play
        df["n_in_deck_draw"] = n_in - n_play
        df["score_draw"] = draw_score(df, when="when_")
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

    print(f"winner mean cards played {sum_played_w / n_games:.3f}  "
          f"loser {sum_played_l / n_games:.3f}")


if __name__ == "__main__":
    sys.exit(main())
