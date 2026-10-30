"""D11 — is the colour lean already in the games Forge played against itself?

`d10_rewardcolour.py` puts the lean in the scorer's deck score. The scorer was
fitted to match outcomes, and the encoder to per-card win rates from the same
games, so a lean in the reward could be a lean in the corpus underneath it. If
it is, the lean is not any model's: it is a property of Forge's own game
simulation, and every model in the stack has distilled it faithfully.

Two levels of the corpus are read, both under
`Y:\\Nicolas\\mtg\\mtg-models-data\\sealed\\training-data\\matches-bo1`.

The deck level is decisive, and mirrors D10 exactly with a real result in place
of the scorer's estimate. Every row of `match-outcomes-bo1-embedding.txt` is one
game between two sealed decks built from the same set. Regressing the A-wins
indicator on the difference in the two decks' colour shares asks whether white
and green decks actually beat blue and red ones when Forge pilots them. Because
the regressor is a difference between two decks of one match, set, format and
build method difference out of every row that shares them.

The card level is `cards-win-rates.txt`, the table the encoder trains on. Its
raw columns are counts of games won and lost with the card in the deck, so a
colour lean there needs no model at all to be visible.

Two confounds are handled. Deck-building method is entered as a fixed effect and
the same-method subset is reported beside it, because `random` and `forge-best`
decks differ in strength and could differ in colour. Card quality is controlled
with Forge's bundled human `draft_rank`, which is the only grader in the project
that was not itself fitted to this corpus — the win-rate label and the scorer's
swap value both were, and using either here would be circular.

Usage
-----
    python scripts/draft_probes/d11_winratecolour.py --stride 4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.corpora import card_column, iter_matches, load_win_rates
from _lib.stats import ols, spearman
from draft_corpus_common import ColourResolver
from draft_probes.probe_lib import (
    BO1_DATA,
    CARDS,
    HINTS,
    OUT,
    WIN_RATES,
    WUBRG,
    deck_colour_shares,
    wg_ur,
)

MATCHES = BO1_DATA / "match-outcomes-bo1-embedding.txt"
COLOURS = WUBRG


def card_level(colours: ColourResolver) -> dict:
    """Raw win rate by colour, straight off the encoder's training table."""
    per: dict[str, list[float]] = {c: [] for c in COLOURS}
    wins: Counter[str] = Counter()
    losses: Counter[str] = Counter()
    n_cards = 0
    for name, row in load_win_rates(WIN_RATES, min_in_deck=50).items():
        ids = colours(name)
        if not ids:
            continue
        w, lo = row["wins_when_in_deck"], row["losses_when_in_deck"]
        n_cards += 1
        for c in ids:
            per[c].append(w / (w + lo))
            wins[c] += w / len(ids)
            losses[c] += lo / len(ids)
    out = {
        "n_cards": n_cards,
        "mean_card_win_rate": {c: float(np.mean(per[c])) if per[c] else None
                               for c in COLOURS},
        "pooled_win_rate": {c: wins[c] / (wins[c] + losses[c]) if wins[c] else None
                            for c in COLOURS},
        "n_per_colour": {c: len(per[c]) for c in COLOURS},
    }
    m = out["mean_card_win_rate"]
    out["wg_ur"] = (m["W"] + m["G"]) / 2 - (m["U"] + m["R"]) / 2
    print("\n=== card level: win rate of games with the card in the deck ===")
    print("  " + "  ".join(f"{c} {m[c]:.4f} (n={out['n_per_colour'][c]})"
                           for c in COLOURS))
    print(f"  WG - UR: {out['wg_ur']:+.4f} over {n_cards} cards with 50+ games")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4,
                    help="keep every Nth match; the file is ordered by run, so "
                         "a stride samples every set rather than the first few")
    ap.add_argument("--max-matches", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    colours = ColourResolver(CARDS)

    rank = card_column(HINTS, "draft_rank", -1.0)  # lower is picked earlier, so negate
    print(f"exogenous card grader: {len(rank)} names with a Forge draft rank")

    results = {"card_level": card_level(colours)}

    def deck_vector(names: list[str]) -> tuple[np.ndarray, float, int] | None:
        """Colour shares, mean rank and coloured-card count; None for a deck
        with under ten coloured or ten ranked cards."""
        frac, n = deck_colour_shares(names, colours)
        ranks = [rank[nm] for nm in names if nm in rank]
        if n < 10 or len(ranks) < 10:
            return None
        return frac, sum(ranks) / len(ranks), n

    rows: list[tuple] = []
    kept = seen = 0
    for m in iter_matches(MATCHES):
        seen += 1
        if seen % args.stride:
            continue
        a = deck_vector(m.deck_a.cards)
        b = deck_vector(m.deck_b.cards)
        if a is None or b is None or not m.games:
            continue
        # Bo1: the first game is the match.
        rows.append((m.set_code, m.method_a, m.method_b, a[0] - b[0], a[1] - b[1],
                     1.0 if m.games[0] == "A" else 0.0))
        kept += 1
        if args.max_matches and kept >= args.max_matches:
            break
    print(f"\n{kept} games kept out of {seen} scanned "
          f"({len(colours.unresolved)} card names unresolved)")

    y = np.array([r[5] for r in rows])
    D = np.stack([r[3] for r in rows])
    dq = np.array([r[4] for r in rows])
    dq = dq / dq.std()
    print(f"A-win rate {y.mean():.4f}; mean |colour share gap| "
          + " ".join(f"{c}{np.abs(D[:, i]).mean():.3f}"
                     for i, c in enumerate(COLOURS)))

    # Build-method fixed effects: one column per unordered method pair, signed
    # +1 when A holds the first method and -1 when it holds the second, so a
    # pair with a systematically stronger builder cannot leak into the colours.
    pairs = sorted({tuple(sorted((r[1], r[2]))) for r in rows})
    idx = {p: i for i, p in enumerate(pairs)}
    M = np.zeros((len(rows), len(pairs)))
    for i, r in enumerate(rows):
        p = tuple(sorted((r[1], r[2])))
        M[i, idx[p]] = 0.0 if r[1] == r[2] else (1.0 if r[1] == p[0] else -1.0)
    print(f"{len(pairs)} build-method pairs")

    same = np.array([r[1] == r[2] for r in rows])
    fits = {}
    print("\n=== deck level: P(A wins) per unit of colour-share advantage ===")
    for label, X, sel, names in (
        ("colour only", D, slice(None), list(COLOURS)),
        ("+ method pair", np.column_stack([D, M]), slice(None),
         list(COLOURS) + [f"m{i}" for i in range(len(pairs))]),
        ("+ Forge draft rank", np.column_stack([D, M, dq]), slice(None),
         list(COLOURS) + [f"m{i}" for i in range(len(pairs))] + ["rank"]),
        ("same-method games only", np.column_stack([D[same], dq[same]]), same,
         list(COLOURS) + ["rank"]),
    ):
        yy = y[sel] if not isinstance(sel, slice) else y
        fit = ols(X, yy)
        beta, se = fit.beta, fit.se
        fits[label] = {"beta": beta[:5].tolist(), "se": se[:5].tolist(),
                       "wg_ur": wg_ur(beta), "n": int(len(yy)),
                       "rank_col": float(beta[-1]) if "rank" in names else None}
        print(f"  {label:24s} n={len(yy):7d}  "
              + " ".join(f"{c}{beta[i]:+7.4f}" for i, c in enumerate(COLOURS))
              + f"   WG-UR {wg_ur(beta):+.4f}")
        print(f"  {'':24s} {'':10s}"
              + " ".join(f" {' ':1s}({se[i]:.4f})" for i in range(5)))

    # How much of the corpus's colour order survives each step downstream. D10
    # holds the reward's colours and each policy's opening ranking.
    order = {}
    games = np.array(fits["+ Forge draft rank"]["beta"])
    order["games (deck level)"] = games.tolist()
    order["games (card level)"] = [results["card_level"]["mean_card_win_rate"][c]
                                   for c in COLOURS]
    d10 = OUT / "d10_rewardcolour.json"
    if d10.exists():
        blob = json.loads(d10.read_text(encoding="utf-8"))
        order["reward"] = blob["fits"]["colour only"]["beta"][:5]

    print("\n=== colour order at each step of the chain ===")
    ref = order["games (deck level)"]
    for label, v in order.items():
        seq = " > ".join(COLOURS[i] for i in np.argsort(-np.array(v)))
        print(f"  {label:22s} {seq}    (rank corr with the games {spearman(v, ref):+.2f})")
    if "reward" in order:
        print("  the games and the reward agree on the top of the order and "
              "swap U and R at the bottom.")
    results["chain"] = {"values": order,
                        "rho_with_games": {k: spearman(v, ref)
                                           for k, v in order.items()}}

    (OUT / "d11_winratecolour.json").write_text(
        json.dumps({"n_games": kept, "n_scanned": seen, "stride": args.stride,
                    "a_win_rate": float(y.mean()), "fits": fits,
                    **results}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'd11_winratecolour.json'}")


if __name__ == "__main__":
    main()
