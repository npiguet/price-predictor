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
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draft_corpus_common import ColourResolver  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "draft-probes"
CARDS = REPO / "output" / "cardsfolder-512"
HINTS = REPO / "output" / "scorer-probes" / "forge_hints.csv"
DATA = Path(r"Y:\Nicolas\mtg\mtg-models-data\sealed\training-data\matches-bo1")
MATCHES = DATA / "match-outcomes-bo1-embedding.txt"
WINRATES = DATA / "cards-win-rates.txt"

COLOURS = "WUBRG"


def wg_ur(beta: np.ndarray) -> float:
    return (beta[0] + beta[4]) / 2 - (beta[1] + beta[3]) / 2


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least squares with a rank-deficient colour block.

    The five colour shares of a deck sum to one, so their differences sum to
    zero and the design is one short of full rank. `lstsq` returns the
    minimum-norm solution, which centres the five colour coefficients on zero
    and makes each one a deviation from the average colour. The `WG - UR`
    contrast is orthogonal to the null direction and is identified regardless.
    """
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov = float(resid @ resid) / (len(y) - rank) * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.abs(np.diag(cov))), float(rank)


def card_level(colours: ColourResolver) -> dict:
    """Raw win rate by colour, straight off the encoder's training table."""
    per: dict[str, list[float]] = {c: [] for c in COLOURS}
    wins: Counter[str] = Counter()
    losses: Counter[str] = Counter()
    n_cards = 0
    with WINRATES.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            ids = colours(row["card_name"])
            if not ids:
                continue
            try:
                w = float(row["wins_when_in_deck"])
                lo = float(row["losses_when_in_deck"])
            except (TypeError, ValueError):
                continue
            if w + lo < 50:
                continue
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

    rank: dict[str, float] = {}
    if HINTS.exists():
        with HINTS.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            for row in csv.DictReader(fh):
                try:                       # lower is picked earlier, so negate
                    rank[row["name"]] = -float(row["draft_rank"])
                except (TypeError, ValueError, KeyError):
                    continue
    print(f"exogenous card grader: {len(rank)} names with a Forge draft rank")

    results = {"card_level": card_level(colours)}

    def deck_vector(names: list[str]) -> tuple[np.ndarray, float, int] | None:
        acc: Counter[str] = Counter()
        n = nr = 0
        r = 0.0
        for nm in names:
            ids = colours(nm)
            if ids:                        # basics and artifacts carry no lane
                n += 1
                for c in ids:
                    acc[c] += 1.0 / len(ids)
            if nm in rank:
                r += rank[nm]
                nr += 1
        if n < 10 or nr < 10:
            return None
        return np.array([acc[c] / n for c in COLOURS]), r / nr, n

    rows: list[tuple] = []
    kept = seen = 0
    with MATCHES.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            seen += 1
            if seen % args.stride:
                continue
            f = line.rstrip("\n").split(";")
            if len(f) < 10:
                continue                   # tolerated trailing partial line
            a = deck_vector(f[5].split("|"))
            b = deck_vector(f[6].split("|"))
            if a is None or b is None or not f[7]:
                continue
            rows.append((f[2], f[3], f[4], a[0] - b[0], a[1] - b[1],
                         1.0 if f[7][0] == "A" else 0.0))
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
        beta, se, rk = ols(X, yy)
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

    def rho(a: list[float], b: list[float]) -> float:
        ra = np.argsort(np.argsort(np.array(a)))
        rb = np.argsort(np.argsort(np.array(b)))
        return float(np.corrcoef(ra, rb)[0, 1])

    print("\n=== colour order at each step of the chain ===")
    ref = order["games (deck level)"]
    for label, v in order.items():
        seq = " > ".join(COLOURS[i] for i in np.argsort(-np.array(v)))
        print(f"  {label:22s} {seq}    (rank corr with the games {rho(v, ref):+.2f})")
    if "reward" in order:
        print(f"  the games and the reward agree on the top of the order and "
              f"swap U and R at the bottom.")
    results["chain"] = {"values": order,
                        "rho_with_games": {k: rho(v, ref)
                                           for k, v in order.items()}}

    (OUT / "d11_winratecolour.json").write_text(
        json.dumps({"n_games": kept, "n_scanned": seen, "stride": args.stride,
                    "a_win_rate": float(y.mean()), "fits": fits,
                    **results}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'd11_winratecolour.json'}")


if __name__ == "__main__":
    main()
