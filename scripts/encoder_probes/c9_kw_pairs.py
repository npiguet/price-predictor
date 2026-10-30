"""C9 — the full two-keyword interaction matrix, encoder-side.

Generalises the c1b deathtouch x trample 2x2 to all 120 pairs of the
sixteen studied keywords. Every arm adds exactly two ``static:`` lines to
the same base creature, so the line-add layout artifact cancels; the slot
a pair does not use is filled with the control keyword ``fear``, which is
outside the studied set, and any additive value it has cancels in the
interaction term::

    interaction(A, B) = P(A, B) - P(A, fear) - P(fear, B) + P(fear, fear)

Positive interaction means the pair is worth more together than the two
keywords separately (deathtouch+trample); negative means the second
keyword is partly wasted (two forms of evasion). Both slot orders are
encoded and averaged. Base creatures are keywordless, at most one ability
line, power+toughness 3-10, as in c1b.

Writes ``c9_kw_pairs.csv`` (one row per unordered pair, bootstrap CI over
base creatures) and prints the strongest interactions in both directions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402

KEYWORDS = cc.KEYWORDS
CONTROL = "fear"
N_BASE = 150
SEED = 7


def main() -> None:
    C = cc.CardCorpus.of(pl.load_frame(text=True))
    power, tough = C.power, C.toughness
    base = cc.sample_rows(C.is_creature & np.isfinite(power) & np.isfinite(tough)
                          & (C.kw_count == 0) & (C.n_abilities <= 1)
                          & (power + tough >= 3) & (power + tough <= 10),
                          N_BASE, np.random.default_rng(SEED))

    pairs = [(a, b) for i, a in enumerate(KEYWORDS) for b in KEYWORDS[i + 1:]]
    # arms per base: control-control, (K, ctrl) and (ctrl, K) singles,
    # and both orders of every pair
    arms: list[tuple[str, str]] = [(CONTROL, CONTROL)]
    arms += [(k, CONTROL) for k in KEYWORDS]
    arms += [(CONTROL, k) for k in KEYWORDS]
    arms += [(a, b) for a, b in pairs]
    arms += [(b, a) for a, b in pairs]
    idx = {arm: i for i, arm in enumerate(arms)}

    print(f"encoding {len(base) * len(arms)} texts ({len(base)} bases x {len(arms)} arms)",
          flush=True)
    scored = pl.score_arms(base, arms, lambda r, kk: cc.add_line(
        cc.add_line(C.text[r], f"static: {kk[0]}"), f"static: {kk[1]}"))
    P = scored.sp

    p00 = P[:, idx[(CONTROL, CONTROL)]]
    rows = []
    for a, b in pairs:
        inter_ab = (P[:, idx[(a, b)]] - P[:, idx[(a, CONTROL)]]
                    - P[:, idx[(CONTROL, b)]] + p00)
        inter_ba = (P[:, idx[(b, a)]] - P[:, idx[(b, CONTROL)]]
                    - P[:, idx[(CONTROL, a)]] + p00)
        inter = (inter_ab + inter_ba) / 2
        joint = ((P[:, idx[(a, b)]] + P[:, idx[(b, a)]]) / 2 - p00)
        off = float(scored.off[:, [idx[(a, b)], idx[(b, a)]]].mean())
        rows.append({
            "kw_a": a, "kw_b": b, "n": len(base),
            **pl.mean_ci(inter, "interaction_sd", n_boot=1000),
            "joint_vs_control_sd": float(joint.mean()),
            "order_gap_sd": float((inter_ab - inter_ba).mean()),
            "frac_pos": float((inter > 0).mean()),
            "off_manifold": off,
        })
    out = pd.DataFrame(rows)

    # the fear-referenced values are only identified relative to fear's own
    # interactions; double-centering the symmetric pair matrix (subtract each
    # keyword's mean interaction, add back the grand mean) removes any
    # additive per-keyword reference and leaves the pair-specific structure
    M = pd.DataFrame(np.nan, index=KEYWORDS, columns=KEYWORDS)
    for r in rows:
        M.loc[r["kw_a"], r["kw_b"]] = r["interaction_sd"]
        M.loc[r["kw_b"], r["kw_a"]] = r["interaction_sd"]
    row_mean = M.mean(axis=1)
    grand = float(np.nanmean(M.to_numpy()))
    out["interaction_centered_sd"] = [
        r["interaction_sd"] - row_mean[r["kw_a"]] - row_mean[r["kw_b"]] + grand
        for r in rows]
    out = out.sort_values("interaction_sd", ascending=False)
    out.to_csv(pl.SCRATCH / "c9_kw_pairs.csv", index=False)

    sig = out[(out["ci_lo"] > 0) | (out["ci_hi"] < 0)]
    print(f"\npairs with CI excluding zero: {len(sig)} of {len(out)}")
    print("\nstrongest positive interactions:")
    print(out.head(10).round(3).to_string(index=False))
    print("\nstrongest negative interactions:")
    print(out.tail(10).round(3).to_string(index=False))
    dt = out[(out["kw_a"] == "deathtouch") & (out["kw_b"] == "trample")]
    print("\nsanity (c1b measured +0.14 with vigilance/reach controls):")
    print(dt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
