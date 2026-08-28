"""R1b — line-attribution index: does `flying` pay only where it belongs?

Three counterfactual families, all read off the fidelity/weighted
``score_play`` probe and all gated through the manifold check:

A. **Single-line fliers** — creatures whose only ability line is
   ``static: flying``. Variants: original / flying deleted / deleted +
   a *spell* line that grants flying to something else / deleted + a
   *static* line that mentions flying under negation. The attribution
   index Δ(iii−ii)/Δ(i−ii) is 0 if the encoder pays the keyword only in
   its own slot and 1 if it pays the bare token wherever it appears.

B. **Real multi-line fliers** — delete ``static: flying`` from 300 random
   ones; the realistic-card estimate of the same premium.

C. **Matched non-fliers** — add ``static: flying`` to 300 non-flying
   creatures matched to (B) on mana value and power/toughness; the
   bidirectional check |Δadd| vs |Δremove|.

Outputs ``output/encoder-probes/r1b_*.csv`` + ``r1b_summary.json``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl  # noqa: E402

ABILITY_PREFIXES = ("static:", "spell[", "activated[", "triggered", "replacement")
GRANT_LINE = "spell[1]: target creature gains flying until end of turn."
NEGATED_LINE = "static: CARDNAME can't block creatures with flying."
FLY_LINE = "static: flying"
GATE = 0.35325
N_SAMPLE = 300
SEED = 4242
BATCH = 96
SD_SCORE_PLAY = 0.06181


def card_lines(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return [l for l in text.splitlines() if l.strip() and not l.startswith("name:")]


def mana_value(lines: list[str]) -> float:
    cost = next((l[len("mana cost:"):].strip() for l in lines
                 if l.startswith("mana cost:")), "")
    total = 0.0
    for sym in re.findall(r"\{([^}]*)\}", cost):
        if sym.isdigit():
            total += int(sym)
        elif sym.upper() == "X":
            continue
        else:
            total += 1
    return total


def power_toughness(lines: list[str]) -> tuple[float, float]:
    pt = next((l[len("power toughness:"):].strip() for l in lines
               if l.startswith("power toughness:")), "")
    m = re.match(r"(-?\d+)\s*/\s*(-?\d+)", pt)
    if not m:
        return (np.nan, np.nan)
    return (float(m.group(1)), float(m.group(2)))


def drop_line(lines: list[str], target: str) -> list[str]:
    return [l for l in lines if l.strip() != target]


def boot_ci(values: np.ndarray, stat=np.mean, n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    draws = stat(values[idx], axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def ratio_boot_ci(num: np.ndarray, den: np.ndarray, n: int = 4000, seed: int = 0):
    """Bootstrap CI of mean(num)/mean(den) over paired cards."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n, len(num)))
    draws = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    join = pl.build_join()
    join = join[join["is_primary"]].reset_index(drop=True)

    lines_by_idx = {i: card_lines(p) for i, p in enumerate(join["txt_path"])}

    singles, multi, nonfly = [], [], []
    for i, lines in lines_by_idx.items():
        types = next((l for l in lines if l.startswith("types:")), "")
        if "creature" not in types:
            continue
        ab = [l for l in lines if l.startswith(ABILITY_PREFIXES)]
        has_fly = any(l.strip() == FLY_LINE for l in ab)
        text_low = "\n".join(lines).lower()
        if has_fly and len(ab) == 1:
            singles.append(i)
        elif has_fly and len(ab) > 1:
            multi.append(i)
        elif "flying" not in text_low:
            nonfly.append(i)

    rng = np.random.default_rng(SEED)
    multi_s = sorted(rng.choice(multi, min(N_SAMPLE, len(multi)), replace=False))

    # matched non-fliers: greedy nearest on (mv, power, toughness)
    feats = np.array([[mana_value(lines_by_idx[i]), *power_toughness(lines_by_idx[i])]
                      for i in nonfly], dtype=float)
    ok = np.isfinite(feats).all(axis=1)
    pool_idx = np.array(nonfly)[ok]
    pool_feats = feats[ok]
    used = set()
    matched = []
    for i in multi_s:
        f = np.array([mana_value(lines_by_idx[i]), *power_toughness(lines_by_idx[i])])
        if not np.isfinite(f).all():
            continue
        d = np.abs(pool_feats - f).sum(axis=1)
        for j in np.argsort(d):
            if int(pool_idx[j]) not in used:
                used.add(int(pool_idx[j]))
                matched.append((i, int(pool_idx[j]), float(d[j])))
                break

    print(f"single-line fliers {len(singles)} | multi-line fliers {len(multi)} "
          f"(sampled {len(multi_s)}) | matched non-fliers {len(matched)}", flush=True)

    # ── build every variant text, encode once ───────────────────────────
    specs: list[tuple[str, int, str, str]] = []  # (family, row_idx, variant, text)

    for i in singles:
        base = lines_by_idx[i]
        stripped = drop_line(base, FLY_LINE)
        specs += [
            ("single", i, "i_original", "\n".join(base)),
            ("single", i, "ii_deleted", "\n".join(stripped)),
            ("single", i, "iii_grant_spell", "\n".join(stripped + [GRANT_LINE])),
            ("single", i, "iv_negated_static", "\n".join(stripped + [NEGATED_LINE])),
        ]
    for i in multi_s:
        base = lines_by_idx[i]
        specs += [
            ("multi", i, "i_original", "\n".join(base)),
            ("multi", i, "ii_deleted", "\n".join(drop_line(base, FLY_LINE))),
        ]
    for i, j, _ in matched:
        base = lines_by_idx[j]
        specs += [
            ("nonfly", j, "i_original", "\n".join(base)),
            ("nonfly", j, "ii_added", "\n".join(base + [FLY_LINE])),
        ]

    runner = pl.EncoderRunner()
    emb = runner.encode_texts([s[3] for s in specs], batch_size=BATCH)
    print(f"encoded {len(specs)} variants on {runner.device}", flush=True)

    probes = pl.load_probes("fidelity", True)
    pred = pl.predict_labels(emb, probes)
    _, corpus_ref = pl.corpus_embedding_matrix()
    mdist, _ = pl.manifold_distance(emb, corpus_ref)

    frame = pd.DataFrame({
        "family": [s[0] for s in specs],
        "row": [s[1] for s in specs],
        "variant": [s[2] for s in specs],
        "name": [join["name"].iloc[s[1]] for s in specs],
        "score_play": pred["score_play"].to_numpy(),
        "played_rate": pred["played_rate"].to_numpy(),
        "manifold_dist": mdist,
    })
    frame.to_csv(pl.SCRATCH / "r1b_variants.csv", index=False)

    wide = frame.pivot_table(index=["family", "row", "name"], columns="variant",
                             values=["score_play", "manifold_dist"])
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(pl.SCRATCH / "r1b_wide.csv", index=False)

    out: dict = {"n_single": len(singles), "n_multi_pool": len(multi),
                 "n_multi_sampled": len(multi_s), "n_matched": len(matched),
                 "label_sd_score_play": SD_SCORE_PLAY}

    # ── family A: attribution index ─────────────────────────────────────
    A = wide[wide["family"] == "single"].copy()
    for excl, tag in ((False, "all"), (True, "on_manifold")):
        sub = A
        if excl:
            cols = [c for c in A.columns if c.startswith("manifold_dist__")]
            sub = A[(A[cols] <= GATE).all(axis=1)]
        if len(sub) < 5:
            continue
        d_true = (sub["score_play__i_original"] - sub["score_play__ii_deleted"]).to_numpy()
        d_grant = (sub["score_play__iii_grant_spell"] - sub["score_play__ii_deleted"]).to_numpy()
        d_neg = (sub["score_play__iv_negated_static"] - sub["score_play__ii_deleted"]).to_numpy()
        rec = {"n": int(len(sub))}
        for key, d in (("true_flying_premium", d_true), ("grant_spell", d_grant),
                       ("negated_static", d_neg)):
            lo, hi = boot_ci(d, seed=1)
            rec[key] = {"mean": float(d.mean()), "mean_sd_units": float(d.mean() / SD_SCORE_PLAY),
                        "ci95": [lo, hi], "ci95_sd_units": [lo / SD_SCORE_PLAY, hi / SD_SCORE_PLAY],
                        "frac_positive": float((d > 0).mean())}
        for key, d in (("attribution_index_grant", d_grant), ("attribution_index_negated", d_neg)):
            lo, hi = ratio_boot_ci(d, d_true, seed=2)
            rec[key] = {"point": float(d.mean() / d_true.mean()), "ci95": [lo, hi]}
        rec["n_off_manifold_any"] = int(len(A) - len(sub)) if excl else None
        out[f"family_A_{tag}"] = rec

    # ── family B / C: bidirectional ─────────────────────────────────────
    B = wide[wide["family"] == "multi"].copy()
    d_rem = (B["score_play__i_original"] - B["score_play__ii_deleted"]).to_numpy()
    lo, hi = boot_ci(d_rem, seed=3)
    out["family_B_remove_from_real_fliers"] = {
        "n": int(len(B)), "mean": float(d_rem.mean()),
        "mean_sd_units": float(d_rem.mean() / SD_SCORE_PLAY), "ci95": [lo, hi],
        "frac_positive": float((d_rem > 0).mean()),
        "n_off_manifold": int((B["manifold_dist__ii_deleted"] > GATE).sum()),
    }
    C = wide[wide["family"] == "nonfly"].copy()
    d_add = (C["score_play__ii_added"] - C["score_play__i_original"]).to_numpy()
    lo, hi = boot_ci(d_add, seed=4)
    out["family_C_add_to_nonfliers"] = {
        "n": int(len(C)), "mean": float(d_add.mean()),
        "mean_sd_units": float(d_add.mean() / SD_SCORE_PLAY), "ci95": [lo, hi],
        "frac_positive": float((d_add > 0).mean()),
        "n_off_manifold": int((C["manifold_dist__ii_added"] > GATE).sum()),
    }
    out["bidirectional_ratio_add_over_remove"] = float(d_add.mean() / d_rem.mean())

    with open(pl.SCRATCH / "r1b_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
