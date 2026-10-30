"""R2c — is the encoder more brittle on cards it was trained on?

Matched pairs of encoder-train and encoder-val cards (same ability-line
count, same total line count, nearest ``log n_in_deck``), each carrying at
least two ability lines so ``swap_ability_lines`` applies. Every placebo
edit is re-encoded and read off the fidelity/weighted ``score_play``
probe; if the *train* half's predictions move more under a
meaning-preserving rewrite, the encoder is keying on the exact text
fingerprint of the cards it memorised.

Outputs ``output/encoder-probes/r2c_*.csv`` + ``r2c_summary.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402

N_PAIRS = 500
SEED = 31337
N_BOOT = 4000
SD_SCORE_PLAY = pl.SD["score_play"]
SD_PLAYED_RATE = pl.SD["played_rate"]


def main() -> None:
    join = pl.load_frame(features=False, text=True)
    texts = ["\n".join(cc.lines(t)) for t in join["text"]]
    join["n_lines"] = [len(cc.lines(t)) for t in texts]
    join["n_ability_lines"] = [len(cc.ability_lines(t)) for t in texts]

    ok = (join["n_ability_lines"] >= 2).to_numpy()
    is_train = (join["split"] == "train").to_numpy() & ok
    is_val = (join["split"] == "val").to_numpy() & ok
    print(f"eligible: {is_train.sum()} train / {is_val.sum()} val", flush=True)

    rng = np.random.default_rng(SEED)
    val_idx = rng.choice(np.flatnonzero(is_val), min(N_PAIRS, int(is_val.sum())),
                         replace=False)
    train_pool = np.flatnonzero(is_train)
    tp_feat = np.column_stack([
        np.log1p(join["n_in_deck"].to_numpy(float)[train_pool]),
        join["n_ability_lines"].to_numpy(float)[train_pool],
        join["n_lines"].to_numpy(float)[train_pool],
    ])
    used, pairs = set(), []
    for vi in val_idx:
        f = np.array([np.log1p(float(join["n_in_deck"].iloc[vi])),
                      float(join["n_ability_lines"].iloc[vi]),
                      float(join["n_lines"].iloc[vi])])
        d = (np.abs(tp_feat - f) * np.array([1.0, 3.0, 1.0])).sum(axis=1)
        for k in np.argsort(d):
            ti = int(train_pool[k])
            if ti not in used:
                used.add(ti)
                pairs.append((int(vi), ti, float(d[k])))
                break
    print(f"matched {len(pairs)} pairs", flush=True)

    freq = pl.subtype_frequencies()
    specs: list[pl.Variant] = []
    for vi, ti, _ in pairs:
        for split, idx in (("val", vi), ("train", ti)):
            base = texts[idx]
            specs.append(pl.Variant(split, idx, "base", base))
            for key, variant in pl.placebo_edits(base, freq).items():
                if variant is not None:
                    specs.append(pl.Variant(split, idx, key, variant))

    frame = pl.score_variants(specs, join, distance=False).rename(
        columns={"family": "split", "variant": "edit"})
    frame.insert(4, "n_in_deck", join["n_in_deck"].iloc[frame["row"]].astype(int).to_numpy())
    frame.to_csv(pl.SCRATCH / "r2c_variants.csv", index=False)

    base = frame[frame["edit"] == "base"].set_index("row")
    out: dict = {"n_pairs": len(pairs), "label_sd_score_play": SD_SCORE_PLAY}
    balance = {}
    for split in ("train", "val"):
        rows = [p[1] if split == "train" else p[0] for p in pairs]
        balance[split] = {
            "median_n_in_deck": float(join["n_in_deck"].iloc[rows].median()),
            "mean_n_ability_lines": float(join["n_ability_lines"].iloc[rows].mean()),
            "mean_n_lines": float(join["n_lines"].iloc[rows].mean()),
        }
    out["match_balance"] = balance

    rec_rows = []
    for edit in ("swap_static", "subtype_swap", "swap_ability_lines"):
        sub = frame[frame["edit"] == edit]
        entry: dict = {}
        samples = {}
        for split in ("train", "val"):
            s = sub[sub["split"] == split]
            if len(s) < 5:
                continue
            d = np.abs(s["score_play"].to_numpy()
                       - base.loc[s["row"], "score_play"].to_numpy())
            dp = np.abs(s["played_rate"].to_numpy()
                        - base.loc[s["row"], "played_rate"].to_numpy())
            samples[split] = d
            entry[split] = {
                "n": int(len(d)),
                "median_abs_delta": float(np.median(d)),
                "median_abs_delta_sd_units": float(np.median(d) / SD_SCORE_PLAY),
                "mean_abs_delta_sd_units": float(d.mean() / SD_SCORE_PLAY),
                "p95_abs_delta_sd_units": float(np.percentile(d, 95) / SD_SCORE_PLAY),
                "played_rate_median_abs_sd_units": float(np.median(dp) / SD_PLAYED_RATE),
            }
            rec_rows.append({"edit": edit, "split": split, **entry[split]})
        if len(samples) == 2:
            u, p = mannwhitneyu(samples["train"], samples["val"], alternative="two-sided")
            entry["mannwhitney_p_two_sided"] = float(p)
            entry["ratio_train_over_val_median"] = float(
                np.median(samples["train"]) / np.median(samples["val"]))
            # bootstrap CI on the median ratio, both halves resampled per draw
            rng2 = np.random.default_rng(7)
            tr, va = samples["train"], samples["val"]
            draws = (np.median(tr[rng2.integers(0, len(tr), (N_BOOT, len(tr)))], axis=1)
                     / np.median(va[rng2.integers(0, len(va), (N_BOOT, len(va)))], axis=1))
            entry["ratio_ci95"] = [float(np.percentile(draws, 2.5)),
                                   float(np.percentile(draws, 97.5))]
        out[edit] = entry

    pd.DataFrame(rec_rows).to_csv(pl.SCRATCH / "r2c_placebo_by_split.csv", index=False)
    pl.write_summary(out, "r2c_summary.json")


if __name__ == "__main__":
    main()
