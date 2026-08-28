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

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl  # noqa: E402

ABILITY_PREFIXES = ("static:", "spell[", "activated[", "triggered", "replacement")
N_PAIRS = 500
SEED = 31337
BATCH = 96
SD_SCORE_PLAY = 0.06181
SD_PLAYED_RATE = 0.1247


def card_text(path: str) -> str:
    return "\n".join(l for l in Path(path).read_text(encoding="utf-8", errors="replace")
                     .splitlines() if l.strip() and not l.startswith("name:"))


def main() -> None:
    join = pl.build_join()
    join = join[join["is_primary"]].reset_index(drop=True)

    texts, n_lines, n_ab = [], [], []
    for p in join["txt_path"]:
        t = card_text(p)
        lines = t.splitlines()
        texts.append(t)
        n_lines.append(len(lines))
        n_ab.append(sum(1 for l in lines if l.startswith(ABILITY_PREFIXES)))
    join["n_lines"], join["n_ability_lines"] = n_lines, n_ab

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
    specs: list[tuple[str, int, str, str]] = []
    for vi, ti, _ in pairs:
        for split, idx in (("val", vi), ("train", ti)):
            base = texts[idx]
            specs.append((split, idx, "base", base))
            for key, variant in pl.placebo_edits(base, freq).items():
                if variant is not None:
                    specs.append((split, idx, key, variant))

    runner = pl.EncoderRunner()
    emb = runner.encode_texts([s[3] for s in specs], batch_size=BATCH)
    print(f"encoded {len(specs)} variants on {runner.device}", flush=True)

    probes = pl.load_probes("fidelity", True)
    pred = pl.predict_labels(emb, probes)
    frame = pd.DataFrame({
        "split": [s[0] for s in specs], "row": [s[1] for s in specs],
        "edit": [s[2] for s in specs],
        "name": [join["name"].iloc[s[1]] for s in specs],
        "n_in_deck": [int(join["n_in_deck"].iloc[s[1]]) for s in specs],
        "score_play": pred["score_play"].to_numpy(),
        "played_rate": pred["played_rate"].to_numpy(),
    })
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
            # bootstrap CI on the median ratio
            rng2 = np.random.default_rng(7)
            draws = []
            for _ in range(4000):
                a = rng2.choice(samples["train"], len(samples["train"]))
                b = rng2.choice(samples["val"], len(samples["val"]))
                draws.append(np.median(a) / np.median(b))
            entry["ratio_ci95"] = [float(np.percentile(draws, 2.5)),
                                   float(np.percentile(draws, 97.5))]
        out[edit] = entry

    pd.DataFrame(rec_rows).to_csv(pl.SCRATCH / "r2c_placebo_by_split.csv", index=False)
    with open(pl.SCRATCH / "r2c_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
