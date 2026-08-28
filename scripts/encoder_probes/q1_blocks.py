"""R6 (a) per-block probes, (b) block ablation, (d) block redundancy.

The gen-4 encoder pools with 8 learned queries whose 64-dim outputs are
concatenated in query order (``_MultiQueryAttentionPool.forward``), so the
512-dim card vector is 8 contiguous 64-dim blocks, block ``k`` = dims
``[64k, 64k+64)``. This script asks whether those blocks carry different
information.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl  # noqa: E402
import q_common as qc  # noqa: E402

LABEL_TARGETS = ("score_play", "played_rate", "cast_lift", "color_lift_U")
FEATURE_TARGETS = ("mv", "power", "toughness", "pips_total", "is_creature")


def main() -> None:
    join, emb = qc.load_frame()
    is_train = (join["split"] == "train").to_numpy()
    is_val = (join["split"] == "val").to_numpy()
    print(f"cards={len(join)} train={is_train.sum()} val={is_val.sum()}", flush=True)

    targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for head in LABEL_TARGETS:
        y = pd.to_numeric(join[f"shrunk_{head}"], errors="coerce").to_numpy(float)
        keep = join[f"w_{head}"].to_numpy(float) > 0
        targets[head] = (y, keep)
    for feat in FEATURE_TARGETS:
        y = pd.to_numeric(join[feat], errors="coerce").to_numpy(float)
        keep = np.isfinite(y)
        if feat in ("power", "toughness"):
            keep &= join["is_creature"].fillna(0).to_numpy(float) > 0
        targets[feat] = (y, keep)

    # ── (a) per-block ridge, plus the full 512 baseline ──────────────────
    rows = []
    for name, (y, keep) in targets.items():
        full = qc.honest_ridge(emb, y, is_train, is_val, keep)
        row = {"target": name, "n_val": full["n_val"], "full512": full["val_r2"]}
        for k in range(qc.N_BLOCKS):
            sl = slice(k * qc.BLOCK, (k + 1) * qc.BLOCK)
            r = qc.honest_ridge(emb[:, sl], y, is_train, is_val, keep)
            row[f"b{k}"] = r["val_r2"]
        rows.append(row)
        print("block-probe", name, {k: round(v, 3) for k, v in row.items()
                                    if k.startswith(("b", "f"))}, flush=True)
    per_block = pd.DataFrame(rows)
    per_block.to_csv(qc.SCRATCH / "q1_block_probe_r2.csv", index=False)

    # ── (b) mean-substitution ablation against the honest probes ─────────
    ps = pl.load_probes("honest", True)
    train_mean = emb[is_train].mean(axis=0)
    abl_rows = []
    for probe_name, probe in ps.probes.items():
        head = probe.head
        y = pd.to_numeric(join[f"shrunk_{head}"], errors="coerce").to_numpy(float)
        if probe.space == "logit":
            y = pl.to_logit(y)
        w_all = join[f"w_{head}"].to_numpy(float)
        m = is_val & np.isfinite(y) & (w_all > 0)
        base = pl._r2(y[m], probe.predict(emb[m]), w_all[m])
        row = {"probe": probe_name, "base_val_r2": base}
        for k in range(qc.N_BLOCKS):
            sl = slice(k * qc.BLOCK, (k + 1) * qc.BLOCK)
            x = emb[m].copy()
            x[:, sl] = train_mean[sl]
            row[f"d_b{k}"] = pl._r2(y[m], probe.predict(x), w_all[m]) - base
        abl_rows.append(row)
        print("ablation", probe_name, flush=True)
    ablation = pd.DataFrame(abl_rows)
    ablation.to_csv(qc.SCRATCH / "q1_block_ablation.csv", index=False)

    # ── (d) redundancy ──────────────────────────────────────────────────
    pr_whole = qc.participation_ratio(emb)
    pcs = np.stack([
        qc.first_pc_scores(emb[:, k * qc.BLOCK:(k + 1) * qc.BLOCK])
        for k in range(qc.N_BLOCKS)
    ])
    pc_corr = np.corrcoef(pcs)
    block_pr = [
        qc.participation_ratio(emb[:, k * qc.BLOCK:(k + 1) * qc.BLOCK])
        for k in range(qc.N_BLOCKS)
    ]
    block_norm = [
        float(np.linalg.norm(emb[:, k * qc.BLOCK:(k + 1) * qc.BLOCK], axis=1).mean())
        for k in range(qc.N_BLOCKS)
    ]
    block_sd = [
        float(emb[:, k * qc.BLOCK:(k + 1) * qc.BLOCK].std(axis=0).mean())
        for k in range(qc.N_BLOCKS)
    ]
    # Cross-block CCA-free redundancy: R2 of predicting block j's PC1 from
    # block i's 64 dims is overkill; the PC1 correlation matrix is the ask.
    np.save(qc.SCRATCH / "q1_pc_corr.npy", pc_corr)
    pd.DataFrame(pc_corr).to_csv(qc.SCRATCH / "q1_pc_corr.csv", index=False)

    summary = {
        "n_cards": int(len(join)),
        "pr_whole": pr_whole,
        "block_pr": block_pr,
        "block_mean_norm": block_norm,
        "block_mean_dim_sd": block_sd,
        "pc1_abs_corr_offdiag_mean": float(
            np.abs(pc_corr[np.triu_indices(qc.N_BLOCKS, 1)]).mean()
        ),
        "pc1_abs_corr_offdiag_max": float(
            np.abs(pc_corr[np.triu_indices(qc.N_BLOCKS, 1)]).max()
        ),
    }
    with open(qc.SCRATCH / "q1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
