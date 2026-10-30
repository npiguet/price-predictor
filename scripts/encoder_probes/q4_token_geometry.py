"""Do integer tokens carry magnitude, and does the label reward it?

Two measurements behind the integer-collapse discussion in
experiments/2026-08-28-encoder-preferences.md:

1. Token geometry: project each integer token's input embedding onto the
   number-line direction fit on the tokens 1-7 (least squares of embedding
   on value). A token that continues the line carries magnitude; the
   observed result is that every integer >= 8 projects to roughly "a four",
   in no consistent order.
2. The label's own utility curve: mean shrunk_score_play of real creatures
   by printed power, which plateaus at 5-6 and declines past 8 - so
   training never paid the encoder to order the big numbers.

Writes q4_token_geometry.csv and q4_power_labels.csv to
output/encoder-probes/ and prints both tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402


def token_projections(max_n: int = 15) -> pd.DataFrame:
    tok = pl.load_tokenizer(pl.VOCAB_PATH)
    ck = torch.load(pl.ENCODER_CKPT, map_location="cpu", weights_only=False)
    emb = ck["model_state_dict"]["token_encoder.token_embedding.weight"].numpy()

    ids = {}
    for n in range(max_n + 1):
        enc, _ = tok.encode(f"power toughness: {n}/{n}", 16)
        ids[n] = enc[3]
    vecs = np.stack([emb[ids[n]] for n in range(max_n + 1)])

    ns = np.arange(1, 8).astype(float)
    coef = np.linalg.lstsq(np.stack([ns, np.ones_like(ns)], 1),
                           vecs[1:8], rcond=None)[0][0]
    direction = coef / np.linalg.norm(coef)
    proj = vecs @ direction
    step = (proj[7] - proj[1]) / 6
    return pd.DataFrame({
        "integer": range(max_n + 1),
        "token_id": [ids[n] for n in range(max_n + 1)],
        "numberline_position": (proj - proj[1]) / step,
    })


def printed_power(text: str) -> float | None:
    """The literal power on the statline; None for ``*`` and the like."""
    pt = cc.get_field(text, "power toughness:")
    if pt is None:
        return None
    try:
        return float(pt.split("/", 1)[0])
    except ValueError:
        return None


def power_label_curve() -> pd.DataFrame:
    join = pl.load_frame(features=False, text=True)
    join = join.assign(power=[printed_power(t) for t in join["text"]])
    cr = join[join["power"].notna()]
    g = (cr.groupby("power")["shrunk_score_play"]
           .agg(mean_score_play="mean", n="count").reset_index())
    return g[(g["n"] >= 8) & (g["power"] <= 13)]


if __name__ == "__main__":
    t = token_projections()
    t.to_csv(pl.SCRATCH / "q4_token_geometry.csv", index=False)
    print("integer-token position on the 1-7 number line (units of one step):")
    print(t.round(2).to_string(index=False))

    c = power_label_curve()
    c.to_csv(pl.SCRATCH / "q4_power_labels.csv", index=False)
    print("\nmean shrunk_score_play by printed power (creatures, n >= 8):")
    print(c.round(4).to_string(index=False))
