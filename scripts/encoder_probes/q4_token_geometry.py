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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from price_predictor.infrastructure.tokenizer_store import load_tokenizer  # noqa: E402

import probe_lib as pl  # noqa: E402

OUT = REPO / "output" / "encoder-probes"


def token_projections(max_n: int = 15) -> pd.DataFrame:
    tok = load_tokenizer(REPO / "models/sealed/encoder/vocab.txt")
    ck = torch.load(pl.ENCODER_CKPT if hasattr(pl, "ENCODER_CKPT") else
                    REPO / "models/sealed/encoder/full-20260517-014759-attn-6l-8h-8q-0.1mlm-512d.pt",
                    map_location="cpu", weights_only=False)
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


def power_label_curve() -> pd.DataFrame:
    join = pl.build_join()
    join = join[join["is_primary"]]
    powers = []
    for _, row in join.iterrows():
        p = None
        for line in Path(row["txt_path"]).read_text(encoding="utf-8").splitlines():
            if line.startswith("power toughness:"):
                head = line.split(":", 1)[1].strip().split("/", 1)[0]
                try:
                    p = float(head)
                except ValueError:
                    p = None
                break
        powers.append(p)
    join = join.assign(power=powers)
    cr = join[join["power"].notna()]
    g = (cr.groupby("power")["shrunk_score_play"]
           .agg(mean_score_play="mean", n="count").reset_index())
    return g[(g["n"] >= 8) & (g["power"] <= 13)]


if __name__ == "__main__":
    t = token_projections()
    t.to_csv(OUT / "q4_token_geometry.csv", index=False)
    print("integer-token position on the 1-7 number line (units of one step):")
    print(t.round(2).to_string(index=False))

    c = power_label_curve()
    c.to_csv(OUT / "q4_power_labels.csv", index=False)
    print("\nmean shrunk_score_play by printed power (creatures, n >= 8):")
    print(c.round(4).to_string(index=False))
