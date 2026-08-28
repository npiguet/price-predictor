"""PCA of the text block ([0:512]) of every card embedding in cardsfolder-512.

Writes ``text_pca_512.npz`` (keys: mean, components, explained) into
``output/scorer-probes/`` — the input ``t6_mechanism.py`` P1 needs — and prints
the variance-concentration numbers quoted in
``experiments/2026-08-27-scorer-preferences.md`` ("Two numbers per card ...":
PC1 share, top-k shares, participation ratio).
"""

import glob
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "scorer-probes"
OUT.mkdir(parents=True, exist_ok=True)

t = time.time()
files = glob.glob(str(REPO / "output" / "cardsfolder-512" / "*" / "*.npz"))
print(len(files), "npz files")
X = np.empty((len(files), 512), dtype=np.float32)
for i, f in enumerate(files):
    with np.load(f) as z:
        X[i] = z["embedding"][:512]
print(f"loaded in {time.time() - t:.0f}s")

mu = X.mean(0)
U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
var = S ** 2
share = var / var.sum()
pr = (var.sum() ** 2) / ((var ** 2).sum())
print("PC1 share: %.1f%%, top2: %.1f%%, top8: %.1f%%, top32: %.1f%%"
      % (100 * share[0], 100 * share[:2].sum(), 100 * share[:8].sum(), 100 * share[:32].sum()))
print("effective rank (participation ratio): %.1f" % pr)
k90 = int(np.searchsorted(np.cumsum(share), 0.90) + 1)
k99 = int(np.searchsorted(np.cumsum(share), 0.99) + 1)
print("90%% of variance in", k90, "axes; 99%% in", k99)

np.savez_compressed(OUT / "text_pca_512.npz", mean=mu, components=Vt, explained=share)
print("saved", OUT / "text_pca_512.npz")
