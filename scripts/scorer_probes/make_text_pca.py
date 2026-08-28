"""PCA of the text block ([0:512]) of every card embedding in cardsfolder-512.

Writes ``text_pca_512.npz`` (keys: mean, components, explained) into
``output/scorer-probes/`` — the input ``t6_mechanism.py`` P1 needs — and prints
the numbers quoted in ``experiments/2026-08-27-scorer-preferences.md``
("Two numbers per card ..."): the variance concentration (PC1 share, top-k
shares, participation ratio) and the meaning of the leading axes (R² of each
encoder training label regressed on the top-k PC coordinates, plus the signed
PC1/PC2 correlations that identify PC1 as the played-rate axis and PC2 as the
winnability axis).
"""

import glob
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "scorer-probes"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

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

# --- what the leading axes mean: regress the encoder labels on top-k PCs ---
import probe_lib as pl  # noqa: E402

wr = pl.load_win_rates()
loc = pl.ConvertedCardLocator(pl.CARDS_PATH)
stems = {Path(f).stem: i for i, f in enumerate(files)}
MAIN = ["shrunk_score_play", "shrunk_score_draw", "shrunk_played_rate",
        "shrunk_cast_lift"]
COLOR = [f"shrunk_color_lift_{c}" for c in "WUBRG"]
keep, rows = [], []
for name, rec in wr.items():
    p = loc.embedding_path(name)
    i = stems.get(p.stem) if p is not None else None
    vals = [rec.get(k) for k in MAIN + COLOR]
    if i is None or any(v is None for v in vals):
        continue
    keep.append(i)
    rows.append(vals)
Y = np.array(rows)
P = (X[np.array(keep)] - mu) @ Vt.T
print(f"\nlabel regression on top-k PCs ({len(keep)} labeled cards)")


def r2(k, y):
    A = np.column_stack([np.ones(len(y)), P[:, :k]])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return 1 - (y - A @ b).var() / y.var()


ks = [1, 2, 4, 8, 32]
table = {}
print(f"{'label':24s}" + "".join(f"  k={k:<4d}" for k in ks))
for j, lab in enumerate(MAIN):
    table[lab] = [r2(k, Y[:, j]) for k in ks]
    print(f"{lab:24s}" + "".join(f"  {v:6.2f}" for v in table[lab]))
table["color_lift_avg"] = [float(np.mean([r2(k, Y[:, len(MAIN) + c])
                                          for c in range(5)])) for k in ks]
print(f"{'color_lift (avg of 5)':24s}"
      + "".join(f"  {v:6.2f}" for v in table["color_lift_avg"]))
pc_corr = {}
for pc in (0, 1):
    cors = [float(np.corrcoef(P[:, pc], Y[:, j])[0, 1]) for j in range(len(MAIN))]
    pc_corr[f"PC{pc + 1}"] = dict(zip(MAIN, cors))
    print(f"PC{pc + 1} corr: " + ", ".join(
        f"{lab.removeprefix('shrunk_')} {c:+.2f}" for lab, c in zip(MAIN, cors)))

import json  # noqa: E402

(OUT / "text_pc_labels.json").write_text(json.dumps(
    {"ks": ks, "n_cards": len(keep), "r2": table, "pc_corr": pc_corr}, indent=1),
    encoding="utf-8")
print("saved", OUT / "text_pc_labels.json")
