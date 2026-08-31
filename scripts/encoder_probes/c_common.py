"""Shared scaffolding for the C-series counterfactual content battery (R10-R16).

Every C script imports this. It owns:

* the join table + the ``card_table.pkl`` feature frame, merged on card name;
* the fidelity/weighted probe set and a ``predict`` helper that returns the
  two headline columns (``score_play``, ``played_rate``) already divided by
  their label SD, so every number in the report is in label-SD units;
* converted-card text surgery (read, split into fields, substitute inside a
  line, add/remove a line) plus a canonical line-order re-serializer;
* paired-contrast bookkeeping: bootstrap CIs, correct-direction fractions,
  off-manifold fractions.

Design rule inherited from R1/R2: positional placebos are large and negatively
biased, token substitutions are clean. So every contrast here is built as a
*paired same-card difference between two arms with identical layout*.
"""

from __future__ import annotations

import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl  # noqa: E402

SCRATCH = pl.SCRATCH
CARD_TABLE = SCRATCH / "card_table.pkl"  # written by p0b_card_table.py

SD = {"score_play": 0.06181, "played_rate": 0.1247}
MANIFOLD_GATE = 0.35325

# ── data ────────────────────────────────────────────────────────────────

_JOIN: pd.DataFrame | None = None
_PROBES = None
_RUNNER: pl.EncoderRunner | None = None
_REF: np.ndarray | None = None


def join_table() -> pd.DataFrame:
    """Primary joined cards merged with the grounding feature table."""
    global _JOIN
    if _JOIN is None:
        j = pl.build_join()
        j = j[j["is_primary"]].reset_index(drop=True)
        with open(CARD_TABLE, "rb") as f:
            ct = pickle.load(f)
        feat_cols = [c for c in ct.columns
                     if c not in set(j.columns) and c != "card_name"]
        ct = ct[["card_name"] + feat_cols].drop_duplicates("card_name")
        j = j.merge(ct, left_on="name", right_on="card_name", how="left")
        j["text"] = [Path(p).read_text(encoding="utf-8", errors="replace")
                     for p in j["txt_path"]]
        _JOIN = j
    return _JOIN


def probes():
    global _PROBES
    if _PROBES is None:
        _PROBES = pl.load_probes("fidelity", True)
    return _PROBES


def runner() -> pl.EncoderRunner:
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = pl.EncoderRunner()
    return _RUNNER


def reference_cloud() -> np.ndarray:
    global _REF
    if _REF is None:
        _, _REF = pl.corpus_embedding_matrix()
    return _REF


def encode(texts: Sequence[str], batch_size: int = 96) -> np.ndarray:
    return runner().encode_texts(list(texts), batch_size=batch_size)


def predict_sd(embeddings: np.ndarray) -> pd.DataFrame:
    """Probe predictions with ``score_play``/``played_rate`` in label-SD units."""
    pred = pl.predict_labels(embeddings, probes())
    out = pd.DataFrame(index=pred.index)
    out["score_play"] = pred["score_play"] / SD["score_play"]
    out["played_rate"] = pred["played_rate"] / SD["played_rate"]
    out["cast_lift"] = pred["cast_lift"]
    return out


def offmanifold(embeddings: np.ndarray) -> np.ndarray:
    dist, _ = pl.manifold_distance(embeddings, reference_cloud())
    return dist


# ── converted-card text surgery ─────────────────────────────────────────

FIELD_ORDER = (
    "name:", "mana cost:", "types:", "power toughness:", "loyalty:",
    "static:", "replacement", "triggered", "activated", "spell", "other",
)
ABILITY_PREFIXES = ("static:", "spell[", "activated[", "triggered", "replacement")


def strip_name(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.startswith("name:"))


def lines(text: str) -> list[str]:
    return [l for l in text.splitlines() if l.strip()]


def find_line(ls: Sequence[str], prefix: str) -> int:
    for i, l in enumerate(ls):
        if l.startswith(prefix):
            return i
    return -1


def get_field(text: str, prefix: str) -> str | None:
    for l in lines(text):
        if l.startswith(prefix):
            return l[len(prefix):].strip()
    return None


def set_field(text: str, prefix: str, value: str) -> str:
    ls = lines(text)
    i = find_line(ls, prefix)
    if i < 0:
        return text
    ls[i] = f"{prefix} {value}"
    return "\n".join(ls)


def static_lines(text: str) -> list[int]:
    return [i for i, l in enumerate(lines(text)) if l.startswith("static:")]


def ability_lines(text: str) -> list[int]:
    return [i for i, l in enumerate(lines(text)) if l.startswith(ABILITY_PREFIXES)]


def replace_in_line(text: str, index: int, new_line: str) -> str:
    ls = lines(text)
    ls[index] = new_line
    return "\n".join(ls)


def _insert_slot(ls: Sequence[str], new_line: str) -> int:
    """Canonical insertion index for ``new_line`` (keeps field order)."""
    if new_line.startswith("static:"):
        st = [i for i, l in enumerate(ls) if l.startswith("static:")]
        if st:
            return st[-1] + 1
        # after power toughness / types
        for pref in ("power toughness:", "types:", "mana cost:"):
            i = find_line(ls, pref)
            if i >= 0:
                return i + 1
        return len(ls)
    return len(ls)


def add_line(text: str, new_line: str) -> str:
    ls = lines(text)
    ls.insert(_insert_slot(ls, new_line), new_line)
    return "\n".join(ls)


def drop_line(text: str, index: int) -> str:
    ls = lines(text)
    del ls[index]
    return "\n".join(ls)


def renumber(text: str) -> str:
    """Renumber ``spell[k]`` / ``triggered[k]`` / ``activated[k]`` indices."""
    counts: dict[str, int] = {}
    out = []
    for l in lines(text):
        m = re.match(r"^(static|spell|activated|triggered|replacement)\[(\d+)\]:", l)
        if m:
            kind = m.group(1)
            counts[kind] = counts.get(kind, 0) + 1
            l = re.sub(r"^(\w+)\[\d+\]:", rf"\1[{counts[kind]}]:", l)
        out.append(l)
    return "\n".join(out)


# ── contrast bookkeeping ────────────────────────────────────────────────


def bootstrap_ci(values: np.ndarray, n_boot: int = 4000, seed: int = 7,
                 alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


@dataclass
class Contrast:
    """One paired arm-A-minus-arm-B result over n base cards."""

    label: str
    delta: np.ndarray          # per-card difference, label-SD units
    off: float = 0.0           # fraction of rows off-manifold in either arm
    head: str = "score_play"

    def row(self) -> dict:
        d = np.asarray(self.delta, dtype=float)
        d = d[np.isfinite(d)]
        lo, hi = bootstrap_ci(d)
        return {
            "contrast": self.label,
            "head": self.head,
            "n": len(d),
            "mean_sd": float(d.mean()) if len(d) else float("nan"),
            "ci_lo": lo,
            "ci_hi": hi,
            "median_sd": float(np.median(d)) if len(d) else float("nan"),
            "frac_pos": float((d > 0).mean()) if len(d) else float("nan"),
            "off_manifold": self.off,
        }


def contrast_frame(contrasts: Iterable[Contrast]) -> pd.DataFrame:
    return pd.DataFrame([c.row() for c in contrasts])


def md_table(df: pd.DataFrame, floatfmt: str = "{:+.3f}") -> str:
    """Markdown table; numeric columns formatted, ints left alone."""
    def fmt(v, col):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, float) and np.isfinite(v):
            if col in ("n",):
                return str(int(v))
            if col in ("frac_pos", "off_manifold", "frac_correct"):
                return f"{v:.3f}"
            return floatfmt.format(v)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return str(v)

    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(fmt(r[c], c) for c in cols) + " |"
        for _, r in df.iterrows()
    ]
    return "\n".join([head, rule] + body)


def save(name: str, obj) -> Path:
    path = SCRATCH / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        import json
        path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    return path
