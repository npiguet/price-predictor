"""R1c — scope and negation flips against a matched placebo null.

Four minimal edits, each of which should push ``score_play`` *down* if the
encoder reads scope rather than tokens:

a. **lockdown aura** — ``enchanted creature can't attack or block`` →
   ``CARDNAME can't attack or block`` (the restriction now binds the aura
   itself, which cannot attack anyway: the card becomes blank).
b. **removal** — ``destroy target [adj] creature.`` → ``... you control.``
c. **pump** — ``gets +N/+N until end of turn`` → ``gets -N/-N ...``.
d. **targeting** — ``target opponent`` → ``you``.

Each family carries two nulls measured on *its own base cards*: the
placebo-edit family from ``probe_lib`` where the card supports one, and a
universal line-order permutation (the converted format is a field list,
so line order is a formatting convention, not meaning).

Outputs ``output/encoder-probes/r1c_*.csv`` + ``r1c_summary.json``.
"""

from __future__ import annotations

import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402
from _lib.stats import bootstrap_ci  # noqa: E402

GATE = pl.MANIFOLD_GATE
SD_SCORE_PLAY = pl.SD["score_play"]
BATCH = 96
SEED = 909
MAX_PER_FAMILY = 400

LOCKDOWN = re.compile(r"enchanted creature can't attack or block", re.I)
REMOVAL = re.compile(r"(destroy target (?:[a-z]+ )?creature)\.", re.I)
PUMP = re.compile(r"\+(\d+)/\+(\d+) until end of turn", re.I)
OPPONENT = re.compile(r"target opponent", re.I)


def edit_lockdown(t: str) -> str | None:
    return LOCKDOWN.sub("CARDNAME can't attack or block", t) if LOCKDOWN.search(t) else None


def edit_removal(t: str) -> str | None:
    if not REMOVAL.search(t) or "you control" in t.lower():
        return None
    return REMOVAL.sub(lambda m: m.group(1) + " you control.", t)


def edit_pump(t: str) -> str | None:
    if not PUMP.search(t):
        return None
    return PUMP.sub(lambda m: f"-{m.group(1)}/-{m.group(2)} until end of turn", t)


def edit_opponent(t: str) -> str | None:
    return OPPONENT.sub("you", t) if OPPONENT.search(t) else None


FAMILIES = {
    "a_lockdown_aura": edit_lockdown,
    "b_removal_you_control": edit_removal,
    "c_pump_to_shrink": edit_pump,
    "d_opponent_to_you": edit_opponent,
}


def line_shuffle(t: str, rng: np.random.Generator) -> str:
    lines = t.splitlines()
    if len(lines) < 2:
        return t
    for _ in range(8):
        order = rng.permutation(len(lines))
        if list(order) != list(range(len(lines))):
            break
    return "\n".join(lines[i] for i in order)


def main() -> None:
    join = pl.load_frame(features=False, text=True)
    texts = {i: "\n".join(cc.lines(t)) for i, t in enumerate(join["text"])}

    rng = np.random.default_rng(SEED)
    freq = pl.subtype_frequencies()

    specs: list[tuple[str, int, str, str]] = []  # family, row, variant, text
    for family, fn in FAMILIES.items():
        hits = [i for i, t in texts.items() if fn(t) is not None]
        if len(hits) > MAX_PER_FAMILY:
            hits = sorted(rng.choice(hits, MAX_PER_FAMILY, replace=False))
        print(f"{family}: {len(hits)} cards", flush=True)
        for i in hits:
            base = texts[i]
            specs.append((family, i, "base", base))
            specs.append((family, i, "flip", fn(base)))
            specs.append((family, i, "null_line_order", line_shuffle(base, rng)))
            for key, variant in pl.placebo_edits(base, freq).items():
                if variant is not None:
                    specs.append((family, i, f"null_{key}", variant))

    runner = pl.runner()
    emb = runner.encode_texts([s[3] for s in specs], batch_size=BATCH)
    print(f"encoded {len(specs)} variants on {runner.device}", flush=True)

    pred = pl.predict_labels(emb, pl.fidelity_probes())
    mdist, _ = pl.manifold_distance(emb, pl.reference_cloud()[1])

    frame = pd.DataFrame({
        "family": [s[0] for s in specs], "row": [s[1] for s in specs],
        "variant": [s[2] for s in specs],
        "name": [join["name"].iloc[s[1]] for s in specs],
        "score_play": pred["score_play"].to_numpy(),
        "played_rate": pred["played_rate"].to_numpy(),
        "manifold_dist": mdist,
    })
    frame.to_csv(pl.SCRATCH / "r1c_variants.csv", index=False)

    out: dict = {"label_sd_score_play": SD_SCORE_PLAY, "gate": GATE, "families": {}}
    for family in FAMILIES:
        sub = frame[frame["family"] == family]
        base = sub[sub["variant"] == "base"].set_index("row")
        rec: dict = {"n": int(len(base))}
        for variant in sorted(sub["variant"].unique()):
            if variant == "base":
                continue
            v = sub[sub["variant"] == variant].set_index("row")
            common = base.index.intersection(v.index)
            d = (v.loc[common, "score_play"] - base.loc[common, "score_play"]).to_numpy()
            if len(d) < 3:
                continue
            # str hash() is salted per process; crc32 keeps the CI reproducible
            lo, hi = bootstrap_ci(d, seed=zlib.crc32(variant.encode()) % 1000)
            off = int((v.loc[common, "manifold_dist"] > GATE).sum())
            entry = {
                "n": int(len(d)),
                "mean_delta": float(d.mean()),
                "mean_delta_sd_units": float(d.mean() / SD_SCORE_PLAY),
                "ci95_sd_units": [lo / SD_SCORE_PLAY, hi / SD_SCORE_PLAY],
                "median_abs_delta_sd_units": float(np.median(np.abs(d)) / SD_SCORE_PLAY),
                "frac_negative": float((d < 0).mean()),
                "n_off_manifold": off,
            }
            if variant != "flip":
                rec.setdefault("nulls", {})[variant] = entry
            else:
                rec["flip"] = entry
        # flip vs each null, on the cards where both exist
        flip = sub[sub["variant"] == "flip"].set_index("row")
        d_flip = (flip["score_play"] - base.loc[flip.index, "score_play"]).abs()
        rec["vs_nulls"] = {}
        for variant in sub["variant"].unique():
            if not variant.startswith("null_"):
                continue
            v = sub[sub["variant"] == variant].set_index("row")
            common = v.index.intersection(flip.index)
            if len(common) < 5:
                continue
            d_null = (v.loc[common, "score_play"] - base.loc[common, "score_play"]).abs()
            u, p = mannwhitneyu(d_flip.loc[common], d_null, alternative="greater")
            rec["vs_nulls"][variant] = {
                "n_paired": int(len(common)),
                "median_abs_flip_sd": float(d_flip.loc[common].median() / SD_SCORE_PLAY),
                "median_abs_null_sd": float(d_null.median() / SD_SCORE_PLAY),
                "frac_flip_exceeds_null": float((d_flip.loc[common].to_numpy()
                                                 > d_null.to_numpy()).mean()),
                "mannwhitney_p": float(p),
            }
        # on-manifold-only recompute of the flip
        keep = flip["manifold_dist"] <= GATE
        if keep.sum() >= 5:
            d = (flip.loc[keep, "score_play"] - base.loc[flip.index[keep], "score_play"]).to_numpy()
            rec["flip_on_manifold_only"] = {
                "n": int(keep.sum()),
                "mean_delta_sd_units": float(d.mean() / SD_SCORE_PLAY),
                "frac_negative": float((d < 0).mean()),
            }
        out["families"][family] = rec

    with open(pl.SCRATCH / "r1c_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
