"""Tier-0 build: join table, split, probes, equivalence classes, gates.

Runs every one-time computation the gen-4 encoder study's later probes
depend on, caches the artifacts under ``output/encoder-probes/``, and
writes the validation report to ``output/encoder-probes/p0_report.md``.

    python -m scripts.encoder_probes.p0_build          # full build
    .venv/Scripts/python.exe scripts/encoder_probes/p0_build.py --smoke

``--smoke`` runs the same pipeline on a 2,000-card sample, on CPU, with a
tiny alpha grid — enough to prove the wiring without touching the GPU.
``--force`` rebuilds the cached join table and corpus matrix.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The report is Markdown with non-cp1252 glyphs; the Windows console is not.
for _stream in (sys.stdout, sys.stderr):
    _stream.reconfigure(encoding="utf-8", errors="replace")

import probe_lib as pl  # noqa: E402

TRAINING_LOG = pl.REPO / (
    "models/sealed/encoder/full-20260517-014759-attn-6l-8h-8q-0.1mlm-512d-training.log"
)
REPORT = pl.SCRATCH / "p0_report.md"
ENCODER_NAME = pl.ENCODER_CKPT.name


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _split_sizes_from_log(path: Path) -> tuple[int, int] | None:
    """Pull ``Card-level split: N train / M val`` out of the training log."""
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Card-level split:" in line:
            body = line.split("Card-level split:", 1)[1]
            train = int(body.split("train")[0].strip())
            val = int(body.split("/", 1)[1].split("val")[0].strip())
            return train, val
    return None


def _md_table(frame: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def cell(v):
        if isinstance(v, float):
            return "" if v != v else floatfmt.format(v)
        return "" if v is None else str(v)

    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body])


# ── sections ────────────────────────────────────────────────────────────


def section_join(join: pd.DataFrame, labels: dict) -> str:
    unjoined = pd.read_csv(pl.SCRATCH / "unjoined.csv") if (
        pl.SCRATCH / "unjoined.csv").exists() else pd.DataFrame()
    methods = join["join_method"].value_counts()
    lines = [
        "## Join rate",
        "",
        f"- Label rows in `cards-win-rates.txt`: **{len(labels)}**",
        f"- Joined to a converted `.txt` + cached `.npz`: **{len(join)}** "
        f"({100 * len(join) / len(labels):.3f}%)",
        f"- Rows with a missing `.npz`: **{int(join['npz_path'].isna().sum())}**",
        f"- Unjoined: **{len(unjoined)}**",
        "",
        "| join method | rows |",
        "| --- | --- |",
    ]
    for method, count in methods.items():
        lines.append(f"| {method} | {count} |")
    lines += ["", "### Unjoined rows", ""]
    if len(unjoined):
        lines.append(_md_table(unjoined))
    else:
        lines.append("_none_")
    dups = join[join["dup_group"]]
    lines += [
        "",
        "### Two label rows, one card file",
        "",
        "The same card spelled two ways across corpus eras. Both rows keep "
        "their labels; `is_primary` is False on the lower-`n_in_deck` member "
        "and every probe fit filters on it.",
        "",
    ]
    if len(dups):
        lines.append(_md_table(
            dups[["name", "txt_path", "n_in_deck", "is_primary"]]
            .assign(txt_path=lambda d: [Path(p).name for p in d["txt_path"]])
        ))
    else:
        lines.append("_none_")
    return "\n".join(lines)


def section_split(split: tuple[set[str], set[str]], join: pd.DataFrame) -> str:
    logged = _split_sizes_from_log(TRAINING_LOG)
    got = (len(split[0]), len(split[1]))
    counts = join["split"].value_counts()
    lines = [
        "## Train/val split reconstruction",
        "",
        "Rebuilt by importing `train_encoder._split_cards` and feeding it "
        "`CardLabels` reconstructed from the label snapshot — the snapshot is "
        "written immediately before the split, so it is the run's own label map.",
        "",
        "| source | train | val |",
        "| --- | --- | --- |",
        f"| training log | {logged[0] if logged else '?'} | {logged[1] if logged else '?'} |",
        f"| reconstruction | {got[0]} | {got[1]} |",
        f"| join table (reconstruction minus unjoined rows) "
        f"| {int(counts.get('train', 0))} | {int(counts.get('val', 0))} |",
        "",
    ]
    if logged and tuple(logged) == got:
        lines.append("**Exact match** — the honest probe's val set is the "
                     "encoder's own held-out set.")
    else:
        lines.append("**MISMATCH** — treat the split as a labelled fallback, "
                     "not as the encoder's own partition.")
    return "\n".join(lines)


def section_bitexact(runner: pl.EncoderRunner, join: pd.DataFrame, n: int = 20) -> str:
    rng = random.Random(42)
    paths = [Path(p) for p in rng.sample(list(join["txt_path"]), n)]
    result = runner.check_bit_exact(paths)
    verdict = "**bit-exact**" if result["n_exact"] == result["n"] else "**MISMATCH**"
    return "\n".join([
        "## Re-encode fidelity",
        "",
        f"{result['n_exact']}/{result['n']} random cards re-encode to their "
        f"cached `.npz` text dims {verdict} "
        f"(max |Δ| = {result['max_abs_diff']:.3g}).",
        "",
        "The cache was written one card at a time on CPU, padded to "
        "`max_seq_len`; that configuration is reproduced by "
        "`EncoderRunner.encode_texts(..., exact=True)`. Batching or moving to "
        "CUDA shifts the last bits (max |Δ| ≈ 5e-7, ~1e-5 of a text dim's "
        "spread) — fine for probe read-off, fatal for an equality check.",
    ])


def section_probes(metrics: pd.DataFrame) -> str:
    lines = ["## Probe metrics", ""]
    lines += [
        "Ridge from the 512-dim text vector to each shrunk label. "
        "**fidelity** fits on every joined card (the read-off model for "
        "counterfactual edits); **honest** fits on the encoder's own train "
        "split only, so its `val_r2` / `val_pearson` are unrecycled. "
        "`w` = per-head `n/(n+20)` sample weights (the training objective's), "
        "`u` = unweighted. `played_rate@logit` is the same head fitted in "
        "logit space with the rate clipped to [1e-3, 1−1e-3].",
        "",
        "Read `cv_r2` and `val_r2` as different questions, not as a discrepancy. "
        "`cv_r2` holds out probe folds drawn from every card, four fifths of "
        "which the *encoder* trained on — so the embedding itself already "
        "carries those labels. `val_r2` holds out cards the encoder never saw. "
        "The gap between them (0.71 → 0.37 on `score_play`) is the R2 "
        "memorization-vs-generalization result arriving early, not a bug: use "
        "the fidelity fit only to read off counterfactual *edits*, and the "
        "honest `val_r2` for any claim about unseen cards.",
        "",
    ]
    for mode in ("fidelity", "honest"):
        for weighted in (True, False):
            sub = metrics[(metrics["mode"] == mode) & (metrics["weighted"] == weighted)]
            if not len(sub):
                continue
            lines += [
                f"### {mode} / {'weighted' if weighted else 'unweighted'}",
                "",
                _md_table(sub[[
                    "probe", "alpha", "n_fit", "cv_r2", "in_sample_r2",
                    "train_r2", "val_r2", "val_pearson",
                ]].rename(columns={"in_sample_r2": "insample_r2"})),
                "",
            ]
    return "\n".join(lines)


def section_classes(classes: list[dict], variances: list[dict], emb_check: dict) -> str:
    sizes = pd.Series([c["size"] for c in classes])
    dist = sizes.value_counts().sort_index()
    lines = [
        "## Equivalence classes (R2 input)",
        "",
        "Joined cards grouped by identical name-stripped **token sequence** — "
        "the encoder's literal input. Members are indistinguishable to the "
        "encoder by construction, so their label spread is irreducible noise "
        "plus whatever the labels see that the text does not.",
        "",
        f"- Classes with ≥2 members: **{len(classes)}**",
        f"- Cards inside a class: **{int(sizes.sum())}** "
        f"({100 * sizes.sum() / max(1, emb_check['n_cards']):.1f}% of joined cards)",
        f"- Largest class: **{int(sizes.max()) if len(sizes) else 0}**",
        "",
        "| class size | classes |",
        "| --- | --- |",
    ]
    for size, count in dist.items():
        lines.append(f"| {size} | {count} |")
    lines += [
        "",
        "### Embedding identity check",
        "",
        f"{emb_check['n_checked']} classes checked; "
        f"{emb_check['n_identical']} have bit-identical member embeddings "
        f"(max within-class |Δ| = {emb_check['max_abs_diff']:.3g}). Same token "
        "ids and the same padding must give the same vector, so anything else "
        "would mean the cache is stale relative to the checkpoint.",
        "",
        "### Variance decomposition — `shrunk_score_play`",
        "",
        "`min_n` is an `n_in_deck` floor applied to class members. Raising it "
        "separates irreducible noise from mere under-observation: a card seen "
        "eighty times has a noisy label a bigger corpus would sharpen.",
        "",
        "| min_n | classes | members | within-class var | within SD | member var "
        "| corpus var | explainable (vs members) | explainable (vs corpus) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    usable = [v for v in variances if v]
    for var in usable:
        lines.append(
            f"| {var['min_n']} | {var['n_classes']} | {var['n_members']} "
            f"| {var['within_class_var']:.3e} | {var['noise_sd']:.4f} "
            f"| {var['member_var']:.3e} | {var['corpus_var']:.3e} "
            f"| {var['explainable_fraction_vs_members']:.3f} "
            f"| {var['explainable_fraction_vs_corpus']:.3f} |"
        )
    if not usable:
        lines.append("| | | | | | | | | |")
    lines += [
        "",
        "Read the corpus-relative column with care: functional reprints skew "
        "toward simple commons, so their within-class noise samples the "
        "low-complexity end of the corpus rather than all of it. The "
        "members-relative column is the tighter bound, and the one to quote.",
    ]
    return "\n".join(lines)


def section_manifold(stats: dict) -> str:
    q = stats["quantiles"]
    lines = [
        "## Manifold gate",
        "",
        "Cosine distance from each real card to its nearest *other* real card, "
        "over all 512-dim cached embeddings. An edited card that sits beyond "
        "the 95th percentile has left the cloud the encoder was fitted on, and "
        "a probe read-off there is extrapolation.",
        "",
        "| quantile | cosine distance |",
        "| --- | --- |",
    ]
    for name, value in q.items():
        lines.append(f"| {name} | {value:.5f} |")
    lines += [
        "",
        f"**Off-manifold gate (p95): {q['p95']:.5f}**  ",
        f"mean {stats['mean']:.5f}; {stats['n_zero']} cards sit within 1e-6 of "
        "another card — the functional reprints, whose distance is exactly "
        "zero up to float32 normalisation.",
    ]
    return "\n".join(lines)


def section_external(join: pd.DataFrame) -> str:
    lines = ["## External joins", ""]
    if "ai_remove_deck" in join.columns:
        hinted = int(join["has_forge_hint"].sum())
        blacklisted = int(
            pd.to_numeric(join["ai_remove_deck"], errors="coerce").fillna(0).sum()
        )
        ranked = int(pd.to_numeric(join["draft_rank"], errors="coerce").notna().sum())
        lines += [
            f"- `scripts/scorer_probes/forge_hints.csv` matched **{hinted}** rows "
            f"({100 * hinted / len(join):.1f}%): **{blacklisted}** carry Forge's "
            f"`ai_remove_deck` blacklist flag, **{ranked}** carry a human draft rank.",
        ]
    else:
        lines.append(
            "- `forge_hints.csv` is absent — regenerate it with "
            "`scripts/scorer_probes/forge_hints.py` (needs the sibling "
            "`../forge` checkout)."
        )
    years = pd.to_numeric(join["first_year"], errors="coerce")
    if years.notna().any():
        rarity = join["first_rarity"].value_counts()
        lines += [
            f"- `AllPrintings.json` matched **{int(years.notna().sum())}** rows "
            f"({100 * years.notna().mean():.1f}%) with an earliest printing "
            f"({int(years.min())}–{int(years.max())}).",
            "",
            "| earliest-printing rarity | cards |",
            "| --- | --- |",
        ]
        for name, count in rarity.items():
            lines.append(f"| {name} | {count} |")
    else:
        lines.append("- `AllPrintings.json` was not scanned.")
    return "\n".join(lines)


def section_placebo(placebo: dict) -> str:
    lines = [
        "## Placebo edits",
        "",
        "Meaning-preserving rewrites whose probe shift is the null any real "
        "counterfactual has to clear. Encoded on the study's default path "
        "(batched, GPU when present), so the ~5e-7 batching noise is inside "
        "every number below.",
        "",
        "| edit | applicable cards | median |Δscore_play| | p95 |Δ| | max |Δ| |",
        "| --- | --- | --- | --- | --- |",
    ]
    for kind, s in placebo["edits"].items():
        if s["n"] == 0:
            lines.append(f"| {kind} | 0 | | | |")
            continue
        lines.append(
            f"| {kind} | {s['n']} | {s['median']:.5f} | {s['p95']:.5f} | "
            f"{s['max']:.5f} |"
        )
    lines += [
        "",
        f"Sampled from {placebo['n_sampled']} cards; the fidelity/weighted "
        f"`score_play` probe reads off the edit. Label SD for scale: "
        f"{placebo['label_sd']:.5f}.",
    ]
    return "\n".join(lines)


# ── build ───────────────────────────────────────────────────────────────


def compute_embedding_identity(
    classes: list[dict], join: pd.DataFrame, limit: int = 200,
) -> dict:
    """Confirm class members really do share one cached embedding."""
    checked = classes[:limit]
    max_diff = 0.0
    identical = 0
    for cls in checked:
        matrix = pl.load_embedding_matrix(cls["names"], join)
        diff = float(np.abs(matrix - matrix[0]).max())
        max_diff = max(max_diff, diff)
        identical += int(diff == 0.0)
    return {
        "n_checked": len(checked),
        "n_identical": identical,
        "max_abs_diff": max_diff,
        "n_cards": len(join),
    }


def compute_manifold_stats(join: pd.DataFrame) -> dict:
    keys, matrix = pl.corpus_embedding_matrix()
    index = {k: i for i, k in enumerate(keys)}
    rows = np.array([index[pl.corpus_key(p)] for p in join["txt_path"]])
    query = matrix[rows]
    dist, _ = pl.manifold_distance(query, matrix, self_rows=rows)
    qs = [0.5, 0.75, 0.9, 0.95, 0.99]
    return {
        "quantiles": {"p05": float(np.quantile(dist, 0.05)),
                      **{f"p{int(q * 100)}": float(np.quantile(dist, q)) for q in qs}},
        "mean": float(dist.mean()),
        "n_zero": int((dist < 1e-6).sum()),
        "distances": dist,
    }


def compute_placebo_null(
    runner: pl.EncoderRunner, join: pd.DataFrame, probe: pl.HeadProbe,
    *, n_sample: int = 400, batch_size: int = 64, seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    sample = rng.sample(list(join["txt_path"]), min(n_sample, len(join)))
    freq = pl.subtype_frequencies()
    base_texts = [Path(p).read_text(encoding="utf-8", errors="replace") for p in sample]
    base = probe.predict(runner.encode_texts(base_texts, batch_size=batch_size))
    out: dict[str, dict] = {}
    for kind in ("swap_static", "subtype_swap", "swap_ability_lines"):
        idxs, texts = [], []
        for i, text in enumerate(base_texts):
            edited = pl.placebo_edits(text, freq)[kind]
            if edited is not None:
                idxs.append(i)
                texts.append(edited)
        if not texts:
            out[kind] = {"n": 0}
            continue
        shifted = probe.predict(runner.encode_texts(texts, batch_size=batch_size))
        delta = np.abs(shifted - base[idxs])
        out[kind] = {
            "n": len(texts),
            "median": float(np.median(delta)),
            "p95": float(np.quantile(delta, 0.95)),
            "max": float(delta.max()),
        }
    return {"edits": out, "n_sampled": len(sample)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="2,000-card CPU sanity run; writes no report")
    parser.add_argument("--force", action="store_true",
                        help="rebuild the cached join table and corpus matrix")
    parser.add_argument("--no-printings", action="store_true",
                        help="skip the AllPrintings.json scan")
    args = parser.parse_args(argv)

    _log("loading labels")
    labels = pl.load_labels()

    _log("building join table")
    join = pl.build_join(force=args.force, with_printings=not args.no_printings)
    if args.smoke:
        join = join.sample(2000, random_state=42).sort_values("name").reset_index(drop=True)

    _log(f"joined {len(join)} / {len(labels)} label rows")
    _log("loading embedding matrix")
    embeddings = pl.load_embedding_matrix(list(join["name"]), join)

    _log("loading encoder")
    runner = pl.EncoderRunner(device="cpu" if args.smoke else None)
    bitexact = section_bitexact(runner, join, n=5 if args.smoke else 20)
    print(bitexact)

    _log("fitting probes")
    folds = 2 if args.smoke else 5
    if args.smoke:
        pl.ALPHA_GRID = (1.0, 10.0)
    probe_sets = []
    for mode in ("fidelity", "honest"):
        for weighted in (True, False):
            ps = pl.fit_probes(join, embeddings, mode=mode, weighted=weighted,
                               folds=folds)
            probe_sets.append(ps)
            if not args.smoke:
                pl.save_probes(ps)
            _log(f"  {ps.key}: "
                 + " ".join(f"{k}={v.metrics.get('val_r2', float('nan')):.3f}"
                            for k, v in ps.probes.items()))
    metrics = pl.probe_metrics_frame(probe_sets)
    if not args.smoke:
        metrics.to_csv(pl.SCRATCH / "probe_metrics.csv", index=False)

    _log("building equivalence classes")
    classes = pl.equivalence_classes(join)
    variances = [
        pl.variance_decomposition(classes, join, min_n=floor)
        for floor in (0, 200, 800)
    ]
    emb_check = compute_embedding_identity(classes, join,
                                           limit=20 if args.smoke else 200)
    if not args.smoke:
        with open(pl.SCRATCH / "equivalence_classes.json", "w", encoding="utf-8") as f:
            json.dump(classes, f, indent=1)

    _log("computing manifold distances")
    manifold = compute_manifold_stats(join)
    if not args.smoke:
        np.save(pl.SCRATCH / "manifold_distance.npy", manifold["distances"])

    _log("running the placebo null")
    fidelity_w = next(p for p in probe_sets if p.key == "fidelity_w")
    placebo = compute_placebo_null(
        runner, join, fidelity_w.probes["score_play"],
        n_sample=40 if args.smoke else 400,
        batch_size=16 if args.smoke else 64,
    )
    placebo["label_sd"] = float(
        pd.to_numeric(join["shrunk_score_play"], errors="coerce").std()
    )

    if args.smoke:
        _log("smoke run complete (no report written)")
        print(json.dumps({
            "classes": len(classes),
            "variances": variances,
            "placebo": placebo["edits"],
        }, indent=1))
        return 0

    report = "\n\n".join([
        "# Encoder probes — Tier 0 build report",
        f"Checkpoint: `{ENCODER_NAME}`  \n"
        f"Labels: `{pl.WIN_RATES}`  \n"
        f"Cards: `{pl.CARDS_PATH}`  \n"
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        section_join(join, labels),
        section_split(pl.reconstruct_split(labels), join),
        bitexact,
        section_probes(metrics),
        section_classes(classes, variances, emb_check),
        section_manifold(manifold),
        section_placebo(placebo),
        section_external(join),
    ])
    REPORT.write_text(report, encoding="utf-8")
    _log(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
