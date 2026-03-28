"""Architecture search script for the transformer price predictor.

Trains combinations of (d_model, n_layers, n_heads) with ff_dim=4*d_model,
evaluates each, and ranks by a composite score that prioritises accuracy in
the €2–50 range.

Default grid: d_model ∈ {64,128,256}, n_layers ∈ {2,4,6}, n_heads ∈ {4}
(9 combos). Override with --d-models, --n-layers, and --n-heads for
wider/deeper exploration, e.g.:
    python scripts/search_transformer_arch.py --d-models 256,512 --n-layers 6,8 --n-heads 4,8

Usage:
    python scripts/search_transformer_arch.py [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from price_predictor.application.evaluate_transformer import evaluate_transformer
from price_predictor.application.train_transformer import train_transformer

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

_D_MODELS_DEFAULT = [64, 128, 256]
_N_LAYERS_DEFAULT = [2, 4, 6]
_N_HEADS_DEFAULT = [4]


def _build_grid(d_models: list[int], n_layers: list[int], n_heads: list[int]) -> list[dict]:
    combos = []
    for d in d_models:
        for h in n_heads:
            if d % h != 0:
                raise ValueError(
                    f"d_model={d} is not divisible by n_heads={h}. "
                    f"Choose a d_model that is a multiple of {h}."
                )
            for l in n_layers:
                combos.append({
                    "d_model": d,
                    "n_layers": l,
                    "n_heads": h,
                    "ff_dim": 4 * d,
                })
    return combos


def _parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of integers (e.g. '64,128,256')."""
    try:
        return [int(x.strip()) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integers, got: {value!r}"
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Weights per bucket label (lower composite score = better model)
_BUCKET_WEIGHTS: dict[str, float] = {
    "€2–10":       4.0,
    "€10–50":      4.0,
    "€0.50–2":     3.0,
    ">€50":        2.0,
    "€0.10–0.50":  1.0,
    "<€0.10":      1.0,
}


def _composite_score(buckets: dict[str, dict]) -> float:
    """Compute weighted sum of median_log_error across buckets.

    Lower is better.  Buckets missing from the result are skipped.
    """
    score = 0.0
    for label, weight in _BUCKET_WEIGHTS.items():
        if label in buckets:
            score += weight * buckets[label]["med_log"]
    return score


# ---------------------------------------------------------------------------
# Results file helpers
# ---------------------------------------------------------------------------

def _load_results(path: Path) -> list[dict]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_results(path: Path, results: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def _already_done(results: list[dict], model_dir: Path) -> bool:
    """Return True if a run for this model_dir already has a saved result."""
    existing_dirs = {r["model_dir"] for r in results}
    return str(model_dir) in existing_dirs


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _print_results_table(results: list[dict], top_n: int = 5) -> None:
    sorted_results = sorted(results, key=lambda r: r["composite_score"])

    header = (
        f"{'Rank':>4}  {'d_model':>7}  {'n_layers':>8}  {'ff_dim':>6}  "
        f"{'best_ep':>7}  {'val_loss':>8}  {'score':>8}  {'MAE':>6}  "
        f"{'top20':>5}  {'early':>5}"
    )
    print()
    print("=" * len(header))
    print("Architecture Search Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for rank, r in enumerate(sorted_results, start=1):
        marker = " <-- best" if rank == 1 else ""
        print(
            f"{rank:>4}  {r['d_model']:>7}  {r['n_layers']:>8}  {r['ff_dim']:>6}  "
            f"{r['best_epoch']:>7}  {r['best_val_loss']:>8.4f}  "
            f"{r['composite_score']:>8.4f}  {r['mae_eur']:>6.2f}  "
            f"{r['top_20_overlap']:>5.2f}  {'yes' if r['stopped_early'] else 'no':>5}"
            f"{marker}"
        )

    print("=" * len(header))
    print()
    print(f"Top {min(top_n, len(sorted_results))} configurations:")
    for rank, r in enumerate(sorted_results[:top_n], start=1):
        print(
            f"  {rank}. d_model={r['d_model']}, n_layers={r['n_layers']}, "
            f"ff_dim={r['ff_dim']}  →  score={r['composite_score']:.4f}, "
            f"MAE=€{r['mae_eur']:.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search transformer architecture hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/cardsfolder/"),
        help="Path to converted card text files",
    )
    parser.add_argument(
        "--prices-path", type=Path, default=Path("resources/AllPricesToday.json"),
        help="Path to AllPricesToday.json",
    )
    parser.add_argument(
        "--printings-path", type=Path, default=Path("resources/AllPrintings.json"),
        help="Path to AllPrintings.json",
    )
    parser.add_argument(
        "--vocab-path", type=Path, default=Path("models/price-predictor/transformer/vocab.txt"),
        help="Path to vocab.txt",
    )
    parser.add_argument(
        "--results-file", type=Path, default=Path("arch_search_results.json"),
        help="Where to write JSON results",
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("models/price-predictor/transformer/search/"),
        help="Base directory for model artifacts",
    )
    parser.add_argument(
        "--d-models", type=_parse_int_list, default=None,
        metavar="INT,INT,...",
        help="Comma-separated d_model values to search (default: 64,128,256)",
    )
    parser.add_argument(
        "--n-layers", type=_parse_int_list, default=None,
        metavar="INT,INT,...",
        help="Comma-separated n_layers values to search (default: 2,4,6)",
    )
    parser.add_argument(
        "--n-heads", type=_parse_int_list, default=None,
        metavar="INT,INT,...",
        help="Comma-separated n_heads values to search (default: 4)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Maximum number of training epochs per combination",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience (epochs without val_loss improvement)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    d_models = args.d_models if args.d_models is not None else _D_MODELS_DEFAULT
    n_layers = args.n_layers if args.n_layers is not None else _N_LAYERS_DEFAULT
    n_heads = args.n_heads if args.n_heads is not None else _N_HEADS_DEFAULT
    grid = _build_grid(d_models, n_layers, n_heads)

    print(f"Architecture search — {len(grid)} combinations")
    print(f"{'d_model':>8}  {'n_layers':>8}  {'n_heads':>7}  {'ff_dim':>6}")
    print("-" * 38)
    for combo in grid:
        print(
            f"{combo['d_model']:>8}  {combo['n_layers']:>8}  "
            f"{combo['n_heads']:>7}  {combo['ff_dim']:>6}"
        )
    print()

    results = _load_results(args.results_file)

    for i, combo in enumerate(grid, start=1):
        d_model = combo["d_model"]
        n_layers = combo["n_layers"]
        n_heads = combo["n_heads"]
        ff_dim = combo["ff_dim"]

        model_dir = args.model_output / f"d{d_model}_l{n_layers}_h{n_heads}"

        print(f"\n[{i}/{len(grid)}] d_model={d_model}, n_layers={n_layers}, "
              f"n_heads={n_heads}, ff_dim={ff_dim}")
        print(f"      model_dir: {model_dir}")

        # Resume: skip if already done
        if _already_done(results, model_dir):
            print("      → skipping (already in results file)")
            continue

        # Skip if latest.pt already exists but no result record yet — retrain
        # (could be a partially-completed run; safer to redo than to evaluate a
        # potentially-incomplete checkpoint)

        # --- Train ---
        print("      Training...")
        train_result = train_transformer(
            output_dir=args.output_dir,
            prices_path=args.prices_path,
            printings_path=args.printings_path,
            model_output=model_dir,
            vocab_path=args.vocab_path,
            epochs=args.epochs,
            patience=args.patience,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
        )

        # --- Evaluate ---
        print("      Evaluating...")
        eval_result = evaluate_transformer(
            model_dir=model_dir,
            output_dir=args.output_dir,
            prices_path=args.prices_path,
            printings_path=args.printings_path,
            vocab_path=args.vocab_path,
        )

        # --- Build buckets dict ---
        buckets: dict[str, dict] = {}
        if eval_result.per_bucket:
            for bm in eval_result.per_bucket:
                buckets[bm.label] = {
                    "n": bm.count,
                    "med_log": bm.median_log_error,
                    "signed": bm.median_signed_log_error,
                }

        score = _composite_score(buckets)

        record: dict = {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ff_dim": ff_dim,
            "model_dir": str(model_dir),
            "best_epoch": train_result.best_epoch,
            "best_val_loss": round(train_result.best_val_loss, 4),
            "stopped_early": train_result.stopped_early,
            "composite_score": round(score, 4),
            "buckets": buckets,
            "mae_eur": eval_result.mean_absolute_error_eur,
            "top_20_overlap": eval_result.top_20_overlap,
        }

        results.append(record)
        _save_results(args.results_file, results)

        print(
            f"      Done — score={score:.4f}, MAE=€{eval_result.mean_absolute_error_eur:.2f}, "
            f"top20={eval_result.top_20_overlap:.2f}"
        )

    if not results:
        print("No results to display.")
        return

    _print_results_table(results)


if __name__ == "__main__":
    main()
