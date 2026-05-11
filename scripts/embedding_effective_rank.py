"""Effective-rank / PCA probe of the sealed encoder's card embeddings.

The encoder maps every card into a `pooled_dim`-wide vector (`2·d_model`
for the default attention‖max pool, `d_model` for `--pool-mode attn`).
This script encodes the whole corpus, stacks the card vectors into a
matrix, and reports the singular-value spectrum: how many dimensions
actually carry information.

  * **effective rank** (participation ratio `(Σλᵢ)² / Σλᵢ²`, λᵢ = σᵢ²) —
    one number in `[1, pooled_dim]`. ≈1 ⇒ all variance in one direction;
    ≈pooled_dim ⇒ the spectrum is flat.
  * **stable rank** (`‖X‖_F² / ‖X‖_2²`) — another single-number variant.
  * **components for X% of variance** + a compact scree of the top-10
    explained-variance ratios.
  * **~dead dims** — coordinates the encoder learned to emit ≈constant.

Reading it: if the variance collapses into far fewer dims than the model
has (steep drop, low effective rank), the dimensionality budget is *not*
the bottleneck — widening `d_model` / the pool is pointless. If it's
spread across most dims, the representation may be cramped.

With `--cards-played`, it also computes, per regression head, the R² of
`(shrunk label ~ first k principal components)` for a grid of k — if R²
plateaus early, the spectrum's tail is genuinely dead *for the task*, not
just low-variance. (Caveat: variance ≠ usefulness; this supervised view
is the corrective.)

    python scripts/embedding_effective_rank.py \
        [--encoder-checkpoint models/sealed/encoder/latest.pt] \
        [--vocab-path models/sealed/encoder/vocab.txt] \
        [--cards-folder output/cardsfolder/] \
        [--batch-size 256] [--device auto] [--standardize] \
        [--cards-played output/sealed/cards-played.txt] [--shrinkage-k 20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# The report (and --help) print a few math symbols (≈, ‖·‖, ², …) that the
# default Windows cp1252 stdout can't encode; force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from price_predictor.domain.card_text import ConvertedCardText  # noqa: E402
from price_predictor.infrastructure.tokenizer_store import (  # noqa: E402
    load_tokenizer,
)
from sealed.infrastructure.encoder_store import SealedEncoderStore  # noqa: E402

DEFAULT_CKPT = Path("models/sealed/encoder/latest.pt")
DEFAULT_VOCAB = Path("models/sealed/encoder/vocab.txt")
DEFAULT_CARDS = Path("output/cardsfolder/")
_K_GRID = (1, 2, 5, 10, 20, 50, 100, 200, 350, 512)


def _pick_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _encode_corpus(model, tokenizer, max_seq_len, txt_paths, device, batch_size):
    """Encode every card; return ``(stems: list[str], X: (N, pooled_dim) f64)``."""
    stems: list[str] = []
    vecs: list[np.ndarray] = []
    buf_ids: list[list[int]] = []
    buf_mask: list[list[int]] = []

    def flush() -> None:
        if not buf_ids:
            return
        ids = torch.tensor(buf_ids, dtype=torch.long, device=device)
        mask = torch.tensor(buf_mask, dtype=torch.long, device=device)
        with torch.no_grad():
            pooled = model.encode(ids, mask)
        vecs.append(pooled.cpu().numpy().astype(np.float64))
        buf_ids.clear()
        buf_mask.clear()

    for i, p in enumerate(txt_paths):
        try:
            converted = ConvertedCardText.from_file(p)
        except Exception:
            continue
        input_ids, attention_mask = tokenizer.encode(
            converted.without_name_line(), max_seq_len,
        )
        stems.append(p.stem)
        buf_ids.append(input_ids)
        buf_mask.append(attention_mask)
        if len(buf_ids) >= batch_size:
            flush()
        if (i + 1) % 5000 == 0:
            print(f"  encoded {i + 1} / {len(txt_paths)} ...", flush=True)
    flush()
    X = np.concatenate(vecs, axis=0) if vecs else np.zeros((0, model.config.pooled_dim))
    return stems, X


def _spectrum_report(X: np.ndarray, *, standardize: bool) -> None:
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    per_dim_var = Xc.var(axis=0)
    dead = int((per_dim_var < 1e-9 * max(per_dim_var.max(), 1e-30)).sum())
    if standardize:
        scale = np.sqrt(per_dim_var)
        scale[scale == 0.0] = 1.0
        Xc = Xc / scale
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2  # variance carried by each component
    total = lam.sum()
    if total == 0.0:
        print("  (degenerate: zero total variance)")
        return
    cum = np.cumsum(lam) / total
    eff_rank = float((lam.sum() ** 2) / (lam ** 2).sum())
    stable_rank = float(lam.sum() / lam.max())
    label = "standardized per-dim" if standardize else "raw, centered"
    print(f"\n--- embedding spectrum [{label}] ---")
    print(f"  cards: {n}   pooled dims: {d}   ~dead dims (≈zero var): {dead}")
    print(f"  effective rank (participation ratio): {eff_rank:.1f}  /  {d}")
    print(f"  stable rank (||X||_F^2 / ||X||_2^2):  {stable_rank:.1f}  /  {d}")
    for frac in (0.50, 0.80, 0.90, 0.95, 0.99):
        k = int(np.searchsorted(cum, frac) + 1)
        print(f"  components for {int(frac * 100):>2d}% of variance: {k}")
    head = (lam[:10] / total)
    print("  top-10 explained-variance ratios: " + " ".join(f"{v:.3f}" for v in head))


def _label_r2_report(
    X: np.ndarray, stems: list[str], cards_played_path: Path,
    cards_folder: Path, shrinkage_k: float,
) -> None:
    from sealed.application.train_encoder import (
        _ALL_HEAD_NAMES,
        _aggregate,
        _build_label_map,
        _label_value,
    )
    from sealed.infrastructure.cards_played_reader import iter_rows
    from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

    locator = ConvertedCardLocator(cards_folder)
    labels = _build_label_map(
        _aggregate(iter_rows(cards_played_path), locator), shrinkage_k,
    )
    # Embeddings are keyed by file stem; the label map by canonical name —
    # ConvertedCardLocator.text_path(name).stem bridges the two.
    stem_to_name: dict[str, str] = {}
    for name in labels:
        path = locator.text_path(name)
        if path is not None:
            stem_to_name[path.stem] = name
    stem_to_row = {stem: i for i, stem in enumerate(stems)}

    # PCA once over all encoded cards (centered, unstandardized).
    Xc = X - X.mean(axis=0, keepdims=True)
    # right singular vectors -> PC scores Z = Xc @ Vt.T  (= U * s)
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ vt.T  # (N, d)
    d = scores.shape[1]
    k_grid = [k for k in _K_GRID if k <= d] + ([d] if d not in _K_GRID else [])
    k_grid = sorted(set(k_grid))

    matched = sum(1 for st in stems if st in stem_to_name)
    print(
        f"\n--- per-head R²(shrunk label ~ top-k PCs) ---"
        f"\n  matched {matched} / {len(stems)} encoded cards to a labelled card"
        f"   (k grid: {k_grid})"
    )
    print(f"  {'head':<16}" + "".join(f"k={k:<7d}" for k in k_grid))
    for h, head_name in enumerate(_ALL_HEAD_NAMES):
        rows: list[int] = []
        ys: list[float] = []
        for st, row in stem_to_row.items():
            name = stem_to_name.get(st)
            if name is None:
                continue
            v = _label_value(labels[name], h)
            if v is None:
                continue
            rows.append(row)
            ys.append(float(v))
        if len(rows) < 2:
            print(f"  {head_name:<16}" + "".join("  --   " for _ in k_grid))
            continue
        Zsub = scores[rows]
        y = np.asarray(ys, dtype=np.float64)
        y = y - y.mean()
        ss_tot = float(y @ y)
        cells: list[str] = []
        for k in k_grid:
            Zk = Zsub[:, :k]
            # least-squares fit y ≈ Zk @ beta (no orthogonality assumed)
            beta, *_ = np.linalg.lstsq(Zk, y, rcond=None)
            resid = y - Zk @ beta
            r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
            cells.append(f"{r2:+.3f} ")
        print(f"  {head_name:<16}" + "".join(f"{c:<8}" for c in cells))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--encoder-checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--cards-folder", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument(
        "--standardize", action="store_true",
        help="also report the spectrum after standardizing each dim to unit "
        "variance (useful since the dual pool concatenates dims of different "
        "scales)",
    )
    parser.add_argument(
        "--cards-played", type=Path, default=None,
        help="if given, also report per-head R²(shrunk label ~ top-k PCs)",
    )
    parser.add_argument("--shrinkage-k", type=float, default=20.0)
    args = parser.parse_args()

    if not args.encoder_checkpoint.exists():
        print(f"error: {args.encoder_checkpoint} not found", file=sys.stderr)
        return 1

    device = _pick_device(args.device)
    model, config = SealedEncoderStore().load_encoder(args.encoder_checkpoint)
    model.to(device).eval()
    tokenizer = load_tokenizer(args.vocab_path)
    print(
        f"Loaded encoder: pool_mode={config.pool_mode}, d_model={config.d_model}, "
        f"pooled_dim={config.pooled_dim}, n_layers={config.n_layers}, "
        f"max_seq_len={config.max_seq_len}; device={device}"
    )

    txt_paths = sorted(args.cards_folder.rglob("*.txt"))
    print(f"Encoding {len(txt_paths)} card files ...")
    stems, X = _encode_corpus(
        model, tokenizer, config.max_seq_len, txt_paths, device, args.batch_size,
    )
    if X.shape[0] == 0:
        print("error: no cards encoded", file=sys.stderr)
        return 1

    _spectrum_report(X, standardize=False)
    if args.standardize:
        _spectrum_report(X, standardize=True)
    if args.cards_played is not None:
        if not args.cards_played.exists():
            print(f"error: {args.cards_played} not found", file=sys.stderr)
            return 1
        _label_r2_report(
            X, stems, args.cards_played, args.cards_folder, args.shrinkage_k,
        )

    print(
        "\nReading: low effective rank / steep scree / early R² plateau "
        "=> the encoder isn't using the dims it has; widening d_model or the "
        "pool won't help. High effective rank / R² keeps rising with k "
        "=> the representation may be dimensionally cramped."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
