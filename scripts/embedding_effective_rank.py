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
        --encoder-checkpoint <encoder .pt> --vocab-path <vocab.txt> \
        --cards-folder <converted cardsfolder> \
        [--batch-size 256] [--device auto] [--standardize] \
        [--cards-played <cards-played.txt>] [--shrinkage-k 20]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from price_predictor.domain.card_text import ConvertedCardText
from price_predictor.infrastructure.tokenizer_store import load_tokenizer
from sealed.infrastructure.encoder_store import SealedEncoderStore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.markdown import utf8_console  # noqa: E402
from _lib.stats import participation_ratio  # noqa: E402

# The report (and --help) print math symbols (≈, ‖·‖, ²) cp1252 cannot encode.
utf8_console()

_K_GRID = (1, 2, 5, 10, 20, 50, 100, 200, 350, 512)


def _pick_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


@dataclass(frozen=True)
class EncodeSettings:
    """How the cards are tokenized and batched for the encoder."""

    max_seq_len: int
    device: torch.device
    batch_size: int


class _PooledBatches:
    """Token batches waiting for the encoder, and the pooled vectors so far."""

    def __init__(self, model, device: torch.device) -> None:
        self.model, self.device = model, device
        self.vecs: list[np.ndarray] = []
        self.ids: list[list[int]] = []
        self.mask: list[list[int]] = []

    def add(self, input_ids: list[int], attention_mask: list[int]) -> None:
        self.ids.append(input_ids)
        self.mask.append(attention_mask)

    def flush(self) -> None:
        if not self.ids:
            return
        ids = torch.tensor(self.ids, dtype=torch.long, device=self.device)
        mask = torch.tensor(self.mask, dtype=torch.long, device=self.device)
        with torch.no_grad():
            pooled = self.model.encode(ids, mask)
        self.vecs.append(pooled.cpu().numpy().astype(np.float64))
        self.ids.clear()
        self.mask.clear()


def _read_converted(path: Path):
    """The converted card at ``path``, or ``None`` when it cannot be read."""
    try:
        return ConvertedCardText.from_file(path)
    except Exception:
        return None


def _encode_corpus(model, tokenizer, txt_paths, settings: EncodeSettings):
    """Encode every card; return ``(stems: list[str], X: (N, pooled_dim) f64)``."""
    stems: list[str] = []
    batches = _PooledBatches(model, settings.device)
    for i, p in enumerate(txt_paths):
        converted = _read_converted(p)
        if converted is None:
            continue
        input_ids, attention_mask = tokenizer.encode(
            converted.without_name_line(), settings.max_seq_len,
        )
        stems.append(p.stem)
        batches.add(input_ids, attention_mask)
        if len(batches.ids) >= settings.batch_size:
            batches.flush()
        if (i + 1) % 5000 == 0:
            print(f"  encoded {i + 1} / {len(txt_paths)} ...", flush=True)
    batches.flush()
    vecs = batches.vecs
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
    eff_rank = participation_ratio(lam)
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


@dataclass(frozen=True)
class LabelSources:
    """Where the per-card labels come from."""

    cards_played_path: Path
    cards_folder: Path
    shrinkage_k: float


def _label_maps(sources: LabelSources):
    """The shrunk label map, and the canonical name of each card file stem."""
    from sealed.application.encoder_counters import _aggregate
    from sealed.application.encoder_labels import _build_label_map
    from sealed.infrastructure.cards_played_reader import iter_rows
    from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

    locator = ConvertedCardLocator(sources.cards_folder)
    labels = _build_label_map(
        _aggregate(iter_rows(sources.cards_played_path), locator), sources.shrinkage_k,
    )
    # Embeddings are keyed by file stem; the label map by canonical name —
    # ConvertedCardLocator.text_path(name).stem bridges the two.
    stem_to_name: dict[str, str] = {}
    for name in labels:
        path = locator.text_path(name)
        if path is not None:
            stem_to_name[path.stem] = name
    return labels, stem_to_name


def _pc_scores(X: np.ndarray) -> np.ndarray:
    """PCA once over all encoded cards (centered, unstandardized)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # right singular vectors -> PC scores Z = Xc @ Vt.T  (= U * s)
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt.T  # (N, d)


def _head_targets(h: int, stem_to_row: dict[str, int], labels: dict, stem_to_name: dict):
    """The rows of the encoded cards labelled on head ``h``, and their values."""
    from sealed.application.encoder_heads import _label_value

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
    return rows, ys


def _r2_cells(Zsub: np.ndarray, ys: list[float], k_grid: list[int]) -> list[str]:
    """R² of the labels against the top-k PCs, one cell per ``k``."""
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
    return cells


def _k_grid(d: int) -> list[int]:
    """The numbers of top PCs the R² is read at, ``d`` among them."""
    return sorted(set([k for k in _K_GRID if k <= d] + ([d] if d not in _K_GRID else [])))


def _print_r2_header(matched: int, total: int, k_grid: list[int]) -> None:
    print(
        f"\n--- per-head R²(shrunk label ~ top-k PCs) ---"
        f"\n  matched {matched} / {total} encoded cards to a labelled card"
        f"   (k grid: {k_grid})"
    )
    print(f"  {'head':<16}" + "".join(f"k={k:<7d}" for k in k_grid))


def _label_r2_report(X: np.ndarray, stems: list[str], sources: LabelSources) -> None:
    from sealed.application.encoder_heads import _ALL_HEAD_NAMES

    labels, stem_to_name = _label_maps(sources)
    stem_to_row = {stem: i for i, stem in enumerate(stems)}
    scores = _pc_scores(X)
    k_grid = _k_grid(scores.shape[1])
    _print_r2_header(sum(1 for st in stems if st in stem_to_name), len(stems), k_grid)
    for h, head_name in enumerate(_ALL_HEAD_NAMES):
        rows, ys = _head_targets(h, stem_to_row, labels, stem_to_name)
        if len(rows) < 2:
            print(f"  {head_name:<16}" + "".join("  --   " for _ in k_grid))
            continue
        cells = _r2_cells(scores[rows], ys, k_grid)
        print(f"  {head_name:<16}" + "".join(f"{c:<8}" for c in cells))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--cards-folder", type=Path, required=True)
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
        model, tokenizer, txt_paths,
        EncodeSettings(config.max_seq_len, device, args.batch_size),
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
            X, stems, LabelSources(args.cards_played, args.cards_folder, args.shrinkage_k),
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
