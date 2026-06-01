"""Builder-validation diagnostic logic (FR-042).

Decides whether the fast picker can stand in for the slow SA ``GreedyDeckBuilder``
as the data-gen label-builder. Over a few hundred drafted pools it builds each
pool both ways, scores both with the frozen scorer, and reports:

- the picker-vs-SA **Spearman** rank correlation (the gating number),
- the SA−picker score-gap **median + IQR** (the named spread statistic),
- the SA-vs-SA reference correlation across two independent SA restarts (the
  ceiling: how well SA even tracks itself).

The pure statistics live in :func:`compute_diagnostic` (unit-tested without
torch/Forge); the surrounding builder/scorer wiring reuses ``sealed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BuilderDiagnostic:
    """The three reported numbers (FR-042, SC-007)."""

    picker_vs_sa_spearman: float
    gap_median: float            # median of (SA score − picker score)
    gap_iqr: float               # IQR of the same gap
    sa_vs_sa_spearman: float     # reference ceiling
    n_pools: int


def compute_diagnostic(
    picker_scores: list[float],
    sa_scores_a: list[float],
    sa_scores_b: list[float],
) -> BuilderDiagnostic:
    """Compute the diagnostic from three aligned per-pool score arrays.

    ``sa_scores_a`` is the reference SA build (compared against the picker);
    ``sa_scores_b`` is a second independent SA restart (for the SA-vs-SA ceiling).
    """
    from scipy.stats import spearmanr

    picker = np.asarray(picker_scores, dtype=np.float64)
    sa_a = np.asarray(sa_scores_a, dtype=np.float64)
    sa_b = np.asarray(sa_scores_b, dtype=np.float64)
    n = len(picker)

    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        if n < 2:
            return float("nan")
        corr, _ = spearmanr(x, y)
        return float(corr)

    gap = sa_a - picker
    if n:
        q1, med, q3 = np.percentile(gap, [25, 50, 75])
    else:
        q1 = med = q3 = float("nan")

    return BuilderDiagnostic(
        picker_vs_sa_spearman=_spearman(picker, sa_a),
        gap_median=float(med),
        gap_iqr=float(q3 - q1),
        sa_vs_sa_spearman=_spearman(sa_a, sa_b),
        n_pools=n,
    )


def format_diagnostic(diag: BuilderDiagnostic) -> str:
    """Human-readable summary of the gating numbers."""
    return (
        f"Builder validation over {diag.n_pools} pools:\n"
        f"  picker-vs-SA Spearman : {diag.picker_vs_sa_spearman:.4f}  (gating)\n"
        f"  SA-vs-SA  Spearman    : {diag.sa_vs_sa_spearman:.4f}  (reference ceiling)\n"
        f"  SA-picker score gap   : median={diag.gap_median:.4f}  IQR={diag.gap_iqr:.4f}"
    )


# --------------------------------------------------------------------------- #
# Pool sourcing + build/score wiring (reuses sealed)
# --------------------------------------------------------------------------- #

@dataclass
class ValidateBuilderConfig:
    pools_from: Path | None = None
    fresh_pools: bool = False
    set_code: str | None = None
    n_pools: int = 300
    scorer_checkpoint: Path = field(
        default_factory=lambda: Path("models/sealed/scorer/latest.pt"),
    )
    picker_checkpoint: Path = field(
        default_factory=lambda: Path("models/sealed/picker/latest.pt"),
    )
    cards_path: Path = field(default_factory=lambda: Path("output/cardsfolder/"))


def drafted_pools_from_corpus(path: Path, limit: int) -> list[list[str]]:
    """Every seat's drafted pool from a ``drafts.jsonl``, up to ``limit`` pools."""
    from draft.domain.draft_geometry import DraftGeometry
    from draft.infrastructure.draft_record_io import read_records

    pools: list[list[str]] = []
    for record in read_records(path):
        geo = DraftGeometry.from_record(record)
        for seat_idx in range(geo.pod_size):
            pools.append(geo.drafted_pool(record, seat_idx))
            if len(pools) >= limit:
                return pools
    return pools


def run_validate(config: ValidateBuilderConfig) -> BuilderDiagnostic:
    """Build each pool with picker + two SA restarts, score, and diagnose."""
    import numpy as np
    import torch

    from sealed.application.deck_assembly import (
        assemble_full_deck,
        load_pool_embeddings,
    )
    from sealed.application.evaluate_scorer import score_decks
    from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE, GreedyDeckBuilder
    from sealed.domain.picker_model import PickerModel, decompose_picks
    from sealed.domain.scorer_model import SetTransformerScorer
    from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
    from sealed.infrastructure.picker_store import PickerStore
    from sealed.infrastructure.scorer_store import ScorerStore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    locator = ConvertedCardLocator(config.cards_path)

    scorer_ckpt = ScorerStore().load_checkpoint(config.scorer_checkpoint)
    scorer = SetTransformerScorer(scorer_ckpt.config)
    scorer.load_state_dict(scorer_ckpt.model_state_dict)
    scorer.eval().to(device)

    picker_ckpt = PickerStore().load_checkpoint(config.picker_checkpoint)
    picker = PickerModel(picker_ckpt.config)
    picker.load_state_dict(picker_ckpt.model_state_dict)
    picker.eval().to(device)

    pools = _source_pools(config)

    picker_scores: list[float] = []
    sa_a: list[float] = []
    sa_b: list[float] = []
    for pool in pools:
        embeddings, valid = load_pool_embeddings(pool, locator)
        if len(valid) < NONLAND_DECK_SIZE:
            continue
        embs = [embeddings[n] for n in valid]
        arr = np.stack(embs).astype(np.float32)
        cards = torch.from_numpy(arr).unsqueeze(0).to(device)
        mask = torch.ones(1, arr.shape[0], dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, _ = picker(cards, mask)
        picker_deck = assemble_full_deck(
            decompose_picks(logits[0].cpu(), embs, valid), locator,
        )
        sa_deck_a = assemble_full_deck(
            GreedyDeckBuilder(scorer, embeddings).build(valid), locator,
        )
        sa_deck_b = assemble_full_deck(
            GreedyDeckBuilder(scorer, embeddings).build(valid), locator,
        )
        scores = score_decks(scorer, [picker_deck, sa_deck_a, sa_deck_b], locator)
        picker_scores.append(scores[0])
        sa_a.append(scores[1])
        sa_b.append(scores[2])

    return compute_diagnostic(picker_scores, sa_a, sa_b)


def _source_pools(config: ValidateBuilderConfig) -> list[list[str]]:
    if config.pools_from is not None:
        return drafted_pools_from_corpus(config.pools_from, config.n_pools)
    if config.fresh_pools:
        return _generate_fresh_pools(config)
    raise ValueError("Provide either --pools-from or --fresh-pools")


def _generate_fresh_pools(config: ValidateBuilderConfig) -> list[list[str]]:
    import tempfile

    from sealed.application.generate_pools import GeneratePoolsUseCase
    from sealed.infrastructure.pool_connector import PoolConnector
    from sealed.infrastructure.pool_file_reader import parse_pools

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        GeneratePoolsUseCase().execute(
            config.set_code, config.n_pools, out_dir, PoolConnector(),
        )
        pools = parse_pools(out_dir / "pools.txt")
    return [p.cards for p in pools]
