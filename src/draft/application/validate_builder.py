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
from datetime import datetime
from pathlib import Path

import numpy as np


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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
    _log(f"Loaded scorer + picker on {device}")

    pools = _source_pools(config)
    _log(f"Sourced {len(pools)} pools; building picker + 2 SA decks each...")

    picker_scores: list[float] = []
    sa_a: list[float] = []
    sa_b: list[float] = []
    skipped = 0
    total = len(pools)
    report_every = max(1, total // 20)
    for i, pool in enumerate(pools, start=1):
        embeddings, valid = load_pool_embeddings(pool, locator)
        if len(valid) < NONLAND_DECK_SIZE:
            skipped += 1
            if i % report_every == 0 or i == total:
                _log(f"  {i}/{total} pools  ({len(picker_scores)} scored, {skipped} skipped)")
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
        if i % report_every == 0 or i == total:
            _log(f"  {i}/{total} pools  ({len(picker_scores)} scored, {skipped} skipped)")

    _log(
        f"Done: {len(picker_scores)} pools scored "
        f"({skipped} skipped for < {NONLAND_DECK_SIZE} embeddable cards)"
    )
    return compute_diagnostic(picker_scores, sa_a, sa_b)


def _source_pools(config: ValidateBuilderConfig) -> list[list[str]]:
    if config.pools_from is not None:
        return drafted_pools_from_corpus(config.pools_from, config.n_pools)
    if config.fresh_pools:
        return fresh_drafted_pools(config.set_code, config.n_pools)
    raise ValueError("Provide either --pools-from or --fresh-pools")


def fresh_drafted_pools(set_code: str | None, limit: int) -> list[list[str]]:
    """Run the draft worker and return freshly-*drafted* pools (``packs × P`` cards).

    "Fresh pools" for this diagnostic means freshly *drafted* pools — the exact
    45-card shape the picker labels in ``generate-draft-data`` — not 6-booster
    sealed pools, whose 23-of-~75 selection problem is a different distribution
    and would make the gating decision misleading (spec §5.3). Drives
    ``DraftWorkerMain`` (restarting on crash) and reconstructs each seat's pool
    from the booster transcript until ``limit`` pools are collected.
    """
    from draft.application.agent_mix import format_agent_mix, parse_agent_mix
    from draft.application.generate_draft_data import parse_sentinel_line
    from draft.domain.draft_geometry import Booster, DraftGeometry, DraftRecord, Seat
    from draft.infrastructure.draft_worker_connector import DraftWorkerConnector
    from price_predictor.infrastructure.forge_jvm import kill_process_tree

    mix = format_agent_mix(parse_agent_mix("forge-full:6,forge-r30:1,forge-r100:1"))
    connector = DraftWorkerConnector()
    pools: list[list[str]] = []
    drafts = 0
    attempts = 0
    max_attempts = 50  # bound restarts so a broken Forge can't loop forever
    _log(f"Drafting fresh pools (target {limit}, set={set_code or 'random per draft'})...")
    while len(pools) < limit and attempts < max_attempts:
        attempts += 1
        proc = connector.start(agent_mix=mix, set_code=set_code)
        try:
            for line in proc.stdout:
                transcript = parse_sentinel_line(line)
                if transcript is None:
                    continue
                boosters = [
                    Booster(b["set_code"], list(b["picks"]))
                    for b in transcript["boosters"]
                ]
                agents = [s["agent"] for s in transcript["seats"]]
                record = DraftRecord(
                    transcript["draft_id"], "", "",
                    [Seat(a, [], None) for a in agents], boosters,
                )
                geo = DraftGeometry.from_record(record)
                for seat_idx in range(geo.pod_size):
                    pools.append(geo.drafted_pool(record, seat_idx))
                    if len(pools) >= limit:
                        _log(f"  collected {len(pools)} pools from {drafts + 1} drafts")
                        return pools
                drafts += 1
                if drafts % 5 == 0:
                    _log(f"  {drafts} drafts -> {len(pools)} pools")
        finally:
            kill_process_tree(proc)
    _log(f"  collected {len(pools)} pools from {drafts} drafts")
    return pools
