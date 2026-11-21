"""The start of a gen-1 training run: the corpus read into split, standardized
examples, the model with its optimizer and LR schedule, and the checkpoint paths.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from draft.application.draft_lr_schedule import _make_scheduler, _PlateauLR, _reapply_resume_lr
from draft.application.imitation_batches import BatchSource
from draft.application.imitation_examples import (
    DraftExample,
    _Loader,
    critic_standardization,
    split_draft_disjoint,
)
from draft.application.imitation_log import _log
from draft.application.imitation_model_setup import CorpusShape, _Resume, _resume_or_build
from draft.application.imitation_training_loop import CheckpointTargets, Learner
from draft.infrastructure.draft_agent_store import DraftAgentStore
from draft.infrastructure.draft_record_io import read_records
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

if TYPE_CHECKING:
    from draft.application.train_draft_agent import TrainDraftAgentConfig


@dataclass(frozen=True)
class PreparedCorpus:
    """The corpus's examples split draft-disjoint, the shared embedding table, the
    shape a model is sized from, and the training split's critic standardization."""

    train: list[DraftExample]
    val: list[DraftExample]
    table: np.ndarray
    shape: CorpusShape
    mean: float
    std: float


def load_corpus(config: TrainDraftAgentConfig) -> PreparedCorpus:
    """Read, convert, split and standardize the corpus; refuse one with nothing to train on."""
    loader, examples = _convert_records(config)
    table = loader.table()  # shared (num_unique_cards, dim) embedding table
    train, val = split_draft_disjoint(examples, config.val_fraction)
    if not train:
        raise ValueError("Empty training split; need more drafts or a smaller --val-fraction")
    mean, std = critic_standardization(train)
    _log(
        f"Examples: {len(train)} train + {len(val)} val "
        f"(embedding_dim={loader.embedding_dim}, packs={loader.max_packs}, "
        f"P={loader.max_pack_size}); critic mean={mean:.4f} std={std:.4f}"
    )
    shape = CorpusShape(loader.embedding_dim, loader.max_packs, loader.max_pack_size)
    return PreparedCorpus(train, val, table, shape, mean, std)


def _convert_records(config: TrainDraftAgentConfig) -> tuple[_Loader, list[DraftExample]]:
    """The corpus's examples and the loader that built them.

    The parsed records (~1 GB of JSON on a big corpus) are freed on return,
    once the examples are built.
    """
    records = list(read_records(config.drafts_path))
    if not records:
        raise ValueError(f"No records in {config.drafts_path}")

    locator = ConvertedCardLocator(
        config.cards_path, embeddings_path=config.embeddings_path,
    )
    loader = _Loader(locator, set(config.imitation_agents))
    examples = loader.build(records)
    if not examples:
        raise ValueError("No usable training examples (all picks missing embeddings?)")
    if loader.embedding_dim is None:
        raise ValueError("Could not determine embedding dimension from the cache")
    return loader, examples


def start_model(
    config: TrainDraftAgentConfig, store: DraftAgentStore, corpus: PreparedCorpus,
    device: torch.device,
) -> tuple[_Resume, BatchSource]:
    """The fresh, resumed or warm-started model, and what its batches are built
    from; a loaded checkpoint's stored standardization replaces the corpus's."""
    resume, ck_mean, ck_std = _resume_or_build(config, store, corpus.shape)
    mean, std = corpus.mean, corpus.std
    if config.resume is not None or config.checkpoint is not None:
        mean, std = ck_mean, ck_std  # reuse stored standardization on reload
    return resume, BatchSource(corpus.table, mean, std, device)


def steps_per_epoch(config: TrainDraftAgentConfig, corpus: PreparedCorpus) -> int:
    return max(1, math.ceil(len(corpus.train) / config.batch_size))


def build_learner(
    config: TrainDraftAgentConfig, resume: _Resume, n_steps: int, device: torch.device,
) -> Learner:
    """The model on ``device``, its optimizer (restored on ``--resume``) and its schedule."""
    model = resume.model.to(device)
    optimizer = torch.optim.AdamW(
        [{"params": list(model.parameters()), "lr": config.lr}]
    )
    if resume.optimizer_state:
        optimizer.load_state_dict(resume.optimizer_state)
        # Re-apply config.lr (a possible --lr override) so the fresh scheduler
        # below recaptures its base_lr from it instead of the inherited one.
        _reapply_resume_lr(optimizer, config.lr)
    min_lr = config.min_lr if config.min_lr is not None else config.lr * 1e-3
    plateau = _PlateauLR(
        base_lr=config.lr, factor=config.lr_decay_factor,
        min_lr=min_lr, decay_count=resume.lr_decay_count,
    )
    # Warmup over a fraction of ONE epoch (not epochs*n_steps): with early
    # stopping a run rarely passes epoch ~2-3, so a warmup sized to the max
    # epoch count would never reach full LR.
    scheduler = _make_scheduler(optimizer, n_steps, config.warmup_frac, plateau)
    return Learner(model, optimizer, scheduler, plateau)


def checkpoint_targets(
    config: TrainDraftAgentConfig, store: DraftAgentStore,
) -> CheckpointTargets:
    """Create ``--checkpoint-dir``; ``latest.pt`` and this run's timestamped best there."""
    out_dir = config.checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_config = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()
    }
    return CheckpointTargets(
        store, out_dir / "latest.pt", out_dir / f"{timestamp}.pt", train_config,
    )
