"""The start of a gen-2 RL run: the corpora read into examples and learner
trajectories, the actor, critic and KL reference loaded and the corpus split, the
optimizer with its LR and coefficient schedules, and the checkpoint paths.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from draft.application.draft_lr_schedule import _make_scheduler, _PlateauLR, _reapply_resume_lr
from draft.application.imitation_batches import BatchSource
from draft.application.rl_batches import PolicyPair
from draft.application.rl_coef_schedule import _CoefSchedule
from draft.application.rl_examples import RLExample, _Loader, split_draft_disjoint
from draft.application.rl_log import _log
from draft.application.rl_model_setup import _Resume, _resume_or_build
from draft.application.rl_training_loop import RLCheckpointTargets, RLLearner
from draft.domain.draft_geometry import DraftRecord
from draft.infrastructure.draft_agent_store import DraftAgentStore
from draft.infrastructure.draft_record_io import read_records
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

if TYPE_CHECKING:
    from draft.application.train_draft_agent_rl import TrainDraftAgentRLConfig


@dataclass(frozen=True)
class RLCorpus:
    """Every example of the corpora, the on-policy learner seats' trajectories, the
    shared embedding table and the shape the loader saw."""

    examples: list[RLExample]
    trajectories: list[list[RLExample]]
    table: np.ndarray
    embedding_dim: int
    max_packs: int
    max_pack_size: int


def load_corpus(config: TrainDraftAgentRLConfig) -> RLCorpus:
    """Read and convert the corpora; refuse one with nothing for the learner."""
    on_policy, critic_only = _read_corpora(config)
    locator = ConvertedCardLocator(
        config.cards_path, embeddings_path=config.embeddings_path,
    )
    loader = _Loader(locator, set(config.learner_agents))
    examples, trajectories = loader.build(on_policy, critic_only)
    embedding_dim = _usable_width(config, loader, examples, trajectories)
    return RLCorpus(
        examples, trajectories, loader.table(), embedding_dim,
        loader.max_packs, loader.max_pack_size,
    )


def _read_corpora(
    config: TrainDraftAgentRLConfig,
) -> tuple[list[DraftRecord], list[DraftRecord]]:
    """The on-policy records (refused when there are none) and the critic-only ones."""
    on_policy = list(read_records(config.drafts_path))
    if not on_policy:
        raise ValueError(f"No records in {config.drafts_path}")
    critic_only: list[DraftRecord] = []
    for path in config.critic_corpus:
        critic_only.extend(read_records(path))
    return on_policy, critic_only


def _usable_width(
    config: TrainDraftAgentRLConfig, loader: _Loader, examples: list[RLExample],
    trajectories: list[list[RLExample]],
) -> int:
    """The cache's embedding width, once the corpus is known to have examples and
    learner trajectories."""
    if not examples:
        raise ValueError("No usable training examples (all picks missing embeddings?)")
    embedding_dim = loader.embedding_dim
    if embedding_dim is None:
        raise ValueError("Could not determine embedding dimension from the cache")
    if not trajectories:
        raise ValueError(
            "No learner trajectories — does --drafts-path contain seats with a "
            f"--learner-agents label ({', '.join(sorted(config.learner_agents))})?"
        )
    return embedding_dim


@dataclass(frozen=True)
class RLStart:
    """The loaded checkpoint, the models on the device, what batches are built from
    (the reference's critic standardization) and the draft-disjoint split."""

    resume: _Resume
    models: PolicyPair
    source: BatchSource
    train: list[RLExample]
    val: list[RLExample]


def start_run(
    config: TrainDraftAgentRLConfig, store: DraftAgentStore, corpus: RLCorpus,
    device: torch.device,
) -> RLStart:
    """Load the models, split the corpus (refusing an empty training split) and log
    the run's shape."""
    resume = _resume_or_build(config, store, corpus.embedding_dim)
    mean, std = resume.critic_mean, resume.critic_std
    models = PolicyPair(resume.model.to(device), resume.reference.to(device))
    source = BatchSource(corpus.table, mean, std, device)
    train_ids, val_ids = split_draft_disjoint(corpus.examples, config.val_fraction)
    train = [ex for ex in corpus.examples if ex.draft_index in train_ids]
    val = [ex for ex in corpus.examples if ex.draft_index in val_ids]
    if not train:
        raise ValueError("Empty training split; need more drafts or a smaller --val-fraction")
    _log(
        f"Examples: {len(train)} train + {len(val)} val; "
        f"{len(corpus.trajectories)} learner trajectories "
        f"(embedding_dim={corpus.embedding_dim}, packs={corpus.max_packs}, "
        f"P={corpus.max_pack_size}); critic mean={mean:.4f} std={std:.4f}; "
        f"generation {resume.generation}; device {device}"
    )
    return RLStart(resume, models, source, train, val)


def steps_per_epoch(config: TrainDraftAgentRLConfig, start: RLStart) -> int:
    return max(1, math.ceil(len(start.train) / config.batch_size))


def build_learner(config: TrainDraftAgentRLConfig, start: RLStart, n_steps: int) -> RLLearner:
    """The optimizer (restored on ``--resume``), its LR schedule and plateau
    controller, and the entropy and KL coefficient schedules."""
    resume = start.resume
    model = start.models.model
    optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": config.lr}])
    if resume.optimizer_state:
        optimizer.load_state_dict(resume.optimizer_state)
        _reapply_resume_lr(optimizer, config.lr)
    min_lr = config.min_lr if config.min_lr is not None else config.lr * 1e-3
    plateau = _PlateauLR(
        base_lr=config.lr, factor=config.lr_decay_factor,
        min_lr=min_lr, decay_count=resume.lr_decay_count,
    )
    scheduler = _make_scheduler(optimizer, n_steps, config.warmup_frac, plateau)
    entropy_sched = _CoefSchedule(config.entropy_coef, config.entropy_decay_after)
    kl_sched = _CoefSchedule(config.kl_coef, config.entropy_decay_after)
    return RLLearner(start.models, optimizer, scheduler, plateau, entropy_sched, kl_sched)


def checkpoint_targets(
    config: TrainDraftAgentRLConfig, store: DraftAgentStore, generation: int,
) -> RLCheckpointTargets:
    """Create ``--checkpoint-dir``; ``latest.pt`` and this run's timestamped best there."""
    out_dir = config.checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_config = {k: _recorded(v) for k, v in asdict(config).items()}
    return RLCheckpointTargets(
        store, out_dir / "latest.pt", out_dir / f"{timestamp}.pt", train_config, generation,
    )


def _recorded(value: Any) -> Any:
    """A config value as the checkpoint's ``train_config`` records it."""
    if isinstance(value, Path):
        return str(value)
    return list(value) if isinstance(value, tuple) else value
