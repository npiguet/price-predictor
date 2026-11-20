"""Training use case for the one-shot sealed deck picker.

REINFORCE-from-random-init against a frozen scorer, with a per-pool
empirical-mean baseline, an entropy bonus on a val-reward-driven decay
schedule, and an auxiliary pool-quality head (spec 017).

shared-trainer (resolved, spec 018): four trainers now share the resume /
bootstrap / best-vs-latest persistence / fail-fast width-check / early-stop
scaffolding — ``train_encoder``, ``train_scorer``, ``train_picker``, and
``draft.application.train_draft_agent`` (supervised CE + MC-regression MSE).
The fourth arrival was *not* REINFORCE-style (the original extraction trigger),
and it diverges yet again on dataset shape (per-(draft,seat,pick) typed-token
states) and on carrying two heads over two seat-subsets. The genuinely
identical pieces remain small and stable — the warmup ``LambdaLR`` lambda and
the per-group ``clip_grad_norm_`` — while loss/metric/val bodies differ across
all four. Decision (re-evaluated in specs/018-draft-agent, research §Third-
instance check): **defer extraction**; a generic loop over four loss/metric/
shape variants would be speculative coupling (Simplicity-First). Each trainer
copies the two small helpers by following the pattern.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from price_predictor.infrastructure.torch_training import (
    clip_per_group,
    kl_divergence,
    policy_entropy,
)
from sealed.application.picker_losses import _compute_losses
from sealed.application.picker_pools import (
    PreparedPool,
    _collate,
    _prepare_pools,
    _shuffle_train,
    _split_pools,
)
from sealed.application.picker_resume import (
    _build_optimizer,
    _check_picker_width,
    _move_optimizer_state,
    _resume_or_build_picker,
)
from sealed.application.picker_sampling import _plackett_luce_log_prob, _sample_decks
from sealed.application.picker_scoring import _load_scorer, _score_sampled, _select_device
from sealed.application.picker_training_loop import (
    _EntropySchedule,
    _EpochStats,
    _log,
    _log_epoch,
    _should_stop,
)
from sealed.application.picker_validation import _validate
from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE
from sealed.domain.picker_model import PickerConfig, PickerModel
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.picker_store import PickerStore
from sealed.infrastructure.pool_file_reader import parse_pools

RANDOM_SEED = 42  # hardcoded (FR-018); governs init, shuffle, sampling, split.


@dataclass
class TrainPickerConfig:
    pools_path: Path
    scorer_checkpoint: Path
    cards_path: Path
    checkpoint_dir: Path
    auditor_scorer_checkpoint: Path | None = None
    #: The ``.npz`` embeddings' tree, laid out like ``cards_path``; None
    #: reads them beside the cards.
    embeddings_path: Path | None = None
    resume: Path | None = None
    picker_checkpoint: Path | None = None
    d_model: int | None = None
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int | None = None
    dropout: float = 0.0
    aux_weight: float = 0.1
    normalize_advantage: bool = False
    objective: str = "reinforce"
    topk: int = 16
    batch_size: int = 16
    n_samples: int = 64
    temperature: float = 1.0
    entropy_coef: float = 0.01
    entropy_decay_after: int = 5
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    epochs: int = 100
    val_fraction: float = 0.2
    patience: int = 10
    evals_per_epoch: int = 1
    kl_coef: float = 0.0

    def picker_config(self, embedding_dim: int) -> PickerConfig:
        """Build the architecture config for a fresh picker sized to the cache."""
        return PickerConfig(
            embedding_dim=embedding_dim,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )


def _build_train_config(config: TrainPickerConfig) -> dict[str, Any]:
    """Flatten ``TrainPickerConfig`` to a JSON-friendly dict (Path -> str)."""
    raw = asdict(config)
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in raw.items()}


@dataclass
class TrainPickerResult:
    model: PickerModel
    best_val_reward: float
    best_epoch: int
    best_path: Path


class TrainPickerUseCase:
    """Train a one-shot picker via REINFORCE against a frozen scorer."""

    def execute(self, config: TrainPickerConfig) -> TrainPickerResult:
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        device = _select_device()
        store = PickerStore()

        t0 = time.monotonic()
        scorer = _load_scorer(config.scorer_checkpoint, device)
        embedding_dim = scorer.config.d_model
        _log(f"Loaded training scorer (d_model={embedding_dim}) on {device}")

        locator = ConvertedCardLocator(
            config.cards_path, embeddings_path=config.embeddings_path,
        )
        prepared = _prepare_pools(parse_pools(config.pools_path), locator)
        if not prepared:
            raise ValueError(
                f"No usable pools in {config.pools_path} (every pool had fewer "
                f"than {NONLAND_DECK_SIZE} embeddable cards under {locator.embeddings_root})."
            )
        cache_width = prepared[0].dim
        _check_picker_width(
            scorer.config.d_model, cache_width,
            f"--scorer-checkpoint {config.scorer_checkpoint}", locator.embeddings_root,
        )

        auditor: SetTransformerScorer | None = None
        if config.auditor_scorer_checkpoint is not None:
            auditor = _load_scorer(config.auditor_scorer_checkpoint, device)
            _check_picker_width(
                auditor.config.d_model, cache_width,
                f"--auditor-scorer-checkpoint {config.auditor_scorer_checkpoint}",
                locator.embeddings_root,
            )
            _log("Auditor scorer loaded; cross-scorer audit enabled")

        val_pools, train_pools = _split_pools(prepared, config.val_fraction)
        _log(
            f"Pools ready: {len(train_pools)} train + {len(val_pools)} val "
            f"({time.monotonic() - t0:.1f}s)"
        )

        resume = _resume_or_build_picker(config, store, embedding_dim)
        picker = resume.model.to(device)
        optimizer = _build_optimizer(picker, config)
        if resume.optimizer_state:
            optimizer.load_state_dict(resume.optimizer_state)
            _move_optimizer_state(optimizer, device)

        reference: PickerModel | None = None
        if config.kl_coef != 0 and resume.reference_state is not None:
            reference = PickerModel(picker.config)
            reference.load_state_dict(resume.reference_state)
            reference.eval()
            reference.to(device)

        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_path = config.checkpoint_dir / "latest.pt"
        best_path = config.checkpoint_dir / f"best_{timestamp}.pt"
        train_config = _build_train_config(config)

        schedule = _EntropySchedule(config.entropy_coef, config.entropy_decay_after)
        rng = random.Random(RANDOM_SEED)
        best_val_reward = resume.best_val_reward
        best_epoch = resume.start_epoch
        evals_since_best = 0

        # Validate `evals_per_epoch` times per epoch (a "mini-epoch" is one such
        # interval). With the default 1 this collapses to one eval at epoch end,
        # i.e. the original behavior. Checkpoints and early-stop act on every
        # eval, so patience is denominated in mini-epochs (FR-019, FR-020).
        n_steps = math.ceil(len(train_pools) / config.batch_size)
        evals_per_epoch = max(1, config.evals_per_epoch)
        eval_interval = max(1, n_steps // evals_per_epoch)

        _log(
            f"Starting training loop ({config.epochs} epochs, {n_steps} steps/epoch, "
            f"validating every {eval_interval} steps = ~{evals_per_epoch}x/epoch; "
            f"patience={config.patience} evals) on {device}"
        )
        last_epoch = resume.start_epoch
        stop = False
        for epoch in range(resume.start_epoch, resume.start_epoch + config.epochs):
            last_epoch = epoch
            shuffled = _shuffle_train(train_pools, rng)
            picker.train()
            stats = _EpochStats()
            for step, start in enumerate(
                range(0, len(shuffled), config.batch_size), start=1,
            ):
                group = shuffled[start:start + config.batch_size]
                policy, entropy, aux, kl = self._train_one_step(
                    picker, scorer, reference, optimizer, group, config,
                    schedule.coef, device,
                )
                stats.add(policy, entropy, aux, kl)

                if step % eval_interval != 0 and step != n_steps:
                    continue

                report = _validate(
                    picker, scorer, auditor, val_pools, config.batch_size, device,
                )
                schedule.update(report.mean_reward)
                _log_epoch(epoch, step, n_steps, stats, report)
                stats = _EpochStats()

                new_best = report.mean_reward > best_val_reward
                if new_best:
                    best_val_reward = report.mean_reward
                    best_epoch = epoch
                    evals_since_best = 0
                else:
                    evals_since_best += 1

                store.save_checkpoint(
                    picker, optimizer, epoch, best_val_reward, picker.config,
                    latest_path, train_config=train_config,
                )
                if new_best:
                    store.save_checkpoint(
                        picker, optimizer, epoch, best_val_reward, picker.config,
                        best_path, train_config=train_config,
                    )

                if _should_stop(evals_since_best, config.patience):
                    _log(
                        f"Early stop: {evals_since_best} evals without val-reward "
                        f"improvement (--patience={config.patience})"
                    )
                    stop = True
                    break
                picker.train()  # _validate switched the picker to eval mode
            if stop:
                break

        _log(
            f"Done (last epoch {last_epoch}). Best val_reward={best_val_reward:.4f} "
            f"at epoch {best_epoch}; best checkpoint: {best_path}"
        )
        return TrainPickerResult(picker, best_val_reward, best_epoch, best_path)

    def _train_one_step(
        self,
        picker: PickerModel,
        scorer: SetTransformerScorer,
        reference: PickerModel | None,
        optimizer: torch.optim.Optimizer,
        group: list[PreparedPool],
        config: TrainPickerConfig,
        entropy_coef: float,
        device: torch.device,
    ) -> tuple[float, float, float, float]:
        """One gradient step over a batch of pools. Returns the detached
        (policy_loss, entropy_loss, aux_loss, kl) for logging."""
        cards, mask, land = _collate(group)
        pool_cards = torch.from_numpy(cards).to(device)
        pool_mask = torch.from_numpy(mask).to(device)
        is_land_mask = torch.from_numpy(land).to(device)

        logits, aux_pred = picker(pool_cards, pool_mask)

        pick_indices, picked_mask = _sample_decks(
            logits.detach(), is_land_mask, pool_mask,
            config.n_samples, config.temperature,
        )

        norm_pool = scorer.normalize_features(pool_cards)
        rewards = _score_sampled(
            scorer, norm_pool, pick_indices, picked_mask,
            config.n_samples, device,
        )  # (B, S)

        # PL log-prob on temperature-scaled, padding-masked logits (B*S, N).
        b, n = logits.shape
        scaled = (logits / config.temperature).masked_fill(
            ~pool_mask, float("-inf"),
        )
        scaled_bs = scaled.unsqueeze(1).expand(b, config.n_samples, n).reshape(
            b * config.n_samples, n,
        )
        log_prob = _plackett_luce_log_prob(
            scaled_bs, pick_indices, picked_mask,
        ).view(b, config.n_samples)

        entropy = policy_entropy(logits, pool_mask)
        losses = _compute_losses(
            rewards, log_prob, entropy, aux_pred,
            entropy_coef, config.aux_weight, config.normalize_advantage,
            config.objective, config.topk,
        )
        total = losses.total

        kl_value = 0.0
        if reference is not None:
            with torch.no_grad():
                ref_logits, _ = reference(pool_cards, pool_mask)
            kl = kl_divergence(logits, ref_logits, pool_mask).mean()
            total = total + config.kl_coef * kl
            kl_value = float(kl)

        optimizer.zero_grad()
        total.backward()
        clip_per_group(optimizer, max_norm=config.max_grad_norm)
        optimizer.step()

        return (
            float(losses.policy_loss.detach()),
            float(losses.entropy_loss.detach()),
            float(losses.aux_loss.detach()), kl_value,
        )
