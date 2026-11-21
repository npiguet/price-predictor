"""Train the two-headed draft agent (policy + critic) on a recorded corpus.

Each ``(draft, seat, pack, pick)`` is one typed-token example (FR-030). The
imitation policy is trained with cross-entropy over the whitelisted seats'
``PACK`` tokens (FR-033); the Monte-Carlo critic is trained with MSE over the
standardized leave-one-out pod-relative reward of every non-failed seat
(FR-032). Loss = ``imitation_weight·CE + critic_weight·MSE``.

Follows the ``train_picker`` / ``train_scorer`` / ``train_encoder`` scaffolding
(warmup-then-constant LR, per-group max-norm clip, resume/bootstrap guard,
best-by-val + ``latest.pt``, early stop) without extracting a shared trainer —
see research §Third-instance check (and the T035 follow-up).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from draft.application.draft_lr_schedule import (
    _make_scheduler,
    _maybe_decay,
    _PlateauLR,
    _reapply_resume_lr,
)
from draft.application.draft_training_common import length_bucketed_batches
from draft.application.imitation_batches import _collate, _compute_loss
from draft.application.imitation_examples import (
    RANDOM_SEED,
    _Loader,
    critic_standardization,
    split_draft_disjoint,
)
from draft.application.imitation_log import _fmt_dur, _log
from draft.application.imitation_model_setup import _resume_or_build, _select_device
from draft.application.imitation_validation import _validate
from draft.infrastructure.draft_agent_store import DraftAgentStore
from draft.infrastructure.draft_record_io import read_records
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

_STEP_LOG_INTERVAL = 15.0  # seconds between within-epoch progress lines


@dataclass(kw_only=True)
class TrainDraftAgentConfig:
    drafts_path: Path
    cards_path: Path
    #: The ``.npz`` embeddings' tree, laid out like ``cards_path``; None
    #: reads them beside the cards.
    embeddings_path: Path | None = None
    #: Where the checkpoints (``latest.pt``, ``<timestamp>.pt``) go.
    checkpoint_dir: Path
    d_model: int | None = None
    n_layers: int = 4
    n_heads: int = 8
    ff_dim: int | None = None
    dropout: float = 0.0
    imitation_weight: float = 1.0
    critic_weight: float = 1.0
    imitation_agents: tuple[str, ...] = ("forge-full",)
    lr: float = 3e-4
    warmup_frac: float = 0.05
    batch_size: int = 32
    max_grad_norm: float = 1.0
    epochs: int = 100
    val_fraction: float = 0.0025
    evals_per_epoch: int = 100  # "mini-epochs": validate + checkpoint this often
    patience: int = 30  # counted in mini-epochs (evals), not full epochs
    # Optional ReduceLROnPlateau-style annealing: after lr_decay_patience
    # mini-epochs without a new best val_loss, LR *= lr_decay_factor, down to
    # min_lr. None disables (constant LR after warmup, the default).
    lr_decay_patience: int | None = None
    lr_decay_factor: float = 0.1
    min_lr: float | None = None  # decay floor; None => lr * 1e-3
    resume: Path | None = None
    checkpoint: Path | None = None


def _validate_lr_decay(config: TrainDraftAgentConfig) -> None:
    """Fail fast on inconsistent LR-annealing flags (no-op when disabled)."""
    if config.lr_decay_patience is None:
        return
    if config.lr_decay_patience < 1:
        raise ValueError("--lr-decay-patience must be >= 1")
    if not 0.0 < config.lr_decay_factor < 1.0:
        raise ValueError("--lr-decay-factor must be in (0, 1)")
    if config.lr_decay_patience >= config.patience:
        raise ValueError(
            f"--lr-decay-patience ({config.lr_decay_patience}) must be < "
            f"--patience ({config.patience}) so a decay pre-empts early stop"
        )


@dataclass
class TrainDraftAgentResult:
    best_val_loss: float
    best_epoch: int
    best_path: Path


class TrainDraftAgentUseCase:
    """Joint policy + critic training (FR-030 … FR-039)."""

    def execute(self, config: TrainDraftAgentConfig) -> TrainDraftAgentResult:
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        device = _select_device()
        store = DraftAgentStore()
        _validate_lr_decay(config)

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
        table = loader.table()  # shared (num_unique_cards, dim) embedding table
        del records  # ~1 GB of parsed JSON no longer needed once examples are built

        train, val = split_draft_disjoint(examples, config.val_fraction)
        if not train:
            raise ValueError("Empty training split; need more drafts or a smaller --val-fraction")
        mean, std = critic_standardization(train)
        _log(
            f"Examples: {len(train)} train + {len(val)} val "
            f"(embedding_dim={loader.embedding_dim}, packs={loader.max_packs}, "
            f"P={loader.max_pack_size}); critic mean={mean:.4f} std={std:.4f}"
        )

        resume, ck_mean, ck_std = _resume_or_build(
            config, store, loader.embedding_dim, loader.max_packs, loader.max_pack_size,
        )
        if config.resume is not None or config.checkpoint is not None:
            mean, std = ck_mean, ck_std  # reuse stored standardization on reload
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
        n_steps = max(1, math.ceil(len(train) / config.batch_size))
        # Warmup over a fraction of ONE epoch (not epochs*n_steps): with early
        # stopping a run rarely passes epoch ~2-3, so a warmup sized to the max
        # epoch count would never reach full LR.
        scheduler = _make_scheduler(optimizer, n_steps, config.warmup_frac, plateau)

        out_dir = config.checkpoint_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_path = out_dir / "latest.pt"
        best_path = out_dir / f"{timestamp}.pt"
        train_config = {
            k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()
        }

        # Sort val once by length so eval batches are low-padding (cheap eval).
        val = sorted(val, key=lambda ex: ex.n_tokens)

        rng = random.Random(RANDOM_SEED)
        best_val_loss = resume.best_val_loss
        best_at = (resume.start_epoch, 0)
        evals_since_best = 0
        evals_per_epoch = max(1, config.evals_per_epoch)
        eval_interval = max(1, n_steps // evals_per_epoch)

        def run_eval(
            epoch: int, step: int, train_imit: float, train_crit: float,
        ) -> bool:
            """Validate + save latest/best; return True if early-stop is reached.

            ``train_imit``/``train_crit`` are the mean training losses over the
            mini-epoch just finished, logged beside the val losses so the
            train-vs-val gap is visible on one line.
            """
            nonlocal best_val_loss, best_at, evals_since_best
            report = _validate(model, val, table, mean, std, config, device)
            mse_str = ", ".join(
                f"p{pk}={v:.4f}" for pk, v in sorted(report.per_pack_mse.items())
            )
            train_loss = (
                config.imitation_weight * train_imit
                + config.critic_weight * train_crit
            )
            # Show the live LR only when annealing is on (keeps default output
            # byte-identical to before).
            lr_str = (
                f" lr={optimizer.param_groups[0]['lr']:.2e}"
                if config.lr_decay_patience is not None else ""
            )
            _log(
                f"  eval epoch {epoch} step {step}/{n_steps} | "
                f"train_loss={train_loss:.4f} train_imit={train_imit:.4f} "
                f"train_crit={train_crit:.4f} | val_loss={report.loss:.4f} "
                f"val_imit={report.imitation:.4f} val_crit={report.critic:.4f} "
                f"top1={report.top1:.3f} top3={report.top3:.3f} "
                f"per_pack_mse[{mse_str}]{lr_str}"
            )
            new_best = report.loss < best_val_loss
            if new_best:
                best_val_loss = report.loss
                best_at = (epoch, step)
                evals_since_best = 0
            else:
                evals_since_best += 1
            # Plateau decay: divide the LR and reset the counter so the new LR
            # gets a fresh patience window (an implicit cooldown). With
            # lr_decay_patience < patience this pre-empts early stop until the
            # floor is reached, so early stop only fires at min_lr.
            new_lr = _maybe_decay(evals_since_best, config.lr_decay_patience, plateau)
            if new_lr is not None:
                _log(
                    f"  LR decay #{plateau.decay_count} -> {new_lr:.2e} after "
                    f"{config.lr_decay_patience} mini-epochs without improvement"
                )
                evals_since_best = 0
            store.save_checkpoint(
                model, optimizer, epoch, best_val_loss, model.config, latest_path,
                critic_mean=mean, critic_std=std, train_config=train_config,
                lr_decay_count=plateau.decay_count,
            )
            if new_best:
                store.save_checkpoint(
                    model, optimizer, epoch, best_val_loss, model.config, best_path,
                    critic_mean=mean, critic_std=std, train_config=train_config,
                    lr_decay_count=plateau.decay_count,
                )
            model.train()  # _validate switched to eval mode
            return evals_since_best >= config.patience

        _log(
            f"Training: up to {config.epochs} epochs, {n_steps} steps/epoch; "
            f"eval+checkpoint every {eval_interval} steps (~{evals_per_epoch} "
            f"mini-epochs/epoch); patience {config.patience} mini-epochs; on {device}"
        )
        stop = False
        for epoch in range(resume.start_epoch, resume.start_epoch + config.epochs):
            model.train()
            batches = length_bucketed_batches(train, config.batch_size, rng)
            nb = len(batches)
            epoch_start = time.monotonic()
            last_log = epoch_start
            last_log_step = 0
            win_imit = win_crit = 0.0
            win_n = 0
            ev_imit = ev_crit = 0.0  # train loss since the last eval (mini-epoch)
            ev_n = 0
            for step, chunk in enumerate(batches, start=1):
                batch = _collate(chunk, table, mean, std, device)
                losses, _, _ = _compute_loss(
                    model, batch, config.imitation_weight, config.critic_weight,
                )
                optimizer.zero_grad()
                losses.total.backward()
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group["params"], config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                imit_v = float(losses.imitation.detach())
                crit_v = float(losses.critic.detach())
                win_imit += imit_v
                win_crit += crit_v
                win_n += 1
                ev_imit += imit_v
                ev_crit += crit_v
                ev_n += 1
                now = time.monotonic()
                if now - last_log >= _STEP_LOG_INTERVAL and step < nb:
                    # Windowed rate = true recent training speed; ETA uses the
                    # cumulative rate (which includes the eval/checkpoint pauses)
                    # so it reflects real end-to-end wall time.
                    rate = (step - last_log_step) / (now - last_log)
                    overall = step / (now - epoch_start)
                    eta = (nb - step) / overall if overall > 0 else 0.0
                    ti, tc = win_imit / win_n, win_crit / win_n
                    tl = config.imitation_weight * ti + config.critic_weight * tc
                    _log(
                        f"  epoch {epoch} step {step}/{nb} ({step / nb * 100:.0f}%) "
                        f"train_loss={tl:.4f} train_imit={ti:.4f} train_crit={tc:.4f} "
                        f"{rate:.1f} steps/s ETA {_fmt_dur(eta)}"
                    )
                    last_log = now
                    last_log_step = step
                    win_imit = win_crit = 0.0
                    win_n = 0
                if step % eval_interval == 0 or step == nb:
                    train_imit = ev_imit / ev_n if ev_n else 0.0
                    train_crit = ev_crit / ev_n if ev_n else 0.0
                    ev_imit = ev_crit = 0.0
                    ev_n = 0
                    if run_eval(epoch, step, train_imit, train_crit):
                        _log(
                            f"Early stop: {evals_since_best} mini-epochs without "
                            f"val improvement (--patience={config.patience})"
                        )
                        stop = True
                        break
            if stop:
                break

        best_epoch, best_step = best_at
        _log(
            f"Done. Best val_loss={best_val_loss:.4f} at epoch {best_epoch} "
            f"step {best_step}; best checkpoint: {best_path}"
        )
        return TrainDraftAgentResult(best_val_loss, best_epoch, best_path)
