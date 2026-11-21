"""Gen-2 RL self-play fine-tuning of the two-headed draft agent (spec 020).

On-policy actor-critic (REINFORCE + GAE baseline + KL anchor + entropy bonus)
over a self-play corpus the reference checkpoint generated. Per learner pick the
loss is

    L = − A_t·logπ_T(a_t|s_t)              policy gradient (learner picks only)
      + value_weight · MSE(V(s_t), R_std)  critic regression (all non-failed seats)
      − entropy_coef · H(π_T(·|s_t))       exploration bonus (learner picks)
      + kl_coef · KL(π_T ‖ π_ref,T)        anchor to the reference policy

where π_T is the PACK-masked softmax at the rollout temperature T (research D4),
A_t is the detached GAE(λ) advantage over the seat's ordered trajectory
(research D2/D3), and R is the pod-relative leave-one-out deck_score (terminal).

Mirrors ``train_draft_agent``'s scaffolding (warmup-then-constant LR, per-group
clip, mini-epoch eval + best/latest + early stop, LR-plateau annealing); the
per-pick state walk is the shared :mod:`draft.application.draft_pick_states`. The
REINFORCE helpers (entropy/KL) follow ``sealed.application.picker_training_step``. On-
policy pairing of corpus↔checkpoint is the operator's responsibility — the only
safeguard is the behaviour-anomaly warning (research D6); the corpus is unchanged
(no provenance field, FR-021).
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
from draft.application.imitation_batches import BatchSource
from draft.application.imitation_model_setup import _select_device
from draft.application.rl_advantage_refresh import (
    AdvantageRefresh,
    CriticPass,
    _precompute_advantages,
)
from draft.application.rl_batches import LossSettings, PolicyPair, _collate, _compute_loss
from draft.application.rl_behaviour_check import (
    _ANOMALY_FRACTION,
    _PROB_FLOOR,
    _behaviour_anomaly_summary,
)
from draft.application.rl_coef_schedule import _CoefSchedule
from draft.application.rl_examples import RANDOM_SEED, _Loader, split_draft_disjoint
from draft.application.rl_log import _STEP_LOG_INTERVAL, _fmt_dur, _log
from draft.application.rl_model_setup import _resume_or_build
from draft.application.rl_validation import _validate
from draft.domain.draft_geometry import DraftRecord
from draft.infrastructure.draft_agent_store import DraftAgentStore
from draft.infrastructure.draft_record_io import read_records
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator


@dataclass(kw_only=True)
class TrainDraftAgentRLConfig:
    """Inputs to the RL trainer (data-model §1).

    ``checkpoint``/``learner_agents``/``rollout_temperature`` are required on a
    fresh run (enforced by the CLI / startup validation); they are resumable so
    ``--resume`` inherits them. Architecture is inherited from the loaded
    checkpoint — there are no architecture flags.
    """

    checkpoint: Path | None = None
    resume: Path | None = None
    drafts_path: Path
    critic_corpus: tuple[Path, ...] = ()
    cards_path: Path
    #: The ``.npz`` embeddings' tree, laid out like ``cards_path``; None
    #: reads them beside the cards.
    embeddings_path: Path | None = None
    #: Where the checkpoints (``latest.pt``, ``<timestamp>.pt``) go.
    checkpoint_dir: Path
    learner_agents: tuple[str, ...] = ()
    rollout_temperature: float | None = None
    gae_lambda: float = 0.95
    kl_coef: float = 0.1
    entropy_coef: float = 0.01
    entropy_decay_after: int = 5
    value_weight: float = 1.0
    lr: float = 3e-4
    warmup_frac: float = 0.05
    batch_size: int = 32
    max_grad_norm: float = 1.0
    epochs: int = 100
    val_fraction: float = 0.0025
    evals_per_epoch: int = 100
    patience: int = 30
    lr_decay_patience: int | None = None
    lr_decay_factor: float = 0.1
    min_lr: float | None = None


@dataclass
class TrainDraftAgentRLResult:
    best_val_loss: float
    best_epoch: int
    best_path: Path
    generation: int


class TrainDraftAgentRLUseCase:
    """On-policy actor-critic RL fine-tuning (spec 020 FR-001 … FR-021)."""

    def execute(self, config: TrainDraftAgentRLConfig) -> TrainDraftAgentRLResult:
        self._validate_config(config)
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        device = _select_device()
        store = DraftAgentStore()

        on_policy = list(read_records(config.drafts_path))
        if not on_policy:
            raise ValueError(f"No records in {config.drafts_path}")
        critic_only: list[DraftRecord] = []
        for path in config.critic_corpus:
            critic_only.extend(read_records(path))

        locator = ConvertedCardLocator(
            config.cards_path, embeddings_path=config.embeddings_path,
        )
        loader = _Loader(locator, set(config.learner_agents))
        examples, trajectories = loader.build(on_policy, critic_only)
        if not examples:
            raise ValueError("No usable training examples (all picks missing embeddings?)")
        if loader.embedding_dim is None:
            raise ValueError("Could not determine embedding dimension from the cache")
        if not trajectories:
            raise ValueError(
                "No learner trajectories — does --drafts-path contain seats with a "
                f"--learner-agents label ({', '.join(sorted(config.learner_agents))})?"
            )
        table = loader.table()

        resume = _resume_or_build(config, store, loader.embedding_dim)
        mean, std = resume.critic_mean, resume.critic_std
        model = resume.model.to(device)
        reference = resume.reference.to(device)
        models = PolicyPair(model, reference)
        source = BatchSource(table, mean, std, device)

        train_ids, val_ids = split_draft_disjoint(examples, config.val_fraction)
        train = [ex for ex in examples if ex.draft_index in train_ids]
        val = [ex for ex in examples if ex.draft_index in val_ids]
        if not train:
            raise ValueError("Empty training split; need more drafts or a smaller --val-fraction")
        _log(
            f"Examples: {len(train)} train + {len(val)} val; "
            f"{len(trajectories)} learner trajectories "
            f"(embedding_dim={loader.embedding_dim}, packs={loader.max_packs}, "
            f"P={loader.max_pack_size}); critic mean={mean:.4f} std={std:.4f}; "
            f"generation {resume.generation}; device {device}"
        )

        anomaly = _behaviour_anomaly_summary(
            reference, examples, source, config,
        )
        msg = (
            f"Behaviour check: sampled {anomaly.n_learner}/{anomaly.n_total} "
            f"learner picks, mean log π_ref={anomaly.mean_behaviour_logprob:.3f}, "
            f"{anomaly.frac_below_floor:.1%} below prob {_PROB_FLOOR:g}"
        )
        if anomaly.frac_below_floor > _ANOMALY_FRACTION:
            _log(
                f"WARNING: {msg} — the corpus may not be on-policy for this "
                "--checkpoint / --rollout-temperature. Proceeding anyway "
                "(on-policy pairing is the operator's responsibility)."
            )
        else:
            _log(msg)

        optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": config.lr}])
        if resume.optimizer_state:
            optimizer.load_state_dict(resume.optimizer_state)
            _reapply_resume_lr(optimizer, config.lr)
        min_lr = config.min_lr if config.min_lr is not None else config.lr * 1e-3
        plateau = _PlateauLR(
            base_lr=config.lr, factor=config.lr_decay_factor,
            min_lr=min_lr, decay_count=resume.lr_decay_count,
        )
        n_steps = max(1, math.ceil(len(train) / config.batch_size))
        scheduler = _make_scheduler(optimizer, n_steps, config.warmup_frac, plateau)
        entropy_sched = _CoefSchedule(config.entropy_coef, config.entropy_decay_after)
        kl_sched = _CoefSchedule(config.kl_coef, config.entropy_decay_after)

        out_dir = config.checkpoint_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_path = out_dir / "latest.pt"
        best_path = out_dir / f"{timestamp}.pt"
        train_config = {
            k: (str(v) if isinstance(v, Path) else (list(v) if isinstance(v, tuple) else v))
            for k, v in asdict(config).items()
        }

        val = sorted(val, key=lambda ex: ex.n_tokens)
        rng = random.Random(RANDOM_SEED)
        best_val_loss = resume.best_val_loss
        best_at = (resume.start_epoch, 0)
        evals_since_best = 0
        evals_per_epoch = max(1, config.evals_per_epoch)
        eval_interval = max(1, n_steps // evals_per_epoch)

        def save(path: Path, epoch: int) -> None:
            store.save_checkpoint(
                model, optimizer, epoch, best_val_loss, model.config, path,
                critic_mean=mean, critic_std=std, train_config=train_config,
                lr_decay_count=plateau.decay_count,
                rl_metadata={
                    "generation": resume.generation,
                    "reference_checkpoint": str(config.resume or config.checkpoint),
                    "algorithm": "reinforce+gae",
                    "gae_lambda": config.gae_lambda,
                    "kl_coef": config.kl_coef,
                    "entropy_coef": config.entropy_coef,
                    "value_weight": config.value_weight,
                    "rollout_temperature": float(config.rollout_temperature or 1.0),
                },
            )

        def run_eval(epoch: int, step: int) -> bool:
            nonlocal best_val_loss, best_at, evals_since_best
            report = _validate(
                models, val, source,
                LossSettings(config, entropy_sched.coef, kl_sched.coef),
            )
            _log(
                f"  eval epoch {epoch} step {step}/{n_steps} | "
                f"val_loss={report.loss:.4f} policy={report.policy:.4f} "
                f"value={report.value:.4f} entropy={report.entropy:.4f} "
                f"kl={report.kl:.4f} | ent_coef={entropy_sched.coef:.4g} "
                f"kl_coef={kl_sched.coef:.4g}"
            )
            entropy_sched.update(-report.loss)
            kl_sched.update(-report.loss)
            new_best = report.loss < best_val_loss
            if new_best:
                best_val_loss = report.loss
                best_at = (epoch, step)
                evals_since_best = 0
            else:
                evals_since_best += 1
            new_lr = _maybe_decay(evals_since_best, config.lr_decay_patience, plateau)
            if new_lr is not None:
                _log(f"  LR decay #{plateau.decay_count} -> {new_lr:.2e}")
                evals_since_best = 0
            save(latest_path, epoch)
            if new_best:
                save(best_path, epoch)
            return evals_since_best >= config.patience

        _log(
            f"Training: up to {config.epochs} epochs, {n_steps} steps/epoch; "
            f"eval every {eval_interval} steps (~{evals_per_epoch}/epoch); "
            f"patience {config.patience}"
        )
        stop = False
        for epoch in range(resume.start_epoch, resume.start_epoch + config.epochs):
            # Refresh GAE advantages from the current critic (research D3).
            _precompute_advantages(
                model, trajectories,
                AdvantageRefresh(CriticPass(source, config.batch_size), config.gae_lambda),
                epoch=epoch,
            )
            model.train()
            batches = length_bucketed_batches(train, config.batch_size, rng)
            nb = len(batches)
            epoch_start = time.monotonic()
            last_log = epoch_start
            for step, chunk in enumerate(batches, start=1):
                batch = _collate(chunk, source)
                out = _compute_loss(
                    models, batch, LossSettings(config, entropy_sched.coef, kl_sched.coef),
                )
                optimizer.zero_grad()
                out.total.backward()
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group["params"], config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                now = time.monotonic()
                if now - last_log >= _STEP_LOG_INTERVAL and step < nb:
                    overall = step / (now - epoch_start)
                    eta = (nb - step) / overall if overall > 0 else 0.0
                    _log(
                        f"  epoch {epoch} step {step}/{nb} "
                        f"({step / nb * 100:.0f}%) total={float(out.total.detach()):.4f} "
                        f"ETA {_fmt_dur(eta)}"
                    )
                    last_log = now
                if step % eval_interval == 0 or step == nb:
                    if run_eval(epoch, step):
                        _log(
                            f"Early stop: {evals_since_best} evals without val "
                            f"improvement (--patience={config.patience})"
                        )
                        stop = True
                        break
            if stop:
                break

        best_epoch, best_step = best_at
        _log(
            f"Done. Best val_loss={best_val_loss:.4f} at epoch {best_epoch} "
            f"step {best_step}; best checkpoint: {best_path} (generation "
            f"{resume.generation})"
        )
        return TrainDraftAgentRLResult(
            best_val_loss, best_epoch, best_path, resume.generation,
        )

    @staticmethod
    def _validate_config(config: TrainDraftAgentRLConfig) -> None:
        """Startup validation (FR-017, SC-003)."""
        if (config.checkpoint is None) == (config.resume is None):
            raise ValueError("exactly one of --checkpoint or --resume is required")
        if not config.learner_agents:
            raise ValueError("--learner-agents must be a non-empty whitelist")
        if config.rollout_temperature is None or config.rollout_temperature <= 0:
            raise ValueError("--rollout-temperature must be supplied and positive")
        if config.lr_decay_patience is not None:
            if config.lr_decay_patience < 1:
                raise ValueError("--lr-decay-patience must be >= 1")
            if not 0.0 < config.lr_decay_factor < 1.0:
                raise ValueError("--lr-decay-factor must be in (0, 1)")
            if config.lr_decay_patience >= config.patience:
                raise ValueError("--lr-decay-patience must be < --patience")
