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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from draft.application.draft_pick_states import iter_seat_pick_states
from draft.application.draft_training_common import (
    leave_one_out_rewards,
    length_bucketed_batches,
)
from draft.domain.draft_agent_model import (
    DraftAgentConfig,
    DraftAgentModel,
)
from draft.domain.draft_geometry import DraftGeometry, DraftRecord
from draft.domain.draft_state import TYPE_PACK
from draft.infrastructure.draft_agent_store import DraftAgentStore
from draft.infrastructure.draft_record_io import read_records
from price_predictor.infrastructure.torch_training import masked_log_softmax
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

RANDOM_SEED = 42  # hardcoded (FR-035): governs init, the draft-disjoint split.
_MISSING_WARN_CAP = 20
_STEP_LOG_INTERVAL = 15.0  # seconds between within-epoch progress lines


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _fmt_dur(seconds: float) -> str:
    """Compact h/m/s duration for progress/ETA lines."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


@dataclass
class TrainDraftAgentConfig:
    drafts_path: Path = field(default_factory=lambda: Path("output/draft/drafts.jsonl"))
    cards_path: Path = field(default_factory=lambda: Path("output/cardsfolder/"))
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


# --------------------------------------------------------------------------- #
# Training examples
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class DraftExample:
    """One ``(draft, seat, pack, pick)`` training example (data-model §2).

    To keep millions of examples in RAM, the card embeddings are *not* stored
    here: ``card_idx`` holds int32 rows into a shared embedding table (see
    ``_Loader.table``) materialized per-batch in :func:`_collate`. The small
    per-token int arrays are downcast (type/packs_ago/pick_ago all fit in int8);
    ``pack_mask`` is derived from ``type_idx == TYPE_PACK`` rather than stored.
    """

    draft_index: int             # groups examples for the draft-disjoint split
    card_idx: np.ndarray         # (N,) int32, rows into the shared table
    type_idx: np.ndarray         # (N,) int8
    packs_ago: np.ndarray        # (N,) int8
    pick_ago: np.ndarray         # (N,) int8
    pack_number: int
    pick_number: int
    imitation_active: bool
    target_token: int            # absolute index of the taken PACK token, or -1
    critic_active: bool
    critic_target: float         # raw (pre-standardization) pod-relative reward

    @property
    def n_tokens(self) -> int:
        return int(self.card_idx.shape[0])


class _Loader:
    """Builds ``DraftExample`` objects from a corpus, dropping missing-``.npz`` picks."""

    def __init__(self, locator: ConvertedCardLocator, whitelist: set[str]) -> None:
        self._locator = locator
        self._whitelist = whitelist
        self.embedding_dim: int | None = None
        self.max_packs = 0
        self.max_pack_size = 0
        self._missing: set[str] = set()
        self._missing_total = 0
        # Shared embedding table: each distinct card is loaded once; examples
        # keep only int rows into it (mirrors train_picker's PreparedPool).
        self._table_rows: list[np.ndarray] = []
        self._name_to_idx: dict[str, int | None] = {}

    def _card_index(self, name: str) -> int | None:
        """Row index of ``name`` in the shared table, loading it once; None if no .npz.

        Counts each distinct missing card once (in ``_missing`` / ``_missing_total``)
        on first lookup.
        """
        idx = self._name_to_idx.get(name, -1)
        if idx != -1:
            return idx
        emb = self._locator.load_embedding(name)
        if emb is None:
            self._name_to_idx[name] = None
            if len(self._missing) < _MISSING_WARN_CAP:
                self._missing.add(name)
            self._missing_total += 1
            return None
        if self.embedding_dim is None:
            self.embedding_dim = int(emb.shape[0])
        new_idx = len(self._table_rows)
        self._table_rows.append(np.asarray(emb, dtype=np.float32))
        self._name_to_idx[name] = new_idx
        return new_idx

    def table(self) -> np.ndarray:
        """The shared ``(num_unique_cards, embedding_dim)`` embedding table."""
        if not self._table_rows:
            return np.zeros((0, self.embedding_dim or 1), dtype=np.float32)
        return np.stack(self._table_rows).astype(np.float32)

    def build(self, records: list[DraftRecord]) -> list[DraftExample]:
        examples: list[DraftExample] = []
        total = len(records)
        log_every = max(200, total // 20)  # ~20 progress lines on a big corpus
        start = time.monotonic()
        last_ri, last_t = 0, start  # windowed-rate anchors
        _log(f"Building examples from {total} drafts...")
        for ri, record in enumerate(records, start=1):
            geo = DraftGeometry.from_record(record)
            self.max_packs = max(self.max_packs, geo.packs)
            self.max_pack_size = max(self.max_pack_size, geo.pack_size)
            rewards = leave_one_out_rewards(record)
            for seat_idx, seat in enumerate(record.seats):
                whitelisted = seat.agent in self._whitelist
                critic_active = rewards[seat_idx] is not None
                if not whitelisted and not critic_active:
                    continue  # neither head learns from this seat
                self._emit_seat(
                    record, geo, ri - 1, seat_idx,
                    whitelisted, critic_active, rewards[seat_idx], examples,
                )
            if ri % log_every == 0 or ri == total:
                now = time.monotonic()
                # Windowed (since-last-log) rate: the cold first chunk reads most
                # of the ~28k unique card .npz files off disk, so a lifetime
                # average would understate the (much faster) cached steady state.
                interval = now - last_t
                rate = (ri - last_ri) / interval if interval > 0 else 0.0
                eta = (total - ri) / rate if rate > 0 else 0.0
                last_ri, last_t = ri, now
                _log(
                    f"  loaded {ri}/{total} drafts -> {len(examples)} examples "
                    f"({rate:.0f} drafts/s, ETA {_fmt_dur(eta)})"
                )
        if self._missing_total:
            sample = ", ".join(sorted(self._missing))
            _log(
                f"Dropped {self._missing_total} distinct cards with no .npz "
                f"embedding (e.g. {sample})"
            )
        return examples

    def _emit_seat(
        self,
        record: DraftRecord,
        geo: DraftGeometry,
        draft_index: int,
        seat: int,
        whitelisted: bool,
        critic_active: bool,
        reward: float | None,
        examples: list[DraftExample],
    ) -> None:
        """Emit one ``DraftExample`` per pick for ``seat``.

        Thin adapter over the shared per-pick state walk
        (:func:`draft.application.draft_pick_states.iter_seat_pick_states`),
        which yields the typed-token state at each pick; this wraps each in a
        ``DraftExample`` with the imitation target + critic label. The walk is
        shared with the gen-2 RL loader so the two cannot drift (research
        §Overlapping vocabulary).
        """
        reward_val = float(reward) if reward is not None else 0.0
        for state in iter_seat_pick_states(
            geo, record.boosters, seat, self._card_index,
        ):
            examples.append(DraftExample(
                draft_index=draft_index,
                card_idx=state.card_idx,
                type_idx=state.type_idx,
                packs_ago=state.packs_ago,
                pick_ago=state.pick_ago,
                pack_number=state.pack_number,
                pick_number=state.pick_number,
                imitation_active=whitelisted and state.action_position >= 0,
                target_token=state.action_position,
                critic_active=critic_active,
                critic_target=reward_val,
            ))


# --------------------------------------------------------------------------- #
# Split, standardization, batching (pure helpers)
# --------------------------------------------------------------------------- #

def split_draft_disjoint(
    examples: list[DraftExample], val_fraction: float, seed: int = RANDOM_SEED,
) -> tuple[list[DraftExample], list[DraftExample]]:
    """Partition examples by ``draft_index`` so a draft is entirely train or val.

    Distinct draft indices are shuffled with ``seed`` and the first
    ``val_fraction`` form the held-out set (FR-035).
    """
    ids = sorted({ex.draft_index for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    val_ids = set(ids[:n_val])
    val = [ex for ex in examples if ex.draft_index in val_ids]
    train = [ex for ex in examples if ex.draft_index not in val_ids]
    return train, val


def critic_standardization(train: list[DraftExample]) -> tuple[float, float]:
    """Mean/std of critic targets over the training split (FR-032)."""
    targets = [ex.critic_target for ex in train if ex.critic_active]
    if not targets:
        return 0.0, 1.0
    arr = np.asarray(targets, dtype=np.float64)
    std = float(arr.std())
    return float(arr.mean()), std if std > 1e-8 else 1.0


def imitation_topk_accuracy(
    predictions: list[tuple[np.ndarray, int]], k: int,
) -> float:
    """Top-k accuracy over (pack-action logits, target index) pairs (FR-037)."""
    if not predictions:
        return float("nan")
    hits = 0
    for logits, target in predictions:
        order = np.argsort(-logits)[:k]
        if target in order:
            hits += 1
    return hits / len(predictions)


def per_pack_critic_mse(
    preds: list[float], targets: list[float], pack_numbers: list[int],
) -> dict[int, float]:
    """MSE between standardized critic preds and targets, sliced by pack number."""
    by_pack: dict[int, list[tuple[float, float]]] = {}
    for p, t, pk in zip(preds, targets, pack_numbers):
        by_pack.setdefault(pk, []).append((p, t))
    out: dict[int, float] = {}
    for pk, pairs in by_pack.items():
        diffs = np.asarray([p - t for p, t in pairs], dtype=np.float64)
        out[pk] = float((diffs ** 2).mean())
    return out


@dataclass
class _Batch:
    card_emb: torch.Tensor
    type_idx: torch.Tensor
    packs_ago: torch.Tensor
    pick_ago: torch.Tensor
    card_mask: torch.Tensor
    pack_mask: torch.Tensor
    pack_number: torch.Tensor
    pick_number: torch.Tensor
    target_token: torch.Tensor      # (B,) absolute index, -1 if inactive
    imitation_active: torch.Tensor  # (B,) bool
    critic_active: torch.Tensor     # (B,) bool
    critic_target: torch.Tensor     # (B,) standardized


def _collate(
    batch: list[DraftExample], table: np.ndarray, mean: float, std: float,
    device: torch.device,
) -> _Batch:
    b = len(batch)
    max_n = max(ex.n_tokens for ex in batch)
    dim = table.shape[1]
    card_emb = np.zeros((b, max_n, dim), dtype=np.float32)
    type_idx = np.zeros((b, max_n), dtype=np.int64)
    packs_ago = np.zeros((b, max_n), dtype=np.int64)
    pick_ago = np.zeros((b, max_n), dtype=np.int64)
    card_mask = np.zeros((b, max_n), dtype=bool)
    pack_mask = np.zeros((b, max_n), dtype=bool)
    for i, ex in enumerate(batch):
        n = ex.n_tokens
        card_emb[i, :n] = table[ex.card_idx]  # materialize embeddings per batch
        type_idx[i, :n] = ex.type_idx
        packs_ago[i, :n] = ex.packs_ago
        pick_ago[i, :n] = ex.pick_ago
        card_mask[i, :n] = True
        pack_mask[i, :n] = ex.type_idx == TYPE_PACK
    return _Batch(
        card_emb=torch.from_numpy(card_emb).to(device),
        type_idx=torch.from_numpy(type_idx).to(device),
        packs_ago=torch.from_numpy(packs_ago).to(device),
        pick_ago=torch.from_numpy(pick_ago).to(device),
        card_mask=torch.from_numpy(card_mask).to(device),
        pack_mask=torch.from_numpy(pack_mask).to(device),
        pack_number=torch.tensor([ex.pack_number for ex in batch], device=device),
        pick_number=torch.tensor([ex.pick_number for ex in batch], device=device),
        target_token=torch.tensor([ex.target_token for ex in batch], device=device),
        imitation_active=torch.tensor(
            [ex.imitation_active for ex in batch], device=device, dtype=torch.bool,
        ),
        critic_active=torch.tensor(
            [ex.critic_active for ex in batch], device=device, dtype=torch.bool,
        ),
        critic_target=torch.tensor(
            [(ex.critic_target - mean) / std for ex in batch],
            device=device, dtype=torch.float32,
        ),
    )


# --------------------------------------------------------------------------- #
# Forward / loss
# --------------------------------------------------------------------------- #

@dataclass
class _LossOut:
    total: torch.Tensor
    imitation: torch.Tensor
    critic: torch.Tensor


def _compute_loss(
    model: DraftAgentModel, batch: _Batch, imitation_weight: float, critic_weight: float,
) -> tuple[_LossOut, torch.Tensor, torch.Tensor]:
    """Return (losses, policy_logits, critic_pred)."""
    policy_logits, critic_pred = model(
        batch.card_emb, batch.type_idx, batch.packs_ago, batch.pick_ago,
        batch.card_mask, batch.pack_number, batch.pick_number,
    )
    device = policy_logits.device

    imitation = torch.zeros((), device=device)
    if bool(batch.imitation_active.any()):
        logp = masked_log_softmax(policy_logits, batch.pack_mask)  # (B, N)
        idx = batch.target_token.clamp(min=0).unsqueeze(1)
        nll = -logp.gather(1, idx).squeeze(1)  # (B,)
        active = batch.imitation_active
        imitation = nll[active].mean()

    critic = torch.zeros((), device=device)
    if bool(batch.critic_active.any()):
        active = batch.critic_active
        critic = F.mse_loss(critic_pred[active], batch.critic_target[active])

    total = imitation_weight * imitation + critic_weight * critic
    return _LossOut(total, imitation, critic), policy_logits, critic_pred


# --------------------------------------------------------------------------- #
# Resume / build
# --------------------------------------------------------------------------- #

def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reapply_resume_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Force ``optimizer`` onto ``lr`` after loading a resumed state dict.

    ``load_state_dict`` restores both the saved ``lr`` and the stale
    ``initial_lr`` a previous run's ``LambdaLR`` baked into each param group.
    Setting ``lr`` alone is not enough: a freshly-built ``LambdaLR`` reads its
    ``base_lr`` from ``initial_lr`` (via ``setdefault``, which won't overwrite an
    existing value), then clobbers ``lr`` with ``base_lr × λ`` on the first
    ``step()`` — silently defeating a ``--lr`` override. Popping ``initial_lr``
    lets the new scheduler recapture its base from this ``lr``.
    """
    for group in optimizer.param_groups:
        group["lr"] = lr
        group.pop("initial_lr", None)


class _PlateauLR:
    """ReduceLROnPlateau-style multiplier folded into the warmup scheduler.

    Holds a discrete ``decay_count``; the effective LR is
    ``base_lr * factor ** decay_count`` (times the warmup ramp). The eval loop
    calls :meth:`decay` after ``lr_decay_patience`` mini-epochs without a new
    best val_loss, bounded below by ``min_lr``. ``decay_count`` is persisted in
    the checkpoint so a resumed run continues annealing where it left off.
    """

    def __init__(
        self, base_lr: float, factor: float, min_lr: float, decay_count: int = 0,
    ) -> None:
        self.base_lr = base_lr
        self.factor = factor
        self.min_lr = min_lr
        self.decay_count = decay_count

    def multiplier(self) -> float:
        return self.factor ** self.decay_count

    def current_lr(self) -> float:
        return self.base_lr * self.multiplier()

    def can_decay(self) -> bool:
        """True if one more decay stays at/above ``min_lr``."""
        return self.base_lr * self.factor ** (self.decay_count + 1) >= self.min_lr

    def decay(self) -> float:
        self.decay_count += 1
        return self.current_lr()


def _maybe_decay(
    evals_since_best: int, patience: int | None, controller: _PlateauLR,
) -> float | None:
    """Decay the LR if stuck for ``patience`` evals and the floor allows it.

    Returns the new LR on decay, else ``None``. ``patience is None`` (the
    feature switch) disables annealing entirely.
    """
    if patience is None:
        return None
    if evals_since_best >= patience and controller.can_decay():
        return controller.decay()
    return None


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


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_frac: float,
    controller: _PlateauLR | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup over ``warmup_frac`` of total steps, then constant.

    When a plateau ``controller`` is given, the per-step multiplier also folds
    in its (live) decay factor, so a decay recorded mid-run takes effect on the
    next batch.
    """
    warmup_steps = max(1, int(math.ceil(total_steps * warmup_frac)))

    def schedule(step: int) -> float:
        warm = float(step) / float(warmup_steps) if step < warmup_steps else 1.0
        return warm * (controller.multiplier() if controller is not None else 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


@dataclass
class _Resume:
    model: DraftAgentModel
    start_epoch: int
    best_val_loss: float
    optimizer_state: dict[str, Any] | None
    lr_decay_count: int = 0


def _resume_or_build(
    config: TrainDraftAgentConfig,
    store: DraftAgentStore,
    embedding_dim: int,
    packs: int,
    pack_size: int,
) -> tuple[_Resume, float, float]:
    """Return (resume bundle, critic_mean, critic_std) — stats from checkpoint when loading."""
    if config.resume is not None:
        ckpt = store.load_checkpoint(config.resume)
        _check_dims(ckpt.config, embedding_dim, config.cards_path)
        model = DraftAgentModel(ckpt.config)
        model.load_state_dict(ckpt.model_state_dict)
        # Inherit the annealing position unless --lr was explicitly overridden
        # (a different base LR restarts annealing from decay_count 0). --lr is a
        # resumable flag, so an *absent* --lr resolves equal to the stored lr.
        resumed_lr = (ckpt.train_config or {}).get("lr")
        decay_count = ckpt.lr_decay_count
        if resumed_lr is not None and resumed_lr != config.lr:
            decay_count = 0
        return (
            _Resume(model, ckpt.epoch + 1, ckpt.best_val_loss,
                    ckpt.optimizer_state_dict or None, decay_count),
            ckpt.critic_mean, ckpt.critic_std,
        )
    if config.checkpoint is not None:
        ckpt = store.load_checkpoint(config.checkpoint)
        _check_dims(ckpt.config, embedding_dim, config.cards_path)
        model = DraftAgentModel(ckpt.config)
        model.load_state_dict(ckpt.model_state_dict)
        return (
            _Resume(model, 0, float("inf"), None),
            ckpt.critic_mean, ckpt.critic_std,
        )
    agent_config = DraftAgentConfig(
        embedding_dim=embedding_dim,
        packs=packs,
        P=pack_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
    )
    return _Resume(DraftAgentModel(agent_config), 0, float("inf"), None), 0.0, 1.0


def _check_dims(
    ckpt_config: DraftAgentConfig, embedding_dim: int, cards_path: Path,
) -> None:
    if ckpt_config.embedding_dim != embedding_dim:
        raise ValueError(
            f"Checkpoint expects {ckpt_config.embedding_dim}-wide card "
            f"embeddings, but the .npz cache under {cards_path} is "
            f"{embedding_dim}-wide. Re-run `sealed encode-cards` with the "
            "encoder this agent was trained on."
        )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

@dataclass
class _ValReport:
    loss: float
    imitation: float
    critic: float
    top1: float
    top3: float
    per_pack_mse: dict[int, float]


def _validate(
    model: DraftAgentModel,
    val: list[DraftExample],
    table: np.ndarray,
    mean: float,
    std: float,
    config: TrainDraftAgentConfig,
    device: torch.device,
) -> _ValReport:
    model.eval()
    if not val:
        return _ValReport(0.0, 0.0, 0.0, float("nan"), float("nan"), {})
    imitation_preds: list[tuple[np.ndarray, int]] = []
    critic_preds: list[float] = []
    critic_targets: list[float] = []
    critic_packs: list[int] = []
    tot_imit = 0.0
    tot_crit = 0.0
    n_imit = 0
    n_crit = 0
    with torch.no_grad():
        for start in range(0, len(val), config.batch_size):
            chunk = val[start:start + config.batch_size]
            batch = _collate(chunk, table, mean, std, device)
            losses, logits, critic_pred = _compute_loss(
                model, batch, config.imitation_weight, config.critic_weight,
            )
            for i, ex in enumerate(chunk):
                if ex.imitation_active:
                    pack_idx = np.flatnonzero(ex.type_idx == TYPE_PACK)
                    row = logits[i, :ex.n_tokens].cpu().numpy()
                    pack_logits = row[pack_idx]
                    target_in_pack = int(np.flatnonzero(pack_idx == ex.target_token)[0])
                    imitation_preds.append((pack_logits, target_in_pack))
                if ex.critic_active:
                    critic_preds.append(float(critic_pred[i].cpu()))
                    critic_targets.append((ex.critic_target - mean) / std)
                    critic_packs.append(ex.pack_number)
            if bool(batch.imitation_active.any()):
                tot_imit += float(losses.imitation) * int(batch.imitation_active.sum())
                n_imit += int(batch.imitation_active.sum())
            if bool(batch.critic_active.any()):
                tot_crit += float(losses.critic) * int(batch.critic_active.sum())
                n_crit += int(batch.critic_active.sum())
    imit = tot_imit / n_imit if n_imit else 0.0
    crit = tot_crit / n_crit if n_crit else 0.0
    return _ValReport(
        loss=config.imitation_weight * imit + config.critic_weight * crit,
        imitation=imit,
        critic=crit,
        top1=imitation_topk_accuracy(imitation_preds, 1),
        top3=imitation_topk_accuracy(imitation_preds, 3),
        per_pack_mse=per_pack_critic_mse(critic_preds, critic_targets, critic_packs),
    )


# --------------------------------------------------------------------------- #
# Use case
# --------------------------------------------------------------------------- #

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

        locator = ConvertedCardLocator(config.cards_path)
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

        out_dir = Path("models/draft/agent")
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
