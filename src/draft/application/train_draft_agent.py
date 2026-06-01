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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from draft.domain.draft_agent_model import (
    DraftAgentConfig,
    DraftAgentModel,
)
from draft.domain.draft_geometry import DraftGeometry, DraftRecord
from draft.domain.draft_state import TYPE_PACK, build_state
from draft.infrastructure.draft_agent_store import DraftAgentStore
from draft.infrastructure.draft_record_io import read_records
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

RANDOM_SEED = 42  # hardcoded (FR-035): governs init, the draft-disjoint split.
_MISSING_WARN_CAP = 20


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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
    val_fraction: float = 0.2
    patience: int = 10
    resume: Path | None = None
    checkpoint: Path | None = None


# --------------------------------------------------------------------------- #
# Training examples
# --------------------------------------------------------------------------- #

@dataclass
class DraftExample:
    """One ``(draft, seat, pack, pick)`` training example (data-model §2)."""

    draft_id: str
    card_emb: np.ndarray         # (N, embedding_dim) float32
    type_idx: np.ndarray         # (N,) int64
    packs_ago: np.ndarray        # (N,) int64
    pick_ago: np.ndarray         # (N,) int64
    pack_mask: np.ndarray        # (N,) bool, True = PACK token
    pack_number: int
    pick_number: int
    imitation_active: bool
    target_token: int            # absolute index of the taken PACK token, or -1
    critic_active: bool
    critic_target: float         # raw (pre-standardization) pod-relative reward

    @property
    def n_tokens(self) -> int:
        return int(self.type_idx.shape[0])


def _leave_one_out_rewards(record: DraftRecord) -> list[float | None]:
    """Per-seat ``deck_score − mean(other non-failed deck_scores)`` (FR-032)."""
    scores = [s.deck_score for s in record.seats]
    rewards: list[float | None] = []
    for i, score in enumerate(scores):
        if score is None:
            rewards.append(None)
            continue
        others = [s for j, s in enumerate(scores) if j != i and s is not None]
        mean_others = sum(others) / len(others) if others else 0.0
        rewards.append(score - mean_others)
    return rewards


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

    def _embedding(self, name: str) -> np.ndarray | None:
        emb = self._locator.load_embedding(name)
        if emb is None:
            if len(self._missing) < _MISSING_WARN_CAP:
                self._missing.add(name)
            self._missing_total += 1
            return None
        if self.embedding_dim is None:
            self.embedding_dim = int(emb.shape[0])
        return emb.astype(np.float32)

    def build(self, records: list[DraftRecord]) -> list[DraftExample]:
        examples: list[DraftExample] = []
        for record in records:
            geo = DraftGeometry.from_record(record)
            self.max_packs = max(self.max_packs, geo.packs)
            self.max_pack_size = max(self.max_pack_size, geo.pack_size)
            rewards = _leave_one_out_rewards(record)
            for seat_idx, seat in enumerate(record.seats):
                whitelisted = seat.agent in self._whitelist
                critic_active = rewards[seat_idx] is not None
                if not whitelisted and not critic_active:
                    continue  # neither head learns from this seat
                for pack in range(1, geo.packs + 1):
                    for pick in range(1, geo.pack_size + 1):
                        ex = self._example(
                            record, geo, seat_idx, pack, pick,
                            whitelisted, critic_active, rewards[seat_idx],
                        )
                        if ex is not None:
                            examples.append(ex)
        if self._missing_total:
            sample = ", ".join(sorted(self._missing))
            _log(
                f"Dropped {self._missing_total} picks with no .npz embedding "
                f"(e.g. {sample})"
            )
        return examples

    def _example(
        self,
        record: DraftRecord,
        geo: DraftGeometry,
        seat_idx: int,
        pack: int,
        pick: int,
        whitelisted: bool,
        critic_active: bool,
        reward: float | None,
    ) -> DraftExample | None:
        state = build_state(record, geo, seat_idx, pack, pick)
        embs: list[np.ndarray] = []
        types: list[int] = []
        packs_ago: list[int] = []
        pick_ago: list[int] = []
        pack_mask: list[bool] = []
        target_token = -1
        # The taken PACK card is the instance at (pool_count + target_index);
        # track the original index so we can map it through the missing-drop.
        pack_positions = [i for i, c in enumerate(state.cards) if c.token_type == TYPE_PACK]
        target_orig = (
            pack_positions[state.target_index] if pack_positions else -1
        )
        for orig_idx, card in enumerate(state.cards):
            emb = self._embedding(card.name)
            if emb is None:
                continue
            if orig_idx == target_orig:
                target_token = len(embs)
            embs.append(emb)
            types.append(card.token_type)
            packs_ago.append(card.packs_ago)
            pick_ago.append(card.pick_ago)
            pack_mask.append(card.token_type == TYPE_PACK)
        if not embs or not any(pack_mask):
            return None  # no usable PACK tokens — nothing to learn here
        imitation_active = whitelisted and target_token >= 0
        return DraftExample(
            draft_id=record.draft_id,
            card_emb=np.stack(embs).astype(np.float32),
            type_idx=np.asarray(types, dtype=np.int64),
            packs_ago=np.asarray(packs_ago, dtype=np.int64),
            pick_ago=np.asarray(pick_ago, dtype=np.int64),
            pack_mask=np.asarray(pack_mask, dtype=bool),
            pack_number=pack,
            pick_number=pick,
            imitation_active=imitation_active,
            target_token=target_token,
            critic_active=critic_active,
            critic_target=float(reward) if reward is not None else 0.0,
        )


# --------------------------------------------------------------------------- #
# Split, standardization, batching (pure helpers)
# --------------------------------------------------------------------------- #

def split_draft_disjoint(
    examples: list[DraftExample], val_fraction: float, seed: int = RANDOM_SEED,
) -> tuple[list[DraftExample], list[DraftExample]]:
    """Partition examples by ``draft_id`` so a draft is entirely train or val.

    Distinct ids are shuffled with ``seed`` and the first ``val_fraction`` form
    the held-out set (FR-035).
    """
    ids = sorted({ex.draft_id for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    val_ids = set(ids[:n_val])
    val = [ex for ex in examples if ex.draft_id in val_ids]
    train = [ex for ex in examples if ex.draft_id not in val_ids]
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
    batch: list[DraftExample], mean: float, std: float, device: torch.device,
) -> _Batch:
    b = len(batch)
    max_n = max(ex.n_tokens for ex in batch)
    dim = batch[0].card_emb.shape[1]
    card_emb = np.zeros((b, max_n, dim), dtype=np.float32)
    type_idx = np.zeros((b, max_n), dtype=np.int64)
    packs_ago = np.zeros((b, max_n), dtype=np.int64)
    pick_ago = np.zeros((b, max_n), dtype=np.int64)
    card_mask = np.zeros((b, max_n), dtype=bool)
    pack_mask = np.zeros((b, max_n), dtype=bool)
    for i, ex in enumerate(batch):
        n = ex.n_tokens
        card_emb[i, :n] = ex.card_emb
        type_idx[i, :n] = ex.type_idx
        packs_ago[i, :n] = ex.packs_ago
        pick_ago[i, :n] = ex.pick_ago
        card_mask[i, :n] = True
        pack_mask[i, :n] = ex.pack_mask
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

def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


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
        logp = _masked_log_softmax(policy_logits, batch.pack_mask)  # (B, N)
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


def _make_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_frac: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup over ``warmup_frac`` of total steps, then constant."""
    warmup_steps = max(1, int(math.ceil(total_steps * warmup_frac)))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


@dataclass
class _Resume:
    model: DraftAgentModel
    start_epoch: int
    best_val_loss: float
    optimizer_state: dict[str, Any] | None


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
        return (
            _Resume(model, ckpt.epoch + 1, ckpt.best_val_loss,
                    ckpt.optimizer_state_dict or None),
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
            batch = _collate(chunk, mean, std, device)
            losses, logits, critic_pred = _compute_loss(
                model, batch, config.imitation_weight, config.critic_weight,
            )
            for i, ex in enumerate(chunk):
                if ex.imitation_active:
                    pack_idx = np.flatnonzero(ex.pack_mask)
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
        n_steps = max(1, math.ceil(len(train) / config.batch_size))
        scheduler = _make_scheduler(
            optimizer, n_steps * config.epochs, config.warmup_frac,
        )

        out_dir = Path("models/draft/agent")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_path = out_dir / "latest.pt"
        best_path = out_dir / f"{timestamp}.pt"
        train_config = {
            k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()
        }

        rng = random.Random(RANDOM_SEED)
        best_val_loss = resume.best_val_loss
        best_epoch = resume.start_epoch
        epochs_since_best = 0

        _log(f"Training: {config.epochs} epochs, {n_steps} steps/epoch on {device}")
        for epoch in range(resume.start_epoch, resume.start_epoch + config.epochs):
            model.train()
            shuffled = list(train)
            rng.shuffle(shuffled)
            ep_imit = ep_crit = 0.0
            for start in range(0, len(shuffled), config.batch_size):
                chunk = shuffled[start:start + config.batch_size]
                batch = _collate(chunk, mean, std, device)
                losses, _, _ = _compute_loss(
                    model, batch, config.imitation_weight, config.critic_weight,
                )
                optimizer.zero_grad()
                losses.total.backward()
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group["params"], config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                ep_imit += float(losses.imitation.detach())
                ep_crit += float(losses.critic.detach())

            report = _validate(model, val, mean, std, config, device)
            mse_str = ", ".join(
                f"p{pk}={v:.4f}" for pk, v in sorted(report.per_pack_mse.items())
            )
            _log(
                f"epoch={epoch} train_imit={ep_imit / n_steps:.4f} "
                f"train_crit={ep_crit / n_steps:.4f} | val_loss={report.loss:.4f} "
                f"val_imit={report.imitation:.4f} val_crit={report.critic:.4f} "
                f"top1={report.top1:.3f} top3={report.top3:.3f} "
                f"per_pack_mse[{mse_str}]"
            )

            new_best = report.loss < best_val_loss
            if new_best:
                best_val_loss = report.loss
                best_epoch = epoch
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            store.save_checkpoint(
                model, optimizer, epoch, best_val_loss, model.config, latest_path,
                critic_mean=mean, critic_std=std, train_config=train_config,
            )
            if new_best:
                store.save_checkpoint(
                    model, optimizer, epoch, best_val_loss, model.config, best_path,
                    critic_mean=mean, critic_std=std, train_config=train_config,
                )
            if epochs_since_best >= config.patience:
                _log(f"Early stop: {epochs_since_best} epochs without val improvement")
                break

        _log(
            f"Done. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}; "
            f"best checkpoint: {best_path}"
        )
        return TrainDraftAgentResult(best_val_loss, best_epoch, best_path)
