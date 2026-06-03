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

from draft.domain.draft_agent_model import (
    DraftAgentConfig,
    DraftAgentModel,
)
from draft.domain.draft_geometry import DraftGeometry, DraftRecord
from draft.domain.draft_state import (
    TYPE_PACK,
    TYPE_PASSED,
    TYPE_POOL,
    TYPE_TAKEN,
)
from draft.infrastructure.draft_agent_store import DraftAgentStore
from draft.infrastructure.draft_record_io import read_records
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
        _log(f"Building examples from {total} drafts...")
        for ri, record in enumerate(records, start=1):
            geo = DraftGeometry.from_record(record)
            self.max_packs = max(self.max_packs, geo.packs)
            self.max_pack_size = max(self.max_pack_size, geo.pack_size)
            rewards = _leave_one_out_rewards(record)
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
                elapsed = time.monotonic() - start
                rate = ri / elapsed if elapsed > 0 else 0.0
                eta = (total - ri) / rate if rate > 0 else 0.0
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
        """Emit one example per pick for ``seat`` in a single incremental pass.

        Walks the seat's boosters once. The dominant POOL/TAKEN blocks are kept
        as parallel ``(idx, last-seen-pack, last-seen-pick)`` lists — each card
        resolved to an embedding row and last-seen exactly once, when it enters
        the block — so at each pick the example is assembled by list-concat +
        **vectorized** recency in numpy, with a small Python loop only over the
        PACK/PASSED blocks. Semantics are identical to building each pick via
        :func:`draft.domain.draft_state.build_state` (locked by an equivalence
        test); recency follows FR-019/FR-021.

        ``passed`` and ``last_seen`` track *all* cards (the wheel/flush logic
        needs them); POOL/TAKEN/PACK/PASSED *emission* drops cards with no
        ``.npz`` (consistent with the missing-embedding policy).
        """
        pod, P, packs = geo.pod_size, geo.pack_size, geo.packs
        boosters = record.boosters
        reward_val = float(reward) if reward is not None else 0.0
        card_index = self._card_index

        # POOL / TAKEN as embeddable (idx, lp, li); built once on entry.
        pool_idx: list[int] = []
        pool_lp: list[int] = []
        pool_li: list[int] = []
        taken_idx: list[int] = []
        taken_lp: list[int] = []
        taken_li: list[int] = []
        passed: dict[str, tuple[int, int]] = {}      # all cards (state)
        last_seen: dict[str, tuple[int, int]] = {}   # all cards (state)

        for p in range(1, packs + 1):
            for i in range(1, P + 1):
                k, off = geo.booster_for_pick(seat, p, i)
                picks = boosters[k].picks
                vis = picks[off:]
                # Wheel diff: cards passed from this booster last time, gone now,
                # were taken by others (persistent; serves emit + advance).
                if off >= pod:
                    for c in picks[off - pod + 1: off]:
                        passed.pop(c, None)
                        idx = card_index(c)
                        if idx is not None:
                            lp, li = last_seen[c]
                            taken_idx.append(idx)
                            taken_lp.append(lp)
                            taken_li.append(li)
                pack_set: set[str] = set()
                pack_actions: list[str] = []
                for c in vis:
                    if c not in pack_set:
                        pack_set.add(c)
                        pack_actions.append(c)

                # --- emit the example for target (p, i) ---
                self._emit_example(
                    draft_index, p, i, P, pack_actions, pack_set,
                    pool_idx, pool_lp, pool_li, taken_idx, taken_lp, taken_li,
                    passed, last_seen, whitelisted, critic_active, reward_val,
                    examples,
                )

                # --- advance past pick (p, i) ---
                taken_card = vis[0]
                t_idx = card_index(taken_card)
                if t_idx is not None:
                    pool_idx.append(t_idx)
                    pool_lp.append(p)
                    pool_li.append(i)
                passed.pop(taken_card, None)
                for c in vis:
                    last_seen[c] = (p, i)
                for c in vis[1:]:
                    passed[c] = (p, i)
            # Pack boundary: remaining PASSED flush to TAKEN (FR-019b).
            for c, (lp, li) in passed.items():
                idx = card_index(c)
                if idx is not None:
                    taken_idx.append(idx)
                    taken_lp.append(lp)
                    taken_li.append(li)
            passed.clear()

    def _emit_example(
        self,
        draft_index: int,
        p: int,
        i: int,
        P: int,
        pack_actions: list[str],
        pack_set: set[str],
        pool_idx: list[int],
        pool_lp: list[int],
        pool_li: list[int],
        taken_idx: list[int],
        taken_lp: list[int],
        taken_li: list[int],
        passed: dict[str, tuple[int, int]],
        last_seen: dict[str, tuple[int, int]],
        whitelisted: bool,
        critic_active: bool,
        reward_val: float,
        examples: list[DraftExample],
    ) -> None:
        """Assemble one example: concat the 4 typed blocks, vectorize recency."""
        card_index = self._card_index
        n_pool = len(pool_idx)

        # PACK block (small Python loop). The taken card is pack_actions[0];
        # its position fixes the imitation target (or -1 if it was dropped).
        pack_i: list[int] = []
        pack_lp: list[int] = []
        pack_li: list[int] = []
        target_token = -1
        for j, name in enumerate(pack_actions):
            idx = card_index(name)
            if idx is None:
                continue
            ls = last_seen.get(name)
            lp, li = ls if ls is not None else (p, i)  # never-seen -> recency 0,0
            if j == 0:
                target_token = n_pool + len(pack_i)
            pack_i.append(idx)
            pack_lp.append(lp)
            pack_li.append(li)
        if not pack_i:
            return  # no usable PACK tokens — nothing to learn here

        # PASSED block (small Python loop); wheel survivors are now in PACK.
        pass_i: list[int] = []
        pass_lp: list[int] = []
        pass_li: list[int] = []
        for name, (lp, li) in passed.items():
            if name in pack_set:
                continue
            idx = card_index(name)
            if idx is None:
                continue
            pass_i.append(idx)
            pass_lp.append(lp)
            pass_li.append(li)

        idx_all = pool_idx + pack_i + pass_i + taken_idx
        lp_all = pool_lp + pack_lp + pass_lp + taken_lp
        li_all = pool_li + pack_li + pass_li + taken_li
        n = len(idx_all)

        # Vectorized recency (FR-021), entirely in int8: packs_ago = min(2,
        # p - lp); pick_ago is (i - li) within the current pack else the frozen
        # end-of-pack value (P - li). Both are provably already in range
        # (lp ≤ p, 1 ≤ li ≤ P), so no clip/astype is needed.
        lpli = np.asarray((lp_all, li_all), dtype=np.int8)  # (2, n)
        lp_arr, li_arr = lpli[0], lpli[1]
        packs_ago = np.minimum(p - lp_arr, 2)
        pick_ago = np.where(packs_ago == 0, i - li_arr, P - li_arr)

        type_idx = np.empty(n, dtype=np.int8)
        a = n_pool
        b = a + len(pack_i)
        c = b + len(pass_i)
        type_idx[:a] = TYPE_POOL
        type_idx[a:b] = TYPE_PACK
        type_idx[b:c] = TYPE_PASSED
        type_idx[c:] = TYPE_TAKEN

        examples.append(DraftExample(
            draft_index=draft_index,
            card_idx=np.asarray(idx_all, dtype=np.int32),
            type_idx=type_idx,
            packs_ago=packs_ago,
            pick_ago=pick_ago,
            pack_number=p,
            pick_number=i,
            imitation_active=whitelisted and target_token >= 0,
            target_token=target_token,
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


_BUCKET_MULTIPLIER = 50  # megabatch = bucket_multiplier * batch_size examples


def length_bucketed_batches(
    examples: list[DraftExample], batch_size: int, rng: random.Random,
) -> list[list[DraftExample]]:
    """Group similar-length examples into batches; reshuffle batch order per call.

    Examples are shuffled, partitioned into megabatches, sorted by token count
    *within* each megabatch, cut into batches, and the batch order is shuffled.
    So each batch holds near-equal-length examples (little padding wasted on
    masks) while still varying composition and order across epochs.
    """
    order = list(examples)
    rng.shuffle(order)
    mega = batch_size * _BUCKET_MULTIPLIER
    batches: list[list[DraftExample]] = []
    for start in range(0, len(order), mega):
        chunk = order[start:start + mega]
        chunk.sort(key=lambda ex: ex.n_tokens)
        for b in range(0, len(chunk), batch_size):
            batches.append(chunk[b:b + batch_size])
    rng.shuffle(batches)
    return batches


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
        # Warmup over a fraction of ONE epoch (not epochs*n_steps): with early
        # stopping a run rarely passes epoch ~2-3, so a warmup sized to the max
        # epoch count would never reach full LR.
        scheduler = _make_scheduler(optimizer, n_steps, config.warmup_frac)

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

        def run_eval(epoch: int, step: int) -> bool:
            """Validate + save latest/best; return True if early-stop is reached."""
            nonlocal best_val_loss, best_at, evals_since_best
            report = _validate(model, val, table, mean, std, config, device)
            mse_str = ", ".join(
                f"p{pk}={v:.4f}" for pk, v in sorted(report.per_pack_mse.items())
            )
            _log(
                f"  eval epoch {epoch} step {step}/{n_steps} | "
                f"val_loss={report.loss:.4f} val_imit={report.imitation:.4f} "
                f"val_crit={report.critic:.4f} top1={report.top1:.3f} "
                f"top3={report.top3:.3f} per_pack_mse[{mse_str}]"
            )
            new_best = report.loss < best_val_loss
            if new_best:
                best_val_loss = report.loss
                best_at = (epoch, step)
                evals_since_best = 0
            else:
                evals_since_best += 1
            store.save_checkpoint(
                model, optimizer, epoch, best_val_loss, model.config, latest_path,
                critic_mean=mean, critic_std=std, train_config=train_config,
            )
            if new_best:
                store.save_checkpoint(
                    model, optimizer, epoch, best_val_loss, model.config, best_path,
                    critic_mean=mean, critic_std=std, train_config=train_config,
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
            win_imit = win_crit = 0.0
            win_n = 0
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
                win_imit += float(losses.imitation.detach())
                win_crit += float(losses.critic.detach())
                win_n += 1
                now = time.monotonic()
                if now - last_log >= _STEP_LOG_INTERVAL and step < nb:
                    rate = step / (now - epoch_start)
                    eta = (nb - step) / rate if rate > 0 else 0.0
                    _log(
                        f"  epoch {epoch} step {step}/{nb} ({step / nb * 100:.0f}%) "
                        f"imit={win_imit / win_n:.4f} crit={win_crit / win_n:.4f} "
                        f"{rate:.1f} steps/s ETA {_fmt_dur(eta)}"
                    )
                    last_log = now
                    win_imit = win_crit = 0.0
                    win_n = 0
                if step % eval_interval == 0 or step == nb:
                    if run_eval(epoch, step):
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
