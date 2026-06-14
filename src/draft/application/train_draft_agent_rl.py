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
REINFORCE helpers (entropy/KL) follow ``sealed.application.train_picker``. On-
policy pairing of corpus↔checkpoint is the operator's responsibility — the only
safeguard is the behaviour-anomaly warning (research D6); the corpus is unchanged
(no provenance field, FR-021).
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
from draft.application.rl_advantage import gae_advantages
from draft.application.train_draft_agent import (
    _make_scheduler,
    _maybe_decay,
    _PlateauLR,
    _reapply_resume_lr,
    _select_device,
)
from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.domain.draft_geometry import DraftGeometry, DraftRecord
from draft.domain.draft_state import TYPE_PACK
from draft.infrastructure.draft_agent_store import DraftAgentStore
from draft.infrastructure.draft_record_io import read_records
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

RANDOM_SEED = 42  # hardcoded; governs init and the draft-disjoint split.
_GAMMA = 1.0  # draft reward is terminal, so the return is R (research D2).
_PROB_FLOOR = 1e-4  # behaviour prob below this counts as an anomaly (D6).
_ANOMALY_FRACTION = 0.05  # warn if more than this fraction of learner picks anomalous.
_BUCKET_MULTIPLIER = 50  # megabatch = bucket_multiplier * batch_size examples.
_MISSING_WARN_CAP = 20
_STEP_LOG_INTERVAL = 15.0


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


@dataclass
class TrainDraftAgentRLConfig:
    """Inputs to the RL trainer (data-model §1).

    ``checkpoint``/``learner_agents``/``rollout_temperature`` are required on a
    fresh run (enforced by the CLI / startup validation); they are resumable so
    ``--resume`` inherits them. Architecture is inherited from the loaded
    checkpoint — there are no architecture flags.
    """

    checkpoint: Path | None = None
    resume: Path | None = None
    drafts_path: Path = field(default_factory=lambda: Path("output/draft/drafts.jsonl"))
    critic_corpus: tuple[Path, ...] = ()
    cards_path: Path = field(default_factory=lambda: Path("output/cardsfolder/"))
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


# --------------------------------------------------------------------------- #
# Training examples & trajectories
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class RLExample:
    """One ``(draft, seat, pack, pick)`` RL training state (data-model §3)."""

    draft_index: int
    card_idx: np.ndarray
    type_idx: np.ndarray
    packs_ago: np.ndarray
    pick_ago: np.ndarray
    pack_number: int
    pick_number: int
    action_token: int        # absolute index of the taken PACK token, or -1
    learner_active: bool      # feeds policy-grad + entropy + KL
    critic_active: bool       # feeds critic MSE
    reward: float             # raw (pre-standardization) terminal pod-relative R
    advantage: float = 0.0    # cached detached GAE A_t (refreshed per epoch)
    value: float = 0.0        # cached critic V(s_t) (refreshed per epoch)

    @property
    def n_tokens(self) -> int:
        return int(self.card_idx.shape[0])


def _leave_one_out_rewards(record: DraftRecord) -> list[float | None]:
    """Per-seat ``deck_score − mean(other non-failed deck_scores)`` (FR-010)."""
    scores = [s.deck_score for s in record.seats]
    rewards: list[float | None] = []
    for i, score in enumerate(scores):
        if score is None:
            rewards.append(None)
            continue
        others = [s for j, s in enumerate(scores) if j != i and s is not None]
        rewards.append(score - (sum(others) / len(others) if others else 0.0))
    return rewards


class _Loader:
    """Builds ``RLExample``s + per-learner-seat trajectories from corpora.

    Mirrors ``train_draft_agent._Loader``'s shared-table memoization (each
    distinct card's ``.npz`` loaded once; examples keep int rows). The on-policy
    corpus contributes both learner-seat trajectories (policy gradient + critic)
    and non-learner critic coverage; ``critic_corpus`` corpora contribute critic
    coverage only — never learner trajectories (FR-013).
    """

    def __init__(self, locator: ConvertedCardLocator, learner_agents: set[str]) -> None:
        self._locator = locator
        self._learner = learner_agents
        self.embedding_dim: int | None = None
        self.max_packs = 0
        self.max_pack_size = 0
        self._missing: set[str] = set()
        self._missing_total = 0
        self._table_rows: list[np.ndarray] = []
        self._name_to_idx: dict[str, int | None] = {}

    def _card_index(self, name: str) -> int | None:
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
        if not self._table_rows:
            return np.zeros((0, self.embedding_dim or 1), dtype=np.float32)
        return np.stack(self._table_rows).astype(np.float32)

    def build(
        self, on_policy: list[DraftRecord], critic_only: list[DraftRecord],
    ) -> tuple[list[RLExample], list[list[RLExample]]]:
        examples: list[RLExample] = []
        trajectories: list[list[RLExample]] = []
        self._ingest(on_policy, examples, trajectories, allow_learner=True)
        self._ingest(critic_only, examples, trajectories, allow_learner=False)
        if self._missing_total:
            sample = ", ".join(sorted(self._missing))
            _log(
                f"Dropped {self._missing_total} distinct cards with no .npz "
                f"embedding (e.g. {sample})"
            )
        return examples, trajectories

    def _ingest(
        self,
        records: list[DraftRecord],
        examples: list[RLExample],
        trajectories: list[list[RLExample]],
        *,
        allow_learner: bool,
    ) -> None:
        for ri, record in enumerate(records):
            geo = DraftGeometry.from_record(record)
            self.max_packs = max(self.max_packs, geo.packs)
            self.max_pack_size = max(self.max_pack_size, geo.pack_size)
            rewards = _leave_one_out_rewards(record)
            for seat_idx, seat in enumerate(record.seats):
                reward = rewards[seat_idx]
                if reward is None:
                    continue  # failed build → excluded entirely (FR-015)
                is_learner = allow_learner and seat.agent in self._learner
                traj: list[RLExample] = []
                for state in iter_seat_pick_states(
                    geo, record.boosters, seat_idx, self._card_index,
                ):
                    ex = RLExample(
                        draft_index=ri,
                        card_idx=state.card_idx,
                        type_idx=state.type_idx,
                        packs_ago=state.packs_ago,
                        pick_ago=state.pick_ago,
                        pack_number=state.pack_number,
                        pick_number=state.pick_number,
                        action_token=state.action_position,
                        learner_active=is_learner and state.action_position >= 0,
                        critic_active=True,
                        reward=float(reward),
                    )
                    examples.append(ex)
                    if is_learner:
                        traj.append(ex)
                if traj:
                    trajectories.append(traj)


# --------------------------------------------------------------------------- #
# Split, batching, helpers (copied small helpers — see research §Third-instance)
# --------------------------------------------------------------------------- #

def split_draft_disjoint(
    examples: list[RLExample], val_fraction: float, seed: int = RANDOM_SEED,
) -> tuple[set[int], set[int]]:
    """Partition draft indices so a draft is entirely train or val (FR-035)."""
    ids = sorted({ex.draft_index for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    val_ids = set(ids[:n_val])
    train_ids = {i for i in ids if i not in val_ids}
    return train_ids, val_ids


def length_bucketed_batches(
    examples: list[RLExample], batch_size: int, rng: random.Random,
) -> list[list[RLExample]]:
    """Group similar-length examples into batches; reshuffle order per call."""
    order = list(examples)
    rng.shuffle(order)
    mega = batch_size * _BUCKET_MULTIPLIER
    batches: list[list[RLExample]] = []
    for start in range(0, len(order), mega):
        chunk = order[start:start + mega]
        chunk.sort(key=lambda ex: ex.n_tokens)
        for b in range(0, len(chunk), batch_size):
            batches.append(chunk[b:b + batch_size])
    rng.shuffle(batches)
    return batches


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def _policy_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-row entropy over PACK positions (B,). NaN-gradient-guarded.

    Copied from ``train_picker._policy_entropy``: zero the ``-inf`` log-probs
    before the ``p·logp`` product so masked positions contribute 0 to both the
    value and the gradient (``0·-inf`` would otherwise be NaN).
    """
    logp = _masked_log_softmax(logits, mask)
    p = logp.exp()
    safe_logp = logp.masked_fill(~mask, 0.0)
    return -(p * safe_logp).sum(dim=-1)


def _kl_divergence(
    logits: torch.Tensor, ref_logits: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Per-row KL(π ‖ π_ref) over PACK positions (B,). NaN-gradient-guarded."""
    logp = _masked_log_softmax(logits, mask)
    logq = _masked_log_softmax(ref_logits, mask)
    p = logp.exp()
    log_ratio = (logp - logq).masked_fill(~mask, 0.0)
    return (p * log_ratio).sum(dim=-1)


class _CoefSchedule:
    """Val-driven coefficient decay (entropy / KL), copied from train_picker.

    Held at ``initial`` until the val metric improves for ``decay_after``
    consecutive evals, then multiplied by ``factor`` on each eval that fails to
    beat the running best. The "metric" is higher-is-better; the trainer passes
    ``-val_loss`` so a falling loss counts as improvement.
    """

    def __init__(self, initial: float, decay_after: int, factor: float = 0.9) -> None:
        self.coef = initial
        self.decay_after = decay_after
        self.factor = factor
        self.armed = False
        self.streak = 0
        self.prev: float | None = None
        self.best = float("-inf")

    def update(self, metric: float) -> float:
        if self.prev is not None and metric > self.prev:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.decay_after:
            self.armed = True
        improved_best = metric > self.best
        if self.armed and not improved_best:
            self.coef *= self.factor
        self.prev = metric
        self.best = max(self.best, metric)
        return self.coef


# --------------------------------------------------------------------------- #
# Batching / forward / loss
# --------------------------------------------------------------------------- #

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
    action_token: torch.Tensor
    learner_active: torch.Tensor
    critic_active: torch.Tensor
    reward_std: torch.Tensor
    advantage: torch.Tensor


def _collate(
    batch: list[RLExample], table: np.ndarray, mean: float, std: float,
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
        card_emb[i, :n] = table[ex.card_idx]
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
        action_token=torch.tensor([ex.action_token for ex in batch], device=device),
        learner_active=torch.tensor(
            [ex.learner_active for ex in batch], device=device, dtype=torch.bool,
        ),
        critic_active=torch.tensor(
            [ex.critic_active for ex in batch], device=device, dtype=torch.bool,
        ),
        reward_std=torch.tensor(
            [(ex.reward - mean) / std for ex in batch],
            device=device, dtype=torch.float32,
        ),
        advantage=torch.tensor(
            [ex.advantage for ex in batch], device=device, dtype=torch.float32,
        ),
    )


@dataclass
class _LossOut:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    kl: torch.Tensor


def _forward_logits_critic(
    model: DraftAgentModel, batch: _Batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model(
        batch.card_emb, batch.type_idx, batch.packs_ago, batch.pick_ago,
        batch.card_mask, batch.pack_number, batch.pick_number,
    )


def _compute_loss(
    model: DraftAgentModel,
    reference: DraftAgentModel,
    batch: _Batch,
    config: TrainDraftAgentRLConfig,
    entropy_coef: float,
    kl_coef: float,
) -> _LossOut:
    """The RL loss (data-model §4). Distributions are π_T = softmax(logits / T)."""
    temp = float(config.rollout_temperature or 1.0)
    policy_logits, critic_pred = _forward_logits_critic(model, batch)
    device = policy_logits.device
    scaled = policy_logits / temp

    policy = torch.zeros((), device=device)
    entropy = torch.zeros((), device=device)
    kl = torch.zeros((), device=device)
    learner = batch.learner_active
    if bool(learner.any()):
        logp = _masked_log_softmax(scaled, batch.pack_mask)  # (B, N)
        idx = batch.action_token.clamp(min=0).unsqueeze(1)
        logp_action = logp.gather(1, idx).squeeze(1)  # (B,)
        adv = batch.advantage[learner]
        policy = -(adv * logp_action[learner]).mean()
        entropy = _policy_entropy(scaled, batch.pack_mask)[learner].mean()
        with torch.no_grad():
            ref_logits, _ = _forward_logits_critic(reference, batch)
        kl = _kl_divergence(scaled, ref_logits / temp, batch.pack_mask)[learner].mean()

    value = torch.zeros((), device=device)
    if bool(batch.critic_active.any()):
        active = batch.critic_active
        value = F.mse_loss(critic_pred[active], batch.reward_std[active])

    total = (
        policy + config.value_weight * value
        - entropy_coef * entropy + kl_coef * kl
    )
    return _LossOut(total, policy, value, entropy, kl)


# --------------------------------------------------------------------------- #
# Per-epoch advantage precompute (research D3) + behaviour-anomaly (D6)
# --------------------------------------------------------------------------- #

def _critic_values(
    model: DraftAgentModel, examples: list[RLExample], table: np.ndarray,
    mean: float, std: float, batch_size: int, device: torch.device,
) -> None:
    """Write the current critic value onto each example (batched, no_grad)."""
    order = sorted(range(len(examples)), key=lambda i: examples[i].n_tokens)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idxs = order[s:s + batch_size]
            batch = _collate([examples[i] for i in idxs], table, mean, std, device)
            _, critic = _forward_logits_critic(model, batch)
            vals = critic.cpu().numpy()  # one device→host transfer per batch
            for j, i in enumerate(idxs):
                examples[i].value = float(vals[j])
    model.train()


def _precompute_advantages(
    model: DraftAgentModel,
    trajectories: list[list[RLExample]],
    table: np.ndarray,
    mean: float,
    std: float,
    gae_lambda: float,
    batch_size: int,
    device: torch.device,
) -> None:
    """Refresh GAE advantages over every learner trajectory (research D3)."""
    flat = [ex for traj in trajectories for ex in traj]
    if not flat:
        return
    _critic_values(model, flat, table, mean, std, batch_size, device)
    for traj in trajectories:
        values = np.asarray([ex.value for ex in traj], dtype=np.float64)
        reward_std = (traj[0].reward - mean) / std
        adv = gae_advantages(values, reward_std, gae_lambda, _GAMMA)
        for ex, a in zip(traj, adv):
            ex.advantage = float(a)


@dataclass
class _AnomalySummary:
    n_learner: int
    frac_below_floor: float
    mean_behaviour_logprob: float


def _behaviour_anomaly_summary(
    reference: DraftAgentModel,
    examples: list[RLExample],
    table: np.ndarray,
    mean: float,
    std: float,
    config: TrainDraftAgentRLConfig,
    device: torch.device,
) -> _AnomalySummary:
    """Recompute log π_ref,T(a_t) over learner picks; summarize anomalies (D6)."""
    temp = float(config.rollout_temperature or 1.0)
    learner = [ex for ex in examples if ex.learner_active]
    if not learner:
        return _AnomalySummary(0, 0.0, float("nan"))
    order = sorted(range(len(learner)), key=lambda i: learner[i].n_tokens)
    below = 0
    logp_sum = 0.0
    reference.eval()
    with torch.no_grad():
        for s in range(0, len(order), config.batch_size):
            idxs = order[s:s + config.batch_size]
            chunk = [learner[i] for i in idxs]
            batch = _collate(chunk, table, mean, std, device)
            ref_logits, _ = _forward_logits_critic(reference, batch)
            logp = _masked_log_softmax(ref_logits / temp, batch.pack_mask)
            idx = batch.action_token.clamp(min=0).unsqueeze(1)
            logp_action = logp.gather(1, idx).squeeze(1).cpu().numpy()
            logp_sum += float(logp_action.sum())
            below += int((np.exp(logp_action) < _PROB_FLOOR).sum())
    n = len(learner)
    return _AnomalySummary(n, below / n, logp_sum / n)


# --------------------------------------------------------------------------- #
# Resume / build (research D7)
# --------------------------------------------------------------------------- #

@dataclass
class _Resume:
    model: DraftAgentModel
    reference: DraftAgentModel
    start_epoch: int
    best_val_loss: float
    optimizer_state: dict[str, Any] | None
    lr_decay_count: int
    critic_mean: float
    critic_std: float
    generation: int


def _check_dims(ckpt_config: DraftAgentConfig, embedding_dim: int, cards_path: Path) -> None:
    if ckpt_config.embedding_dim != embedding_dim:
        raise ValueError(
            f"Checkpoint expects {ckpt_config.embedding_dim}-wide card "
            f"embeddings, but the .npz cache under {cards_path} is "
            f"{embedding_dim}-wide. Re-run `sealed encode-cards` with the "
            "encoder this agent was trained on."
        )


def _resume_or_build(
    config: TrainDraftAgentRLConfig, store: DraftAgentStore, embedding_dim: int,
) -> _Resume:
    """Load actor+critic from --checkpoint or --resume; freeze a KL reference.

    The KL reference is a frozen copy of the model weights at run start: the
    reference πₖ on a fresh ``--checkpoint`` run; the resumed-from weights on
    ``--resume`` (research D7). Generation index = reference generation + 1.
    """
    source = config.resume if config.resume is not None else config.checkpoint
    assert source is not None  # CLI guarantees exactly one
    ckpt = store.load_checkpoint(source)
    _check_dims(ckpt.config, embedding_dim, config.cards_path)
    model = DraftAgentModel(ckpt.config)
    model.load_state_dict(ckpt.model_state_dict)
    reference = DraftAgentModel(ckpt.config)
    reference.load_state_dict(ckpt.model_state_dict)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    if config.resume is not None:
        start_epoch = ckpt.epoch + 1
        best = ckpt.best_val_loss
        opt_state = ckpt.optimizer_state_dict or None
        decay = ckpt.lr_decay_count
        # Continuing an RL run: keep this run's generation index.
        generation = int((ckpt.rl_metadata or {}).get("generation", 2))
    else:
        start_epoch = 0
        best = float("inf")
        opt_state = None
        decay = 0
        # Fresh kickoff: reference generation (gen-1 if no rl_metadata) + 1.
        ref_gen = int((ckpt.rl_metadata or {}).get("generation", 1))
        generation = ref_gen + 1
    return _Resume(
        model, reference, start_epoch, best, opt_state, decay,
        ckpt.critic_mean, ckpt.critic_std, generation,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

@dataclass
class _ValReport:
    loss: float
    policy: float
    value: float
    entropy: float
    kl: float


def _validate(
    model: DraftAgentModel,
    reference: DraftAgentModel,
    val: list[RLExample],
    table: np.ndarray,
    mean: float,
    std: float,
    config: TrainDraftAgentRLConfig,
    entropy_coef: float,
    kl_coef: float,
    device: torch.device,
) -> _ValReport:
    """Held-out RL objective (the best-checkpoint metric, research D9)."""
    model.eval()
    if not val:
        return _ValReport(0.0, 0.0, 0.0, 0.0, 0.0)
    tot = pol = val_l = ent = kl = 0.0
    n = 0
    with torch.no_grad():
        for s in range(0, len(val), config.batch_size):
            chunk = val[s:s + config.batch_size]
            batch = _collate(chunk, table, mean, std, device)
            out = _compute_loss(
                model, reference, batch, config, entropy_coef, kl_coef,
            )
            bn = len(chunk)
            tot += float(out.total) * bn
            pol += float(out.policy) * bn
            val_l += float(out.value) * bn
            ent += float(out.entropy) * bn
            kl += float(out.kl) * bn
            n += bn
    model.train()
    return _ValReport(tot / n, pol / n, val_l / n, ent / n, kl / n)


# --------------------------------------------------------------------------- #
# Use case
# --------------------------------------------------------------------------- #

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

        locator = ConvertedCardLocator(config.cards_path)
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
            reference, examples, table, mean, std, config, device,
        )
        msg = (
            f"Behaviour check: {anomaly.n_learner} learner picks, "
            f"mean log π_ref={anomaly.mean_behaviour_logprob:.3f}, "
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

        out_dir = Path("models/draft/agent")
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
                model, reference, val, table, mean, std, config,
                entropy_sched.coef, kl_sched.coef, device,
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
                model, trajectories, table, mean, std,
                config.gae_lambda, config.batch_size, device,
            )
            model.train()
            batches = length_bucketed_batches(train, config.batch_size, rng)
            nb = len(batches)
            epoch_start = time.monotonic()
            last_log = epoch_start
            for step, chunk in enumerate(batches, start=1):
                batch = _collate(chunk, table, mean, std, device)
                out = _compute_loss(
                    model, reference, batch, config,
                    entropy_sched.coef, kl_sched.coef,
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
                        f"({step / nb * 100:.0f}%) total={float(out.total):.4f} "
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
