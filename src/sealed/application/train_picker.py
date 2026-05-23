"""Training use case for the one-shot sealed deck picker.

REINFORCE-from-random-init against a frozen scorer, with a per-pool
empirical-mean baseline, an entropy bonus on a val-reward-driven decay
schedule, and an auxiliary pool-quality head (spec 017).

TODO(shared-trainer): three trainers now share the resume / bootstrap /
best-vs-latest persistence / fail-fast width-check / early-stop scaffolding —
``train_encoder``, ``train_scorer``, and this ``train_picker``. They diverge on
loss type (weighted MSE+MLM / pairwise BCE / REINFORCE-with-baseline), metric
direction (val_loss / val_acc / val_reward), and per-step dataset shape, so a
unifying abstraction would be a generic training loop (premature today). If a
fourth REINFORCE-style trainer arrives, extract the common scaffolding then.
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

from sealed.domain.card_embedding_layout import (
    COLOR_FLAGS,
    FEATURE_COUNT,
    MANA_VALUE,
    POWER,
    TOUGHNESS,
    is_land_embedding,
)
from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE
from sealed.domain.picker_model import PickerConfig, PickerModel, walk_pick_indices
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.picker_store import PickerStore
from sealed.infrastructure.pool_file_reader import parse_pools
from sealed.infrastructure.scorer_store import ScorerStore

RANDOM_SEED = 42  # hardcoded (FR-018); governs init, shuffle, sampling, split.
_SCORE_CHUNK = 256  # sub-batch size for frozen-scorer forwards over many decks.
_ADVANTAGE_STD_EPS = 1e-6  # floor for the per-pool std in GRPO advantage norm.


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


@dataclass
class TrainPickerConfig:
    pools_path: Path
    scorer_checkpoint: Path = field(
        default_factory=lambda: Path("models/sealed/scorer/latest.pt"),
    )
    auditor_scorer_checkpoint: Path | None = None
    cards_path: Path = field(default_factory=lambda: Path("output/cardsfolder/"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("models/sealed/picker/"))
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


# --------------------------------------------------------------------------- #
# Prepared pools and batching
# --------------------------------------------------------------------------- #

class PreparedPool:
    """A sealed pool plus its card embeddings.

    Cards repeat heavily across pools of the same set, so a 100k-pool file of
    per-pool ``(N, dim)`` float32 copies costs ~20 GB. To bound that, pools may
    instead store ``int32`` indices into a shared embedding table built once by
    ``_prepare_pools``; ``.embeddings`` materializes the per-pool ``(N, dim)``
    view on demand (transient, GC'd after each batch). The dense form — passing
    a real ``(N, dim)`` array — is kept for tests and small callers.
    """

    def __init__(
        self,
        set_code: str,
        names: list[str],
        embeddings: np.ndarray,
        is_land: np.ndarray,
    ) -> None:
        self.set_code = set_code
        self.names = names
        self.is_land = is_land
        self._dense: np.ndarray | None = embeddings
        self._table: np.ndarray | None = None
        self._indices: np.ndarray | None = None

    @classmethod
    def from_shared(
        cls,
        set_code: str,
        names: list[str],
        table: np.ndarray,
        indices: np.ndarray,
        is_land: np.ndarray,
    ) -> "PreparedPool":
        obj = cls.__new__(cls)
        obj.set_code = set_code
        obj.names = names
        obj.is_land = is_land
        obj._dense = None
        obj._table = table
        obj._indices = indices
        return obj

    @property
    def embeddings(self) -> np.ndarray:
        if self._dense is not None:
            return self._dense
        assert self._table is not None and self._indices is not None
        return self._table[self._indices]

    @property
    def num_cards(self) -> int:
        return int(self.is_land.shape[0])

    @property
    def dim(self) -> int:
        if self._dense is not None:
            return int(self._dense.shape[1])
        assert self._table is not None
        return int(self._table.shape[1])


def _prepare_pools(
    pools: list[tuple[str, list[str]]], locator: ConvertedCardLocator,
) -> list[PreparedPool]:
    """Load embeddings for every pool; skip pools with < 23 embeddable cards.

    Each distinct card is loaded once into a shared table; pools keep only int
    indices into it (see ``PreparedPool``) so a large pool file stays at the
    size of the unique-card set rather than the sum of all pool copies.
    """
    table_rows: list[np.ndarray] = []
    table_is_land: list[bool] = []
    name_to_idx: dict[str, int | None] = {}

    def _intern(name: str) -> int | None:
        idx = name_to_idx.get(name, -1)
        if idx != -1:
            return idx
        emb = locator.load_embedding(name)
        if emb is None:
            name_to_idx[name] = None
            return None
        new_idx = len(table_rows)
        table_rows.append(np.asarray(emb, dtype=np.float32))
        table_is_land.append(bool(is_land_embedding(emb)))
        name_to_idx[name] = new_idx
        return new_idx

    pending: list[tuple[str, list[str], list[int]]] = []
    for set_code, names in pools:
        valid_names: list[str] = []
        idxs: list[int] = []
        for name in names:
            idx = _intern(name)
            if idx is not None:
                valid_names.append(name)
                idxs.append(idx)
        if len(valid_names) < NONLAND_DECK_SIZE:
            continue
        pending.append((set_code, valid_names, idxs))

    if not pending:
        return []
    table = np.stack(table_rows).astype(np.float32)
    land_table = np.array(table_is_land, dtype=bool)
    prepared: list[PreparedPool] = []
    for set_code, valid_names, idxs in pending:
        index_arr = np.asarray(idxs, dtype=np.int64)
        prepared.append(
            PreparedPool.from_shared(
                set_code, valid_names, table, index_arr, land_table[index_arr],
            )
        )
    return prepared


def _split_pools(
    pools: list[PreparedPool], val_fraction: float,
) -> tuple[list[PreparedPool], list[PreparedPool]]:
    """Front ``val_fraction`` is the fixed validation slice; rest is training."""
    n_val = int(len(pools) * val_fraction)
    return pools[:n_val], pools[n_val:]


def _shuffle_train(
    pools: list[PreparedPool], rng: random.Random,
) -> list[PreparedPool]:
    out = list(pools)
    rng.shuffle(out)
    return out


def _collate(pool_slice: list[PreparedPool]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-pad a slice of pools to ``(B, max_N, dim)`` + masks."""
    b = len(pool_slice)
    max_n = max(p.num_cards for p in pool_slice)
    dim = pool_slice[0].dim
    cards = np.zeros((b, max_n, dim), dtype=np.float32)
    mask = np.zeros((b, max_n), dtype=bool)
    land = np.zeros((b, max_n), dtype=bool)
    for i, p in enumerate(pool_slice):
        n = p.num_cards
        cards[i, :n] = p.embeddings
        mask[i, :n] = True
        land[i, :n] = p.is_land
    return cards, mask, land


# --------------------------------------------------------------------------- #
# Sampler, log-prob, entropy (pure helpers)
# --------------------------------------------------------------------------- #

def _sample_decks(
    logits: torch.Tensor,
    is_land_mask: torch.Tensor,
    pool_mask: torch.Tensor,
    n_samples: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-batched sequential without-replacement sampler (spec § 3.2, FR-013).

    Operates on the full ``(B*S, N)`` tensor in vectorized steps. Each row picks
    one card per step via ``torch.multinomial`` until its 23-spell quota fills
    (per-row stop; finished rows still draw but their picks are discarded).

    Returns ``(pick_indices (B*S, K), picked_mask (B*S, K))`` where ``K`` is the
    number of steps taken and ``picked_mask`` marks the real picks.
    """
    b, n = logits.shape
    device = logits.device
    bs = b * n_samples

    base = (logits / temperature).masked_fill(~pool_mask, float("-inf"))
    work = base.unsqueeze(1).expand(b, n_samples, n).reshape(bs, n)
    land_bs = is_land_mask.unsqueeze(1).expand(b, n_samples, n).reshape(bs, n)
    valid_bs = pool_mask.unsqueeze(1).expand(b, n_samples, n).reshape(bs, n)

    picked = torch.zeros(bs, n, dtype=torch.bool, device=device)
    spells = torch.zeros(bs, dtype=torch.long, device=device)
    arange = torch.arange(bs, device=device)

    pick_steps: list[torch.Tensor] = []
    real_steps: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(n):
            remaining = (~picked) & valid_bs
            has_remaining = remaining.any(dim=-1)
            active = spells < NONLAND_DECK_SIZE
            real = active & has_remaining
            if not bool(real.any()):
                break
            cur = work.masked_fill(~remaining, float("-inf"))
            # Dead rows (no remaining) have an all -inf row -> give them a safe
            # finite slot so softmax/multinomial don't NaN; their pick is masked.
            dead = ~has_remaining
            if bool(dead.any()):
                cur[dead, 0] = 0.0
            probs = torch.softmax(cur, dim=-1)
            pick = torch.multinomial(probs, 1).squeeze(-1)

            upd = torch.zeros_like(picked)
            upd[arange, pick] = real
            picked = picked | upd

            is_land_pick = land_bs[arange, pick]
            spells = spells + (real & ~is_land_pick).long()

            pick_steps.append(pick)
            real_steps.append(real)

    pick_indices = torch.stack(pick_steps, dim=1)  # (bs, K)
    picked_mask = torch.stack(real_steps, dim=1)   # (bs, K)
    return pick_indices, picked_mask


def _plackett_luce_log_prob(
    logits: torch.Tensor,
    pick_indices: torch.Tensor,
    picked_mask: torch.Tensor,
) -> torch.Tensor:
    """Differentiable PL log-prob of each sampled deck (spec § 3.5, FR-014).

    ``logits`` is ``(B*S, N)`` (already temperature-scaled, padding = -inf).
    Returns ``(B*S,)`` log-probabilities, summed over real pick steps.
    """
    bs, n = logits.shape
    k = pick_indices.shape[1]
    arange = torch.arange(bs, device=logits.device)
    removed = torch.zeros(bs, n, device=logits.device)
    log_prob = torch.zeros(bs, device=logits.device)
    for step in range(k):
        cur = logits + removed
        lse = torch.logsumexp(cur, dim=-1)
        idx = pick_indices[:, step]
        step_lp = logits[arange, idx] - lse
        log_prob = log_prob + step_lp * picked_mask[:, step].to(log_prob.dtype)
        removed[arange, idx] = float("-inf")
    return log_prob


def _masked_log_softmax(logits: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.masked_fill(~pool_mask, float("-inf")), dim=-1)


def _policy_entropy(logits: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
    """Per-pool entropy of the picker's softmax over valid positions (B,).

    Masked positions have ``logp = -inf`` and ``p = 0``. Computing ``p * logp``
    there is ``0 * -inf = NaN``; ``torch.where`` would mask the NaN out of the
    forward value but autograd still backpropagates NaN through the discarded
    branch. Zeroing the ``-inf`` log-probs *before* the product keeps both the
    value and the gradient finite (the contribution is genuinely 0 there).
    """
    logp = _masked_log_softmax(logits, pool_mask)
    p = logp.exp()
    safe_logp = logp.masked_fill(~pool_mask, 0.0)
    return -(p * safe_logp).sum(dim=-1)


def _kl_penalty(
    logits: torch.Tensor, ref_logits: torch.Tensor, pool_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean per-pool KL(picker || reference) over valid positions (FR-025).

    Same ``0 * -inf`` / NaN-gradient trap as ``_policy_entropy``: at masked
    positions ``logp - logq`` is ``-inf - -inf = NaN``. Zero the per-position
    log-ratio before weighting by ``p`` (which is 0 there anyway).
    """
    logp = _masked_log_softmax(logits, pool_mask)
    logq = _masked_log_softmax(ref_logits, pool_mask)
    p = logp.exp()
    log_ratio = (logp - logq).masked_fill(~pool_mask, 0.0)
    kl = (p * log_ratio).sum(dim=-1)
    return kl.mean()


class _EntropySchedule:
    """Val-reward-driven entropy-coefficient decay (FR-016).

    Held constant at ``initial`` until val reward has improved monotonically for
    ``decay_after`` consecutive epochs; thereafter multiplied by ``factor`` at
    the end of every epoch that fails to improve on the previous best.
    """

    def __init__(self, initial: float, decay_after: int, factor: float = 0.9) -> None:
        self.coef = initial
        self.decay_after = decay_after
        self.factor = factor
        self.armed = False
        self.streak = 0
        self.prev: float | None = None
        self.best = float("-inf")

    def update(self, val_reward: float) -> float:
        if self.prev is not None and val_reward > self.prev:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.decay_after:
            self.armed = True
        improved_best = val_reward > self.best
        if self.armed and not improved_best:
            self.coef *= self.factor
        self.prev = val_reward
        self.best = max(self.best, val_reward)
        return self.coef


def _should_stop(epochs_since_best: int, patience: int) -> bool:
    return epochs_since_best >= patience


@dataclass
class _Losses:
    policy_loss: torch.Tensor
    entropy_loss: torch.Tensor
    aux_loss: torch.Tensor
    total: torch.Tensor
    baseline: torch.Tensor   # (B,)
    advantage: torch.Tensor  # (B, S), detached


def _compute_losses(
    rewards: torch.Tensor,
    log_prob: torch.Tensor,
    entropy: torch.Tensor,
    aux_pred: torch.Tensor,
    entropy_coef: float,
    aux_weight: float,
    normalize_advantage: bool = False,
    objective: str = "reinforce",
    topk: int = 16,
) -> _Losses:
    """Assemble the picker training loss (spec § 3.3, § 3.4, FR-039).

    ``rewards`` and ``log_prob`` are ``(B, S)``; ``entropy`` and ``aux_pred``
    are ``(B,)``. The per-pool baseline is the mean reward; the aux target is
    the detached per-pool mean reward (FR-015). The entropy and aux terms are
    identical across objectives; only the policy-loss term differs.

    ``objective="reinforce"`` (default): advantage-weighted policy gradient
    with a detached per-pool baseline (FR-014). When ``normalize_advantage``
    is set, the centered reward is divided by the per-pool reward std
    (GRPO-style group normalization) — within-pool reward variance is small
    (the sampled decks share most cards), so raw advantages are tiny and the
    gradient weak; rescaling to unit variance gives a consistent step. A
    degenerate pool (all samples equal) keeps a ~0 advantage.

    ``objective="topk"`` (FR-039): reward-ranked / best-of-N. Per pool, keep
    the ``topk`` highest-reward sampled decks and maximize their log-prob by
    plain max-likelihood (no baseline/advantage term). Depends only on the
    ranking, not the (tiny) advantage magnitudes, and has no high-variance
    negative-gradient term.
    """
    baseline = rewards.mean(dim=1)
    centered = rewards - baseline.unsqueeze(1)
    if normalize_advantage:
        std = rewards.std(dim=1, unbiased=False, keepdim=True)
        centered = centered / (std + _ADVANTAGE_STD_EPS)
    advantage = centered.detach()
    if objective == "topk":
        k = min(topk, rewards.shape[1])
        topk_idx = rewards.topk(k, dim=1).indices  # (B, k)
        selected_log_prob = log_prob.gather(1, topk_idx)
        policy_loss = -selected_log_prob.mean()
    else:
        policy_loss = -(advantage * log_prob).mean()
    entropy_loss = -entropy_coef * entropy.mean()
    aux_loss = F.mse_loss(aux_pred, baseline.detach())
    total = policy_loss + entropy_loss + aux_weight * aux_loss
    return _Losses(policy_loss, entropy_loss, aux_loss, total, baseline, advantage)


# --------------------------------------------------------------------------- #
# Frozen-scorer reward path
# --------------------------------------------------------------------------- #

def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_scorer(path: Path, device: torch.device) -> SetTransformerScorer:
    checkpoint = ScorerStore().load_checkpoint(path)
    model = SetTransformerScorer(checkpoint.config)
    model.load_state_dict(checkpoint.model_state_dict)
    model.eval()
    model.to(device)
    return model


def _score_sampled(
    scorer: SetTransformerScorer,
    norm_pool: torch.Tensor,
    pick_indices: torch.Tensor,
    picked_mask: torch.Tensor,
    n_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Score every sampled deck with the frozen scorer (FR-012). Returns (B, S)."""
    b, max_n, dim = norm_pool.shape
    bs = b * n_samples
    k = pick_indices.shape[1]
    pool_bs = norm_pool.unsqueeze(1).expand(b, n_samples, max_n, dim).reshape(bs, max_n, dim)
    gather_idx = pick_indices.unsqueeze(-1).expand(bs, k, dim)
    deck_cards = torch.gather(pool_bs, 1, gather_idx)
    scores = torch.empty(bs, device=device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda"),
    ):
        for start in range(0, bs, _SCORE_CHUNK):
            stop = min(start + _SCORE_CHUNK, bs)
            out = scorer.forward_prenormalized(
                deck_cards[start:stop], picked_mask[start:stop],
            ).squeeze(-1)
            scores[start:stop] = out.float()
    return scores.view(b, n_samples)


def _score_chosen_decks(
    scorer: SetTransformerScorer,
    pools: list[PreparedPool],
    chosen: list[list[int]],
    device: torch.device,
) -> np.ndarray:
    """Score deterministic chosen-card decks. Returns (n_decks,) numpy scores."""
    if not pools:
        return np.zeros(0, dtype=np.float32)
    dim = pools[0].dim
    n = len(pools)
    scores = np.zeros(n, dtype=np.float32)
    for start in range(0, n, _SCORE_CHUNK):
        stop = min(start + _SCORE_CHUNK, n)
        group = list(range(start, stop))
        max_k = max(len(chosen[i]) for i in group)
        cards = np.zeros((len(group), max_k, dim), dtype=np.float32)
        mask = np.zeros((len(group), max_k), dtype=bool)
        for row, i in enumerate(group):
            idxs = chosen[i]
            cards[row, :len(idxs)] = pools[i].embeddings[idxs]
            mask[row, :len(idxs)] = True
        cards_t = torch.from_numpy(cards).to(device)
        mask_t = torch.from_numpy(mask).to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            norm = scorer.normalize_features(cards_t)
            out = scorer.forward_prenormalized(norm, mask_t).squeeze(-1)
        scores[start:stop] = out.float().cpu().numpy()
    return scores


# --------------------------------------------------------------------------- #
# Distributional summaries (US3, FR-032)
# --------------------------------------------------------------------------- #

@dataclass
class DistribSummary:
    colors_mean: float
    creatures_mean: float
    type_creature_share: float
    cmc_hist: list[float]  # 5 bins: CMC<=2, 3, 4, 5, 6+


def _feat(emb: np.ndarray, slot: int) -> float:
    return float(emb[-FEATURE_COUNT + slot])


def _distrib_summaries(
    pools: list[PreparedPool], chosen: list[list[int]],
) -> DistribSummary:
    if not pools:
        return DistribSummary(0.0, 0.0, 0.0, [0.0] * 5)
    color_start = -FEATURE_COUNT + COLOR_FLAGS.start
    color_stop = -FEATURE_COUNT + COLOR_FLAGS.stop
    colors_per_deck: list[int] = []
    creatures_per_deck: list[int] = []
    shares: list[float] = []
    hist = np.zeros(5, dtype=np.float64)
    n_decks = 0
    for i, idxs in enumerate(chosen):
        if not idxs:
            continue
        n_decks += 1
        embs = pools[i].embeddings[idxs]
        color_union = (embs[:, color_start:color_stop] > 0.5).any(axis=0).sum()
        colors_per_deck.append(int(color_union))
        is_creature = (embs[:, -FEATURE_COUNT + POWER] > 0) | (
            embs[:, -FEATURE_COUNT + TOUGHNESS] > 0
        )
        n_creatures = int(is_creature.sum())
        creatures_per_deck.append(n_creatures)
        shares.append(n_creatures / len(idxs))
        cmc = embs[:, -FEATURE_COUNT + MANA_VALUE]
        for v in cmc:
            if v <= 2:
                hist[0] += 1
            elif v <= 3:
                hist[1] += 1
            elif v <= 4:
                hist[2] += 1
            elif v <= 5:
                hist[3] += 1
            else:
                hist[4] += 1
    if n_decks == 0:
        return DistribSummary(0.0, 0.0, 0.0, [0.0] * 5)
    return DistribSummary(
        colors_mean=float(np.mean(colors_per_deck)),
        creatures_mean=float(np.mean(creatures_per_deck)),
        type_creature_share=float(np.mean(shares)),
        cmc_hist=[float(x / n_decks) for x in hist],
    )


# --------------------------------------------------------------------------- #
# Resume / build
# --------------------------------------------------------------------------- #

def _check_picker_width(
    checkpoint_dim: int, embedding_dim: int, source: str, cards_path: Path,
) -> None:
    if checkpoint_dim != embedding_dim:
        raise ValueError(
            f"{source} expects {checkpoint_dim}-wide card embeddings, but the "
            f".npz cache under {cards_path} is {embedding_dim}-wide. Re-run "
            "`sealed encode-cards` with the encoder this picker was trained on, "
            "or start a fresh picker (drop --resume)."
        )


@dataclass
class _PickerResume:
    model: PickerModel
    start_epoch: int
    best_val_reward: float
    optimizer_state: dict[str, Any] | None
    reference_state: dict[str, Any] | None  # for KL penalty (bootstrap only)


def _resume_or_build_picker(
    config: TrainPickerConfig, store: PickerStore, embedding_dim: int,
) -> _PickerResume:
    if config.resume is not None:
        ckpt = store.load_checkpoint(config.resume)
        _check_picker_width(
            ckpt.config.embedding_dim, embedding_dim,
            f"--resume {config.resume}", config.cards_path,
        )
        model = PickerModel(ckpt.config)
        model.load_state_dict(ckpt.model_state_dict)
        return _PickerResume(
            model=model,
            start_epoch=ckpt.epoch + 1,
            best_val_reward=ckpt.best_val_reward,
            optimizer_state=ckpt.optimizer_state_dict or None,
            reference_state=None,
        )

    if config.picker_checkpoint is not None:
        ckpt = store.load_checkpoint(config.picker_checkpoint)
        _check_picker_width(
            ckpt.config.embedding_dim, embedding_dim,
            f"--picker-checkpoint {config.picker_checkpoint}", config.cards_path,
        )
        model = PickerModel(ckpt.config)
        model.load_state_dict(ckpt.model_state_dict)
        return _PickerResume(
            model=model,
            start_epoch=0,
            best_val_reward=float("-inf"),
            optimizer_state=None,
            reference_state=ckpt.model_state_dict if config.kl_coef != 0 else None,
        )

    model = PickerModel(config.picker_config(embedding_dim))
    return _PickerResume(
        model=model,
        start_epoch=0,
        best_val_reward=float("-inf"),
        optimizer_state=None,
        reference_state=None,
    )


def _build_optimizer(
    model: PickerModel, config: TrainPickerConfig,
) -> torch.optim.Optimizer:
    groups = [{"params": list(model.parameters()), "lr": config.lr, "name": "picker"}]
    return torch.optim.AdamW(groups)


def _clip_per_group(optimizer: torch.optim.Optimizer, *, max_norm: float) -> float:
    norm = 0.0
    for group in optimizer.param_groups:
        pre = torch.nn.utils.clip_grad_norm_(group["params"], max_norm=max_norm)
        norm = max(norm, float(pre))
    return norm


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

@dataclass
class ValidationReport:
    mean_reward: float
    audit_corr: float | None
    distrib: DistribSummary


def _deterministic_chosen(
    model: PickerModel, pools: list[PreparedPool], batch_size: int, device: torch.device,
) -> list[list[int]]:
    """Run deterministic inference (FR-006) over every pool, return chosen indices."""
    chosen: list[list[int]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pools), batch_size):
            group = pools[start:start + batch_size]
            cards, mask, _land = _collate(group)
            cards_t = torch.from_numpy(cards).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            logits, _ = model(cards_t, mask_t)
            for i, p in enumerate(group):
                n = p.num_cards
                row = logits[i, :n].cpu()
                chosen.append(walk_pick_indices(row, p.is_land))
    return chosen


def _validate(
    model: PickerModel,
    scorer: SetTransformerScorer,
    auditor: SetTransformerScorer | None,
    pools: list[PreparedPool],
    batch_size: int,
    device: torch.device,
) -> ValidationReport:
    if not pools:
        return ValidationReport(0.0, None, DistribSummary(0.0, 0.0, 0.0, [0.0] * 5))
    chosen = _deterministic_chosen(model, pools, batch_size, device)
    train_scores = _score_chosen_decks(scorer, pools, chosen, device)
    audit_corr: float | None = None
    if auditor is not None:
        audit_scores = _score_chosen_decks(auditor, pools, chosen, device)
        audit_corr = _audit_correlation(train_scores, audit_scores)
    distrib = _distrib_summaries(pools, chosen)
    return ValidationReport(float(train_scores.mean()), audit_corr, distrib)


def _audit_correlation(train_scores: np.ndarray, audit_scores: np.ndarray) -> float:
    """Spearman rank correlation between two scorers on the val decks (FR-030)."""
    from scipy.stats import spearmanr

    if len(train_scores) < 2:
        return float("nan")
    corr, _ = spearmanr(train_scores, audit_scores)
    return float(corr)


# --------------------------------------------------------------------------- #
# Use case
# --------------------------------------------------------------------------- #

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

        locator = ConvertedCardLocator(config.cards_path)
        prepared = _prepare_pools(parse_pools(config.pools_path), locator)
        if not prepared:
            raise ValueError(
                f"No usable pools in {config.pools_path} (every pool had fewer "
                f"than {NONLAND_DECK_SIZE} embeddable cards under {config.cards_path})."
            )
        cache_width = prepared[0].dim
        _check_picker_width(
            scorer.config.d_model, cache_width,
            f"--scorer-checkpoint {config.scorer_checkpoint}", config.cards_path,
        )

        auditor: SetTransformerScorer | None = None
        if config.auditor_scorer_checkpoint is not None:
            auditor = _load_scorer(config.auditor_scorer_checkpoint, device)
            _check_picker_width(
                auditor.config.d_model, cache_width,
                f"--auditor-scorer-checkpoint {config.auditor_scorer_checkpoint}",
                config.cards_path,
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

        entropy = _policy_entropy(logits, pool_mask)
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
            kl = _kl_penalty(logits, ref_logits, pool_mask)
            total = total + config.kl_coef * kl
            kl_value = float(kl)

        optimizer.zero_grad()
        total.backward()
        _clip_per_group(optimizer, max_norm=config.max_grad_norm)
        optimizer.step()

        return (
            float(losses.policy_loss.detach()),
            float(losses.entropy_loss.detach()),
            float(losses.aux_loss.detach()), kl_value,
        )


@dataclass
class _EpochStats:
    policy_loss: float = 0.0
    entropy_loss: float = 0.0
    aux_loss: float = 0.0
    kl: float = 0.0
    n: int = 0

    def add(self, policy: float, entropy: float, aux: float, kl: float) -> None:
        self.policy_loss += policy
        self.entropy_loss += entropy
        self.aux_loss += aux
        self.kl += kl
        self.n += 1

    def mean(self) -> tuple[float, float, float, float]:
        d = max(self.n, 1)
        return (
            self.policy_loss / d, self.entropy_loss / d,
            self.aux_loss / d, self.kl / d,
        )


def _log_epoch(
    epoch: int, step: int, n_steps: int, stats: _EpochStats,
    report: ValidationReport,
) -> None:
    policy, entropy, aux, kl = stats.mean()
    parts = [
        f"epoch={epoch}",
        f"step={step}/{n_steps}",
        f"policy_loss={policy:.4f}",
        f"entropy_loss={entropy:.4f}",
        f"aux_loss={aux:.4f}",
        f"val_reward={report.mean_reward:.4f}",
    ]
    if kl != 0.0:
        parts.append(f"kl={kl:.4f}")
    if report.audit_corr is not None and not math.isnan(report.audit_corr):
        parts.append(f"audit_corr={report.audit_corr:.4f}")
    d = report.distrib
    hist = "[" + ", ".join(f"{x:.1f}" for x in d.cmc_hist) + "]"
    parts.append(
        f"| colors_mean={d.colors_mean:.2f} creatures_mean={d.creatures_mean:.2f} "
        f"type_creature_share={d.type_creature_share:.2f} cmc_hist={hist}"
    )
    _log("  ".join(parts))
