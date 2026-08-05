"""Gen-3 online self-play GRPO fine-tuning of the two-headed draft agent (spec 021).

A streaming generate → update → discard → regenerate loop in one process. Each
round drives **one resident Forge draft worker** for ``drafts_per_round`` fresh
drafts whose learner seats are piloted by the live in-training policy, builds and
scores every seat's deck to get the reward, takes **one minibatch pass** of the
single critic-free term

    L = − mean( A · logπ_T(a|s) )        over learner-seat picks only

then discards the batch and drafts the next round with the updated weights. π_T is
the PACK-masked softmax at the rollout temperature ``T``; ``A`` is the round-
standardised pod-relative leave-one-out ``deck_score`` (terminal, γ=1), shared by
all of a seat's picks and detached.

There is deliberately no critic term, GAE, KL anchor, entropy bonus, validation
split, early stop, or best-checkpoint guard — see :class:`TrainDraftAgentOnlineConfig`.
The per-pick state walk is the shared :mod:`draft.application.draft_pick_states`;
the reward and batching helpers are :mod:`draft.application.draft_training_common`;
the masked-distribution math is
:mod:`price_predictor.infrastructure.torch_training`.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from draft.application.draft_pick_states import iter_seat_pick_states
from draft.application.draft_training_common import (
    leave_one_out_rewards,
    length_bucketed_batches,
)
from draft.domain.draft_agent_model import DraftAgentModel
from draft.domain.draft_geometry import DraftGeometry, DraftRecord
from draft.domain.draft_state import TYPE_PACK
from draft.infrastructure.draft_agent_store import DraftAgentStore
from price_predictor.infrastructure.torch_training import (
    clip_per_group,
    kl_divergence,
    masked_log_softmax,
    policy_entropy,
)

CHECKPOINT_DIR = Path("models/draft/agent")

_MISSING_WARN_CAP = 20

# Degenerate-round guard (FR-023): below this reward std the standardisation
# would amplify float noise into arbitrary advantages.
_ADV_STD_EPS = 1e-8
# Advantage-spread reporting bands (FR-014).
_ADV_NEAR_ZERO = 0.1
_ADV_LARGE = 0.5


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


_DEFAULT_MIX = "gen-3:5,gen-1:3,forge-r30:1,forge-r100:1"


@dataclass
class TrainDraftAgentOnlineConfig:
    """Resolved inputs to the online GRPO loop (data-model §1).

    ``learner_label``/``learner_checkpoint`` and ``rollout_temperature`` are
    required; the dataclass defaults exist only so the CLI can construct and then
    validate (:func:`_validate_config`, data-model §1.1). Architecture is
    inherited from the learner checkpoint — there are no architecture flags.

    **Absent by design** (spec FR-006, Out of Scope) — gen-2 knobs that do not
    exist here and must not be added: ``value_weight``, ``gae_lambda``,
    ``kl_coef``, ``entropy_coef`` and every coefficient-decay schedule,
    ``val_fraction``, ``patience``, ``epochs``, ``lr_decay_*``, ``resume``, and
    ``pick_mode`` (rollouts are always sampled at ``rollout_temperature``). The
    operator is expected to tune exactly three knobs — ``lr``,
    ``rollout_temperature``, and ``drafts_per_round``; everything else carries a
    fixed default.
    """

    # Agent wiring (FR-002, FR-003)
    learner_label: str = ""
    learner_checkpoint: Path | None = None
    frozen: dict[str, Path] = field(default_factory=dict)
    anchor: str | None = None
    mix: list[tuple[str, int]] = field(default_factory=list)

    # Deck building & reward (FR-005)
    scorer_checkpoint: Path = field(
        default_factory=lambda: Path("models/sealed/scorer/latest.pt"),
    )
    build_method: str = "greedy"
    picker_checkpoint: Path = field(
        default_factory=lambda: Path("models/sealed/picker/latest.pt"),
    )
    cards_path: Path = field(default_factory=lambda: Path("output/cardsfolder/"))

    # Rollout & optimisation (FR-004, FR-006, FR-025)
    rollout_temperature: float | None = None
    lr: float = 1e-4
    drafts_per_round: int = 10
    anchor_window: int = 100
    snapshot_every: int = 25
    max_rounds: int | None = None
    set_code: str | None = None
    batch_size: int = 32
    max_grad_norm: float = 1.0
    warmup_steps: int = 200
    seed: int = 42

    # Corpus & fault handling (FR-020, FR-031)
    output_path: Path = field(
        default_factory=lambda: Path("output/draft/drafts.jsonl"),
    )
    max_consecutive_faults: int = 5


# --------------------------------------------------------------------------- #
# Startup validation (data-model §1.1, FR-024)
# --------------------------------------------------------------------------- #

class OnlineConfigError(ValueError):
    """An invalid gen-3 run configuration, rejected before any update."""


def parse_learner_spec(specs: list[str] | None) -> tuple[str, Path]:
    """Parse the single ``--learner LABEL=PATH``.

    A bare ``PATH`` is deliberately **not** accepted (unlike
    ``--agent-checkpoint``): the learner's mix label is load-bearing — it names
    the seats that feed the gradient and the anchor margin's learner side.
    """
    specs = list(specs or [])
    if not specs:
        raise OnlineConfigError(
            "--learner LABEL=PATH is required: exactly one learner agent names "
            "the mix label piloted by the live policy and the checkpoint that "
            "warm-starts it"
        )
    if len(specs) > 1:
        raise OnlineConfigError(
            f"exactly one --learner is allowed, got {len(specs)}: "
            f"{', '.join(specs)}"
        )
    label, sep, path_str = specs[0].partition("=")
    label, path_str = label.strip(), path_str.strip()
    if not sep or not label or not path_str:
        raise OnlineConfigError(
            f"--learner {specs[0]!r} must be 'LABEL=PATH' — the mix label is "
            "load-bearing, so a bare PATH is not accepted"
        )
    return label, Path(path_str)


def parse_frozen_specs(specs: list[str] | None) -> dict[str, Path]:
    """Parse the repeatable ``--frozen LABEL=PATH`` bindings."""
    frozen: dict[str, Path] = {}
    for spec in specs or []:
        label, sep, path_str = spec.partition("=")
        label, path_str = label.strip(), path_str.strip()
        if not sep or not label or not path_str:
            raise OnlineConfigError(f"--frozen {spec!r} must be 'LABEL=PATH'")
        if label in frozen:
            raise OnlineConfigError(f"duplicate --frozen label {label!r}")
        frozen[label] = Path(path_str)
    return frozen


def resolve_anchor(config: TrainDraftAgentOnlineConfig) -> str:
    """Resolve (and validate) the anchor label — data-model §1.1 rule 4.

    The anchor is the frozen label the margin is measured against. With exactly
    one frozen agent it defaults to that agent; otherwise ``--anchor`` is
    required. It must be a frozen label present in the mix — never the learner
    (whose margin against itself is 0) and never a Forge built-in (which carries
    no checkpoint and so cannot be held fixed as a generation).
    """
    mix_labels = {label for label, _ in config.mix}
    if config.anchor is None:
        if not config.frozen:
            raise OnlineConfigError(
                "no anchor available: bind a reference generation with "
                "--frozen LABEL=PATH (its windowed mean deck_score is the "
                "baseline the anchor margin is measured against)"
            )
        if len(config.frozen) > 1:
            raise OnlineConfigError(
                "--anchor is required when more than one --frozen agent is "
                f"bound (got {', '.join(sorted(config.frozen))}); it names "
                "which one the anchor margin is measured against"
            )
        anchor = next(iter(config.frozen))
    else:
        anchor = config.anchor
        if anchor not in config.frozen:
            raise OnlineConfigError(
                f"--anchor {anchor!r} must be a --frozen label (got "
                f"{', '.join(sorted(config.frozen)) or 'none'}); the learner "
                "and Forge built-ins cannot serve as the fixed baseline"
            )
    if anchor not in mix_labels:
        raise OnlineConfigError(
            f"--anchor {anchor!r} must also appear in --mix, or no anchor seats "
            "are ever drafted and the margin is never defined"
        )
    return anchor


def validate_config(config: TrainDraftAgentOnlineConfig) -> str:
    """Run every data-model §1.1 rule in order; return the resolved anchor label.

    Raises :class:`OnlineConfigError` on the first violation. Every check here
    runs before the Forge worker launches and before any update (FR-024,
    SC-006); the checkpoint-width / geometry checks (rule 7) need the embedding
    cache and so run inside :meth:`TrainDraftAgentOnlineUseCase.execute`.
    """
    from draft.application.agent_registry import FORGE_BUILTINS

    # Rule 1 — the learner binding resolves to an existing checkpoint.
    if not config.learner_label or config.learner_checkpoint is None:
        raise OnlineConfigError("--learner LABEL=PATH is required")
    if not config.learner_checkpoint.exists():
        raise OnlineConfigError(
            f"learner checkpoint not found: {config.learner_checkpoint}"
        )
    for label, path in sorted(config.frozen.items()):
        if not path.exists():
            raise OnlineConfigError(
                f"frozen checkpoint for {label!r} not found: {path}"
            )

    # Rule 2 — the learner label is in the mix and is not also frozen.
    mix_labels = {label for label, _ in config.mix}
    if config.learner_label not in mix_labels:
        raise OnlineConfigError(
            f"learner label {config.learner_label!r} must appear in --mix, or "
            "no learner seats are ever drafted and there is nothing to train on"
        )
    if config.learner_label in config.frozen:
        raise OnlineConfigError(
            f"label {config.learner_label!r} is bound both as the learner and "
            "as a --frozen agent; it cannot be trained and held fixed at once"
        )

    # Rule 3 — every mix label is a built-in, a frozen agent, or the learner.
    known = FORGE_BUILTINS | set(config.frozen) | {config.learner_label}
    for label in sorted(mix_labels - known):
        raise OnlineConfigError(
            f"mix label {label!r} is neither a Forge built-in "
            f"({', '.join(sorted(FORGE_BUILTINS))}), a --frozen agent, nor the "
            "learner"
        )

    # Rule 4 — the anchor resolves unambiguously to a frozen label in the mix.
    anchor = resolve_anchor(config)

    # Rule 5 — the rollout temperature is supplied and positive.
    if config.rollout_temperature is None:
        raise OnlineConfigError(
            "-T/--rollout-temperature is required (no default): it is both the "
            "sampling temperature and the temperature every policy "
            "distribution in the update is evaluated at"
        )
    if config.rollout_temperature <= 0:
        raise OnlineConfigError(
            f"--rollout-temperature must be > 0, got {config.rollout_temperature}"
        )

    # Rule 6 — the reward inputs exist.
    if not config.scorer_checkpoint.exists():
        raise OnlineConfigError(
            f"scorer checkpoint not found: {config.scorer_checkpoint} (the "
            "frozen scorer produces every seat's deck_score, i.e. the reward)"
        )
    if config.build_method == "picker" and not config.picker_checkpoint.exists():
        raise OnlineConfigError(
            f"picker checkpoint not found: {config.picker_checkpoint} "
            "(required by --build-method picker)"
        )

    # Rule 8 — positive counts.
    for name, value in (
        ("--drafts-per-round", config.drafts_per_round),
        ("--anchor-window", config.anchor_window),
        ("--batch-size", config.batch_size),
        ("--snapshot-every", config.snapshot_every),
    ):
        if value < 1:
            raise OnlineConfigError(f"{name} must be >= 1, got {value}")
    if config.max_rounds is not None and config.max_rounds < 1:
        raise OnlineConfigError(
            f"--max-rounds must be >= 1 when given, got {config.max_rounds}"
        )
    return anchor


# --------------------------------------------------------------------------- #
# Training examples (data-model §3) — a trimmed sibling of RLExample
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class OnlineExample:
    """One learner ``(draft, seat, pack, pick)`` state.

    Trimmed against ``train_draft_agent_rl.RLExample``: no critic target, no GAE
    value cache, and no ``learner_active`` flag — non-learner picks are never
    materialised at all, since nothing but the policy gradient consumes them.
    """

    card_idx: np.ndarray      # (N,) int32, rows into RoundBatch.table
    type_idx: np.ndarray      # (N,) int8, TYPE_*
    packs_ago: np.ndarray     # (N,) int8
    pick_ago: np.ndarray      # (N,) int8
    pack_number: int          # 1-based
    pick_number: int          # 1-based
    action_token: int         # absolute index of the taken PACK token
    advantage: float = 0.0    # the seat's shared, detached A (§4)

    @property
    def n_tokens(self) -> int:
        return int(self.card_idx.shape[0])


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
    advantage: torch.Tensor


def _collate(
    batch: list[OnlineExample], table: np.ndarray, device: torch.device,
) -> _Batch:
    """Pad to the batch's longest example and place every tensor on ``device``.

    Padding is per batch (length bucketing keeps it small) and the whole batch
    moves in a handful of transfers — never one per item (Principle VIII).
    """
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
        advantage=torch.tensor(
            [ex.advantage for ex in batch], device=device, dtype=torch.float32,
        ),
    )


def _forward(model: DraftAgentModel, batch: _Batch) -> tuple[torch.Tensor, torch.Tensor]:
    return model(
        batch.card_emb, batch.type_idx, batch.packs_ago, batch.pick_ago,
        batch.card_mask, batch.pack_number, batch.pick_number,
    )


# --------------------------------------------------------------------------- #
# Round loader (data-model §2, research D6/D14)
# --------------------------------------------------------------------------- #

@dataclass
class RoundBatch:
    """One round's fresh, single-use data. Built, consumed once, then dropped."""

    examples: list[OnlineExample] = field(default_factory=list)
    seat_examples: list[list[OnlineExample]] = field(default_factory=list)
    learner_rewards: list[float] = field(default_factory=list)
    table: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 1), dtype=np.float32),
    )
    dropped_seats: int = 0
    index: int = 0
    records: list[DraftRecord] = field(default_factory=list)
    gen_seconds: float = 0.0


def learner_seat_rewards(
    record: DraftRecord, learner_label: str,
) -> tuple[list[tuple[int, float]], int]:
    """``(seat_index, R)`` for every surviving learner seat, plus the drop count.

    ``R`` is the pod-relative leave-one-out ``deck_score`` (FR-008). Two kinds of
    learner seat are excluded and counted as dropped (FR-022, data-model §4):

    * a failed build (``deck_score is None``) — no score, so no reward; and
    * a seat whose pod has **no other** scored seat, whose leave-one-out
      baseline is undefined. ``leave_one_out_rewards`` substitutes ``0.0`` for
      the empty mean there, which would silently turn the raw ``deck_score``
      into the reward on a different scale from every other seat's.
    """
    rewards = leave_one_out_rewards(record)
    n_scored = sum(1 for seat in record.seats if seat.deck_score is not None)
    surviving: list[tuple[int, float]] = []
    dropped = 0
    for seat_idx, seat in enumerate(record.seats):
        if seat.agent != learner_label:
            continue
        reward = rewards[seat_idx]
        if reward is None or n_scored < 2:
            dropped += 1
            continue
        surviving.append((seat_idx, float(reward)))
    return surviving, dropped


class RoundLoader:
    """Turns one round's records into learner examples over a per-round table.

    Mirrors ``train_draft_agent_rl._Loader``'s shared-table memoization (each
    distinct card resolved once, examples keep int rows) minus the multi-corpus
    split. The table is rebuilt per round (research D14): a round touches only a
    few hundred cards, and a table that accumulated across an hours-long run
    would grow without bound and be re-``stack``ed every round.
    """

    def __init__(self, locator) -> None:
        self._locator = locator
        self.embedding_dim: int | None = None
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

    def _table(self) -> np.ndarray:
        if not self._table_rows:
            return np.zeros((0, self.embedding_dim or 1), dtype=np.float32)
        return np.stack(self._table_rows).astype(np.float32)

    def build(self, records: list[DraftRecord], learner_label: str) -> RoundBatch:
        """Walk every learner seat of every record into a fresh :class:`RoundBatch`."""
        self._table_rows = []
        self._name_to_idx = {}

        batch = RoundBatch(records=list(records))
        for record in records:
            geo = DraftGeometry.from_record(record)
            surviving, dropped = learner_seat_rewards(record, learner_label)
            batch.dropped_seats += dropped
            for seat_idx, reward in surviving:
                seat_examples = [
                    OnlineExample(
                        card_idx=state.card_idx,
                        type_idx=state.type_idx,
                        packs_ago=state.packs_ago,
                        pick_ago=state.pick_ago,
                        pack_number=state.pack_number,
                        pick_number=state.pick_number,
                        action_token=state.action_position,
                    )
                    for state in iter_seat_pick_states(
                        geo, record.boosters, seat_idx, self._card_index,
                    )
                    # action_position == -1 ⇒ the taken card has no .npz, so
                    # there is no usable action to reinforce at this pick.
                    if state.action_position >= 0
                ]
                batch.seat_examples.append(seat_examples)
                batch.learner_rewards.append(reward)
                batch.examples.extend(seat_examples)

        if self._missing_total:
            sample = ", ".join(sorted(self._missing))
            _log(
                f"Dropped {self._missing_total} cards with no .npz embedding "
                f"(e.g. {sample})"
            )
            self._missing.clear()
            self._missing_total = 0

        batch.table = self._table()
        return batch


# --------------------------------------------------------------------------- #
# Reward → advantage (data-model §4, FR-009/FR-023)
# --------------------------------------------------------------------------- #

@dataclass
class AdvantageStats:
    """The round's reward-signal axis (FR-014)."""

    reward_mean: float = 0.0
    reward_std: float = 0.0
    adv_std: float = 0.0
    adv_near_zero_frac: float = 0.0
    adv_large_frac: float = 0.0
    adv_absmax: float = 0.0
    degenerate: bool = False
    reason: str | None = None


def standardize_round_advantages(
    rewards: list[float],
) -> tuple[list[float], AdvantageStats]:
    """Centre and scale the round's learner rewards to unit variance.

    Returns ``(advantages, stats)`` in reward order. A **degenerate** round
    (fewer than two surviving rewards, or a reward std below ``1e-8``) returns
    no advantages and ``stats.degenerate``: the caller takes no optimizer step,
    so the weights do not move (FR-023). This is the guard that keeps a
    zero-variance round from dividing by zero and amplifying float noise into
    arbitrary advantages.
    """
    n = len(rewards)
    if n < 2:
        return [], AdvantageStats(
            reward_mean=float(np.mean(rewards)) if n else 0.0,
            degenerate=True,
            reason=f"{n} surviving learner reward{'' if n == 1 else 's'}",
        )
    arr = np.asarray(rewards, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < _ADV_STD_EPS:
        return [], AdvantageStats(
            reward_mean=mean, reward_std=std, degenerate=True,
            reason="zero-variance learner rewards",
        )
    adv = (arr - mean) / std
    abs_adv = np.abs(adv)
    return [float(a) for a in adv], AdvantageStats(
        reward_mean=mean,
        reward_std=std,
        adv_std=float(adv.std()),
        adv_near_zero_frac=float((abs_adv < _ADV_NEAR_ZERO).mean()),
        adv_large_frac=float((abs_adv > _ADV_LARGE).mean()),
        adv_absmax=float(abs_adv.max()),
    )


def assign_advantages(batch: RoundBatch) -> AdvantageStats:
    """Standardise the round's rewards and stamp each seat's shared ``A``.

    The reward is terminal (γ=1), so one detached scalar covers all of a seat's
    picks. On a degenerate round every advantage stays ``0.0`` and the caller
    skips the update entirely.
    """
    advantages, stats = standardize_round_advantages(batch.learner_rewards)
    if stats.degenerate:
        return stats
    for seat_examples, advantage in zip(batch.seat_examples, advantages):
        for example in seat_examples:
            example.advantage = advantage
    return stats


# --------------------------------------------------------------------------- #
# Loss (data-model §5, FR-010)
# --------------------------------------------------------------------------- #

def _compute_loss(
    model: DraftAgentModel, batch: _Batch, temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``-mean(A · logπ_T(a|s))`` over the batch. Returns ``(loss, policy_logits)``.

    The single term: no critic/value regression, no GAE, no KL anchor, no
    entropy bonus. The critic head is forwarded (the model returns it) and
    discarded, so it receives no gradient and is carried through the run
    unchanged (FR-027).
    """
    policy_logits, _critic = _forward(model, batch)
    logp = masked_log_softmax(policy_logits / temperature, batch.pack_mask)
    taken = logp.gather(1, batch.action_token.unsqueeze(1)).squeeze(1)
    loss = -(batch.advantage * taken).mean()
    return loss, policy_logits


# --------------------------------------------------------------------------- #
# Diagnostics (data-model §6, research D9)
# --------------------------------------------------------------------------- #

@dataclass
class SweepDiagnostics:
    """Exploration + movement figures from the one post-update no-grad pass."""

    entropy: float = 0.0
    perplexity: float = 1.0
    off_argmax_rate: float = 0.0
    mean_logp: float = 0.0
    kl_prev_new: float = 0.0


def diagnostics_sweep(
    prev_model: DraftAgentModel,
    model: DraftAgentModel,
    examples: list[OnlineExample],
    table: np.ndarray,
    *,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> SweepDiagnostics:
    """One batched ``no_grad`` sweep over the round's picks forwarding both models.

    ``prev_model`` holds πₖ — the exact policy that generated the round — so the
    entropy, perplexity and off-argmax rate are exact rather than blurred across
    a pass whose weights were moving. The same sweep yields ``KL(πₖ ‖ πₖ₊₁)``,
    which no pre-update measurement could.

    **Both models must be in ``eval()``**: with dropout active these figures
    measure noise rather than the policy. The caller owns mode, since it also
    owns the update that precedes this call.
    """
    n = len(examples)
    if n == 0:
        return SweepDiagnostics()

    entropy_sum = 0.0
    logp_sum = 0.0
    kl_sum = 0.0
    off_argmax = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = examples[start:start + batch_size]
            batch = _collate(chunk, table, device)
            prev_logits, _ = _forward(prev_model, batch)
            new_logits, _ = _forward(model, batch)
            prev_scaled = prev_logits / temperature
            new_scaled = new_logits / temperature

            entropy_sum += float(policy_entropy(prev_scaled, batch.pack_mask).sum())
            logp = masked_log_softmax(prev_scaled, batch.pack_mask)
            actions = batch.action_token.unsqueeze(1)
            logp_sum += float(logp.gather(1, actions).squeeze(1).sum())
            argmax = prev_scaled.masked_fill(
                ~batch.pack_mask, float("-inf"),
            ).argmax(dim=-1)
            off_argmax += int((argmax != batch.action_token).sum())
            kl_sum += float(
                kl_divergence(prev_scaled, new_scaled, batch.pack_mask).sum()
            )

    entropy = entropy_sum / n
    return SweepDiagnostics(
        entropy=entropy,
        perplexity=math.exp(entropy),
        off_argmax_rate=off_argmax / n,
        mean_logp=logp_sum / n,
        kl_prev_new=kl_sum / n,
    )


# --------------------------------------------------------------------------- #
# Anchor window (data-model §7, FR-017/FR-019/FR-021)
# --------------------------------------------------------------------------- #

class AnchorWindow:
    """Sliding window of per-draft, per-label ``deck_score``s behind the margin.

    The anchor and learner labels are bound once at construction and never
    re-bound during a run (FR-021) — the moment the anchor moves, the margin
    stops meaning "improvement over a fixed point".
    """

    def __init__(self, maxlen: int, learner_label: str, anchor_label: str) -> None:
        self._window: deque[dict[str, list[float]]] = deque(maxlen=maxlen)
        self.learner_label = learner_label
        self.anchor_label = anchor_label
        self.best_margin: float | None = None
        self.best_round: int | None = None

    def add(self, record: DraftRecord) -> None:
        """Append one draft's scored seats, evicting the oldest when full."""
        entry: dict[str, list[float]] = {}
        for seat in record.seats:
            if seat.deck_score is None:
                continue  # failed build — excluded from every mean
            entry.setdefault(seat.agent, []).append(float(seat.deck_score))
        self._window.append(entry)

    @property
    def window_drafts(self) -> int:
        return len(self._window)

    def label_mean(self, label: str) -> float | None:
        """Mean ``deck_score`` for ``label`` across the window; ``None`` if unseen."""
        total = 0.0
        count = 0
        for entry in self._window:
            for score in entry.get(label, ()):
                total += score
                count += 1
        return total / count if count else None

    def label_means(self) -> dict[str, float]:
        """Every label's windowed mean, for the ``progress`` line's raw components."""
        totals: dict[str, list[float]] = {}
        for entry in self._window:
            for label, scores in entry.items():
                totals.setdefault(label, []).extend(scores)
        return {
            label: sum(scores) / len(scores)
            for label, scores in totals.items() if scores
        }

    @property
    def margin(self) -> float | None:
        """``mean(learner) − mean(anchor)``; ``None`` until both are populated."""
        learner = self.label_mean(self.learner_label)
        anchor = self.label_mean(self.anchor_label)
        if learner is None or anchor is None:
            return None
        return learner - anchor

    def observe_round(self, round_index: int) -> None:
        """Record this round's margin as the new best if it is."""
        margin = self.margin
        if margin is None:
            return
        if self.best_margin is None or margin > self.best_margin:
            self.best_margin = margin
            self.best_round = round_index


# --------------------------------------------------------------------------- #
# The one-pass update (FR-011, research D8)
# --------------------------------------------------------------------------- #

@dataclass
class UpdateResult:
    """What one round's pass over its own fresh batch produced."""

    policy_loss: float = 0.0
    grad_norm: float = 0.0
    steps: int = 0


def run_update_pass(
    model: DraftAgentModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batch: RoundBatch,
    *,
    temperature: float,
    batch_size: int,
    max_grad_norm: float,
    device: torch.device,
    rng: random.Random,
) -> UpdateResult:
    """One shuffled, length-bucketed pass over the round's learner picks.

    Exactly one epoch — the batch is then discarded and never revisited
    (FR-011). The mild within-pass off-policyness (later minibatches see
    already-stepped weights) is the accepted trade-off for getting enough
    gradient steps out of a round; ``KL(prev‖new)`` in the movement diagnostics
    exists to size it.
    """
    result = UpdateResult()
    if not batch.examples:
        return result

    model.train()
    minibatches = length_bucketed_batches(batch.examples, batch_size, rng)
    loss_sum = 0.0
    norm_sum = 0.0
    for chunk in minibatches:
        collated = _collate(chunk, batch.table, device)
        loss, _ = _compute_loss(model, collated, temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Pre-clip norms are the movement signal; post-clip ones are bounded by
        # max_grad_norm and carry no shape information.
        norms = clip_per_group(optimizer, max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        loss_sum += float(loss.detach())
        norm_sum += max(norms.values()) if norms else 0.0
        result.steps += 1

    if result.steps:
        result.policy_loss = loss_sum / result.steps
        result.grad_norm = norm_sum / result.steps
    return result


# --------------------------------------------------------------------------- #
# Per-round diagnostics record + stdout formatting (contract §2, FR-014…FR-018)
# --------------------------------------------------------------------------- #

@dataclass
class RoundDiagnostics:
    """Everything printed for one round (data-model §6)."""

    learner_label: str = ""
    anchor_label: str = ""
    index: int = 0
    drafts: int = 0
    total_drafts: int = 0
    learner_seats: int = 0
    dropped_seats: int = 0
    learner_picks: int = 0
    gen_seconds: float = 0.0
    train_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None
    reward: AdvantageStats = field(default_factory=AdvantageStats)
    sweep: SweepDiagnostics = field(default_factory=SweepDiagnostics)
    policy_loss: float = 0.0
    grad_norm: float = 0.0
    anchor_margin: float | None = None
    label_means: dict[str, float] = field(default_factory=dict)
    window_drafts: int = 0


def _fmt_margin(margin: float | None) -> str:
    return "n/a" if margin is None else f"{margin:+.3f}"


def _ordered_label_means(diag: "RoundDiagnostics") -> str:
    """Learner first, then the anchor, then the rest — the margin's two terms
    sit side by side rather than being separated by alphabetical order."""
    lead = [
        label for label in (diag.learner_label, diag.anchor_label)
        if label in diag.label_means
    ]
    rest = sorted(label for label in diag.label_means if label not in lead)
    return " ".join(f"{label}={diag.label_means[label]:.3f}" for label in lead + rest)


def format_round_lines(diag: RoundDiagnostics) -> list[str]:
    """The round's stdout block: one summary line + four detail lines.

    A degenerate round (FR-023) collapses to a single summary line marked
    ``skipped (no signal)`` — it still carries the anchor margin, so no round is
    ever silent (SC-003).
    """
    head = (
        f"round {diag.index} | "
        f"drafts {diag.drafts} ({diag.total_drafts})"
    )
    if diag.skipped:
        return [
            f"{head} | skipped (no signal): {diag.skip_reason} | "
            f"margin {_fmt_margin(diag.anchor_margin)}"
        ]

    reward, sweep = diag.reward, diag.sweep
    summary = (
        f"{head} | "
        f"picks {diag.learner_picks} ({diag.dropped_seats} seats dropped) | "
        f"gen {diag.gen_seconds:.0f}s train {diag.train_seconds:.0f}s | "
        f"R {reward.reward_mean:+.3f}+-{reward.reward_std:.3f} | "
        f"|A|<0.1 {reward.adv_near_zero_frac * 100:.0f}% | "
        f"ppl {sweep.perplexity:.2f} | "
        f"KL {sweep.kl_prev_new:.4f} | "
        f"margin {_fmt_margin(diag.anchor_margin)}"
    )
    label_means = _ordered_label_means(diag)
    return [
        summary,
        (
            f"  reward   : learner seats={diag.learner_seats} "
            f"R mean={reward.reward_mean:+.3f} std={reward.reward_std:.3f} | "
            f"A std={reward.adv_std:.3f} "
            f"|A|<0.1={reward.adv_near_zero_frac * 100:.1f}% "
            f"|A|>0.5={reward.adv_large_frac * 100:.1f}% "
            f"max|A|={reward.adv_absmax:.2f}"
        ),
        (
            f"  explore  : H={sweep.entropy:.3f} ppl={sweep.perplexity:.3f} "
            f"off-argmax={sweep.off_argmax_rate * 100:.1f}%   "
            "(band: ppl 2-3 / off-argmax 25-40%)"
        ),
        (
            f"  movement : mean logpi={sweep.mean_logp:.3f} "
            f"policy_loss={diag.policy_loss:+.4f} "
            f"grad_norm={diag.grad_norm:.2f} "
            f"KL(prev||new)={sweep.kl_prev_new:.5f}"
        ),
        (
            f"  progress : anchor margin={_fmt_margin(diag.anchor_margin)} | "
            f"{label_means or 'no scored seats yet'} | "
            f"window={diag.window_drafts} drafts"
        ),
    ]


def format_startup_echo(
    config: TrainDraftAgentOnlineConfig,
    *,
    run_id: str,
    anchor: str,
    generation: int,
    device: torch.device,
    embedding_dim: int,
) -> list[str]:
    """The resolved run configuration, echoed once before the worker launches.

    ``generation`` is the **lineage counter** read from the base checkpoint, not
    the ``--learner`` mix label: a run labeled ``gen-3`` warm-started from a
    gen-1 base prints ``generation 1 -> 2``. The label names a kind of seat; the
    counter records how many training generations deep the weights are.
    """
    from draft.application.agent_mix import format_agent_mix

    lines = [
        f"Online GRPO run {run_id}: generation {generation} -> {generation + 1}",
        f"  learner   : {config.learner_label} <- {config.learner_checkpoint}",
    ]
    for label in sorted(config.frozen):
        suffix = "  (anchor)" if label == anchor else ""
        lines.append(f"  frozen    : {label} <- {config.frozen[label]}{suffix}")
    reward = f"  reward    : scorer {config.scorer_checkpoint} | build-method {config.build_method}"
    if config.build_method == "picker":
        reward += f" | picker {config.picker_checkpoint}"
    lines += [
        f"  mix       : {format_agent_mix(config.mix)}  (>=1 learner seat forced)",
        reward,
        (
            f"  rollout   : T={config.rollout_temperature} | "
            f"drafts/round={config.drafts_per_round} | "
            f"set={config.set_code or 'random'} | seed={config.seed} "
            "(Forge-side rollouts unseeded)"
        ),
        (
            f"  optimiser : lr={config.lr:.0e} batch={config.batch_size} "
            f"clip={config.max_grad_norm} warmup={config.warmup_steps} steps"
        ),
        (
            f"  runtime   : device {device} | embedding width {embedding_dim} | "
            f"anchor window {config.anchor_window} drafts"
        ),
        (
            f"  outputs   : corpus {config.output_path} (append) | "
            f"checkpoints {CHECKPOINT_DIR}/ "
            f"(snapshot every {config.snapshot_every} rounds)"
        ),
    ]
    return lines


def format_final_summary(
    *,
    rounds: int,
    total_drafts: int,
    learner_picks: int,
    elapsed: float,
    latest_path: Path,
    snapshot_path: Path | None,
    window: AnchorWindow,
) -> list[str]:
    """The FR-019 closing block, printed on every one of the three exit paths."""
    lines = [
        f"Done after {rounds} rounds | {total_drafts} drafts | "
        f"{learner_picks} learner picks | {_fmt_dur(elapsed)}",
        f"  latest checkpoint : {latest_path}",
    ]
    if snapshot_path is not None:
        lines.append(f"  final snapshot    : {snapshot_path}")
    if window.best_margin is None:
        lines.append("  best anchor margin: n/a (never defined during the run)")
    else:
        lines.append(
            f"  best anchor margin: {window.best_margin:+.3f} at round "
            f"{window.best_round} (current {_fmt_margin(window.margin)})"
        )
    return lines


# --------------------------------------------------------------------------- #
# Checkpoint payload (data-model §8, FR-027/FR-028)
# --------------------------------------------------------------------------- #

def _stringify(value: Any) -> Any:
    """Recursively render Paths as strings so the config survives torch.save."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify(item) for item in value]
    return value


def base_generation(rl_metadata: dict[str, Any] | None) -> int:
    """The base checkpoint's lineage counter (gen-1 checkpoints carry none)."""
    if not rl_metadata:
        return 1
    return int(rl_metadata.get("generation", 1))


def build_rl_metadata(
    config: TrainDraftAgentOnlineConfig, generation: int,
) -> dict[str, Any]:
    """Gen-3's self-describing metadata — no critic/GAE/KL/entropy knobs exist."""
    return {
        "generation": generation + 1,
        "base_checkpoint": str(config.learner_checkpoint),
        "algorithm": "online-grpo",
        "lr": config.lr,
        "rollout_temperature": config.rollout_temperature,
        "drafts_per_round": config.drafts_per_round,
    }


def _write_checkpoint(
    store: DraftAgentStore,
    model: DraftAgentModel,
    optimizer: torch.optim.Optimizer,
    config: TrainDraftAgentOnlineConfig,
    base: Any,
    generation: int,
    round_index: int,
    path: Path,
) -> None:
    """Write one gen-1-format checkpoint carrying the critic head unchanged.

    ``best_val_loss`` is ``inf`` because no held-out metric exists (FR-026), and
    ``critic_mean``/``critic_std`` are copied verbatim from the base so the
    (untrained) critic still de-standardizes to raw scorer-score space.
    """
    store.save_checkpoint(
        model,
        optimizer,
        round_index,
        float("inf"),
        base.config,
        path,
        critic_mean=base.critic_mean,
        critic_std=base.critic_std,
        train_config=_stringify(asdict(config)),
        rl_metadata=build_rl_metadata(config, generation),
    )


# --------------------------------------------------------------------------- #
# The loop (FR-005, FR-029, FR-031)
# --------------------------------------------------------------------------- #

class TrainDraftAgentOnlineUseCase:
    """Owns the whole online loop in one process (FR-029).

    At startup it validates, loads the learner, builds one shared card locator,
    one deck labeler, one pick registry (learner served by the *live* model), and
    launches one resident Forge worker. Each round it pulls
    ``--drafts-per-round`` fresh records from the suspended record stream,
    appends them to the corpus, applies the one-pass GRPO update, writes
    ``latest.pt``, and drafts the next round with the updated weights.
    """

    def execute(self, config: TrainDraftAgentOnlineConfig) -> int:
        """Run until ``--max-rounds``, an interrupt, or the pick-fault abort.

        Returns the number of completed rounds.
        """
        from draft.application.agent_pick_service import AgentPickService
        from draft.application.agent_registry import AgentRegistry
        from draft.application.generate_draft_data import (
            GenerateDraftDataSupervisor,
            build_labeler,
        )
        from draft.application.train_draft_agent import _make_scheduler, _select_device
        from draft.infrastructure.draft_record_io import append_record
        from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

        anchor = validate_config(config)
        temperature = float(config.rollout_temperature)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        rng = random.Random(config.seed)

        # --- learner: the model the loop optimises AND the model that drafts ---
        store = DraftAgentStore()
        base = store.load_checkpoint(config.learner_checkpoint)
        embedding_dim = check_embedding_width(base.config, config)
        device = _select_device()
        model = DraftAgentModel(base.config)
        model.load_state_dict(base.model_state_dict)
        model.to(device)
        model.eval()
        # πₖ for the movement diagnostics: resident, never in train mode.
        prev_model = DraftAgentModel(base.config)
        prev_model.to(device)
        prev_model.eval()
        for parameter in prev_model.parameters():
            parameter.requires_grad_(False)

        # One memoizing locator for the labeler, every pick service, and the
        # trainer, so each card's .npz is decompressed once for the whole run.
        locator = ConvertedCardLocator(config.cards_path)
        loader = RoundLoader(locator)

        generation = base_generation(base.rl_metadata)

        optimizer = torch.optim.AdamW(
            [{"params": list(model.parameters()), "lr": config.lr, "name": "agent"}],
        )
        # One warmup at the start of the run, then constant — never rebuilt per
        # round, so the LR schedule and optimizer moments are continuous across
        # rounds (FR-025). An online run has no total step count, so the ramp is
        # expressed in optimizer steps rather than a fraction of a horizon (D15).
        scheduler = _make_scheduler(
            optimizer, total_steps=config.warmup_steps, warmup_frac=1.0,
            controller=None,
        )

        gen_config = self._generation_config(config, temperature)
        labeler = build_labeler(gen_config, locator=locator)
        learner_service = AgentPickService.from_model(
            model, base.config, locator, device=device,
            pick_mode="sample", temperature=temperature, seed=config.seed,
        )
        registry = AgentRegistry.build(
            dict(config.frozen),
            {label for label, _ in config.mix},
            locator=locator,
            pick_mode="sample",
            temperature=temperature,
            seed=config.seed,
            device=device,
            preloaded={config.learner_label: learner_service},
        )
        supervisor = GenerateDraftDataSupervisor(
            gen_config, labeler=labeler, registry=registry,
        )

        for line in format_startup_echo(
            config, run_id=supervisor.run_id, anchor=anchor, generation=generation,
            device=device, embedding_dim=embedding_dim,
        ):
            _log(line)

        window = AnchorWindow(
            config.anchor_window, config.learner_label, anchor,
        )
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        latest_path = CHECKPOINT_DIR / "latest.pt"

        supervisor._install_signal_handlers()
        launch = supervisor._default_launch_worker(registry.external_labels)
        records = supervisor.iter_records(launch, labeler)

        rounds = 0
        total_drafts = 0
        total_picks = 0
        snapshot_path: Path | None = None
        start_time = time.monotonic()
        # Append-only, never "w": output/draft/drafts.jsonl is the canonical
        # shared corpus and truncating it would destroy the gen-1 / live-play
        # data (FR-020).
        with open(config.output_path, "a", buffering=1, encoding="utf-8") as out:
            try:
                while config.max_rounds is None or rounds < config.max_rounds:
                    gen_start = time.monotonic()
                    model.eval()  # generation: the live policy drafts
                    fresh = self._pull_round(
                        records, config.drafts_per_round, out, window, append_record,
                    )
                    if not fresh:
                        break  # the stream ended (shutdown requested)
                    gen_seconds = time.monotonic() - gen_start
                    total_drafts += len(fresh)

                    train_start = time.monotonic()
                    batch = loader.build(fresh, config.learner_label)
                    batch.index = rounds
                    batch.gen_seconds = gen_seconds
                    stats = assign_advantages(batch)

                    diag = RoundDiagnostics(
                        learner_label=config.learner_label,
                        anchor_label=anchor,
                        index=rounds,
                        drafts=len(fresh),
                        total_drafts=total_drafts,
                        learner_seats=len(batch.learner_rewards),
                        dropped_seats=batch.dropped_seats,
                        learner_picks=len(batch.examples),
                        gen_seconds=gen_seconds,
                        reward=stats,
                    )
                    if stats.degenerate:
                        diag.skipped = True
                        diag.skip_reason = stats.reason
                    else:
                        total_picks += len(batch.examples)
                        prev_model.load_state_dict(model.state_dict())
                        update = run_update_pass(
                            model, optimizer, scheduler, batch,
                            temperature=temperature,
                            batch_size=config.batch_size,
                            max_grad_norm=config.max_grad_norm,
                            device=device,
                            rng=rng,
                        )
                        # Back to eval before the sweep: with dropout active the
                        # entropy/off-argmax/KL figures would measure noise.
                        model.eval()
                        diag.policy_loss = update.policy_loss
                        diag.grad_norm = update.grad_norm
                        diag.sweep = diagnostics_sweep(
                            prev_model, model, batch.examples, batch.table,
                            batch_size=config.batch_size,
                            temperature=temperature,
                            device=device,
                        )
                    diag.train_seconds = time.monotonic() - train_start

                    window.observe_round(rounds)
                    diag.anchor_margin = window.margin
                    diag.label_means = window.label_means()
                    diag.window_drafts = window.window_drafts
                    for line in format_round_lines(diag):
                        _log(line)

                    rounds += 1
                    _write_checkpoint(
                        store, model, optimizer, config, base, generation,
                        rounds, latest_path,
                    )
                    _log(f"saved {latest_path} (round {rounds - 1})")
                    if rounds % config.snapshot_every == 0:
                        snapshot_path = self._snapshot(
                            store, model, optimizer, config, base, generation, rounds,
                        )
                    if supervisor._shutdown.is_set():
                        break
            finally:
                records.close()

        snapshot_path = self._snapshot(
            store, model, optimizer, config, base, generation, rounds,
        )
        for line in format_final_summary(
            rounds=rounds, total_drafts=total_drafts, learner_picks=total_picks,
            elapsed=time.monotonic() - start_time, latest_path=latest_path,
            snapshot_path=snapshot_path, window=window,
        ):
            _log(line)
        return rounds

    @staticmethod
    def _generation_config(
        config: TrainDraftAgentOnlineConfig, temperature: float,
    ) -> Any:
        """The supervisor's config, every field set explicitly.

        Not relying on ``GenerateDraftDataConfig``'s defaults is deliberate: its
        ``build_method`` defaults to ``"picker"`` while gen-3's default is
        ``"greedy"``, so an omitted field would silently label every seat with a
        different builder than the operator asked for.
        """
        from draft.application.generate_draft_data import GenerateDraftDataConfig

        return GenerateDraftDataConfig(
            n_drafts=0,  # unused: the online loop consumes iter_records directly
            agent_mix=config.mix,
            set_code=config.set_code,
            scorer_checkpoint=config.scorer_checkpoint,
            build_method=config.build_method,
            picker_checkpoint=config.picker_checkpoint,
            cards_path=config.cards_path,
            output_path=config.output_path,
            resume=True,
            agent_checkpoints=dict(config.frozen),
            pick_mode="sample",
            temperature=temperature,
            seed=config.seed,
            max_consecutive_faults=config.max_consecutive_faults,
            # The link that actually arms the Java rule: without it, learner-free
            # pods are drafted and trained on as if valid (FR-003).
            required_agent=config.learner_label,
        )

    @staticmethod
    def _pull_round(
        records: Iterator[DraftRecord],
        n: int,
        out: Any,
        window: AnchorWindow,
        append_record,
    ) -> list[DraftRecord]:
        """Pull ``n`` fresh records, persisting and windowing each as it arrives.

        Between rounds the generator stays suspended and the worker stays
        resident. Each record is appended the moment it arrives, so a round that
        later no-ops still leaves its drafts on disk.
        """
        fresh: list[DraftRecord] = []
        for record in records:
            append_record(out, record)
            window.add(record)
            fresh.append(record)
            if len(fresh) >= n:
                break
        return fresh

    @staticmethod
    def _snapshot(
        store: DraftAgentStore,
        model: DraftAgentModel,
        optimizer: torch.optim.Optimizer,
        config: TrainDraftAgentOnlineConfig,
        base: Any,
        generation: int,
        rounds: int,
    ) -> Path:
        path = CHECKPOINT_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        _write_checkpoint(
            store, model, optimizer, config, base, generation, rounds, path,
        )
        _log(f"snapshot {path} (round {rounds})")
        return path


def probe_embedding_width(cards_path: Path) -> int | None:
    """Width of the ``.npz`` cache, read from the first cached card found.

    The trainers that read a corpus learn this from the loader after ingesting
    it; an online run has no corpus at startup, so the cache is probed directly
    to get the width *before* the Forge worker launches.
    """
    for path in sorted(cards_path.rglob("*.npz")):
        try:
            with np.load(path) as data:
                return int(data["embedding"].shape[0])
        except Exception:
            continue  # a truncated/foreign .npz — try the next one
    return None


def check_embedding_width(
    agent_config: Any, config: TrainDraftAgentOnlineConfig,
) -> int:
    """Fail fast when the checkpoint and the ``.npz`` cache disagree (rule 7).

    Without this the mismatch would surface as a torch shape error deep inside
    the first round's forward — after Forge has already started and drafted.
    """
    from draft.application.train_draft_agent import _check_dims

    width = probe_embedding_width(config.cards_path)
    if width is None:
        raise OnlineConfigError(
            f"no .npz card embeddings found under {config.cards_path}; run "
            "`python -m sealed encode-cards` with the encoder this agent was "
            "trained on"
        )
    _check_dims(agent_config, width, config.cards_path)
    return width
