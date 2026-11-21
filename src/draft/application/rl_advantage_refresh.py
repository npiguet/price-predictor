"""The per-epoch refresh of the critic values and the GAE(λ) advantages over every
learner trajectory (research D3), logged so the pass is not silent.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from draft.application.imitation_batches import BatchSource
from draft.application.rl_advantage import gae_advantages
from draft.application.rl_batches import _collate, _forward_logits_critic
from draft.application.rl_examples import RLExample
from draft.application.rl_log import _STEP_LOG_INTERVAL, _fmt_dur, _log
from draft.domain.draft_agent_model import DraftAgentModel

_GAMMA = 1.0  # draft reward is terminal, so the return is R (research D2).


@dataclass(frozen=True)
class CriticPass:
    """A batched no-grad critic forward: what its batches are built from and their size."""

    source: BatchSource
    batch_size: int


@dataclass(frozen=True)
class AdvantageRefresh:
    """The critic forward over every learner pick and the GAE(λ) that turns its
    values into advantages."""

    critic_pass: CriticPass
    gae_lambda: float


class _ApproxAdvantage:
    """Running MC-advantage approx (R_std − V) over picks processed so far, for the
    live progress line. It's the λ=1 form; the exact GAE(λ) summary is logged by
    the caller once the full forward is done. (Picks are length-sorted, so early
    lines are biased toward short/early-draft picks until the pass completes.)
    """

    def __init__(self) -> None:
        self.adv_sum = self.adv_sumsq = 0.0
        self.adv_n = self.adv_gt1 = self.adv_gt5 = 0

    def add(self, approx: np.ndarray) -> None:
        self.adv_sum += float(approx.sum())
        self.adv_sumsq += float((approx ** 2).sum())
        self.adv_n += approx.size
        abs_a = np.abs(approx)
        self.adv_gt1 += int((abs_a > 0.1).sum())
        self.adv_gt5 += int((abs_a > 0.5).sum())

    def summary(self) -> str:
        a_mean = self.adv_sum / self.adv_n
        a_std = math.sqrt(max(self.adv_sumsq / self.adv_n - a_mean ** 2, 0.0))
        return (
            f"adv≈ mean={a_mean:+.3f} std={a_std:.3f} "
            f"|A|>0.1={self.adv_gt1 / self.adv_n:.0%} |A|>0.5={self.adv_gt5 / self.adv_n:.0%}"
        )


class _PassProgress:
    """A progress line every ``_STEP_LOG_INTERVAL`` when the pass has a label."""

    def __init__(self, log_label: str | None, n: int) -> None:
        self._label = log_label
        self._n = n
        self._start = time.monotonic()
        self._last_log = self._start

    def after_batch(self, end: int, approx: _ApproxAdvantage) -> None:
        """Called once the picks before ``end`` (in pass order) are done."""
        now = time.monotonic()
        if self._label is not None and now - self._last_log >= _STEP_LOG_INTERVAL:
            self._log_line(min(end, self._n), now, approx)

    def _log_line(self, done: int, now: float, approx: _ApproxAdvantage) -> None:
        n = self._n
        rate = done / (now - self._start) if now > self._start else 0.0
        eta = (n - done) / rate if rate > 0 else 0.0
        _log(
            f"  {self._label}: {done}/{n} picks ({rate:.0f}/s, ETA "
            f"{_fmt_dur(eta)}) | {approx.summary()}"
        )
        self._last_log = now


def _record_values(
    chunk: list[RLExample], vals: np.ndarray, source: BatchSource, approx: _ApproxAdvantage,
) -> None:
    """Write the batch's critic values onto its examples and add their (R_std − V)."""
    for j, ex in enumerate(chunk):
        ex.value = float(vals[j])
    rewards = np.asarray([ex.reward for ex in chunk], dtype=np.float64)
    approx.add((rewards - source.mean) / source.std - vals.astype(np.float64))


def _critic_values(
    model: DraftAgentModel, examples: list[RLExample], critic_pass: CriticPass,
    *, log_label: str | None = None,
) -> None:
    """Write the current critic value onto each example (batched, no_grad).

    Logs a progress line every ``_STEP_LOG_INTERVAL`` when ``log_label`` is set
    (this is otherwise a long silent pass on a large corpus).
    """
    order = sorted(range(len(examples)), key=lambda i: examples[i].n_tokens)
    batch_size = critic_pass.batch_size
    model.eval()
    progress = _PassProgress(log_label, len(order))
    approx = _ApproxAdvantage()
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            chunk = [examples[i] for i in order[s:s + batch_size]]
            batch = _collate(chunk, critic_pass.source)
            _, critic = _forward_logits_critic(model, batch)
            vals = critic.cpu().numpy()  # one device→host transfer per batch
            _record_values(chunk, vals, critic_pass.source, approx)
            progress.after_batch(s + batch_size, approx)
    model.train()


def _precompute_advantages(
    model: DraftAgentModel,
    trajectories: list[list[RLExample]],
    refresh: AdvantageRefresh,
    *, epoch: int | None = None,
) -> None:
    """Refresh GAE advantages over every learner trajectory (research D3).

    Needs the critic value at *every* learner pick (each feeds the GAE over its
    full trajectory → the policy gradient), so — unlike the behaviour check — it
    cannot be subsampled. It logs start/progress/done so the pass is not silent.
    """
    flat = [ex for traj in trajectories for ex in traj]
    if not flat:
        return
    label = f"epoch {epoch} GAE precompute" if epoch is not None else "GAE precompute"
    _log(f"{label}: critic forward over {len(flat)} learner picks...")
    t0 = time.monotonic()
    _critic_values(model, flat, refresh.critic_pass, log_label=label)
    source = refresh.critic_pass.source
    adv_chunks: list[np.ndarray] = []
    for traj in trajectories:
        values = np.asarray([ex.value for ex in traj], dtype=np.float64)
        reward_std = (traj[0].reward - source.mean) / source.std
        adv = gae_advantages(values, reward_std, refresh.gae_lambda, _GAMMA)
        for ex, a in zip(traj, adv):
            ex.advantage = float(a)
        adv_chunks.append(adv)
    # Advantage distribution = the learnable signal. std is the effective signal
    # magnitude (in standardized-reward units); the |A|>t fractions show how many
    # picks carry actionable signal. Near-zero std/fractions ⇒ little to learn at
    # this rollout temperature (spec 020 diagnostics).
    all_adv = np.concatenate(adv_chunks) if adv_chunks else np.zeros(0)
    if all_adv.size:
        abs_adv = np.abs(all_adv)
        _log(
            f"{label}: advantage mean={all_adv.mean():+.3f} std={all_adv.std():.3f} "
            f"|A|>0.1={np.mean(abs_adv > 0.1):.1%} |A|>0.5={np.mean(abs_adv > 0.5):.1%} "
            f"max|A|={abs_adv.max():.2f}"
        )
    _log(f"{label}: done in {_fmt_dur(time.monotonic() - t0)}")
