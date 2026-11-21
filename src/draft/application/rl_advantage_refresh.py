"""The per-epoch refresh of the critic values and the GAE(λ) advantages over every
learner trajectory (research D3), logged so the pass is not silent.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from draft.application.rl_advantage import gae_advantages
from draft.application.rl_batches import _collate, _forward_logits_critic
from draft.application.rl_examples import RLExample
from draft.application.rl_log import _STEP_LOG_INTERVAL, _fmt_dur, _log
from draft.domain.draft_agent_model import DraftAgentModel

_GAMMA = 1.0  # draft reward is terminal, so the return is R (research D2).


def _critic_values(
    model: DraftAgentModel, examples: list[RLExample], table: np.ndarray,
    mean: float, std: float, batch_size: int, device: torch.device,
    *, log_label: str | None = None,
) -> None:
    """Write the current critic value onto each example (batched, no_grad).

    Logs a progress line every ``_STEP_LOG_INTERVAL`` when ``log_label`` is set
    (this is otherwise a long silent pass on a large corpus).
    """
    order = sorted(range(len(examples)), key=lambda i: examples[i].n_tokens)
    n = len(order)
    model.eval()
    start = time.monotonic()
    last_log = start
    # Running MC-advantage approx (R_std − V) over picks processed so far, for the
    # live progress line. It's the λ=1 form; the exact GAE(λ) summary is logged by
    # the caller once the full forward is done. (Picks are length-sorted, so early
    # lines are biased toward short/early-draft picks until the pass completes.)
    adv_sum = adv_sumsq = 0.0
    adv_n = adv_gt1 = adv_gt5 = 0
    with torch.no_grad():
        for s in range(0, n, batch_size):
            idxs = order[s:s + batch_size]
            batch = _collate([examples[i] for i in idxs], table, mean, std, device)
            _, critic = _forward_logits_critic(model, batch)
            vals = critic.cpu().numpy()  # one device→host transfer per batch
            for j, i in enumerate(idxs):
                examples[i].value = float(vals[j])
            rewards = np.asarray([examples[i].reward for i in idxs], dtype=np.float64)
            approx = (rewards - mean) / std - vals.astype(np.float64)
            adv_sum += float(approx.sum())
            adv_sumsq += float((approx ** 2).sum())
            adv_n += approx.size
            abs_a = np.abs(approx)
            adv_gt1 += int((abs_a > 0.1).sum())
            adv_gt5 += int((abs_a > 0.5).sum())
            now = time.monotonic()
            if log_label is not None and now - last_log >= _STEP_LOG_INTERVAL:
                done = min(s + batch_size, n)
                rate = done / (now - start) if now > start else 0.0
                eta = (n - done) / rate if rate > 0 else 0.0
                a_mean = adv_sum / adv_n
                a_std = math.sqrt(max(adv_sumsq / adv_n - a_mean ** 2, 0.0))
                _log(
                    f"  {log_label}: {done}/{n} picks ({rate:.0f}/s, ETA "
                    f"{_fmt_dur(eta)}) | adv≈ mean={a_mean:+.3f} std={a_std:.3f} "
                    f"|A|>0.1={adv_gt1 / adv_n:.0%} |A|>0.5={adv_gt5 / adv_n:.0%}"
                )
                last_log = now
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
    _critic_values(model, flat, table, mean, std, batch_size, device, log_label=label)
    adv_chunks: list[np.ndarray] = []
    for traj in trajectories:
        values = np.asarray([ex.value for ex in traj], dtype=np.float64)
        reward_std = (traj[0].reward - mean) / std
        adv = gae_advantages(values, reward_std, gae_lambda, _GAMMA)
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
