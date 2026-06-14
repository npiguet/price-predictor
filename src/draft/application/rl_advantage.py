"""Generalized Advantage Estimation for a terminal-reward draft trajectory.

A draft seat's reward is **terminal**: a single scalar `R` (the pod-relative
leave-one-out `deck_score`) arrives after the last pick; intermediate picks have
no reward. Given the critic's value at each of the seat's ordered picks, GAE(λ)
turns that into a per-pick advantage `A_t` (spec 020 FR-011; research D2).

With `γ = 1` (draft reward is terminal, so the return is `R` for every pick):

- `δ_t = γ·V(s_{t+1}) − V(s_t)` for non-terminal picks, `δ_T = R − V(s_T)`,
- `A_t = Σ_{l≥0} (γλ)^l δ_{t+l}`.

At `λ = 1` this telescopes to the Monte-Carlo advantage `A_t = R − V(s_t)`
(matching the gen-1 critic's MC target); at `λ = 0` it is the one-step TD residual
`δ_t`. The critic itself regresses toward the MC return `R` (research D2), so this
module returns advantages only — the return target is simply `R`.

Pure numpy, no torch.
"""

from __future__ import annotations

import numpy as np


def gae_advantages(
    values: np.ndarray,
    reward: float,
    gae_lambda: float,
    gamma: float = 1.0,
) -> np.ndarray:
    """Per-pick GAE(λ) advantages for one ordered trajectory.

    ``values`` is ``(T,)`` — the critic value at each of the seat's picks in
    (pack, pick) order. ``reward`` is the terminal scalar ``R``. Returns a
    ``(T,)`` float64 array of detached advantages. An empty trajectory returns an
    empty array.
    """
    t = int(np.asarray(values).shape[0])
    adv = np.zeros(t, dtype=np.float64)
    if t == 0:
        return adv
    v = np.asarray(values, dtype=np.float64)
    gae = 0.0
    for i in range(t - 1, -1, -1):
        v_next = 0.0 if i == t - 1 else v[i + 1]
        r = reward if i == t - 1 else 0.0
        delta = r + gamma * v_next - v[i]
        gae = delta + gamma * gae_lambda * gae
        adv[i] = gae
    return adv
