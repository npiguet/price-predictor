"""The ``train-picker`` deck sampler and the Plackett-Luce log-probability of
each sampled deck.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE


@dataclass(frozen=True)
class SamplingSpec:
    """How the sampler draws: ``n_samples`` decks per pool, from the logits
    divided by ``temperature``."""

    n_samples: int
    temperature: float


@dataclass(frozen=True)
class SampledDecks:
    """The decks drawn for a batch of pools: ``(B*S, K)`` pick indices, the
    mask of their real picks, and ``n_samples`` (S)."""

    pick_indices: torch.Tensor
    picked_mask: torch.Tensor
    n_samples: int


def _sample_decks(
    logits: torch.Tensor,
    is_land_mask: torch.Tensor,
    pool_mask: torch.Tensor,
    spec: SamplingSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-batched sequential without-replacement sampler (spec § 3.2, FR-013).

    Operates on the full ``(B*S, N)`` tensor in vectorized steps. Each row picks
    one card per step via ``torch.multinomial`` until its 23-spell quota fills
    (per-row stop; finished rows still draw but their picks are discarded).

    Returns ``(pick_indices (B*S, K), picked_mask (B*S, K))`` where ``K`` is the
    number of steps taken and ``picked_mask`` marks the real picks.
    """
    n_samples, temperature = spec.n_samples, spec.temperature
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


def _sampled_log_prob(
    logits: torch.Tensor, pool_mask: torch.Tensor, decks: SampledDecks, temperature: float,
) -> torch.Tensor:
    """The ``(B, S)`` PL log-prob of each sampled deck, on the logits scaled by
    ``temperature`` with the padding masked out, repeated per sample."""
    b, n = logits.shape
    n_samples = decks.n_samples
    scaled = (logits / temperature).masked_fill(
        ~pool_mask, float("-inf"),
    )
    scaled_bs = scaled.unsqueeze(1).expand(b, n_samples, n).reshape(
        b * n_samples, n,
    )
    return _plackett_luce_log_prob(
        scaled_bs, decks.pick_indices, decks.picked_mask,
    ).view(b, n_samples)
