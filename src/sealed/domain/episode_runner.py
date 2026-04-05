"""EpisodeRunner: runs one episode of the sealed pick game."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from sealed.domain.replay_buffer import Episode

if TYPE_CHECKING:
    from sealed.domain.card_embedding_port import CardEmbeddingPort

BASIC_LAND_NAMES = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]
MAX_PICKS = 40
IDEAL_SPELLS = 23  # target non-land picks for a 40-card sealed deck
IDEAL_LANDS = 17   # target land picks for a 40-card sealed deck


def _build_base_tensor(
    pool_names: str,
    card_port: "CardEmbeddingPort",
    n_slots: int,
    d_model: int | None = None,
) -> np.ndarray:
    """Build initial pool tensor for an episode.

    Returns float32 ndarray of shape [n_slots, d_model].

    d_model should be passed from model.config.d_model so that the tensor
    matches the Pool Transformer's expected input size exactly. When not
    provided, falls back to embed_dim + 8 (correct only when embed_dim + 8
    is already divisible by n_heads).

    Flag layout (indices embed_dim .. embed_dim+7):
      +0  pick_count   — how many times this slot has been picked (0/1 for booster, 0..N for basic lands)
      +1  available    — 1 if still available to pick (booster: cleared after first pick; basic: always 1)
      +2..+7  (reserved/padding — always 0)

    Initial state:
      Booster slots  (0 .. n_booster-1): pick_count=0, available=1.
      Basic land slots (n_booster .. n_slots-1): pick_count=0, available=1.
    """
    names = pool_names.split(";")
    n_booster = len(names)
    n_basic = n_slots - n_booster
    if n_basic < 0:
        raise ValueError(f"n_slots={n_slots} < n_booster={n_booster}")

    booster_embeds = [card_port.get_embedding(name) for name in names]
    basic_embeds = [card_port.get_embedding(BASIC_LAND_NAMES[i]) for i in range(n_basic)]

    all_embeds = booster_embeds + basic_embeds
    embed_dim = all_embeds[0].shape[0] if all_embeds else 0
    if d_model is None:
        d_model = embed_dim + 8

    tensor = np.zeros((n_slots, d_model), dtype=np.float32)

    for i, emb in enumerate(booster_embeds):
        tensor[i, :embed_dim] = emb
        tensor[i, embed_dim + 1] = 1.0  # available_flag

    for i, emb in enumerate(basic_embeds):
        j = n_booster + i
        tensor[j, :embed_dim] = emb
        tensor[j, embed_dim + 1] = 1.0  # available_flag

    return tensor


class EpisodeRunner:
    def run(
        self,
        pool_names: str,
        card_port: "CardEmbeddingPort",
        model: object,
        rng_seed: int,
        best_run: int = 1,
    ) -> Episode:
        """Run one episode.

        Action masking prevents duplicate picks: before each pick, the logits of
        already-picked booster slots (available_flag == 0) are set to -inf so they
        cannot be selected. Every episode always completes all best_run steps.

        Land picks are allowed throughout all picks. Per-step rewards incentivise
        picking exactly IDEAL_SPELLS=23 non-land cards and IDEAL_LANDS=17 land cards:
          +1 for a spell pick when n_spell < IDEAL_SPELLS
          -1 for a spell pick when n_spell >= IDEAL_SPELLS
          +1 for a land pick when n_land < IDEAL_LANDS
          -1 for a land pick when n_land >= IDEAL_LANDS
        """
        n_slots: int = model.config.n_slots  # type: ignore[attr-defined]
        booster_names = pool_names.split(";")
        n_booster = len(booster_names)

        current = _build_base_tensor(pool_names, card_port, n_slots, model.config.d_model)  # type: ignore[attr-defined]
        embed_dim = model.config.card_embed_dim  # type: ignore[attr-defined]

        actions: list[int] = []
        log_probs_list: list[float] = []
        step_rewards_list: list[float] = []
        n_spell = 0  # non-land picks so far
        n_land = 0   # land picks so far

        rng = np.random.default_rng(rng_seed)
        seeds = rng.integers(0, 2**31 - 1, size=best_run, dtype=np.int64)

        device = next(model.parameters()).device  # type: ignore[attr-defined]
        model.eval()  # type: ignore[attr-defined]
        for step in range(best_run):
            step_rng = np.random.default_rng(int(seeds[step]))
            perm = step_rng.permutation(n_booster)  # perm[shuffled_pos] = pool_index

            shuffled = current.copy()
            for sp in range(n_booster):
                shuffled[sp] = current[perm[sp]]
            # Basic land slots stay at indices n_booster..n_slots-1 (unchanged)

            input_t = torch.from_numpy(shuffled).unsqueeze(0).to(device)  # [1, n_slots, d_model]
            with torch.no_grad():
                logits = model(input_t)  # type: ignore[operator]  # [1, n_slots]

            # Action masking: set logits of unavailable slots to -inf
            available = torch.from_numpy(shuffled[:, embed_dim + 1]).to(device)  # [n_slots]
            logits[0] = logits[0].masked_fill(available == 0, -1e9)

            logits_1d = logits[0]  # [n_slots]

            log_probs_all = torch.log_softmax(logits_1d, dim=-1)
            probs = torch.softmax(logits_1d, dim=-1)
            sampled_sp = int(torch.multinomial(probs, 1).item())

            if sampled_sp < n_booster:
                pool_index = int(perm[sampled_sp])
            else:
                pool_index = sampled_sp  # basic land slot

            # Classify the pick (needed for step rewards)
            if pool_index >= n_booster:
                is_land = True
            elif card_port.is_land(booster_names[pool_index]):
                is_land = True
            else:
                is_land = False

            log_prob = float(log_probs_all[sampled_sp].item())

            actions.append(pool_index)
            log_probs_list.append(log_prob)

            # Per-step reward: +1 for picks within budget, -1 for picks over budget
            if is_land:
                step_rewards_list.append(1.0 if n_land < IDEAL_LANDS else -1.0)
            else:
                step_rewards_list.append(1.0 if n_spell < IDEAL_SPELLS else -1.0)

            # Update tensor state
            if pool_index >= n_booster:
                current[pool_index, embed_dim] += 1.0      # pick_count (can exceed 1 for basic lands)
            else:
                current[pool_index, embed_dim] = 1.0       # pick_count (0→1 for booster cards)
                current[pool_index, embed_dim + 1] = 0.0   # available_flag

            if is_land:
                n_land += 1
            else:
                n_spell += 1

        n_total = len(actions)
        excess = max(n_spell - IDEAL_SPELLS, 0) + max(n_land - IDEAL_LANDS, 0)
        effective_run = n_total - excess
        reward = float(effective_run) / float(best_run) * 2.0 - 1.0

        return Episode(
            pool_names=pool_names,
            shuffle_seeds=seeds,
            actions=np.array(actions, dtype=np.int32),
            log_probs=np.array(log_probs_list, dtype=np.float32),
            step_rewards=np.array(step_rewards_list, dtype=np.float32),
            reward=reward,
            effective_run=effective_run,
        )
