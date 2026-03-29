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


def _build_base_tensor(pool_names: str, card_port: "CardEmbeddingPort", n_slots: int) -> np.ndarray:
    """Build initial pool tensor for an episode.

    Returns float32 ndarray of shape [n_slots, d_model] where d_model = embed_dim + 4.

    Booster slots  (0 .. n_booster-1): picked=0, available=1, is_land=0, count=0.
    Basic land slots (n_booster .. n_slots-1): picked=0, available=1, is_land=1, count=0.
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
    d_model = embed_dim + 4

    tensor = np.zeros((n_slots, d_model), dtype=np.float32)

    for i, emb in enumerate(booster_embeds):
        tensor[i, :embed_dim] = emb
        tensor[i, embed_dim + 1] = 1.0  # available_flag

    for i, emb in enumerate(basic_embeds):
        j = n_booster + i
        tensor[j, :embed_dim] = emb
        tensor[j, embed_dim + 1] = 1.0  # available_flag
        tensor[j, embed_dim + 2] = 1.0  # is_land

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

        Returns an Episode with pool indices in ``actions``.
        """
        n_slots: int = model.config.n_slots  # type: ignore[attr-defined]
        n_booster = len(pool_names.split(";"))

        current = _build_base_tensor(pool_names, card_port, n_slots)
        embed_dim = current.shape[1] - 4

        picked_set: set[int] = set()
        actions: list[int] = []
        log_probs_list: list[float] = []

        rng = np.random.default_rng(rng_seed)
        seeds = rng.integers(0, 2**31 - 1, size=MAX_PICKS, dtype=np.int64)

        model.eval()  # type: ignore[attr-defined]
        for step in range(MAX_PICKS):
            step_rng = np.random.default_rng(int(seeds[step]))
            perm = step_rng.permutation(n_booster)  # perm[shuffled_pos] = pool_index

            shuffled = current.copy()
            for sp in range(n_booster):
                shuffled[sp] = current[perm[sp]]
            # Basic land slots stay at indices n_booster..n_slots-1 (unchanged)

            input_t = torch.from_numpy(shuffled).unsqueeze(0)  # [1, n_slots, d_model]
            with torch.no_grad():
                logits = model(input_t)  # type: ignore[operator]  # [1, n_slots]
            logits_1d = logits[0]  # [n_slots]

            log_probs_all = torch.log_softmax(logits_1d, dim=-1)
            probs = torch.softmax(logits_1d, dim=-1)
            sampled_sp = int(torch.multinomial(probs, 1).item())

            if sampled_sp < n_booster:
                pool_index = int(perm[sampled_sp])
            else:
                pool_index = sampled_sp  # basic land slot

            # Terminate if the booster slot was already picked
            if pool_index < n_booster and pool_index in picked_set:
                break

            log_prob = float(log_probs_all[sampled_sp].item())
            actions.append(pool_index)
            log_probs_list.append(log_prob)

            # Update current tensor flags
            if pool_index < n_booster:
                picked_set.add(pool_index)
                current[pool_index, embed_dim] = 1.0      # picked_flag
                current[pool_index, embed_dim + 1] = 0.0  # available_flag
            else:
                # Basic land slot: increment count, set picked
                current[pool_index, embed_dim + 3] += 1.0  # basic_land_count
                current[pool_index, embed_dim] = 1.0        # picked_flag

        current_run = len(actions)
        reward = float(current_run) / float(best_run) * 2.0 - 1.0

        return Episode(
            pool_names=pool_names,
            shuffle_seeds=seeds,
            actions=np.array(actions, dtype=np.int32),
            log_probs=np.array(log_probs_list, dtype=np.float32),
            reward=reward,
        )
