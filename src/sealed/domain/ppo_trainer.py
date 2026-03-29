"""PPOTrainer: computes PPO loss and updates model weights."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from sealed.domain.episode_runner import _build_base_tensor
from sealed.domain.replay_buffer import Episode

if TYPE_CHECKING:
    from sealed.domain.card_embedding_port import CardEmbeddingPort


@dataclass
class TrainBatchResult:
    mean_reward: float
    episode_runs: list[int]


class PPOTrainer:
    def __init__(
        self,
        model: object,
        optimizer: object,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef

    def update(
        self,
        episodes: list[Episode],
        card_port: "CardEmbeddingPort",
    ) -> TrainBatchResult:
        """Reconstruct episode states, compute PPO loss, perform one gradient step."""
        episode_runs: list[int] = []
        reward_sum = 0.0

        self.model.train()  # type: ignore[attr-defined]
        device = next(self.model.parameters()).device  # type: ignore[attr-defined]

        total_steps = sum(len(ep.actions) for ep in episodes)

        # Normalise advantages over the batch so exactly half are positive/negative
        # regardless of absolute reward scale.
        raw_rewards = [float(ep.reward) for ep in episodes]
        mean_r = statistics.mean(raw_rewards)
        std_r = statistics.stdev(raw_rewards) if len(raw_rewards) > 1 else 1.0

        self.optimizer.zero_grad()  # type: ignore[attr-defined]

        for ep in episodes:
            n_steps = len(ep.actions)
            if n_steps == 0:
                episode_runs.append(0)
                continue

            n_slots: int = self.model.config.n_slots  # type: ignore[attr-defined]
            n_booster = len(ep.pool_names.split(";"))

            current = _build_base_tensor(ep.pool_names, card_port, n_slots)
            embed_dim = current.shape[1] - 4

            advantage = (float(ep.reward) - mean_r) / (std_r + 1e-8)

            for step in range(n_steps):
                action = int(ep.actions[step])
                old_lp = float(ep.log_probs[step])

                step_rng = np.random.default_rng(int(ep.shuffle_seeds[step]))
                perm = step_rng.permutation(n_booster)
                inv_perm = np.argsort(perm)  # inv_perm[pool_index] = shuffled_position

                shuffled = current.copy()
                for sp in range(n_booster):
                    shuffled[sp] = current[perm[sp]]

                input_t = torch.from_numpy(shuffled).unsqueeze(0).float().to(device)
                logits = self.model(input_t)  # type: ignore[operator]  # [1, n_slots]
                new_log_probs_all = torch.log_softmax(logits[0], dim=-1)  # [n_slots]

                # Map pool_index back to the shuffled input position
                if action < n_booster:
                    shuffled_pos = int(inv_perm[action])
                else:
                    shuffled_pos = action

                new_lp = new_log_probs_all[shuffled_pos]

                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
                entropy = -(new_log_probs_all * new_log_probs_all.exp()).sum()
                step_loss = (-torch.min(surr1, surr2) - self.entropy_coef * entropy) / max(total_steps, 1)
                step_loss.backward()  # releases graph immediately; gradients accumulate

                # Simulate the pick to maintain correct tensor state for next steps
                if action < n_booster:
                    current[action, embed_dim] = 1.0       # picked_flag
                    current[action, embed_dim + 1] = 0.0   # available_flag
                else:
                    current[action, embed_dim + 3] += 1.0  # basic_land_count
                    current[action, embed_dim] = 1.0        # picked_flag

            reward_sum += float(ep.reward)
            episode_runs.append(n_steps)

        if total_steps > 0:
            self.optimizer.step()  # type: ignore[attr-defined]

        mean_reward = reward_sum / len(episodes) if episodes else 0.0
        return TrainBatchResult(
            mean_reward=mean_reward,
            episode_runs=episode_runs,
        )
