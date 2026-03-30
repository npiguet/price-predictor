"""PPOTrainer: computes PPO loss and updates model weights."""
from __future__ import annotations

from dataclasses import dataclass
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

        n_terminals = sum(1 for ep in episodes if ep.term_action >= 0)
        total_steps = sum(len(ep.actions) for ep in episodes) + n_terminals

        # Normalise per-step rewards across all steps in the batch, including
        # the terminal pick (reward=-1) for each failed episode.
        step_reward_arrays = [ep.step_rewards for ep in episodes if len(ep.actions) > 0]
        parts = step_reward_arrays.copy()
        if n_terminals > 0:
            parts.append(np.full(n_terminals, -1.0, dtype=np.float32))
        if parts:
            all_step_rewards = np.concatenate(parts)
            mean_r = float(all_step_rewards.mean())
            std_r = float(all_step_rewards.std()) if len(all_step_rewards) > 1 else 1.0
        else:
            mean_r, std_r = 0.0, 1.0

        self.optimizer.zero_grad()  # type: ignore[attr-defined]

        for ep in episodes:
            n_steps = len(ep.actions)
            has_terminal = ep.term_action >= 0

            if n_steps == 0 and not has_terminal:
                episode_runs.append(0)
                continue

            n_slots: int = self.model.config.n_slots  # type: ignore[attr-defined]
            n_booster = len(ep.pool_names.split(";"))

            current = _build_base_tensor(ep.pool_names, card_port, n_slots)
            embed_dim = current.shape[1] - 8

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

                advantage = (float(ep.step_rewards[step]) - mean_r) / (std_r + 1e-8)
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
                entropy = -(new_log_probs_all * new_log_probs_all.exp()).sum()
                step_loss = (-torch.min(surr1, surr2) - self.entropy_coef * entropy) / max(total_steps, 1)
                step_loss.backward()  # releases graph immediately; gradients accumulate

                # Simulate the pick to maintain correct tensor state for next steps
                if action < n_booster:
                    current[action, embed_dim] = 1.0       # pick_count (0→1 for booster cards)
                    current[action, embed_dim + 1] = 0.0   # available_flag
                else:
                    current[action, embed_dim] += 1.0      # pick_count (can exceed 1 for basic lands)

            # Process the terminating (illegal) pick with a -1 advantage so the
            # model receives a gradient signal pushing it away from bad picks.
            if has_terminal:
                term_step = n_steps  # seeds[n_steps] was used for the terminal pick
                step_rng = np.random.default_rng(int(ep.shuffle_seeds[term_step]))
                perm = step_rng.permutation(n_booster)
                inv_perm = np.argsort(perm)

                shuffled = current.copy()
                for sp in range(n_booster):
                    shuffled[sp] = current[perm[sp]]

                input_t = torch.from_numpy(shuffled).unsqueeze(0).float().to(device)
                logits = self.model(input_t)  # type: ignore[operator]
                new_log_probs_all = torch.log_softmax(logits[0], dim=-1)

                term_action = ep.term_action
                if term_action < n_booster:
                    term_shuffled_pos = int(inv_perm[term_action])
                else:
                    term_shuffled_pos = term_action

                new_lp = new_log_probs_all[term_shuffled_pos]
                old_lp = ep.term_log_prob

                advantage = (-1.0 - mean_r) / (std_r + 1e-8)
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
                entropy = -(new_log_probs_all * new_log_probs_all.exp()).sum()
                step_loss = (-torch.min(surr1, surr2) - self.entropy_coef * entropy) / max(total_steps, 1)
                step_loss.backward()

            reward_sum += float(ep.reward)
            episode_runs.append(n_steps)

        if total_steps > 0:
            self.optimizer.step()  # type: ignore[attr-defined]

        mean_reward = reward_sum / len(episodes) if episodes else 0.0
        return TrainBatchResult(
            mean_reward=mean_reward,
            episode_runs=episode_runs,
        )
