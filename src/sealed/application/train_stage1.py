"""TrainStage1UseCase: PPO training loop for Stage 1 sealed deck-picker."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.domain.replay_buffer import ReplayBuffer
from sealed.domain.episode_runner import EpisodeRunner
from sealed.domain.ppo_trainer import PPOTrainer
from sealed.infrastructure.pool_loader import PoolLoader, card_npz_path
from sealed.infrastructure.pool_model_store import PoolModelStore
from sealed.infrastructure.embedding_store import EmbeddingStore


@dataclass
class TrainingState:
    best_run: int = 1
    episode_count: int = 0
    consecutive_successes: int = 0
    reward_baseline: float = 0.0


class _EmbeddingAdapter:
    """Wraps EmbeddingStore to satisfy CardEmbeddingPort (structural protocol)."""

    def __init__(self, store: EmbeddingStore, cards_path: Path) -> None:
        self._store = store
        self._cards_path = cards_path

    def get_embedding(self, card_name: str) -> np.ndarray:
        return self._store.load(card_npz_path(self._cards_path, card_name))


class TrainStage1UseCase:
    def execute(
        self,
        pools_path: Path,
        cards_path: Path,
        model_path: Path,
        batch_size: int = 32,
        set_code: str = "RVR",
    ) -> None:
        pool_loader = PoolLoader()
        model_store = PoolModelStore()
        embedding_store = EmbeddingStore()

        # Load and validate pools
        pools_file = pools_path / "pools.txt"
        pools = pool_loader.load_pools(pools_file)  # raises ValueError if empty/missing

        # Validate card embeddings present (check first pool to fail fast)
        pool_loader.assemble_pool_tensor(pools[0], cards_path)  # raises FileNotFoundError

        # Ensure model directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize model, optimizer, state
        config = PoolTransformerConfig()
        model = PoolTransformerModel(config)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        replay_buffer = ReplayBuffer(max_size=1000)
        state = TrainingState()

        if model_path.exists():
            ckpt = model_store.load(model_path)
            model.load_state_dict(ckpt.pool_transformer_state_dict)
            optimizer.load_state_dict(ckpt.optimizer_state_dict)
            if isinstance(ckpt.training_state, TrainingState):
                state = ckpt.training_state
            elif isinstance(ckpt.training_state, dict):
                s = ckpt.training_state
                state = TrainingState(
                    best_run=s.get("best_run", 1),
                    episode_count=s.get("episode_count", 0),
                    consecutive_successes=s.get("consecutive_successes", 0),
                    reward_baseline=s.get("reward_baseline", 0.0),
                )
            if ckpt.replay_buffer:
                replay_buffer = ReplayBuffer.from_list(ckpt.replay_buffer)

        card_port = _EmbeddingAdapter(embedding_store, cards_path)
        runner = EpisodeRunner()
        trainer = PPOTrainer(model, optimizer)
        trainer.reward_baseline = state.reward_baseline

        pool_idx = 0

        while True:
            batch_episodes = []
            for _ in range(batch_size):
                pool_names = pools[pool_idx % len(pools)]
                pool_idx += 1
                ep = runner.run(
                    pool_names=pool_names,
                    card_port=card_port,
                    model=model,
                    rng_seed=state.episode_count,
                    best_run=state.best_run,
                )
                state.episode_count += 1
                current_run = len(ep.actions)

                if current_run > state.best_run:
                    state.best_run = current_run

                if current_run == MAX_COMPLETE_RUN:
                    state.consecutive_successes += 1
                else:
                    state.consecutive_successes = 0

                replay_buffer.append(ep)
                batch_episodes.append(ep)

            sampled = replay_buffer.sample(batch_size)
            result = trainer.update(sampled, card_port, state.best_run)
            state.reward_baseline = trainer.reward_baseline

            runs_str = ",".join(str(r) for r in result.episode_runs)
            print(
                f"[ep {state.episode_count}] batch runs: {runs_str}"
                f"  best_run={state.best_run}  mean_reward={result.mean_reward:.3f}"
            )

            model_store.save(model_path, model, optimizer, state, replay_buffer)

            if state.episode_count % 1000 == 0:
                model_store.save_timestamped(model_path, model, optimizer, state, replay_buffer)

            if state.consecutive_successes >= 100:
                print(
                    f"Stage 1 complete: 100 consecutive episodes with 40 legal picks."
                    f" Model saved to {model_path}."
                )
                return


# A "perfect" episode picks all 40 booster cards without collision.
MAX_COMPLETE_RUN = 40
