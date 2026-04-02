"""TrainStage1UseCase: PPO training loop for Stage 1 sealed deck-picker."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torch.optim as optim

from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.domain.episode_runner import EpisodeRunner, MAX_PICKS
from sealed.domain.ppo_trainer import PPOTrainer
from sealed.infrastructure.pool_loader import PoolLoader
from sealed.infrastructure.pool_model_store import PoolModelStore
from sealed.infrastructure.embedding_store import EmbeddingStore
from sealed.infrastructure.embedding_adapter import EmbeddingAdapter


@dataclass
class TrainingState:
    best_run: int = 1
    episode_count: int = 0


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
        print(f"Loaded {len(pools)} pools from {pools_file}")

        # Validate card embeddings present (check first pool to fail fast)
        print(f"Validating card embeddings in {cards_path} ...")
        pool_loader.assemble_pool_tensor(pools[0], cards_path)  # raises FileNotFoundError
        print("Embeddings OK")

        # Ensure model directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Select compute device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # Initialize model, optimizer, state
        config = PoolTransformerConfig()
        model = PoolTransformerModel(config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        state = TrainingState()

        if model_path.exists():
            ckpt = model_store.load(model_path)
            model.load_state_dict(ckpt.pool_transformer_state_dict)
            model.to(device)
            optimizer.load_state_dict(ckpt.optimizer_state_dict)
            if isinstance(ckpt.training_state, TrainingState):
                state = ckpt.training_state
            elif isinstance(ckpt.training_state, dict):
                s = ckpt.training_state
                state = TrainingState(
                    best_run=s.get("best_run", 1),
                    episode_count=s.get("episode_count", 0),
                )
            print(
                f"Resumed from {model_path}"
                f"  episode={state.episode_count}"
                f"  best_run={state.best_run}"
            )
        else:
            print(f"No checkpoint found at {model_path} — starting fresh")

        print(
            f"Model: {n_params:,} parameters"
            f"  n_slots={config.n_slots}  d_model={config.d_model}"
            f"  n_layers={config.n_layers}  n_heads={config.n_heads}"
        )
        print(f"Training  set={set_code}  batch_size={batch_size}  checkpoint={model_path}")
        print("Collecting first batch ...")

        card_port = EmbeddingAdapter(embedding_store, cards_path)
        runner = EpisodeRunner()
        trainer = PPOTrainer(model, optimizer)

        pool_idx = 0

        while True:
            card_port.reset_timing()

            t0 = time.perf_counter()
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
                batch_episodes.append(ep)
            t_collect = time.perf_counter() - t0

            t0 = time.perf_counter()
            result = trainer.update(batch_episodes, card_port)
            t_update = time.perf_counter() - t0

            t_embed = card_port.total_load_s

            # Curriculum advancement: all episodes must complete best_run picks
            batch_all_succeeded = (
                all(ep.termination == "success" for ep in batch_episodes)
                and result.mean_reward >= 0.90
            )

            n_dup = sum(1 for ep in batch_episodes if ep.termination == "duplicate")

            runs_str = ",".join(str(r) for r in result.episode_runs)
            print(
                f"[ep {state.episode_count}] batch runs: {runs_str}"
                f"  best_run={state.best_run}  mean_reward={result.mean_reward:.3f}"
                f"  | dup={n_dup}"
                f"  | collect={t_collect:.2f}s  update={t_update:.2f}s  embed={t_embed:.2f}s"
            )

            if batch_all_succeeded:
                if state.best_run == MAX_PICKS:
                    model_store.save(model_path, model, optimizer, state)
                    print(
                        f"Stage 1 complete: full batch succeeded at best_run={MAX_PICKS}."
                        f" Model saved to {model_path}."
                    )
                    return
                state.best_run += 1
                print(f"  -> curriculum advanced: best_run={state.best_run}")

            model_store.save(model_path, model, optimizer, state)

            if state.episode_count % 1000 == 0:
                model_store.save_timestamped(model_path, model, optimizer, state)
