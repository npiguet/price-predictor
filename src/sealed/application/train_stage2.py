"""TrainStage2UseCase: PPO training loop for Stage 2 (heuristic-gate) sealed deck-picker."""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.optim as optim

from sealed.application.train_stage1 import TrainingState
from sealed.domain.episode_runner import BASIC_LAND_NAMES, MAX_PICKS, EpisodeRunner
from sealed.domain.mana_scorer import (
    ManaScore,
    compute_ideal_distribution,
    compute_mana_score,
    count_actual_sources,
    count_pips,
)
from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.domain.ppo_trainer import PPOTrainer
from sealed.domain.replay_buffer import Episode
from sealed.infrastructure.embedding_adapter import EmbeddingAdapter
from sealed.infrastructure.embedding_store import EmbeddingStore
from sealed.infrastructure.pool_loader import PoolLoader
from sealed.infrastructure.pool_model_store import PoolModelStore

if TYPE_CHECKING:
    from sealed.domain.card_embedding_port import CardEmbeddingPort

_CONVERGENCE_THRESHOLD = 0.90


def _compute_episode_mana_score(ep: Episode, card_port: "CardEmbeddingPort") -> ManaScore:
    """Compute mana score for a completed episode."""
    booster_names = ep.pool_names.split(";")
    n_booster = len(booster_names)

    land_texts: list[str] = []
    non_land_texts: list[str] = []

    for pool_index in ep.actions:
        if pool_index < n_booster:
            card_name = booster_names[pool_index]
        elif pool_index - n_booster < len(BASIC_LAND_NAMES):
            card_name = BASIC_LAND_NAMES[pool_index - n_booster]
        else:
            continue

        if card_port.is_land(card_name):
            land_texts.append(card_port.get_card_text(card_name))
        else:
            non_land_texts.append(card_port.get_card_text(card_name))

    n_lands = len(land_texts)
    pip_counts = count_pips(non_land_texts)
    ideal = compute_ideal_distribution(pip_counts)
    actual = count_actual_sources(land_texts)
    return compute_mana_score(ideal, actual, n_lands)


class TrainStage2UseCase:
    def execute(
        self,
        pools_path: Path,
        cards_path: Path,
        model_path: Path,
        init_from: Path,
        batch_size: int = 32,
        set_code: str = "RVR",
    ) -> None:
        pool_loader = PoolLoader()
        model_store = PoolModelStore()
        embedding_store = EmbeddingStore()

        # Load and validate pools
        pools_file = pools_path / "pools.txt"
        pools = pool_loader.load_pools(pools_file)
        print(f"Loaded {len(pools)} pools from {pools_file}")

        # Validate card embeddings present
        print(f"Validating card embeddings in {cards_path} ...")
        pool_loader.assemble_pool_tensor(pools[0], cards_path)
        print("Embeddings OK")

        # Ensure model directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        first_npz = next(cards_path.rglob("*.npz"), None)
        if first_npz is None:
            raise FileNotFoundError(f"No .npz embedding files found in {cards_path}")
        import numpy as _np
        card_embed_dim = int(_np.load(first_npz)["embedding"].shape[0])
        print(f"Detected card embedding dimension: {card_embed_dim}")
        config = PoolTransformerConfig.from_embed_dim(card_embed_dim)
        model = PoolTransformerModel(config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        state = TrainingState(best_run=MAX_PICKS, episode_count=0)

        if model_path.exists():
            # Resume from Stage 2 checkpoint (full state)
            ckpt = model_store.load(model_path)
            model.load_state_dict(ckpt.pool_transformer_state_dict)
            model.to(device)
            optimizer.load_state_dict(ckpt.optimizer_state_dict)
            if isinstance(ckpt.training_state, TrainingState):
                state = ckpt.training_state
            elif isinstance(ckpt.training_state, dict):
                s = ckpt.training_state
                state = TrainingState(
                    best_run=s.get("best_run", MAX_PICKS),
                    episode_count=s.get("episode_count", 0),
                )
            print(
                f"Resumed from {model_path}"
                f"  episode={state.episode_count}"
            )
        elif init_from.exists():
            # Init from Stage 1 checkpoint (model weights only, fresh optimizer)
            ckpt = model_store.load(init_from)
            model.load_state_dict(ckpt.pool_transformer_state_dict)
            model.to(device)
            state = TrainingState(best_run=MAX_PICKS, episode_count=0)
            print(f"Initialized from Stage 1 checkpoint: {init_from}")
        else:
            raise ValueError(f"Stage 1 checkpoint not found at {init_from}")

        print(
            f"Model: {n_params:,} parameters"
            f"  n_slots={config.n_slots}  d_model={config.d_model}"
        )
        print(f"Stage 2 training  set={set_code}  batch_size={batch_size}  checkpoint={model_path}")

        card_port = EmbeddingAdapter(embedding_store, cards_path)
        runner = EpisodeRunner()
        trainer = PPOTrainer(model, optimizer)

        pool_idx = 0

        while True:
            card_port.reset_timing()

            t0 = time.perf_counter()
            batch_episodes: list[Episode] = []
            for _ in range(batch_size):
                pool_names = pools[pool_idx % len(pools)]
                pool_idx += 1
                ep = runner.run(
                    pool_names=pool_names,
                    card_port=card_port,
                    model=model,
                    rng_seed=state.episode_count,
                    best_run=MAX_PICKS,
                )
                state.episode_count += 1
                batch_episodes.append(ep)
            t_collect = time.perf_counter() - t0

            # Compute mana scores for completed episodes, override rewards
            batch_scores: list[float] = []
            n_dup = 0
            for ep in batch_episodes:
                if ep.termination == "success":
                    ms = _compute_episode_mana_score(ep, card_port)
                    ep.step_rewards = np.full(
                        len(ep.actions), ms.reward, dtype=np.float32
                    )
                    ep.reward = ms.reward
                    batch_scores.append(ms.score)
                else:
                    n_dup += 1
                    batch_scores.append(ep.reward)  # Stage 1 reward for duplicates

            t0 = time.perf_counter()
            trainer.update(batch_episodes, card_port)
            t_update = time.perf_counter() - t0

            t_embed = card_port.total_load_s

            mean_score = sum(batch_scores) / len(batch_scores) if batch_scores else 0.0
            scores_str = ",".join(f"{s:.3f}" for s in batch_scores)
            print(
                f"[ep {state.episode_count}] batch scores: {scores_str}"
                f"  mean_score={mean_score:.3f}"
                f"  | dup={n_dup}"
                f"  | collect={t_collect:.2f}s  update={t_update:.2f}s  embed={t_embed:.2f}s"
            )

            # Convergence: all episodes completed and scored above threshold
            batch_converged = (
                n_dup == 0
                and all(s > _CONVERGENCE_THRESHOLD for s in batch_scores)
            )

            model_store.save(model_path, model, optimizer, state)

            if state.episode_count % 1000 == 0:
                model_store.save_timestamped(model_path, model, optimizer, state)

            if batch_converged:
                print(
                    f"Stage 2 complete: full batch scored > {_CONVERGENCE_THRESHOLD}."
                    f" Model saved to {model_path}."
                )
                return
