"""SampleStage1UseCase: run the trained Stage 1 model interactively."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.domain.episode_runner import EpisodeRunner, BASIC_LAND_NAMES, MAX_PICKS
from sealed.infrastructure.pool_loader import PoolLoader
from sealed.infrastructure.pool_model_store import PoolModelStore
from sealed.infrastructure.embedding_store import EmbeddingStore
from sealed.infrastructure.embedding_adapter import EmbeddingAdapter


class SampleStage1UseCase:
    def execute(
        self,
        pools_path: Path,
        cards_path: Path,
        model_path: Path,
        n_samples: int = 10,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(str(model_path))

        model_store = PoolModelStore()
        ckpt = model_store.load(model_path)

        config = PoolTransformerConfig()
        model = PoolTransformerModel(config)
        model.load_state_dict(ckpt.pool_transformer_state_dict)
        model.eval()

        pool_loader = PoolLoader()
        pools_file = pools_path / "pools.txt"
        pools = pool_loader.load_pools(pools_file)

        embedding_store = EmbeddingStore()
        card_port = EmbeddingAdapter(embedding_store, cards_path)
        runner = EpisodeRunner()

        selected = random.sample(pools, min(n_samples, len(pools)))

        with torch.no_grad():
            for pool_idx, pool_names in enumerate(selected):
                ep = runner.run(
                    pool_names=pool_names,
                    card_port=card_port,
                    model=model,
                    rng_seed=pool_idx,
                    best_run=MAX_PICKS,
                )

                booster_names = pool_names.split(";")
                n_booster = len(booster_names)

                picks = []
                for pool_index in ep.actions:
                    if pool_index < n_booster:
                        picks.append(booster_names[pool_index])
                    else:
                        land_idx = pool_index - n_booster
                        if 0 <= land_idx < len(BASIC_LAND_NAMES):
                            picks.append(BASIC_LAND_NAMES[land_idx])
                        else:
                            picks.append(f"<slot {pool_index}>")

                n_picks = len(ep.actions)
                status = "SUCCESS" if n_picks == MAX_PICKS else f"TERMINATED pick={n_picks + 1}"
                print(f"--- Sample {pool_idx + 1} [{status}] picks={n_picks} ---")
                for i, name in enumerate(picks, 1):
                    print(f"  {i:2d}. {name}")
                print()
