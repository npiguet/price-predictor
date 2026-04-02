"""SampleStage2UseCase: run the trained Stage 2 model and display mana analysis."""
from __future__ import annotations

import random
from pathlib import Path

import torch

from sealed.domain.episode_runner import BASIC_LAND_NAMES, MAX_PICKS, EpisodeRunner
from sealed.domain.mana_scorer import (
    compute_ideal_distribution,
    compute_mana_score,
    count_actual_sources,
    count_pips,
)
from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.infrastructure.embedding_adapter import EmbeddingAdapter
from sealed.infrastructure.embedding_store import EmbeddingStore
from sealed.infrastructure.pool_loader import PoolLoader
from sealed.infrastructure.pool_model_store import PoolModelStore

_COLORS = ("W", "U", "B", "R", "G", "C")


class SampleStage2UseCase:
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

                picks: list[str] = []
                land_texts: list[str] = []
                non_land_texts: list[str] = []

                for pool_index in ep.actions:
                    if pool_index < n_booster:
                        card_name = booster_names[pool_index]
                    elif pool_index - n_booster < len(BASIC_LAND_NAMES):
                        card_name = BASIC_LAND_NAMES[pool_index - n_booster]
                    else:
                        card_name = f"<slot {pool_index}>"
                    picks.append(card_name)

                    if card_name.startswith("<slot"):
                        continue
                    if card_port.is_land(card_name):
                        land_texts.append(card_port.get_card_text(card_name))
                    else:
                        non_land_texts.append(card_port.get_card_text(card_name))

                n_picks = len(ep.actions)
                n_lands = len(land_texts)
                status = "SUCCESS" if n_picks == MAX_PICKS else f"TERMINATED pick={n_picks + 1}"

                # Compute mana score
                pip_counts = count_pips(non_land_texts)
                ideal = compute_ideal_distribution(pip_counts)
                actual = count_actual_sources(land_texts)
                ms = compute_mana_score(ideal, actual, n_lands)

                header = (
                    f"--- Sample {pool_idx + 1} [{status}]"
                    f" picks={n_picks} score={ms.score:.3f} ---"
                )
                print(header)
                for i, name in enumerate(picks, 1):
                    print(f"  {i:2d}. {name}")
                print()
                print("  Mana sources (ideal → actual):")
                for c in _COLORS:
                    ideal_val = ideal.ideal.get(c, 0.0)
                    actual_val = actual.sources.get(c, 0.0)
                    print(f"    {c}:  {ideal_val:.1f} →  {actual_val:.0f}")
                print(f"  Lands: {n_lands}  Score: {ms.score:.3f}")
                print()
