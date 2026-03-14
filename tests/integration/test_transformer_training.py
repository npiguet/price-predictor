"""Integration test: train tiny transformer model on fixture data."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from price_predictor.domain.entities import TransformerConfig
from price_predictor.infrastructure.transformer_dataset import TransformerTrainingDataset
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
from price_predictor.infrastructure.transformer_store import save_model, load_model

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures" / "converted_cards"


def _load_fixture_texts() -> list[tuple[str, str, float]]:
    """Load fixture card texts with fake prices."""
    return [
        ("Lightning Bolt", (FIXTURES_DIR / "lightning_bolt.txt").read_text(encoding="utf-8"), 2.50),
        ("Grizzly Bears", (FIXTURES_DIR / "grizzly_bears.txt").read_text(encoding="utf-8"), 0.10),
        ("Jace, the Mind Sculptor", (FIXTURES_DIR / "jace_the_mind_sculptor.txt").read_text(encoding="utf-8"), 45.00),
    ]


@pytest.mark.integration
class TestTransformerTrainingIntegration:
    def test_train_tiny_model_and_save(self, tmp_path: Path):
        """Train a 2-epoch model on 3 fixture cards and verify artifact."""
        card_tuples = _load_fixture_texts()
        max_seq_len = 64

        dataset = TransformerTrainingDataset(card_tuples, max_seq_len=max_seq_len)
        assert len(dataset) == 3

        config = TransformerConfig(
            d_model=32, n_layers=1, n_heads=2, ff_dim=64,
            max_seq_len=max_seq_len, vocab_size=30522, dropout=0.0,
        )
        model = CardPriceTransformerModel(config)
        model.train()

        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        losses = []
        for epoch in range(2):
            for batch in loader:
                optimizer.zero_grad()
                preds = model(batch["input_ids"], batch["attention_mask"])
                loss = loss_fn(preds, batch["target"])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        # Loss should be a real number
        assert all(loss > 0 for loss in losses)

        # Save and verify
        model_path = save_model(model, config, tmp_path)
        assert model_path.exists()

        # Reload and verify predictions
        loaded_model, loaded_config = load_model(tmp_path)
        loaded_model.eval()
        assert loaded_config.d_model == 32

        with torch.no_grad():
            sample = dataset[0]
            pred = loaded_model(
                sample["input_ids"].unsqueeze(0),
                sample["attention_mask"].unsqueeze(0),
            )
            assert pred.shape == (1,)
            assert torch.isfinite(pred).all()
