"""PickDecksUseCase: build 40-card decks from a pools file with a trained picker.

The inference counterpart to ``build-decks``: one picker forward + the
deterministic § 1.1 walk per pool (instead of a simulated-annealing search),
then ``compute_basic_lands`` fills to 40 cards. Output is the same
``LABEL;SET_CODE;Card1|...|Card40`` format ``build-decks`` produces, so the
file is a drop-in input for ``match-outcomes`` self-play.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sealed.application.build_decks import _count_complete_lines_and_truncate_partial
from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE
from sealed.domain.manabase import compute_basic_lands
from sealed.domain.picker_model import PickerModel, decompose_picks
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.picker_store import PickerStore
from sealed.infrastructure.pool_file_reader import parse_pools


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


@dataclass
class PickDecksConfig:
    pools_path: Path
    label: str
    picker_checkpoint: Path = field(
        default_factory=lambda: Path("models/sealed/picker/latest.pt"),
    )
    cards_path: Path = field(default_factory=lambda: Path("output/cardsfolder/"))
    output: Path = field(
        default_factory=lambda: Path("output/sealed/generated-decks.txt"),
    )
    resume: bool = False


class PickDecksUseCase:
    """Build one picker-guided 40-card deck per pool and write the result file."""

    def execute(self, config: PickDecksConfig) -> int:
        pools = parse_pools(config.pools_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, embedding_dim = self._load_picker(config.picker_checkpoint, device)
        locator = ConvertedCardLocator(config.cards_path)

        config.output.parent.mkdir(parents=True, exist_ok=True)

        total = len(pools)
        if config.resume:
            skip = _count_complete_lines_and_truncate_partial(config.output)
            open_mode = "a"
            if skip > 0:
                _log(
                    f"Resume: {skip} decks already in {config.output}, "
                    f"skipping pools 1..{skip}."
                )
                pools = pools[skip:]
        else:
            skip = 0
            open_mode = "w"

        _log(f"Building decks for {total} pools (label={config.label})...")
        written = 0
        progress_interval = max(1, total // 100)
        width_checked = False
        with open(config.output, open_mode, buffering=1, encoding="utf-8") as out:
            for i, (set_code, pool_names) in enumerate(pools, start=skip + 1):
                embs, valid_names = self._load_pool(pool_names, locator)
                if len(valid_names) >= NONLAND_DECK_SIZE:
                    if not width_checked:
                        self._check_width(
                            embedding_dim, embs[0].shape[0],
                            config.picker_checkpoint, config.cards_path,
                        )
                        width_checked = True
                    deck = self._build_one_deck(model, embs, valid_names, locator, device)
                    out.write(f"{config.label};{set_code};{'|'.join(deck)}\n")
                    written += 1
                if i % progress_interval == 0 or i == total:
                    _log(f"  {i}/{total} pools processed ({written} decks written)")
        _log(f"Done: {written} decks written to {config.output}")
        return written

    def _load_picker(
        self, checkpoint_path: Path, device: torch.device,
    ) -> tuple[PickerModel, int]:
        checkpoint = PickerStore().load_checkpoint(checkpoint_path)
        model = PickerModel(checkpoint.config)
        model.load_state_dict(checkpoint.model_state_dict)
        model.eval()
        model.to(device)
        return model, checkpoint.config.embedding_dim

    @staticmethod
    def _check_width(
        embedding_dim: int, cache_width: int, source: Path, cards_path: Path,
    ) -> None:
        if embedding_dim != cache_width:
            raise ValueError(
                f"Picker checkpoint {source} expects {embedding_dim}-wide card "
                f"embeddings, but the .npz cache under {cards_path} is "
                f"{cache_width}-wide. Re-run `sealed encode-cards` with the "
                "encoder this picker was trained on."
            )

    @staticmethod
    def _load_pool(
        pool_names: list[str], locator: ConvertedCardLocator,
    ) -> tuple[list[np.ndarray], list[str]]:
        embs: list[np.ndarray] = []
        valid_names: list[str] = []
        for name in pool_names:
            emb = locator.load_embedding(name)
            if emb is not None:
                embs.append(emb)
                valid_names.append(name)
        return embs, valid_names

    def _build_one_deck(
        self,
        model: PickerModel,
        embs: list[np.ndarray],
        valid_names: list[str],
        locator: ConvertedCardLocator,
        device: torch.device,
    ) -> list[str]:
        arr = np.stack(embs).astype(np.float32)
        cards = torch.from_numpy(arr).unsqueeze(0).to(device)
        mask = torch.ones(1, arr.shape[0], dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, _ = model(cards, mask)
        chosen = decompose_picks(logits[0].cpu(), embs, valid_names)

        nonland_texts = [
            text for n in chosen if (text := locator.load_text(n)) is not None
        ]
        lands = compute_basic_lands(nonland_texts)
        full_deck: list[str] = list(chosen)
        for land_name, count in lands.items():
            full_deck.extend([land_name] * count)
        return full_deck
