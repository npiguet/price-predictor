"""Persistence for the sealed encoder.

Saves only encoder weights (token_encoder + card_encoder); strips the
regression head before serialization (FR-020). On load, the prefix check
fails immediately if any non-encoder key sneaked into the saved file
(``strict=True`` semantic from data-model.md, applied at the prefix level
because the live model still owns a freshly-initialized regression head
that is intentionally not loaded).
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sealed.domain.encoder_model import SealedEncoderConfig, SealedEncoderModel

_ENCODER_PREFIXES: tuple[str, ...] = ("token_encoder.", "card_encoder.")
_LATEST_FILENAME: str = "latest.pt"


def _encoder_only_state_dict(model: SealedEncoderModel) -> dict[str, Any]:
    full = model.state_dict()
    return {
        key: value
        for key, value in full.items()
        if any(key.startswith(prefix) for prefix in _ENCODER_PREFIXES)
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


class SealedEncoderStore:
    """Save and load sealed encoder checkpoints under ``models/sealed/encoder/``."""

    def save_encoder(
        self,
        model: SealedEncoderModel,
        config: SealedEncoderConfig,
        output_dir: Path,
        *,
        version: str | None = None,
    ) -> Path:
        """Write ``{version}.pt`` and overwrite ``latest.pt`` with a byte copy.

        The persisted payload has two keys:
        - ``model_state_dict``: filtered to ``token_encoder.*`` / ``card_encoder.*``
        - ``config``: ``dataclasses.asdict(config)`` for round-tripping
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        version = version or _timestamp()
        target = output_dir / f"{version}.pt"
        payload: dict[str, Any] = {
            "model_state_dict": _encoder_only_state_dict(model),
            "config": asdict(config),
        }
        torch.save(payload, target)

        latest = output_dir / _LATEST_FILENAME
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        shutil.copy2(target, latest)
        return target

    def load_encoder(
        self, path: Path,
    ) -> tuple[SealedEncoderModel, SealedEncoderConfig]:
        """Reconstruct ``(model, config)`` from a saved ``.pt``.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            RuntimeError: if the saved ``model_state_dict`` contains keys
                outside the encoder prefixes (e.g. a leaked regression head)
                or is missing keys that the encoder children require.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Encoder file not found: {path}")
        raw = torch.load(path, map_location="cpu", weights_only=False)
        config = SealedEncoderConfig(**raw["config"])
        model = SealedEncoderModel(config)
        saved_state: dict[str, torch.Tensor] = raw["model_state_dict"]

        unexpected = [
            key for key in saved_state
            if not any(key.startswith(prefix) for prefix in _ENCODER_PREFIXES)
        ]
        if unexpected:
            raise RuntimeError(
                f"Encoder checkpoint at {path} contains non-encoder keys: "
                f"{sorted(unexpected)[:5]} (and {len(unexpected) - 5} more)"
                if len(unexpected) > 5
                else f"Encoder checkpoint at {path} contains non-encoder keys: "
                f"{sorted(unexpected)}"
            )

        token_state = {
            key.removeprefix("token_encoder."): value
            for key, value in saved_state.items()
            if key.startswith("token_encoder.")
        }
        card_state = {
            key.removeprefix("card_encoder."): value
            for key, value in saved_state.items()
            if key.startswith("card_encoder.")
        }
        model.token_encoder.load_state_dict(token_state, strict=True)
        model.card_encoder.load_state_dict(card_state, strict=True)
        return model, config
