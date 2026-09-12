"""Saving and loading effect-model checkpoints.

A checkpoint carries more than weights, and the extras are what make a run's
numbers mean anything:

- **the split it trained against** — the held-out card list and the ``game_id``
  set across both strata. The corpus is append-only and grows between runs, so
  a recomputed split would not be the trained-against one and the gates would
  score partly on games the model had seen.
- **the vocabulary and keyword-definition paths, and their content hashes.**
  Every inference command re-hashes what it actually loaded and fails fast on a
  mismatch. Encoding a cache with a different vocabulary than the model trained
  on produces vectors that look fine and mean nothing.
- **the keyword withheld from training**, read by the evaluator rather than
  passed as a flag, so the zero-shot check measures the model that was trained
  rather than a keyword an operator remembers choosing.

Training-only heads (MLM, script-API, pairing) are filtered out at save time
(FR-076): they exist to shape the encoder and have no meaning at inference.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from effects.domain.ability_encoder import AbilityEncoderConfig
from effects.domain.effect_model import EffectModel, EffectModelConfig
from price_predictor.infrastructure.torch_checkpoint import (
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_OUTPUT = Path("models/effects/effect-model")
LATEST_NAME = "latest.pt"

#: `--variant full` ships; every other variant writes under its own name so a
#: baseline run can never overwrite the checkpoint the cache was built from.
VARIANT_FULL = "full"
VARIANTS: tuple[str, ...] = (
    VARIANT_FULL, "identity", "state-only", "no-state", "taxonomy",
)


def model_output_for(variant: str, root: Path = DEFAULT_MODEL_OUTPUT) -> Path:
    """``models/effects/effect-model/`` for ``full``, a subdirectory otherwise."""
    root = Path(root)
    return root if variant == VARIANT_FULL else root / variant


def content_hash(path: Path) -> str:
    """SHA-256 of a file's bytes, or ``""`` when the file is absent.

    Absent hashes as empty rather than raising: a run with no keyword-definition
    file is degraded, not broken, and the mismatch check below still catches a
    file that appears or changes between training and inference.
    """
    path = Path(path)
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class HashMismatchError(RuntimeError):
    """A vocabulary or keyword-definition file changed since training."""


class SplitMismatchError(RuntimeError):
    """A variant checkpoint records a different split than the main one."""


@dataclass(frozen=True, slots=True)
class SplitProvenance:
    """What a checkpoint records about the data it saw.

    ``card_disjoint_games`` and ``game_disjoint_games`` are enumerated rather
    than derived, because deriving them again against a grown corpus would give
    a different answer.
    """

    held_out_cards: tuple[str, ...] = ()
    card_disjoint_games: tuple[str, ...] = ()
    game_disjoint_games: tuple[str, ...] = ()
    vocab_path: str = ""
    keyword_definitions_path: str = ""
    vocab_hash: str = ""
    keyword_definitions_hash: str = ""
    #: The implemented keyword held out of training, or None.
    withheld_keyword: str | None = None
    #: The holdout flags the split was computed under (FR-134). Recorded because
    #: a depleted corpus was composed against these values: a run that inherits
    #: the split but reads a corpus depleted against different ones would train
    #: on cards it believes are held out, with nothing to say so.
    holdout_permille: int = 0
    holdout_max_carriers: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitProvenance:
        return cls(
            held_out_cards=tuple(data.get("held_out_cards", ())),
            card_disjoint_games=tuple(data.get("card_disjoint_games", ())),
            game_disjoint_games=tuple(data.get("game_disjoint_games", ())),
            vocab_path=data.get("vocab_path", ""),
            keyword_definitions_path=data.get("keyword_definitions_path", ""),
            vocab_hash=data.get("vocab_hash", ""),
            keyword_definitions_hash=data.get("keyword_definitions_hash", ""),
            withheld_keyword=data.get("withheld_keyword"),
            holdout_permille=int(data.get("holdout_permille", 0)),
            holdout_max_carriers=int(data.get("holdout_max_carriers", 0)),
        )

    @property
    def validation_games(self) -> frozenset[str]:
        return frozenset(self.card_disjoint_games) | frozenset(
            self.game_disjoint_games
        )

    def describes_same_split(self, other: SplitProvenance) -> bool:
        return (
            self.held_out_cards == other.held_out_cards
            and set(self.card_disjoint_games) == set(other.card_disjoint_games)
            and set(self.game_disjoint_games) == set(other.game_disjoint_games)
        )

    def verify_hashes(self, *, vocab_path: Path, keyword_path: Path) -> None:
        """Re-hash the files actually loaded; raise on a mismatch.

        Raises:
            HashMismatchError: naming the recorded and actual hashes, because
                the operator needs to know which file moved under them.
        """
        for label, path, recorded in (
            ("vocabulary", vocab_path, self.vocab_hash),
            ("keyword definitions", keyword_path, self.keyword_definitions_hash),
        ):
            actual = content_hash(path)
            if recorded and actual != recorded:
                raise HashMismatchError(
                    f"{label} at {path} has changed since training: recorded "
                    f"{recorded[:12]}…, actual {actual[:12] or '(missing)'}…. "
                    "Encoding against a different vocabulary produces vectors "
                    "that look valid and mean nothing."
                )


@dataclass(frozen=True, slots=True)
class EffectCheckpoint:
    """One saved effect model, weights plus provenance."""

    encoder_config: AbilityEncoderConfig
    model_config: EffectModelConfig
    encoder_state: dict[str, Any]
    model_state: dict[str, Any]
    provenance: SplitProvenance
    variant: str = VARIANT_FULL
    best_val_loss: float = float("inf")
    epoch: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def filter_training_only(state: dict[str, Any]) -> dict[str, Any]:
    """Drop the training-only heads from a model state dict (FR-076).

    They shape the encoder during training and have no meaning at inference, so
    shipping them would only invite someone to read them.
    """
    prefixes = tuple(f"{name}." for name in EffectModel.TRAINING_ONLY_HEADS)
    return {
        key: value for key, value in state.items()
        if not key.startswith(prefixes)
    }


#: Payload keys the checkpoint's own fields already carry; everything else in a
#: loaded payload is a variant's ``extra``. ``model_config`` is written by
#: ``save_checkpoint`` itself.
_RESERVED_PAYLOAD_KEYS = frozenset({
    "encoder_state_dict", "model_state_dict", "encoder_config", "model_config",
    "provenance", "variant", "best_val_loss", "epoch", "config",
})


class EffectModelStore:
    """Reads and writes ``{timestamp}.pt`` plus a rolling ``latest.pt``."""

    def __init__(self, directory: Path = DEFAULT_MODEL_OUTPUT) -> None:
        self.directory = Path(directory)

    def latest_path(self) -> Path:
        return self.directory / LATEST_NAME

    def save(self, checkpoint: EffectCheckpoint) -> Path:
        """Write a timestamped checkpoint and refresh ``latest.pt``.

        Both files are written rather than one being a link, matching the
        convention every other trainer in this repo uses: an operator points a
        long-lived command at ``latest.pt`` and an experiment at a timestamp.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder_state_dict": checkpoint.encoder_state,
            "model_state_dict": filter_training_only(checkpoint.model_state),
            "encoder_config": asdict(checkpoint.encoder_config),
            "provenance": checkpoint.provenance.as_dict(),
            "variant": checkpoint.variant,
            "best_val_loss": checkpoint.best_val_loss,
            "epoch": checkpoint.epoch,
            **checkpoint.extra,
        }
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        path = self.directory / f"{stamp}.pt"
        save_checkpoint(path, payload, checkpoint.model_config)
        save_checkpoint(self.latest_path(), payload, checkpoint.model_config)
        logger.info("Saved %s checkpoint to %s", checkpoint.variant, path)
        return path

    def load(self, path: Path | None = None) -> EffectCheckpoint:
        """Load a checkpoint, defaulting to this store's ``latest.pt``."""
        path = Path(path) if path is not None else self.latest_path()
        if not path.exists():
            raise FileNotFoundError(
                f"No effect-model checkpoint at {path}. Run "
                "'python -m effects train-effect-model' first."
            )
        payload, model_config = load_checkpoint(
            path, EffectModelConfig, weights_only=False,
        )
        return EffectCheckpoint(
            encoder_config=AbilityEncoderConfig(**payload["encoder_config"]),
            model_config=model_config,
            encoder_state=payload["encoder_state_dict"],
            model_state=payload["model_state_dict"],
            provenance=SplitProvenance.from_dict(payload.get("provenance", {})),
            variant=payload.get("variant", VARIANT_FULL),
            best_val_loss=payload.get("best_val_loss", float("inf")),
            epoch=payload.get("epoch", 0),
            # Whatever a variant needed to save beyond the two state dicts —
            # the identity baseline's embedding table is the one that exists.
            # Round-tripped rather than dropped: a baseline reloaded without it
            # scores on random vectors.
            extra={
                key: value for key, value in payload.items()
                if key not in _RESERVED_PAYLOAD_KEYS
            },
        )


def require_same_split(
    main: EffectCheckpoint, variant: EffectCheckpoint,
    *, main_path: Path, variant_path: Path,
) -> None:
    """Fail fast when a variant checkpoint's split or hashes differ.

    A baseline compared against a model that saw different games is not a
    baseline. The error names both checkpoints, because the operator's next
    move is to retrain one of them with ``--split-from``.
    """
    if not main.provenance.describes_same_split(variant.provenance):
        raise SplitMismatchError(
            f"{variant_path} records a different split than {main_path}. "
            f"Retrain it with --split-from {main_path}: a baseline that saw "
            "different games is not a baseline."
        )
    for label, recorded, actual in (
        ("vocabulary", main.provenance.vocab_hash, variant.provenance.vocab_hash),
        (
            "keyword definitions",
            main.provenance.keyword_definitions_hash,
            variant.provenance.keyword_definitions_hash,
        ),
    ):
        if recorded != actual:
            raise SplitMismatchError(
                f"{variant_path} recorded a different {label} hash than "
                f"{main_path} ({actual[:12] or '(none)'}… vs "
                f"{recorded[:12] or '(none)'}…)."
            )


def resolve_inference_paths(
    provenance: SplitProvenance,
    *,
    vocab_path: Path | None,
    keyword_path: Path | None,
) -> tuple[Path, Path]:
    """The vocabulary and keyword paths an inference command should use (FR-097).

    Both default to what the checkpoint recorded at training time. An explicit
    value overrides — and is still hash-checked, so overriding to a file that
    differs fails rather than silently encoding against the wrong vocabulary.
    """
    resolved_vocab = Path(vocab_path) if vocab_path else Path(provenance.vocab_path)
    resolved_keyword = (
        Path(keyword_path) if keyword_path
        else Path(provenance.keyword_definitions_path)
    )
    return resolved_vocab, resolved_keyword
