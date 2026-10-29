"""Saving and loading effect-model checkpoints: weights plus the provenance
``src/effects/CLAUDE.md`` § Model artifacts lists, and the resume file."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import torch

from effects.domain.ability_encoder import AbilityEncoderConfig
from effects.domain.aux_heads import TRAINING_ONLY_MODULES
from effects.domain.baselines import Baseline
from effects.domain.effect_model import EffectModelConfig
from price_predictor.infrastructure.torch_checkpoint import (
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_OUTPUT = Path("models/effects/effect-model")
LATEST_NAME = "latest.pt"



def model_output_for(variant: str, root: Path = DEFAULT_MODEL_OUTPUT) -> Path:
    """``models/effects/effect-model/`` for ``full``, a subdirectory otherwise."""
    root = Path(root)
    return root if variant == Baseline.FULL else root / variant


def content_hash(path: Path) -> str:
    """SHA-256 of a file's bytes, or ``""`` when the file is absent: a run
    with no keyword-definition file is degraded, not broken."""
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


def _like(default, value):
    """A stored ``value`` as its field's type, which a tuple's or a count's
    serialized form may not keep."""
    return type(default)(value) if isinstance(default, (tuple, int)) else value


@dataclass(frozen=True, slots=True)
class SplitProvenance:
    """What a checkpoint records about the data it saw; the game sets are
    enumerated, since re-deriving them against a grown corpus would differ."""

    held_out_cards: tuple[str, ...] = ()
    card_disjoint_games: tuple[str, ...] = ()
    game_disjoint_games: tuple[str, ...] = ()
    vocab_path: str = ""
    keyword_definitions_path: str = ""
    vocab_hash: str = ""
    keyword_definitions_hash: str = ""
    withheld_keyword: str | None = None
    #: The holdout flags the split was computed under (FR-134).
    holdout_permille: int = 0
    holdout_max_carriers: int = 0
    #: The curated dataset read and its manifest digest (FR-147); ``""`` on a
    #: checkpoint saved before curated corpora existed.
    corpus_path: str = ""
    corpus_digest: str = ""
    #: ``template`` or ``text`` (FR-042). ``text`` is written as no field at
    #: all, so a checkpoint that records none reads as it trained.
    holdout_unit: str = "text"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # A plain string: an enum member would tie loading to its import path.
        data["holdout_unit"] = str(self.holdout_unit)
        if self.holdout_unit == "text":
            del data["holdout_unit"]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitProvenance:
        """The provenance ``as_dict`` wrote; a field a checkpoint predates
        takes its default, which is how such a checkpoint trained."""
        return cls(**{
            spec.name: _like(spec.default, data[spec.name])
            for spec in fields(cls) if spec.name in data
        })

    def file_hashes(self) -> dict[str, str]:
        return {
            "vocabulary": self.vocab_hash,
            "keyword definitions": self.keyword_definitions_hash,
        }

    def describes_same_split(self, other: SplitProvenance) -> bool:
        return (
            self.held_out_cards == other.held_out_cards
            and set(self.card_disjoint_games) == set(other.card_disjoint_games)
            and set(self.game_disjoint_games) == set(other.game_disjoint_games)
        )

    def verify_hashes(self, *, vocab_path: Path, keyword_path: Path) -> None:
        """Re-hash the files actually loaded; raise ``HashMismatchError`` on a
        mismatch."""
        for (label, recorded), path in zip(
            self.file_hashes().items(), (vocab_path, keyword_path),
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
    variant: str = Baseline.FULL
    best_val_loss: float = float("inf")
    epoch: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    #: The sidecar roots the run resolved keys against (FR-063b); empty on a
    #: checkpoint that predates the field.
    cards_folders: tuple[str, ...] = ()
    #: ``TrainingLoop._training_settings`` (FR-063); empty or partial on a
    #: checkpoint that predates the field or a key.
    training_settings: dict[str, Any] = field(default_factory=dict)


def filter_training_only(state: dict[str, Any]) -> dict[str, Any]:
    """Drop the training-only heads from a model state dict (FR-076)."""
    prefixes = tuple(f"{name}." for name in TRAINING_ONLY_MODULES)
    return {
        key: value for key, value in state.items()
        if not key.startswith(prefixes)
    }


#: Payload keys the checkpoint's own fields carry; anything else is ``extra``.
_RESERVED_PAYLOAD_KEYS = frozenset({
    "encoder_state_dict", "model_state_dict", "encoder_config", "model_config",
    "provenance", "variant", "best_val_loss", "epoch", "config",
    "cards_folders", "training_settings",
})

#: Encoder-config keys a gen-1 checkpoint carries that the config no longer
#: has, dropped on load (FR-057, FR-061).
_REMOVED_ENCODER_KEYS = frozenset({"e_noise"})


def encoder_config_from(data: dict[str, Any]) -> AbilityEncoderConfig:
    """An encoder config from a checkpoint's dict; one that records no size
    builds at the defaults, which every pre-size-flag checkpoint trained at."""
    return AbilityEncoderConfig(**{
        key: value for key, value in data.items()
        if key not in _REMOVED_ENCODER_KEYS
    })


class EffectModelStore:
    """Reads and writes ``{timestamp}.pt`` plus a rolling ``latest.pt``."""

    def __init__(self, directory: Path = DEFAULT_MODEL_OUTPUT) -> None:
        self.directory = Path(directory)

    def latest_path(self) -> Path:
        return self.directory / LATEST_NAME

    def save(self, checkpoint: EffectCheckpoint) -> Path:
        """Write a timestamped checkpoint and copy it to ``latest.pt``."""
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder_state_dict": checkpoint.encoder_state,
            "model_state_dict": filter_training_only(checkpoint.model_state),
            "encoder_config": asdict(checkpoint.encoder_config),
            "provenance": checkpoint.provenance.as_dict(),
            "variant": checkpoint.variant,
            "best_val_loss": checkpoint.best_val_loss,
            "epoch": checkpoint.epoch,
            "cards_folders": list(checkpoint.cards_folders),
            "training_settings": dict(checkpoint.training_settings),
            **checkpoint.extra,
        }
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        path = self.directory / f"{stamp}.pt"
        save_checkpoint(path, payload, checkpoint.model_config)
        shutil.copyfile(path, self.latest_path())
        logger.info("Saved %s checkpoint to %s", checkpoint.variant, path)
        return path

    def load(self, path: Path | None = None) -> EffectCheckpoint:
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
            encoder_config=encoder_config_from(payload["encoder_config"]),
            model_config=model_config,
            encoder_state=payload["encoder_state_dict"],
            model_state=payload["model_state_dict"],
            provenance=SplitProvenance.from_dict(payload.get("provenance", {})),
            variant=payload.get("variant", Baseline.FULL),
            best_val_loss=payload.get("best_val_loss", float("inf")),
            epoch=payload.get("epoch", 0),
            # Round-tripped: the identity baseline reloaded without its table
            # would score on random vectors.
            extra={
                key: value for key, value in payload.items()
                if key not in _RESERVED_PAYLOAD_KEYS
            },
            cards_folders=tuple(payload.get("cards_folders", ())),
            training_settings=dict(payload.get("training_settings", {})),
        )


def require_same_split(
    main: EffectCheckpoint, baseline: EffectCheckpoint,
    *, main_path: Path, baseline_path: Path,
) -> None:
    """Fail fast when a baseline checkpoint's split or hashes differ."""
    if not main.provenance.describes_same_split(baseline.provenance):
        raise SplitMismatchError(
            f"{baseline_path} records a different split than {main_path}. "
            f"Retrain it on the --corpus {main_path} trained on: a baseline "
            "that saw different games is not a baseline."
        )
    recorded_by_main = main.provenance.file_hashes()
    for label, actual in baseline.provenance.file_hashes().items():
        recorded = recorded_by_main[label]
        if recorded != actual:
            raise SplitMismatchError(
                f"{baseline_path} recorded a different {label} hash than "
                f"{main_path} ({actual[:12] or '(none)'}… vs "
                f"{recorded[:12] or '(none)'}…)."
            )


def resolve_inference_paths(
    provenance: SplitProvenance,
    *,
    vocab_path: Path | None,
    keyword_path: Path | None,
) -> tuple[Path, Path]:
    """The vocabulary and keyword paths an inference command should use
    (FR-097): the recorded ones unless overridden."""
    resolved_vocab = Path(vocab_path) if vocab_path else Path(provenance.vocab_path)
    resolved_keyword = (
        Path(keyword_path) if keyword_path
        else Path(provenance.keyword_definitions_path)
    )
    return resolved_vocab, resolved_keyword


def load_effect_checkpoint(path: Path) -> EffectCheckpoint:
    """The checkpoint at ``path``, read through the store of its own directory."""
    path = Path(path)
    return EffectModelStore(path.parent).load(path)


def verified_inference_paths(
    checkpoint: EffectCheckpoint,
    *,
    vocab_path: Path | None = None,
    keyword_path: Path | None = None,
) -> tuple[Path, Path]:
    """:func:`resolve_inference_paths`, then the hash check on what it
    resolved, an override included."""
    vocab, keywords = resolve_inference_paths(
        checkpoint.provenance, vocab_path=vocab_path, keyword_path=keyword_path,
    )
    checkpoint.provenance.verify_hashes(vocab_path=vocab, keyword_path=keywords)
    return vocab, keywords


class InferenceCheckpoint(NamedTuple):
    """A checkpoint with the vocabulary and keyword files it verified against."""

    checkpoint: EffectCheckpoint
    vocab_path: Path
    keyword_path: Path


def load_for_inference(
    path: Path,
    *,
    vocab_path: Path | None = None,
    keyword_path: Path | None = None,
) -> InferenceCheckpoint:
    """Load ``path`` and resolve and hash-check its inference files (FR-097)."""
    checkpoint = load_effect_checkpoint(path)
    vocab, keywords = verified_inference_paths(
        checkpoint, vocab_path=vocab_path, keyword_path=keyword_path,
    )
    return InferenceCheckpoint(checkpoint, vocab, keywords)


#: Rewritten at the end of every epoch; not a checkpoint (see CLAUDE.md
#: § Model artifacts).
RESUME_NAME = "resume.pt"
#: The resume state of the run's best card-disjoint epoch, copied from
#: ``resume.pt`` whenever an epoch sets a new best (FR-063f). Copied into
#: another run's directory as its ``resume.pt``, it continues from that epoch.
RESUME_BEST_NAME = "resume-best.pt"
#: Where ``--save-every-epoch`` keeps one checkpoint per epoch.
EPOCHS_DIR = "epochs"


class ResumeMismatchError(RuntimeError):
    """A resume file written by a run whose settings differ from this one's."""


def save_resume_state(directory: Path, state: dict[str, Any]) -> Path:
    """Write ``state`` to ``directory/resume.pt`` atomically, by rename."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RESUME_NAME
    temporary = directory / (RESUME_NAME + ".writing")
    torch.save(state, temporary)
    os.replace(temporary, path)
    return path


def load_resume_state(directory: Path) -> dict[str, Any]:
    """The state :func:`save_resume_state` wrote, on the CPU."""
    path = Path(directory) / RESUME_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No resume file at {path}: nothing to resume. A run writes one at "
            "the end of each epoch it completes."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def resume_mismatches(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Every setting whose value differs between a resume file and this run,
    over both sides' names: a setting recorded only when set can be missing
    from either."""
    return [
        f"{name}: resume file {saved.get(name)!r}, this run {current.get(name)!r}"
        for name in sorted(saved.keys() | current.keys())
        if saved.get(name) != current.get(name)
    ]
