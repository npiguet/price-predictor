"""Saving and loading effect-model checkpoints: weights plus the provenance
``src/effects/docs/checkpoints.md`` lists, and the resume file."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import torch

from effects.domain.ability_encoder import AbilityEncoder, AbilityEncoderConfig
from effects.domain.aux_heads import TRAINING_ONLY_PREFIXES
from effects.domain.baselines import Baseline
from effects.domain.checkpoint_layout import require_input_layout
from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.domain.settings_shape import require_settings_shape
from effects.domain.split_provenance import SplitProvenance
from effects.domain.tokenizer_files import TokenizerFiles
from price_predictor.infrastructure.torch_checkpoint import (
    copy_atomically,
    save_checkpoint,
    write_atomically,
)

logger = logging.getLogger(__name__)

LATEST_NAME = "latest.pt"



def content_hash(path: Path | None) -> str:
    """SHA-256 of a file's bytes, or ``""`` when there is no such file or no
    path: a run with no keyword-definition file is degraded, not broken."""
    if path is None or not Path(path).is_file():
        return ""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class HashMismatchError(RuntimeError):
    """A vocabulary or keyword-definition file changed since training."""


class SplitMismatchError(RuntimeError):
    """A variant checkpoint records a different split than the main one."""


def tokenizer_hashes(files: TokenizerFiles) -> dict[str, str]:
    """Both files' content hashes, under the names a checkpoint's provenance
    and a resume file record them by."""
    return {
        "vocab_hash": content_hash(files.vocab),
        "keyword_definitions_hash": content_hash(files.keywords),
    }


def verify_hashes(provenance: SplitProvenance, files: TokenizerFiles) -> None:
    """Re-hash the files actually loaded; raise ``HashMismatchError`` on a
    mismatch. A hash recorded empty means no file then: a file present now
    differs, and none now agrees."""
    for (label, recorded), path in zip(
        provenance.file_hashes().items(), (files.vocab, files.keywords),
    ):
        actual = content_hash(path)
        if actual != recorded:
            raise HashMismatchError(
                f"{label} at {path} has changed since training: recorded "
                f"{recorded[:12] or '(none)'}…, actual {actual[:12] or '(missing)'}…. "
                "Encoding against a different vocabulary produces vectors "
                "that look valid and mean nothing."
            )


class ScoringFlags(NamedTuple):
    """The flags a saved checkpoint's control score is computed under: its
    run's ``--rule-mask``, ``--stratum-weight`` and ``--deviance-floor``."""

    mask: str
    stratum_weight: float
    deviance_floor: float


@dataclass(frozen=True, slots=True)
class EffectCheckpoint:
    """One saved effect model, weights plus provenance."""

    encoder_config: AbilityEncoderConfig
    model_config: EffectModelConfig
    encoder_state: dict[str, Any]
    model_state: dict[str, Any]
    provenance: SplitProvenance
    variant: Baseline = Baseline.FULL
    epoch: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    #: The sidecar roots the run resolved keys against (FR-063b).
    cards_folders: tuple[str, ...] = ()
    #: The run's settings, ``run_context.training_settings`` (FR-063): every
    #: flag its field records, ``--rule-mask``, ``--deviance-floor``,
    #: ``--ema-epochs``, ``--stratum-weight`` and ``--relation-values``
    #: included, as ``resume.pt`` holds them.
    training_settings: dict[str, Any] = field(default_factory=dict)
    #: ``LossDenominators.as_dict()`` of the run (FR-062).
    loss_denominators: dict[str, Any] = field(default_factory=dict)
    # Gen-3 (data-model.md § EffectCheckpoint). The relation types and their
    # target indices are in ``training_settings`` (``layout_settings``), and
    # the weights saved are the averaged ones once warmup has ended.
    #: The best control score of the run (FR-069): the score of the epoch
    #: whose weights these are; None until a control score exists.
    best_control_score: float | None = None
    #: The control score of the epoch these weights were saved at; None when
    #: that epoch had none. Equals ``best_control_score`` in a best checkpoint.
    control_score: float | None = None
    #: Whether the weights are the averaged ones; False for an epoch that
    #: ended before warmup did, saved with the raw weights.
    averaged: bool = True
    #: The corpus manifest's rule-map digest (FR-060a).
    rule_map_digest: str = ""
    #: The digest of the selection set the run's control score read (E1);
    #: None when the run read none.
    selection_set_digest: str | None = None

    def scored_under(self) -> ScoringFlags:
        """The flags this checkpoint's control score is computed under, as
        its settings record them (every loaded checkpoint records them)."""
        settings = self.training_settings
        return ScoringFlags(
            settings["rule_mask"], float(settings["stratum_weight"]),
            float(settings["deviance_floor"]),
        )


def filter_training_only(state: dict[str, Any]) -> dict[str, Any]:
    """Drop the training-only heads from a model state dict (FR-076)."""
    return {
        key: value for key, value in state.items()
        if not key.startswith(TRAINING_ONLY_PREFIXES)
    }


#: Model state a shipped checkpoint may lack: the training-only heads (FR-076).
#: Anything else missing is refused, or it would run at random initial weights.
OPTIONAL_STATE_PREFIXES: tuple[str, ...] = TRAINING_ONLY_PREFIXES


def load_weights(checkpoint, device) -> tuple[AbilityEncoder, EffectModel]:
    """The checkpoint's encoder and model on ``device``, in evaluation mode,
    at the size the checkpoint recorded (FR-061)."""
    encoder = AbilityEncoder(checkpoint.encoder_config).to(device)
    encoder.load_state_dict(checkpoint.encoder_state)
    model = EffectModel(checkpoint.model_config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint.model_state, strict=False)
    absent = [key for key in missing if not key.startswith(OPTIONAL_STATE_PREFIXES)]
    if absent or unexpected:
        raise RuntimeError(
            f"the checkpoint's model state does not fit the model it declares: "
            f"missing {absent}, unexpected {list(unexpected)}"
        )
    return encoder.eval(), model.eval()


def _config_from(config_cls: type, data: dict[str, Any]) -> Any:
    """A config dataclass from a checkpoint's dict: every field is read, and one
    the dict leaves out raises ``KeyError`` naming it, never built at a default."""
    return config_cls(**{spec.name: data[spec.name] for spec in fields(config_cls)})


def encoder_config_from(data: dict[str, Any]) -> AbilityEncoderConfig:
    """An encoder config from a checkpoint's dict (``_config_from``)."""
    return _config_from(AbilityEncoderConfig, data)


def model_config_from(data: dict[str, Any]) -> EffectModelConfig:
    """A model config from a checkpoint's ``config`` dict (``_config_from``)."""
    return _config_from(EffectModelConfig, data)


def _nonempty_encoder_state(state: dict[str, Any]) -> dict[str, Any]:
    if not state:
        raise ValueError(
            "the checkpoint's encoder_state_dict is empty: the encoder would run at "
            "random initial weights, so a checkpoint saved without its encoder is refused"
        )
    return state


def _same(value: Any) -> Any:
    return value


@dataclass(frozen=True, slots=True)
class PayloadMember:
    """One member of a checkpoint's payload: the ``EffectCheckpoint`` attribute
    it holds, the key it is saved under, and how it is written and read.
    Every member is required on load."""

    attribute: str
    key: str
    write: Callable[[Any], Any] = _same
    read: Callable[[Any], Any] = _same


def _member(name: str, write=_same, read=_same, *, key: str | None = None) -> PayloadMember:
    return PayloadMember(name, key or name, write, read)


#: The checkpoint's payload, declared once: ``save`` writes these members and
#: ``load`` reads them, each required. The model config rides beside them under
#: ``config`` (``torch_checkpoint.save_checkpoint``).
PAYLOAD_MEMBERS: tuple[PayloadMember, ...] = (
    _member("encoder_state", read=_nonempty_encoder_state, key="encoder_state_dict"),
    _member("model_state", filter_training_only, key="model_state_dict"),
    _member("encoder_config", asdict, encoder_config_from),
    _member("provenance", SplitProvenance.as_dict, SplitProvenance.from_dict),
    _member("variant", lambda variant: Baseline(variant).value, Baseline),
    _member("epoch"),
    _member("cards_folders", list, tuple),
    _member("training_settings", dict, dict),
    _member("loss_denominators", dict, dict),
    _member("best_control_score"), _member("control_score"), _member("averaged"),
    _member("rule_map_digest"), _member("selection_set_digest"),
    #: What a variant carries beyond the model (the identity baseline's
    #: embedding table, ``surface_batching.IDENTITY_TABLE_KEY``), nested here so
    #: no name in it can collide with a member.
    _member("extra", dict, dict),
)


class EffectModelStore:
    """Reads and writes ``{timestamp}.pt`` plus a rolling ``latest.pt``."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def latest_path(self) -> Path:
        return self.directory / LATEST_NAME

    def save(self, checkpoint: EffectCheckpoint) -> Path:
        """Write a timestamped checkpoint and copy it to ``latest.pt``."""
        path = self._unused_path(datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
        self._write(path, checkpoint)
        copy_atomically(path, self.latest_path())
        return path

    def save_epoch(self, checkpoint: EffectCheckpoint) -> Path:
        """Write ``epoch-NN.pt`` for the checkpoint's epoch, numbered as the
        training log numbers it; a later save of the same epoch replaces it."""
        path = self.directory / f"epoch-{checkpoint.epoch:02d}.pt"
        self._write(path, checkpoint)
        return path

    def _write(self, path: Path, checkpoint: EffectCheckpoint) -> None:
        payload = {
            member.key: member.write(getattr(checkpoint, member.attribute))
            for member in PAYLOAD_MEMBERS
        }
        save_checkpoint(path, payload, checkpoint.model_config)
        logger.info("Saved %s checkpoint to %s", checkpoint.variant, path)

    def _unused_path(self, stamp: str) -> Path:
        """``{stamp}.pt``, or a numbered sibling when a save in the same second
        took it: a second write would replace the first epoch's checkpoint."""
        path = self.directory / f"{stamp}.pt"
        copy = 1
        while path.exists():
            path = self.directory / f"{stamp}-{copy}.pt"
            copy += 1
        return path

    def load(self, path: Path | None = None) -> EffectCheckpoint:
        path = Path(path) if path is not None else self.latest_path()
        if not path.exists():
            raise FileNotFoundError(
                f"No effect-model checkpoint at {path}. Run "
                "'python -m effects train-effect-model' first."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model_config = model_config_from(payload.pop("config"))
        # Before anything is read from it (FR-045): a checkpoint trained on
        # another input layout would read inputs this code no longer builds,
        # and an older one is told apart by its recorded layout alone.
        require_input_layout(payload["training_settings"])
        require_settings_shape(payload["training_settings"], holder="checkpoint")
        return EffectCheckpoint(
            model_config=model_config,
            **{member.attribute: member.read(payload[member.key]) for member in PAYLOAD_MEMBERS},
        )


def require_same_split(
    main: EffectCheckpoint, baseline: EffectCheckpoint,
    *, main_path: Path, baseline_path: Path,
) -> None:
    """Fail fast when a baseline checkpoint's split, corpus, withheld keyword
    or hashes differ."""
    if not main.provenance.describes_same_split(baseline.provenance):
        raise SplitMismatchError(
            f"{baseline_path} records a different split than {main_path}. "
            f"Retrain it on the --corpus {main_path} trained on: a baseline "
            "that saw different games is not a baseline."
        )
    mismatches = main.provenance.training_mismatches(baseline.provenance)
    if mismatches:
        raise SplitMismatchError(
            f"{baseline_path} trained otherwise than {main_path} (the main one's "
            f"first): {'; '.join(mismatches)}. Retrain it on the same --corpus "
            "and --withhold-keyword."
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


def load_effect_checkpoint(path: Path) -> EffectCheckpoint:
    """The checkpoint at ``path``, read through the store of its own directory."""
    path = Path(path)
    return EffectModelStore(path.parent).load(path)


def inference_files(
    checkpoint: EffectCheckpoint, *,
    vocab: Path | None = None, keywords: Path | None = None,
) -> TokenizerFiles:
    """The vocabulary and keyword files an inference command reads (FR-097):
    the ones ``checkpoint`` recorded unless overridden, hash-checked either
    way, so no command can resolve its files and forget to verify them."""
    recorded = checkpoint.provenance.tokenizer_files
    files = TokenizerFiles(
        Path(vocab) if vocab else recorded.vocab,
        Path(keywords) if keywords else recorded.keywords,
    )
    verify_hashes(checkpoint.provenance, files)
    return files


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
    path = Path(directory) / RESUME_NAME
    write_atomically(path, lambda temporary: torch.save(state, temporary))
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
