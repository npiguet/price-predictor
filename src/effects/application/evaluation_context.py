"""What every stage of one ``evaluate-effect-model`` run reads, gathered once.

Kept apart from the orchestration so the gate modules can name the context
without importing the command.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from effects.domain import default_paths
from effects.domain.baselines import Baseline

if TYPE_CHECKING:
    from effects.domain.corpus_manifest import CorpusManifest
    from effects.infrastructure.effect_model_store import (
        EffectCheckpoint,
        SplitProvenance,
    )
    from effects.infrastructure.model_runner import LoadedModel, Runnable
    from effects.infrastructure.sidecar_io import SidecarCache


@dataclass
class EvaluationContext:
    """What every stage of one evaluation reads, gathered once.

    The main checkpoint is loaded on first use and kept; each record set still
    gets a batcher of its own, since the slot widths and active fields are
    measured from the records it scores.
    """

    config: EvaluateEffectModelConfig
    main: EffectCheckpoint
    vocab_path: Path
    keyword_path: Path
    corpus_path: Path
    manifest: CorpusManifest
    _main_model: LoadedModel | None = field(default=None, repr=False)
    _sidecars: SidecarCache | None = field(default=None, repr=False)

    def main_runnable(self, records) -> Runnable:
        """The main checkpoint, ready to run over ``records``."""
        if self._main_model is None:
            self._main_model = self._load(self.main)
        return self._main_model.runnable(records)

    def runnable_of(self, checkpoint, records) -> Runnable:
        """Another checkpoint over ``records``, loaded now and not kept."""
        return self._load(checkpoint).runnable(records)

    @property
    def sidecars(self) -> SidecarCache:
        """The run's one sidecar cache, which every loaded checkpoint shares."""
        if self._sidecars is None:
            from effects.infrastructure.model_runner import build_sidecars

            self._sidecars = build_sidecars(self.config)
        return self._sidecars

    def _load(self, checkpoint) -> LoadedModel:
        from effects.infrastructure.model_runner import load_model

        return load_model(
            self.config, checkpoint,
            vocab_path=self.vocab_path, keyword_path=self.keyword_path,
            sidecars=self.sidecars,
        )


@dataclass
class EvaluateEffectModelConfig:
    checkpoint: Path = field(
        default_factory=lambda: Path(default_paths.DEFAULT_CHECKPOINT),
    )
    baseline_checkpoints: dict[str, Path] = field(default_factory=dict)
    cards_folders: tuple[Path, ...] = default_paths.DEFAULT_CARDS_FOLDER_PATHS
    variant_scripts: Path | None = None
    vocab_path: Path | None = None
    keyword_definitions: Path | None = None
    #: A curated dataset directory (`build-corpus`); defaults to the one
    #: `--checkpoint` recorded (FR-147).
    corpus: Path | None = None
    abilities_root: Path = field(
        default_factory=lambda: Path(default_paths.DEFAULT_ABILITIES_ROOT),
    )
    #: The per-card win-rate table the decodability battery reads (FR-070).
    win_rates: Path = field(
        default_factory=lambda: Path(default_paths.DEFAULT_WIN_RATES),
    )


def parse_baseline_checkpoint(value: str) -> tuple[Baseline, Path]:
    """``NAME=PATH`` for ``--variant-checkpoint``, ``NAME`` a baseline.

    An unknown name is refused: a misspelt ``identity`` would otherwise read
    as gate 1's "no identity checkpoint" skip.
    """
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(
            f"--variant-checkpoint takes NAME=PATH, got {value!r}"
        )
    known = [b.value for b in Baseline if b is not Baseline.FULL]
    if name.strip() not in known:
        raise ValueError(
            f"--variant-checkpoint names an unknown baseline {name.strip()!r}; "
            f"known: {known}"
        )
    return Baseline(name.strip()), Path(path.strip())


class CorpusMismatchError(RuntimeError):
    """The checkpoint records no curated corpus, or it was rebuilt since."""


def check_corpus(provenance: SplitProvenance, *, actual_digest: str) -> None:
    """Refuse a dataset that is not the one the checkpoint read (FR-147): a
    rebuild is a different split."""
    if provenance.corpus_digest != actual_digest:
        raise CorpusMismatchError(
            f"{provenance.corpus_path} has been rebuilt since this checkpoint "
            f"trained on it (recorded {provenance.corpus_digest}, found "
            f"{actual_digest}). Evaluate against the dataset it read, or "
            "retrain against this one."
        )


def resolve_corpus_path(
    provenance: SplitProvenance, *, corpus: Path | None, checkpoint: Path,
) -> Path:
    """``--corpus``, else the corpus the checkpoint recorded; a checkpoint
    recording none is refused, since nothing else says which games it
    trained on."""
    if not provenance.corpus_digest:
        raise CorpusMismatchError(
            f"{checkpoint} records no curated corpus, so nothing says which "
            "games it trained on. Retrain it with train-effect-model --corpus."
        )
    if corpus is not None:
        return Path(corpus)
    return Path(provenance.corpus_path)


def with_corpus_defaults(
    config: EvaluateEffectModelConfig, manifest: CorpusManifest,
) -> EvaluateEffectModelConfig:
    """``config`` with an unset variant-script tree taken from the manifest;
    without it every variant ability reads as no text."""
    if config.variant_scripts is not None or not manifest.variant_scripts:
        return config
    return replace(config, variant_scripts=Path(manifest.variant_scripts))


def load_context(config: EvaluateEffectModelConfig) -> EvaluationContext:
    """The main checkpoint, hash-checked, with its corpus verified (FR-147)."""
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.effect_model_store import load_for_inference
    from effects.infrastructure.record_io import iter_shards, refuse_mixed_generations

    main_path = Path(config.checkpoint)
    main, vocab_path, keyword_path = load_for_inference(
        main_path,
        vocab_path=config.vocab_path, keyword_path=config.keyword_definitions,
    )
    corpus_path = resolve_corpus_path(
        main.provenance, corpus=config.corpus, checkpoint=main_path,
    )
    manifest = CorpusStore(corpus_path).load()
    check_corpus(main.provenance, actual_digest=manifest.digest())
    # Before any record is read (FR-033).
    refuse_mixed_generations(iter_shards(Path(corpus_path)))
    return EvaluationContext(
        with_corpus_defaults(config, manifest), main, vocab_path, keyword_path,
        corpus_path=corpus_path, manifest=manifest,
    )


def load_baselines(config: EvaluateEffectModelConfig, main) -> dict[str, EffectCheckpoint]:
    """Every ``--variant-checkpoint``, refused unless it shares main's split."""
    from effects.infrastructure.effect_model_store import (
        load_effect_checkpoint,
        require_same_split,
    )

    variants = {}
    for name, path in config.baseline_checkpoints.items():
        loaded = load_effect_checkpoint(path)
        require_same_split(
            main, loaded,
            main_path=Path(config.checkpoint), baseline_path=Path(path),
        )
        variants[name] = loaded
    return variants
