"""What every stage of one ``evaluate-effect-model`` run reads, gathered once.

Kept apart from the orchestration so the gate modules can name the context
without importing the command.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from effects.domain.baselines import Baseline

if TYPE_CHECKING:
    from effects.application.model_loading import LoadedModel, Runnable
    from effects.domain.corpus_manifest import CorpusManifest
    from effects.domain.split_provenance import SplitProvenance
    from effects.domain.tokenizer_files import TokenizerFiles
    from effects.infrastructure.converted_trees import SourceTrees
    from effects.infrastructure.effect_model_store import EffectCheckpoint
    from effects.infrastructure.sidecar_io import SidecarCache


@dataclass
class EvaluationContext:
    """What every stage of one evaluation reads, gathered once.

    The main checkpoint is loaded on first use and kept; each record set still
    gets a batcher of its own, and its active fields follow the classes of the
    records it scores.
    """

    config: EvaluateEffectModelConfig
    main: EffectCheckpoint
    #: The vocabulary and keyword files every loaded checkpoint reads with,
    #: hash-checked against ``main``; a baseline must record the same hashes.
    tokenizer_files: TokenizerFiles
    corpus_path: Path
    manifest: CorpusManifest
    #: Each output stratum's shards as the manifest lists them, checked against
    #: the disk once at load (finding 11.4): every reader of the corpus reads these.
    shards: Mapping[str, tuple[Path, ...]]
    _main_model: LoadedModel | None = field(default=None, repr=False)
    _sidecars: SidecarCache | None = field(default=None, repr=False)

    def main_runnable(self, records) -> Runnable:
        """The main checkpoint, ready to run over ``records``."""
        return self.main_model().runnable(records)

    def main_model(self) -> LoadedModel:
        """The main checkpoint's weights on the device, loaded once."""
        if self._main_model is None:
            self._main_model = self._load(self.main)
        return self._main_model

    def runnable_of(self, checkpoint, records) -> Runnable:
        """Another checkpoint over ``records``, loaded now and not kept."""
        return self._load(checkpoint).runnable(records)

    @property
    def sidecars(self) -> SidecarCache:
        """The run's one sidecar cache, which every loaded checkpoint shares."""
        if self._sidecars is None:
            self._sidecars = self.config.source_trees.sidecar_cache()
        return self._sidecars

    def _load(self, checkpoint) -> LoadedModel:
        from effects.application.model_loading import load_model

        return load_model(checkpoint, files=self.tokenizer_files, sidecars=self.sidecars)


@dataclass
class EvaluateEffectModelConfig:
    checkpoint: Path = field(kw_only=True)
    baseline_checkpoints: dict[str, Path] = field(default_factory=dict)
    #: ``--cards-folder``; None reads the folders ``--checkpoint`` recorded
    #: (:func:`with_checkpoint_defaults`), which the loaded context always holds.
    cards_folders: tuple[Path, ...] | None = None
    variant_scripts: Path | None = None
    vocab_path: Path | None = None
    keyword_definitions: Path | None = None
    #: A curated dataset directory (`build-corpus`); defaults to the one
    #: `--checkpoint` recorded (FR-147).
    corpus: Path | None = None
    abilities_root: Path = field(kw_only=True)
    #: The per-card win-rate table the decodability battery reads (FR-070).
    win_rates: Path = field(kw_only=True)
    #: Where the selection set frozen for the corpus is found, by its digest:
    #: ``--reports-dir``, as on ``train-effect-model``.
    reports_dir: Path = field(kw_only=True)
    #: ``--output``: the directory the files the report names are written to.
    output: Path | None = None

    @property
    def source_trees(self) -> SourceTrees:
        """``--cards-folder`` and ``--variant-scripts`` as one value."""
        from effects.infrastructure.converted_trees import SourceTrees

        return SourceTrees.of(self.cards_folders, self.variant_scripts)

    @property
    def output_dir(self) -> Path:
        """Where the files the report names are written: ``--output``, else
        a directory beside ``--checkpoint`` named for it, since the report
        itself is printed and two checkpoints in one directory must not
        overwrite each other's."""
        return evaluation_dir(self.checkpoint, self.output)


def evaluation_dir(checkpoint: Path, output: Path | None) -> Path:
    """``output``, else ``<checkpoint stem>.evaluation/`` beside ``checkpoint``."""
    if output is not None:
        return Path(output)
    checkpoint = Path(checkpoint)
    return checkpoint.parent / f"{checkpoint.stem}.evaluation"


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


class BaselineVariantError(RuntimeError):
    """A ``--variant-checkpoint NAME=PATH`` whose checkpoint was trained as
    another variant than NAME."""


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
    """``config`` with an unset variant-script tree taken from the manifest
    (:meth:`CorpusManifest.variant_scripts_or`)."""
    return replace(
        config, variant_scripts=manifest.variant_scripts_or(config.variant_scripts),
    )


def with_checkpoint_defaults(
    config: EvaluateEffectModelConfig, main, manifest: CorpusManifest, *,
    corpus: Path | str = "the corpus",
) -> EvaluateEffectModelConfig:
    """``config`` reading the converted trees ``--cards-folder`` names, else
    the ones ``main`` recorded, refused unless their sidecars are the ones
    the corpus was built against (finding 11.3); and the corpus's variant
    tree where none is named (:func:`with_corpus_defaults`)."""
    from effects.infrastructure.converted_trees import SourceTrees

    folders = tuple(Path(folder) for folder in (config.cards_folders or main.cards_folders))
    SourceTrees.of(folders).require_sidecar_digest(
        manifest.sidecar_digest, corpus=corpus,
    )
    return with_corpus_defaults(replace(config, cards_folders=folders), manifest)


def load_context(config: EvaluateEffectModelConfig) -> EvaluationContext:
    """The main checkpoint, hash-checked, with its corpus verified (FR-147)."""
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.curated_shards import require_listed_shards
    from effects.infrastructure.effect_model_store import (
        inference_files,
        load_effect_checkpoint,
    )
    from effects.infrastructure.shard_io import require_generation

    main_path = Path(config.checkpoint)
    main = load_effect_checkpoint(main_path)
    files = inference_files(
        main, vocab=config.vocab_path, keywords=config.keyword_definitions,
    )
    corpus_path = resolve_corpus_path(
        main.provenance, corpus=config.corpus, checkpoint=main_path,
    )
    store = CorpusStore(corpus_path)
    manifest = store.load()
    check_corpus(main.provenance, actual_digest=manifest.digest())
    # The listed shards alone are read (finding 11.4), each checked before any
    # record is (FR-020, FR-033).
    listed = require_listed_shards(store, manifest)
    require_generation(path for paths in listed.values() for path in paths)
    return EvaluationContext(
        with_checkpoint_defaults(config, main, manifest, corpus=corpus_path), main, files,
        corpus_path=corpus_path, manifest=manifest, shards=listed,
    )


def load_baselines(config: EvaluateEffectModelConfig, main) -> dict[str, EffectCheckpoint]:
    """Every ``--variant-checkpoint``, refused unless it is the variant it is
    named as and shares main's split, corpus and withheld keyword."""
    from effects.infrastructure.effect_model_store import (
        load_effect_checkpoint,
        require_same_split,
    )

    variants = {}
    for name, path in config.baseline_checkpoints.items():
        loaded = load_effect_checkpoint(path)
        require_variant(loaded, name, path=Path(path))
        require_same_split(
            main, loaded,
            main_path=Path(config.checkpoint), baseline_path=Path(path),
        )
        variants[name] = loaded
    return variants


def require_variant(loaded: EffectCheckpoint, name: str, *, path: Path) -> None:
    """Refuse a checkpoint trained as another variant than ``name`` (finding
    11.8): the batcher takes its masks from the checkpoint, so
    ``identity=<a full model>`` would run as a full model and gate 1 would
    compare the model with itself."""
    if loaded.variant != name:
        raise BaselineVariantError(
            f"{path} is a {str(loaded.variant)!r} checkpoint, named as "
            f"{str(name)!r} by --variant-checkpoint: pass a checkpoint trained "
            f"with --variant {name}."
        )
