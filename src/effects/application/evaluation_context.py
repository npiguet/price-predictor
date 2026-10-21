"""What every stage of one ``evaluate-effect-model`` run reads, gathered once.

The command's configuration, the main checkpoint with its verified inference
files and curated corpus, and the variant checkpoints compared against it. Kept
apart from the orchestration so the gate modules can name the context they are
handed without importing the command that hands it to them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from effects.infrastructure import default_paths

if TYPE_CHECKING:
    from effects.domain.corpus_manifest import CorpusManifest
    from effects.infrastructure.effect_model_store import (
        EffectCheckpoint,
        SplitProvenance,
    )
    from effects.infrastructure.model_runner import LoadedModel, Runnable
    from effects.infrastructure.sidecar_io import SidecarCache

logger = logging.getLogger(__name__)


# ── what every stage reads ──────────────────────────────────────────────


@dataclass
class EvaluationContext:
    """What every stage of one evaluation reads, gathered once in :func:`run`.

    The main checkpoint's weights are loaded on first use and kept: gate 1,
    the breakdowns and gate 2 all run it, and each used to load it afresh with
    a sidecar cache of its own. Each record set still gets a batcher of its own
    (``LoadedModel.runnable``), since the slot widths and the active fields are
    measured from the records it scores.
    """

    config: EvaluateEffectModelConfig
    main: EffectCheckpoint
    vocab_path: Path
    keyword_path: Path
    corpus_path: Path | None = None
    #: The curated dataset's manifest, read once; None for a raw-corpus run.
    manifest: CorpusManifest | None = None
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


#: The per-card win-rate table the decodability battery reads by default;
#: defined beside the other default paths so ``--help`` can print it without
#: importing this module, which loads torch.
DEFAULT_WIN_RATES = Path(default_paths.DEFAULT_WIN_RATES)


@dataclass
class EvaluateEffectModelConfig:
    checkpoint: Path = field(
        default_factory=lambda: Path(default_paths.DEFAULT_CHECKPOINT),
    )
    baseline_checkpoints: dict[str, Path] = field(default_factory=dict)
    records_dir: Path = field(
        default_factory=lambda: Path(default_paths.DEFAULT_RECORDS_DIR),
    )
    cards_folders: tuple[Path, ...] = tuple(
        Path(folder) for folder in default_paths.DEFAULT_CARDS_FOLDERS
    )
    variant_scripts: Path | None = None
    vocab_path: Path | None = None
    keyword_definitions: Path | None = None
    #: A curated dataset directory (`build-corpus`); defaults to the one
    #: `--checkpoint` recorded, if any (FR-147).
    corpus: Path | None = None
    abilities_root: Path = field(
        default_factory=lambda: Path(default_paths.DEFAULT_ABILITIES_ROOT),
    )
    #: The per-card win-rate table the decodability battery reads (FR-070).
    win_rates: Path = field(default_factory=lambda: DEFAULT_WIN_RATES)


def parse_baseline_checkpoint(value: str) -> tuple[str, Path]:
    """``NAME=PATH`` for ``--variant-checkpoint``."""
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(
            f"--variant-checkpoint takes NAME=PATH, got {value!r}"
        )
    return name.strip(), Path(path.strip())


# ── corpus provenance (FR-147) ──────────────────────────────────────────


class CorpusMismatchError(RuntimeError):
    """The curated corpus was rebuilt since the checkpoint trained on it."""


def check_corpus(provenance: SplitProvenance, *, actual_digest: str) -> None:
    """Refuse a dataset that is not the one the checkpoint read (FR-147).

    A rebuild is a different split, so scoring the gates against it would score
    them partly on games the model trained on — the failure the recorded
    ``game_id`` sets exist to prevent, arriving through the corpus instead.
    """
    if not provenance.corpus_digest:
        return
    if provenance.corpus_digest != actual_digest:
        raise CorpusMismatchError(
            f"{provenance.corpus_path} has been rebuilt since this checkpoint "
            f"trained on it (recorded {provenance.corpus_digest}, found "
            f"{actual_digest}). Evaluate against the dataset it read, or "
            "retrain against this one."
        )


def resolve_corpus_path(
    provenance: SplitProvenance, *, corpus: Path | None, checkpoint: Path,
) -> Path | None:
    """Where to read the curated corpus ``check_corpus`` verifies, if anywhere.

    Defaults to what the checkpoint recorded; an explicit ``--corpus``
    overrides it. That much matches ``resolve_inference_paths``, but the two
    diverge on a checkpoint with no recorded corpus: ``resolve_inference_paths``
    always resolves a path and hash-checks it afterwards, override or not,
    while here there is no digest to check an override against, so the
    override is discarded rather than resolved. Refusing outright would be too
    strong — it would break a batch-eval script that passes ``--corpus``
    uniformly across a mix of curated and legacy checkpoints — and honouring
    it would be meaningless, so a warning names the checkpoint and says why,
    rather than the flag silently doing nothing.
    """
    if not provenance.corpus_digest:
        if corpus is not None:
            logger.warning(
                "%s records no curated corpus, so --corpus %s has nothing "
                "to check its digest against and is ignored.",
                checkpoint, corpus,
            )
        return None
    if corpus is not None:
        return Path(corpus)
    return Path(provenance.corpus_path)


# ── loading ─────────────────────────────────────────────────────────────


def with_corpus_defaults(
    config: EvaluateEffectModelConfig, manifest: CorpusManifest | None,
) -> EvaluateEffectModelConfig:
    """``config`` with what it left unset taken from the corpus's manifest.

    The variant-script tree: the curated corpus holds records of variant cards,
    whose ability keys name that tree, so a run without it reads every variant
    ability as no text at all — the same records the trainer read as text.
    """
    if config.variant_scripts is not None or manifest is None:
        return config
    if not manifest.variant_scripts:
        return config
    return replace(config, variant_scripts=Path(manifest.variant_scripts))


def load_context(config: EvaluateEffectModelConfig) -> EvaluationContext:
    """The main checkpoint, hash-checked, with its corpus verified (FR-147)."""
    from effects.infrastructure.effect_model_store import load_for_inference

    main_path = Path(config.checkpoint)
    main, vocab_path, keyword_path = load_for_inference(
        main_path,
        vocab_path=config.vocab_path, keyword_path=config.keyword_definitions,
    )
    corpus_path = resolve_corpus_path(
        main.provenance, corpus=config.corpus, checkpoint=main_path,
    )
    manifest = None
    # No path means the checkpoint recorded no digest, so nothing to check.
    if corpus_path is not None:
        from effects.infrastructure.corpus_store import CorpusStore
        from effects.infrastructure.record_io import iter_shards, refuse_mixed_generations

        manifest = CorpusStore(corpus_path).load()
        check_corpus(main.provenance, actual_digest=manifest.digest())
        # Before any record is read (FR-033): the two generations read
        # actor_player on playability records differently.
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
