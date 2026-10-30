"""``train-effect-model``: its flags, and reading the curated corpus it trains on.

The trainer derives nothing about the corpus (FR-125 withdrawn): the manifest
``build-corpus`` writes decides the split, the holdout, the validation
samples, the gate-1 slice and the rarity table.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from effects.domain import default_paths
from effects.domain.ability_tokenizer import Surface
from effects.domain.baselines import Baseline
from effects.domain.corpus_manifest import CorpusManifest
from effects.domain.split import CorpusSplit, HoldoutRule
from effects.domain.strata import VALIDATION_STRATA, Stratum
from effects.domain.text_holdout import HoldoutUnit, normalize_script_text
from effects.domain.tokenizer_files import TokenizerFiles
from effects.infrastructure.converted_trees import SourceTrees
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import iter_shards, refuse_mixed_generations

logger = logging.getLogger(__name__)


#: Unique-text resolution records the card-disjoint stratum should hold
#: before gate 1's margins mean anything: below it the run warns.
MIN_HOLDOUT_RECORDS = 2000

#: Shards an epoch reads: ``--steps-per-epoch`` fixes the wall clock, this how
#: many shards those steps spread across. A shard load costs well under a
#: second, so the limit is how thinly a shard is worth sampling, not time.
DEFAULT_SHARDS_PER_EPOCH = 256

#: Worker processes preparing training steps (``--prefetch-workers``): a
#: step's host half costs about as much CPU as its device half takes, so two
#: keep up with no margin and the third covers one parsing its next shard.
DEFAULT_PREFETCH_WORKERS = 3


class EmptyHoldoutError(RuntimeError):
    """The card-disjoint stratum holds no records."""


def check_holdout(
    *,
    card_disjoint_records: int,
    unique_text_records: int,
    minimum: int = MIN_HOLDOUT_RECORDS,
) -> str | None:
    """Raise ``EmptyHoldoutError`` on an empty card-disjoint stratum; return a
    warning for a thin one, None otherwise (FR-088b).

    Empty is fatal because validation would be ``nan`` every epoch, which
    never sets a best, so no checkpoint would ever be saved.
    """
    if card_disjoint_records <= 0:
        raise EmptyHoldoutError(
            "The card-disjoint stratum holds no records, so there is nothing "
            "to select a checkpoint on and gate 1 has nothing to measure. "
            "Collect a full-strength run — pools generated without "
            "--exclude-cards — into the same shard directory; its games hold "
            "the held-out cards and become this stratum."
        )
    if unique_text_records < minimum:
        return (
            f"The card-disjoint stratum holds {unique_text_records} resolution "
            f"records whose acting text is on no training card, below the "
            f"{minimum} gate 1 wants. Its margins will be noisy, and so will "
            "best-checkpoint selection. Collect more full-strength games."
        )
    return None


@dataclass
class TrainEffectModelConfig:
    """The trainer's flag surface (contracts/cli.md § train-effect-model).

    ``help`` metadata is the flag's help, so a default lives here alone.
    """

    #: A curated dataset directory built by ``build-corpus``.
    corpus: str | None = None
    cards_folders: tuple[Path, ...] = default_paths.DEFAULT_CARDS_FOLDER_PATHS
    variant_scripts: Path | None = None
    #: ``--holdout-unit``: None takes the manifest's recorded unit (FR-041).
    holdout_unit: HoldoutUnit | None = None
    vocab_path: Path = field(
        default_factory=lambda: Path(default_paths.DEFAULT_VOCAB_PATH),
    )
    #: ``None`` draws one, so runs do not all measure one arrangement of the
    #: corpus; the checkpoint records the drawn value.
    seed: int | None = field(default=None, metadata={"help": (
        "Seeds weight init, batch planning and each epoch's shard draw; drawn "
        "from the OS and logged when omitted, so any run can be repeated"
    )})
    keyword_definitions: Path = field(
        default_factory=lambda: Path(default_paths.DEFAULT_KEYWORD_DEFINITIONS),
    )
    model_output: Path | None = field(default=None, metadata={"help": (
        "Checkpoint directory (default: models/effects/effect-model/, under "
        "{variant}/ for a baseline)"
    )})
    variant: Baseline = Baseline.FULL
    e_dim: int = 64
    e_noise: float = field(default=0.1, metadata={"help": (
        "Noise ratio on e relative to e's running covariance (FR-056), ramped "
        "in over the first --steps-per-epoch steps (default: %(default)s)"
    )})
    value_weight: float = field(default=0.05, metadata={"help": (
        "Loss weight of the training-only value head (default: %(default)s)"
    )})
    encoder_layers: int = field(default=4, metadata={"help": (
        "Ability-encoder layers, recorded in the checkpoint (default: %(default)s)"
    )})
    encoder_d_model: int = field(default=256, metadata={"help": (
        "Ability-encoder width, recorded in the checkpoint; must divide by the "
        "encoder's head count (default: %(default)s)"
    )})
    keyword_expand_p: float = 0.25
    context_dropout: float = 0.15
    lethality_copies: int = field(default=0, metadata={"help": (
        "Counterfactual copies per qualifying damage or -X/-X resolution per "
        "step, toughness and amount redrawn and the death labelled by the rules "
        "(default: %(default)s)"
    )})
    lr_scale: float = field(default=1.0, metadata={"help": (
        "Factor on the learning rate; may differ from the run a --resume continues"
    )})
    save_every_epoch: bool = field(default=False, metadata={"help": (
        "Also keep every epoch's checkpoint under <model-output>/epochs/, so "
        "any epoch can be probed"
    )})
    mlm_weight: float = 0.1
    mlm_mask_prob: float = 0.15
    #: Small because the script-API term sums a BCE per parameter key, which
    #: starts two orders of magnitude above the value term.
    api_weight: float = 0.003
    #: An epoch rather than a step, so the field set never changes inside an
    #: epoch whose end validates the new fields' untrained heads.
    curriculum_epoch: int = field(default=3, metadata={"help": (
        "Epoch whose first step enables the sparse field group (default: %(default)s)"
    )})
    batch_size: int = 32
    grad_accum: int = 1
    steps_per_epoch: int = 5_000
    shards_per_epoch: int = field(default=DEFAULT_SHARDS_PER_EPOCH, metadata={"help": (
        "Record shards an epoch reads, drawn at random across the corpus: "
        "--steps-per-epoch fixes how long an epoch takes, this how many shards "
        "its steps spread over (default: %(default)s)"
    )})
    prefetch_workers: int = field(default=DEFAULT_PREFETCH_WORKERS, metadata={"help": (
        "Worker processes preparing each step's host half while this process "
        "trains; 0 prepares each step here, just before it trains. Changes speed, never results "
        "(default: %(default)s)"
    )})
    epochs: int = 40
    patience: int = field(default=5, metadata={"help": (
        "Epochs without a new card-disjoint best before stopping; 0 never stops early"
    )})
    withhold_keyword: str | None = field(default=None, metadata={"help": (
        "Hold one implemented keyword's token out of training, always expanding "
        "it instead, so the zero-shot check has something to measure"
    )})
    resume: bool = field(default=False, metadata={"help": (
        "Continue an interrupted run from the resume.pt in --model-output; "
        "every setting that shapes the run, --seed included, must match"
    )})

    def __post_init__(self) -> None:
        # Imported here because the encoder module loads torch (FR-062).
        from effects.domain.ability_encoder import ENCODER_N_HEADS

        if self.encoder_d_model <= 0 or self.encoder_d_model % ENCODER_N_HEADS:
            raise ValueError(
                f"--encoder-d-model {self.encoder_d_model} is not divisible by "
                f"the encoder's {ENCODER_N_HEADS} attention heads; pick a "
                f"multiple of {ENCODER_N_HEADS}."
            )
        if self.encoder_layers <= 0:
            raise ValueError(
                f"--encoder-layers must be at least 1, got {self.encoder_layers}."
            )

    @property
    def source_trees(self) -> SourceTrees:
        """``--cards-folder`` and ``--variant-scripts`` as one value."""
        return SourceTrees.of(self.cards_folders, self.variant_scripts)

    @property
    def tokenizer_files(self) -> TokenizerFiles:
        """``--vocab-path`` and ``--keyword-definitions`` as one value."""
        return TokenizerFiles.of(self.vocab_path, self.keyword_definitions)

    def resolved_model_output(self) -> Path:
        from effects.infrastructure.effect_model_store import model_output_for

        if self.model_output is not None:
            return Path(self.model_output)
        return model_output_for(self.variant)


#: The flags a sweep arm differs by: the checkpoint's training settings record
#: each under its own name, and a resume compares them there, as recorded.
SWEEP_FLAGS: tuple[str, ...] = (
    "e_noise", "value_weight", "mlm_weight", "mlm_mask_prob", "api_weight",
)
#: Flags the training settings record only when set, so a run without one
#: resumes against the settings it always had.
RECORDED_WHEN_SET: tuple[str, ...] = ("lethality_copies",)
#: Every flag the checkpoint's training settings record, which a resume
#: compares there as recorded rather than off the config: ``seed`` as the
#: value the run drew, and :data:`RECORDED_WHEN_SET` only when set.
RECORDED_IN_TRAINING_SETTINGS: frozenset[str] = frozenset({
    "seed", *RECORDED_WHEN_SET, *SWEEP_FLAGS,
})


def recorded_flags(config: TrainEffectModelConfig, *, seed: int) -> dict:
    """:data:`RECORDED_IN_TRAINING_SETTINGS` as the training settings write
    them, in the order they always have (FR-063): ``seed`` is the one the run
    drew, which ``config`` may not hold."""
    return {
        **{name: getattr(config, name) for name in RECORDED_WHEN_SET if getattr(config, name)},
        "seed": seed,
        **{name: getattr(config, name) for name in SWEEP_FLAGS},
    }


@dataclass(frozen=True)
class CorpusFacts:
    """What a run reads off its curated corpus's manifest at startup and
    stamps onto every checkpoint it saves."""

    split: CorpusSplit = field(default_factory=CorpusSplit)
    #: The holdout rule the corpus was composed against; no trainer flag, as
    #: re-deriving it could only disagree.
    holdout: HoldoutRule = HoldoutRule(0, 0, HoldoutUnit.TEXT)
    #: Records in the manifest's gate-1 slice (FR-088b).
    gate_one_records: int = 0
    #: The corpus-wide ``text -> games`` table (FR-146).
    rarity: Mapping[str, int] | None = None
    #: Records per legality mode, so the epoch line places a legality record
    #: in the family the build balanced it in; None without one.
    legality_mode_counts: Mapping[str, int] | None = None
    #: The manifest's digest at read time (FR-147).
    digest: str = ""

    @classmethod
    def of(cls, manifest) -> CorpusFacts:
        return cls(
            split=CorpusSplit(
                held_out_cards=manifest.held_out_cards,
                card_disjoint_games=frozenset(manifest.card_disjoint_games),
                game_disjoint_games=frozenset(manifest.game_disjoint_games),
            ),
            holdout=manifest.holdout,
            gate_one_records=manifest.per_stratum.get(Stratum.GATE_ONE, 0),
            rarity=manifest.rarity,
            legality_mode_counts=manifest.legality_mode_counts or None,
            digest=manifest.digest(),
        )

    def provenance(self, config: TrainEffectModelConfig):
        """What a checkpoint of this run records about the data it saw."""
        from effects.domain.split_provenance import SplitProvenance
        from effects.infrastructure.effect_model_store import tokenizer_hashes

        return SplitProvenance(
            held_out_cards=self.split.held_out_cards,
            card_disjoint_games=tuple(sorted(self.split.card_disjoint_games)),
            game_disjoint_games=tuple(sorted(self.split.game_disjoint_games)),
            vocab_path=str(config.vocab_path),
            keyword_definitions_path=str(config.keyword_definitions),
            **tokenizer_hashes(config.tokenizer_files),
            withheld_keyword=config.withhold_keyword,
            holdout_permille=self.holdout.permille,
            holdout_max_carriers=self.holdout.max_carriers,
            corpus_path=config.corpus or "",
            corpus_digest=self.digest,
            holdout_unit=self.holdout.unit,
        )


@dataclass(frozen=True)
class RunSpec:
    """What one training run is: its flags, the facts its corpus's manifest
    gave, and the seed, resolved once so the seed logged and checkpointed is
    the one every draw used."""

    config: TrainEffectModelConfig
    facts: CorpusFacts
    seed: int


class SurfaceMismatchError(ValueError):
    """A curated dataset's rarity table was keyed on the other surface."""


def require_matching_surface(
    *, manifest_surface: Surface, vocab_path: Path, corpus: str,
) -> None:
    """Refuse a vocabulary whose encoding surface is not the dataset's (FR-141).

    The rarity table is keyed by surface; on the other one every lookup
    misses, and ``sample_weights`` silently falls back to per-shard counts.
    """
    from effects.domain.ability_encoder import surface_of

    actual = surface_of(vocab_path)
    if actual != manifest_surface:
        raise SurfaceMismatchError(
            f"the curated dataset at {corpus} was built on the "
            f"{manifest_surface!r} surface, but --vocab-path {vocab_path} is "
            f"the {actual!r} one. Its rarity table keys every ability text by "
            "surface, so every lookup would miss and rarity weighting would "
            "fall back to counting the resident shard — which is the thing a "
            "curated dataset exists to stop. Pass the vocabulary the manifest "
            "records, or rebuild the dataset against this one."
        )


@dataclass
class EpochResult:
    """One epoch's validation, on both strata."""

    epoch: int
    train_loss: float
    card_disjoint_loss: float
    game_disjoint_loss: float


class HoldoutMismatchError(ValueError):
    """The trainer's own holdout disagrees with the one its corpus recorded."""


class ValidationGameInTrainingError(ValueError):
    """A training shard holds a game its corpus's manifest places in a
    validation stratum.

    ``build-corpus`` routes each game to one stratum, so the shard and the
    manifest describe different builds; training on it would score the
    validation strata partly on games the model trained on.
    """


def check_manifest_holdout(manifest, config: TrainEffectModelConfig) -> None:
    """Refuse a holdout unit or a held-out text set the manifest does not
    record (FR-041); recomputed over this run's sidecars, since different
    sidecars would train on texts the run believes held out."""
    unit = manifest.holdout_unit
    if config.holdout_unit is not None and config.holdout_unit != unit:
        raise HoldoutMismatchError(
            f"--holdout-unit {config.holdout_unit} differs from the unit the "
            f"corpus at {config.corpus} was built under ({unit}). The split is "
            "the corpus's; rebuild it with build-corpus --holdout-unit "
            f"{config.holdout_unit}, or drop the flag."
        )
    trees = SourceTrees.of(config.cards_folders).converted()
    cards_folder = trees.cards_folder
    if cards_folder is None or not cards_folder.is_dir():
        raise HoldoutMismatchError(
            "the holdout cannot be recomputed: no converted card tree among "
            f"--cards-folder {', '.join(str(f) for f in config.cards_folders)}"
        )
    started = time.monotonic()
    logger.info(
        "Recomputing the %s holdout over %d cards' sidecars under %s.",
        unit, len(trees.card_files), cards_folder,
    )
    recomputed = trees.holdout(manifest.holdout)
    logger.info(
        "Holdout recomputed in %.0fs: %d held-out text(s).",
        time.monotonic() - started, len(recomputed.texts),
    )
    recorded = {normalize_script_text(text) for text in manifest.held_out_texts}
    if recomputed.texts != recorded:
        missing = sorted(recorded - recomputed.texts)
        extra = sorted(recomputed.texts - recorded)
        raise HoldoutMismatchError(
            f"the holdout recomputed under {unit} over {cards_folder} holds "
            f"{len(recomputed.texts)} text(s), the corpus recorded "
            f"{len(recorded)}: {len(missing)} recorded text(s) not held out "
            f"now, {len(extra)} held out now and not recorded. The sidecars "
            "were reconverted since the corpus was built, or --cards-folder "
            "names a different tree; point it at the sidecars the corpus was "
            "built against. First difference: "
            f"{(missing or extra)[0]!r}"
        )


def with_variant_scripts(
    config: TrainEffectModelConfig, manifest: CorpusManifest,
) -> TrainEffectModelConfig:
    """``config`` reading the variant-script tree ``--variant-scripts`` names,
    else the one the corpus was built against, as evaluation does; logged,
    because a run reading none trains every variant ability as no text."""
    tree = manifest.variant_scripts_or(config.variant_scripts)
    if tree is None:
        logger.info(
            "No variant-script tree: neither --variant-scripts nor the corpus "
            "manifest names one."
        )
    else:
        source = (
            "--variant-scripts" if config.variant_scripts is not None
            else "the corpus manifest's"
        )
        logger.info("Variant scripts: %s (%s).", tree, source)
    return replace(config, variant_scripts=tree)


def run(config: TrainEffectModelConfig) -> int:
    """Train the encoder and effect head jointly. Returns an exit code.

    Torch is imported only here, so ``python -m effects --help`` stays fast.
    """
    if config.corpus is None:
        logger.error(
            "train-effect-model reads a curated corpus: pass --corpus DIR, built "
            "with `python -m effects build-corpus`. Training straight from raw "
            "shards was withdrawn with FR-125."
        )
        return 1
    store = CorpusStore(Path(config.corpus))
    manifest = store.load()
    config = with_variant_scripts(config, manifest)
    require_matching_surface(
        manifest_surface=manifest.surface, vocab_path=Path(config.vocab_path),
        corpus=config.corpus,
    )
    training_shards = iter_shards(store.training_dir)
    if not training_shards:
        logger.error("No training shards under %s.", store.training_dir)
        return 1
    validation_samples = {
        stratum: store.sample_path(stratum) for stratum in VALIDATION_STRATA
    }
    missing = [str(p) for p in validation_samples.values() if not p.exists()]
    if missing:
        logger.error(
            "The corpus at %s has no validation samples (%s). Rebuild it with a "
            "build-corpus that writes validation/samples/.", config.corpus, ", ".join(missing),
        )
        return 1
    # Before any record is read (FR-033).
    refuse_mixed_generations(iter_shards(Path(config.corpus)))
    check_manifest_holdout(manifest, config)
    facts = CorpusFacts.of(manifest)
    logger.info(
        "Curated corpus at %s: %d training shards, %d gate-one record(s), "
        "%d held-out ability text(s) on %d card(s).",
        config.corpus, len(training_shards), facts.gate_one_records,
        len(manifest.held_out_texts), len(manifest.held_out_cards),
    )

    from effects.application.training_loop import TrainingLoop

    return TrainingLoop(
        config, facts=facts, training_shards=training_shards,
        validation_samples=validation_samples,
    ).execute()
