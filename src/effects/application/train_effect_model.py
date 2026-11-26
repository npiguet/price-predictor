"""``train-effect-model``: its flags, and reading the curated corpus it trains on.

The trainer derives nothing about the corpus (FR-125 withdrawn): the manifest
``build-corpus`` writes decides the split, the holdout, the validation
samples, the gate-1 slice and the rarity table, and the selection set frozen
for it (``--reports-dir``, research E1) what the control score reads.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from effects.application.closed_list_check import require_closed_lists
from effects.application.run_settings import OnResume, Recorded, recorded_settings, setting
from effects.domain import default_paths
from effects.domain.ability_tokenizer import Surface
from effects.domain.baselines import Baseline
from effects.domain.corpus_manifest import CorpusManifest
from effects.domain.rule_mask import RuleMask
from effects.domain.sampling import SAMPLING_CLASSES
from effects.domain.split import CorpusSplit, HoldoutRule
from effects.domain.strata import VALIDATION_STRATA, Stratum
from effects.domain.text_holdout import normalize_script_text
from effects.domain.tokenizer_files import TokenizerFiles
from effects.infrastructure.converted_trees import SourceTrees
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.curated_shards import require_listed_shards
from effects.infrastructure.shard_io import require_generation

logger = logging.getLogger(__name__)

#: The classes each config field declares (``run_settings.setting``).
COMPARED, MAY_CHANGE = OnResume.COMPARED, OnResume.MAY_CHANGE
ALWAYS, WHEN_SET, NEVER = Recorded.ALWAYS, Recorded.WHEN_SET, Recorded.NEVER

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


class MissingTrainingInputError(RuntimeError):
    """Something a run reads before its first step is not there: ``--corpus``
    itself, its training shards, its validation samples, or any record in
    them. Refused before a model is built, rather than training on nothing."""


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


def on_off(value: str) -> str:
    """A switch spelled ``on`` or ``off``, as the CLI contract writes it."""
    if value not in ("on", "off"):
        raise ValueError(f"expected on or off, not {value!r}")
    return value


def require_runnable_schedule(config: TrainEffectModelConfig) -> None:
    """Refuse the flags no run can train with, before any start-up work: a
    count below one fails minutes in (or writes nothing), a negative
    ``--lr-scale`` ascends, a ``--curriculum-epoch`` past ``--epochs``
    never trains the sparse group, and a negative re-warm has no ramp."""
    for name in ("grad_accum", "batch_size", "steps_per_epoch", "epochs"):
        if getattr(config, name) < 1:
            raise ValueError(
                f"--{name.replace('_', '-')} must be at least 1, got {getattr(config, name)}."
            )
    if not config.lr_scale >= 0:
        raise ValueError(
            f"--lr-scale must not be negative, got {config.lr_scale}: a negative "
            "factor makes the optimizer ascend."
        )
    _require_whole_accumulations(config)
    _require_runnable_curriculum(config)


def _require_whole_accumulations(config: TrainEffectModelConfig) -> None:
    if config.steps_per_epoch % config.grad_accum:
        raise ValueError(
            f"--steps-per-epoch {config.steps_per_epoch} is not a multiple of "
            f"--grad-accum {config.grad_accum}: each epoch would drop its last "
            "partial accumulation."
        )


def _require_runnable_curriculum(config: TrainEffectModelConfig) -> None:
    if config.curriculum_epoch > config.epochs:
        raise ValueError(
            f"--curriculum-epoch {config.curriculum_epoch} is above --epochs "
            f"{config.epochs}: the sparse field group would never train."
        )
    if config.curriculum_rewarm_steps < 0:
        raise ValueError(
            f"--curriculum-rewarm-steps must not be negative, got "
            f"{config.curriculum_rewarm_steps}: 0 turns the re-warm off."
        )


@dataclass
class TrainEffectModelConfig:
    """The trainer's flag surface (contracts/cli.md § train-effect-model).

    ``help`` metadata is the flag's help, so a default lives here alone. Each
    field also declares whether a resume may change it and whether the run's
    settings record it (``run_settings.setting``): the settings every
    checkpoint and ``resume.pt`` hold, and what a resume compares, are read
    off those classes.
    """

    #: A curated dataset directory built by ``build-corpus``.
    corpus: str | None = setting(MAY_CHANGE, NEVER, default=None)
    cards_folders: tuple[Path, ...] = setting(
        MAY_CHANGE, NEVER, default=default_paths.DEFAULT_CARDS_FOLDER_PATHS,
    )
    variant_scripts: Path | None = setting(MAY_CHANGE, ALWAYS, default=None)
    vocab_path: Path = setting(MAY_CHANGE, NEVER, kw_only=True, help=(
        "The vocabulary the encoder tokenizes with; the checkpoint records it"
    ))
    #: ``None`` draws one, so runs do not all measure one arrangement of the
    #: corpus; the checkpoint records the drawn value.
    seed: int | None = setting(COMPARED, ALWAYS, default=None, help=(
        "Seeds weight init, batch planning and each epoch's shard draw; drawn "
        "from the OS and logged when omitted, so any run can be repeated"
    ))
    keyword_definitions: Path = setting(MAY_CHANGE, NEVER, kw_only=True, help=(
        "The keyword table the tokenizer expands with; the checkpoint records it"
    ))
    reports_dir: Path = setting(MAY_CHANGE, NEVER, kw_only=True, help=(
        "Where the selection set frozen for --corpus is found, by the corpus "
        "digest: the training run's reports run"
    ))
    model_output: Path = setting(MAY_CHANGE, NEVER, kw_only=True, help=(
        "Checkpoint directory (models/effects/<gen>/effect-model/<run>/)"
    ))
    variant: Baseline = setting(COMPARED, ALWAYS, default=Baseline.FULL)
    e_dim: int = setting(COMPARED, ALWAYS, default=256)
    relation_values: str = setting(COMPARED, ALWAYS, default="on", type=on_off, help=(
        "on: the trunk's relations change what a slot receives as well as how "
        "much it attends; off: the value tables stay at zero and untrained "
        "(default: %(default)s)"
    ))
    rule_mask: RuleMask = setting(COMPARED, ALWAYS, default=RuleMask.HARD, type=RuleMask, help=(
        "hard: a copy rule budgets added trains only the outputs the rule map "
        "names for the rules its record was drawn for, and the affected gate on "
        "its rule's entities; soft: every copy trains every output, the rest "
        "weighing natural share / copies (default: %(default)s)"
    ))
    deviance_floor: float = setting(COMPARED, ALWAYS, default=0.04, help=(
        "Floor on each output term's loss denominator, its constant-prediction "
        "loss on a sample of training shards, in nats per record "
        "(default: %(default)s)"
    ))
    ema_epochs: float = setting(COMPARED, ALWAYS, default=1.0, help=(
        "Span of the averaged weights validation, stopping and checkpoints "
        "read, in epochs: each optimizer step after the average starts (warmup's "
        "end, or one epoch after the curriculum switch when later) weighs "
        "grad-accum / (steps-per-epoch x this) (default: %(default)s)"
    ))
    stratum_weight: float = setting(COMPARED, ALWAYS, default=0.5, help=(
        "Card-disjoint share of the control score, in [0, 1]; the game-disjoint "
        "stratum takes the rest (default: %(default)s)"
    ))
    e_noise: float = setting(COMPARED, ALWAYS, default=0.1, help=(
        "Noise ratio on e relative to e's running covariance (FR-056), ramped "
        "in over the first --steps-per-epoch steps (default: %(default)s)"
    ))
    value_weight: float = setting(COMPARED, ALWAYS, default=0.05, help=(
        "Loss weight of the training-only value head (default: %(default)s)"
    ))
    encoder_layers: int = setting(COMPARED, ALWAYS, default=4, help=(
        "Ability-encoder layers, recorded in the checkpoint (default: %(default)s)"
    ))
    encoder_d_model: int = setting(COMPARED, ALWAYS, default=256, help=(
        "Ability-encoder width, recorded in the checkpoint; must divide by the "
        "encoder's head count (default: %(default)s)"
    ))
    keyword_expand_p: float = setting(COMPARED, ALWAYS, default=0.25)
    context_dropout: float = setting(COMPARED, ALWAYS, default=0.15)
    lethality_copies: int = setting(COMPARED, WHEN_SET, default=0, help=(
        "Counterfactual copies per qualifying damage or -X/-X resolution per "
        "step, toughness and amount redrawn and the death labelled by the rules "
        "(default: %(default)s)"
    ))
    lr_scale: float = setting(MAY_CHANGE, ALWAYS, default=1.0, help=(
        "Factor on the learning rate; may differ from the run a --resume continues"
    ))
    save_every_epoch: bool = setting(MAY_CHANGE, NEVER, default=False, help=(
        "Also keep every epoch's checkpoint under <model-output>/epochs/, so "
        "any epoch can be probed"
    ))
    mlm_weight: float = setting(COMPARED, ALWAYS, default=0.1)
    mlm_mask_prob: float = setting(COMPARED, ALWAYS, default=0.15)
    #: Small because the script-API term sums a BCE per parameter key, which
    #: starts two orders of magnitude above the value term.
    api_weight: float = setting(COMPARED, ALWAYS, default=0.003)
    #: An epoch rather than a step, so the field set never changes inside an
    #: epoch whose end validates the new fields' untrained heads.
    curriculum_epoch: int = setting(COMPARED, ALWAYS, default=3, help=(
        "Epoch whose first step enables the sparse field group (default: %(default)s)"
    ))
    curriculum_rewarm_steps: int = setting(COMPARED, ALWAYS, default=500, help=(
        "Training steps over which the learning rate ramps linearly from 0 back "
        "to the scheduled rate at the curriculum switch; 0 turns the re-warm off "
        "(default: %(default)s)"
    ))
    batch_size: int = setting(COMPARED, ALWAYS, default=32)
    grad_accum: int = setting(COMPARED, ALWAYS, default=1)
    steps_per_epoch: int = setting(COMPARED, ALWAYS, default=5_000)
    shards_per_epoch: int = setting(COMPARED, ALWAYS, default=DEFAULT_SHARDS_PER_EPOCH, help=(
        "Record shards an epoch reads, drawn at random across the corpus: "
        "--steps-per-epoch fixes how long an epoch takes, this how many shards "
        "its steps spread over (default: %(default)s)"
    ))
    prefetch_workers: int = setting(MAY_CHANGE, NEVER, default=DEFAULT_PREFETCH_WORKERS, help=(
        "Worker processes preparing each step's host half while this process "
        "trains; 0 prepares each step here, just before it trains. Changes speed, never results "
        "(default: %(default)s)"
    ))
    epochs: int = setting(MAY_CHANGE, ALWAYS, default=40)
    patience: int = setting(MAY_CHANGE, ALWAYS, default=5, help=(
        "Epochs without a new best control score before stopping, counted from "
        "the averaged weights' start; 0 never stops early (default: %(default)s)"
    ))
    withhold_keyword: str | None = setting(COMPARED, ALWAYS, default=None, help=(
        "Hold one implemented keyword's token out of training, always expanding "
        "it instead, so the zero-shot check has something to measure"
    ))
    resume: bool = setting(MAY_CHANGE, NEVER, default=False, help=(
        "Continue an interrupted run from the resume.pt in --model-output; "
        "every setting that shapes the run, --seed included, must match"
    ))

    def __post_init__(self) -> None:
        require_runnable_schedule(self)
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
        if not self.deviance_floor > 0:
            raise ValueError(
                f"--deviance-floor must be above zero, got {self.deviance_floor}: "
                "it is what a loss denominator is never below."
            )
        if not self.ema_epochs > 0:
            raise ValueError(
                f"--ema-epochs must be above zero, got {self.ema_epochs}: it is "
                "the span the averaged weights cover."
            )
        if not 0.0 <= self.stratum_weight <= 1.0:
            raise ValueError(
                f"--stratum-weight must be in [0, 1], got {self.stratum_weight}: "
                "it is the card-disjoint stratum's share of the control score."
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
        return Path(self.model_output)

    @classmethod
    def unpathed(cls, **options) -> TrainEffectModelConfig:
        """A config for code that scores or batches and reads none of the paths
        a run names (``--vocab-path``, ``--keyword-definitions``,
        ``--reports-dir``, ``--model-output``): each is empty unless
        ``options`` gives it."""
        return cls(**{
            "vocab_path": Path(), "keyword_definitions": Path(), "reports_dir": Path(),
            "model_output": Path(),
            **options,
        })


def recorded_flags(config: TrainEffectModelConfig, *, seed: int) -> dict:
    """Every flag the run's settings record, as each field's ``recorded``
    class says, in declaration order: ``seed`` is the one the run drew, which
    ``config`` may not hold."""
    return recorded_settings(replace(config, seed=seed))


@dataclass(frozen=True)
class CorpusFacts:
    """What a run reads off its curated corpus's manifest at startup and
    stamps onto every checkpoint it saves."""

    split: CorpusSplit = field(default_factory=CorpusSplit)
    #: The holdout rule the corpus was composed against; no trainer flag, as
    #: re-deriving it could only disagree.
    holdout: HoldoutRule = HoldoutRule(0, 0)
    #: Records in the manifest's gate-1 slice (FR-088b).
    gate_one_records: int = 0
    #: The corpus-wide ``text -> games`` table (FR-146).
    rarity: Mapping[str, int] | None = None
    #: Records per legality mode, so the epoch line places a legality record
    #: in the family the build balanced it in; empty without one.
    legality_mode_counts: Mapping[str, int] = field(default_factory=dict)
    #: The rule map's outputs per rule, as the manifest recorded them, which
    #: the rule mask reads (FR-060a, FR-061).
    rule_outputs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: The manifest's digest at read time (FR-147).
    digest: str = ""
    #: The rule map's content hash, which every checkpoint records (FR-060a).
    rule_map_digest: str = ""
    #: The sampling classes the manifest counts training records of
    #: (``per_class``), which fix every epoch's field set (FR-085).
    trained_classes: frozenset[str] = frozenset(SAMPLING_CLASSES)
    #: The digest of the selection set the control score reads, which every
    #: checkpoint records and a resume compares; empty without one.
    selection_set_digest: str = ""
    #: The hash of the manifest's shard listing, which a resume compares
    #: because the corpus digest leaves the listing out (finding 11.4).
    shards_digest: str = ""

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
            legality_mode_counts=manifest.legality_mode_counts,
            rule_outputs=manifest.effective_rule_outputs,
            digest=manifest.digest(),
            shards_digest=manifest.shards_digest(),
            rule_map_digest=manifest.rule_map_digest,
            trained_classes=frozenset(
                name for name, counts in manifest.per_class.items() if counts.kept
            ),
        )

    def untrained_classes(self) -> list[str]:
        """The sampling classes, in report order, the run trains no record of."""
        return [name for name in SAMPLING_CLASSES if name not in self.trained_classes]

    def provenance(self, config: TrainEffectModelConfig, tokenizer_hashes: Mapping[str, str]):
        """What a checkpoint of this run records about the data it saw;
        ``tokenizer_hashes`` are the ones taken when the tokenizer was checked."""
        from effects.domain.split_provenance import SplitProvenance

        return SplitProvenance(
            held_out_cards=self.split.held_out_cards,
            card_disjoint_games=tuple(sorted(self.split.card_disjoint_games)),
            game_disjoint_games=tuple(sorted(self.split.game_disjoint_games)),
            vocab_path=str(config.vocab_path),
            keyword_definitions_path=str(config.keyword_definitions),
            **tokenizer_hashes,
            withheld_keyword=config.withhold_keyword,
            holdout_permille=self.holdout.permille,
            holdout_max_carriers=self.holdout.max_carriers,
            corpus_path=config.corpus or "",
            corpus_digest=self.digest,
        )


@dataclass(frozen=True)
class RunSpec:
    """What one training run is: its flags, the facts its corpus's manifest
    gave, and the seed, resolved once so the seed logged and checkpointed is
    the one every draw used."""

    config: TrainEffectModelConfig
    facts: CorpusFacts
    seed: int
    #: ``vocab_hash`` and ``keyword_definitions_hash``, taken once when the
    #: tokenizer's files were checked and never re-read: every save records
    #: these, so a file rewritten mid-run cannot pass for the one trained on.
    tokenizer_hashes: Mapping[str, str] = field(default_factory=dict)


class SurfaceMismatchError(ValueError):
    """A curated dataset's rarity table was keyed on the other surface."""


def require_matching_surface(
    *, manifest_surface: Surface, vocab_path: Path, corpus: str,
) -> None:
    """Refuse a vocabulary whose encoding surface is not the dataset's (FR-141).

    The rarity table is keyed by surface; on the other one every lookup
    misses, and ``sample_weights`` silently falls back to per-shard counts.
    """
    from effects.infrastructure.vocabulary_file import vocabulary_surface

    actual = vocabulary_surface(vocab_path)
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
    """Refuse a held-out text set the manifest does not record (FR-041);
    recomputed over this run's sidecars, since different sidecars would train
    on texts the run believes held out."""
    trees = SourceTrees.of(config.cards_folders).converted()
    cards_folder = trees.cards_folder
    if cards_folder is None or not cards_folder.is_dir():
        raise HoldoutMismatchError(
            "the holdout cannot be recomputed: no converted card tree among "
            f"--cards-folder {', '.join(str(f) for f in config.cards_folders)}"
        )
    started = time.monotonic()
    logger.info(
        "Recomputing the holdout over %d cards' sidecars under %s.",
        len(trees.card_files), cards_folder,
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
            f"the holdout recomputed over {cards_folder} holds "
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


def _corpus_inputs(store: CorpusStore, manifest: CorpusManifest):
    """The shards the manifest lists (each stratum's), the training ones among
    them and the validation sample of each stratum; refuses a corpus lacking
    training shards or samples, or whose shards are not the listed ones."""
    corpus = store.directory
    listed = require_listed_shards(store, manifest)
    training_shards = list(listed[str(Stratum.TRAINING)])
    if not training_shards:
        raise MissingTrainingInputError(f"No training shards under {store.training_dir}.")
    validation_samples = {
        stratum: store.sample_path(stratum) for stratum in VALIDATION_STRATA
    }
    missing = [str(p) for p in validation_samples.values() if not p.exists()]
    if missing:
        raise MissingTrainingInputError(
            f"The corpus at {corpus} has no validation samples "
            f"({', '.join(missing)}). Rebuild it with a build-corpus that writes "
            "validation/samples/."
        )
    return listed, training_shards, validation_samples


def run(config: TrainEffectModelConfig) -> int:
    """Train the encoder and effect head jointly. Returns an exit code; a
    corpus it cannot train on is refused by exception, which the CLI maps to
    one.

    Torch is imported only here, so ``python -m effects --help`` stays fast.
    """
    if config.corpus is None:
        raise MissingTrainingInputError(
            "train-effect-model reads a curated corpus: pass --corpus DIR, built "
            "with `python -m effects build-corpus`. Training straight from raw "
            "shards was withdrawn with FR-125."
        )
    store = CorpusStore(Path(config.corpus))
    manifest = store.load()
    config = with_variant_scripts(config, manifest)
    # Reconverted since the build, the trees join the keys to other lines (11.3).
    config.source_trees.require_sidecar_digest(manifest.sidecar_digest, corpus=config.corpus)
    require_matching_surface(
        manifest_surface=manifest.surface, vocab_path=Path(config.vocab_path),
        corpus=config.corpus,
    )
    from effects.application.tokenizer_inputs import checked_tokenizer_hashes

    tokenizer_hashes = checked_tokenizer_hashes(config.tokenizer_files, manifest)
    listed, training_shards, validation_samples = _corpus_inputs(store, manifest)
    # Before any record is read (FR-020, FR-033).
    require_generation(path for paths in listed.values() for path in paths)
    if not manifest.rule_counts:
        raise MissingTrainingInputError(
            f"The manifest of the corpus at {config.corpus} has no per-rule "
            "counts (FR-066): the rule mask and the control score read the rules "
            "it records. Rebuild it with a gen-3 build-corpus."
        )
    require_closed_lists(manifest)
    selection = frozen_selection(manifest, listed, Path(config.reports_dir))
    check_manifest_holdout(manifest, config)
    facts = CorpusFacts.of(manifest)
    warn_untrained_classes(facts, store.manifest_path)
    logger.info(
        "Curated corpus at %s: %d training shards, %d gate-one record(s), "
        "%d held-out ability text(s) on %d card(s).",
        config.corpus, len(training_shards), facts.gate_one_records,
        len(manifest.held_out_texts), len(manifest.held_out_cards),
    )

    from effects.application.training_loop import TrainingLoop

    return TrainingLoop(
        config, facts=facts, training_shards=training_shards,
        validation_samples=validation_samples, selection=selection,
        tokenizer_hashes=tokenizer_hashes,
    ).execute()


def warn_untrained_classes(facts: CorpusFacts, manifest_path: Path) -> None:
    """Say which sampling classes the manifest counts no training record of:
    the fields only those classes supervise carry no loss for the whole run."""
    missing = facts.untrained_classes()
    if missing:
        logger.warning(
            "The corpus manifest %s counts no training record of %s, so the "
            "fields only those classes supervise carry no loss this run.",
            manifest_path, ", ".join(missing),
        )


def frozen_selection(
    manifest: CorpusManifest, listed: Mapping[str, tuple[Path, ...]], reports_dir: Path,
):
    """The selection set frozen for the corpus, found by its digest under
    ``reports_dir``, with a reader of each validation stratum's ``listed`` shards; raises
    ``SelectionSetError`` (exit 1) when none is there, or one frozen against
    another corpus or edited (E1)."""
    from effects.application.control_score import FrozenSelection
    from effects.application.freeze_selection_set import stratum_reader
    from effects.domain.selection_set import load_selection_set, selection_set_path

    digest = manifest.digest()
    selection = load_selection_set(selection_set_path(reports_dir, digest), digest)
    logger.info("Selection set %s… frozen for the corpus, under %s.", selection.digest[:12],
                reports_dir)
    return FrozenSelection(selection, {
        str(stratum): stratum_reader(listed[str(stratum)]) for stratum in VALIDATION_STRATA})
