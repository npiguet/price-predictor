"""``train-effect-model``: reading a curated corpus, and the schedule it trains on.

The trainer derives nothing about the corpus any more (FR-125 withdrawn):
``build-corpus`` decides the split, the holdout, the two validation samples,
the gate-1 slice and the rarity table, records all of it in a manifest, and
``--corpus`` is the trainer's only input. Two things here still decide whether
the numbers the evaluator reports mean anything.

**The split excludes whole games, not held-out rows.** Records from one game
share a board, so a held-out card sitting in the *context* of a training record
leaks through its context role even when it is not the acting ability — and the
context role is exactly how the design trains each embedding from both
directions. Excluding the row and keeping the game would make gate 1, the
shipping gate, score partly on cards the encoder had already seen. The manifest
enumerates that split; this module reads it.

**Rarity weighting counts games, not records.** One long game can produce
hundreds of records of one ability; weighting by records would make that ability
look well-observed when it has been seen in one board state. The weights feed a
weighted shuffle *without replacement* over the resident shard (FR-086) rather
than a per-batch class draw with replacement, which replays a small shard's few
records many times over within one epoch.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from effects.domain import default_paths
from effects.domain.baselines import Baseline
from effects.domain.effect_head_layout import active_fields
from effects.domain.records import (
    EffectRecord,
)
from effects.domain.split import CorpusSplit
from effects.domain.strata import VALIDATION_STRATA, Stratum
from effects.domain.text_holdout import normalize_script_text
from effects.infrastructure.converted_trees import ConvertedTrees
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.record_io import (
    MixedGenerationsError,
    iter_shards,
    read_shard,
    refuse_mixed_generations,
)

logger = logging.getLogger(__name__)


#: Unique-text resolution records the card-disjoint stratum should hold
#: before gate 1's margins mean anything. A floor, not a rule: below it
#: the run warns, at zero it stops.
MIN_HOLDOUT_RECORDS = 2000

#: Shards an epoch reads.
#:
#: Sized so an epoch is a representative sample of the corpus rather than a
#: pass over it: ``--steps-per-epoch`` fixes the wall clock, and this fixes how
#: many different shards those steps are spread across. Raising it costs only
#: the per-shard load — about a third of a second against a 45-minute epoch —
#: so the ceiling is how thinly a shard can be sampled and still be worth
#: opening, not time. It scales with the shard count ``build-corpus`` writes:
#: an epoch that reads a sliver of the corpus, and a run that early-stops having
#: seen a few percent of it, measures one corner of the data.
DEFAULT_SHARDS_PER_EPOCH = 256

#: Worker processes preparing training steps (``--prefetch-workers``). Three,
#: because a step's host half costs about as much CPU as its device half does
#: wall time, and a worker also spends a shard's first seconds parsing it: two
#: workers keep up with no margin, three keep up with one parsing. More buy
#: nothing once the device is the bottleneck and cost each a shard of memory.
DEFAULT_PREFETCH_WORKERS = 3


# ── reading the corpus one shard at a time ──────────────────────────────


def epoch_shards(
    shards: Sequence[Path], *, epoch: int, per_epoch: int, seed: int,
) -> list[Path]:
    """The shards epoch ``epoch`` (1-based) reads, drawn across the corpus.

    Drawn at random rather than taken as a contiguous run of the list, because
    a contiguous block is the worst composition an epoch can have. The list is
    in path order, so a block is one collection run's consecutive worker
    lifetimes: the games in it share pools, share a deck-building pass, and are
    about as correlated as two games in the corpus ever get. Worse, it never
    leaves the family that sorts first. On a 3,194-shard curated corpus the
    full-strength shards start at index 2,426 and the variants at 3,125, so
    eighteen shards an epoch walked to index 161 in nine epochs and a full
    forty-epoch run would have reached 720 — reading neither, and training a
    model that never saw a synthetic variant on a corpus built to supply them.

    ``epoch`` no longer selects the block, but it stays in the signature and
    seeds the draw alongside ``seed``, so an epoch's composition is a function
    of those two alone. Deliberately not the training RNG: drawn from that,
    an epoch's shards would depend on how many batches the epochs before it
    planned, and a run resumed or reconfigured anywhere upstream would read a
    different corpus while reporting the same seed.
    """
    if not shards or per_epoch <= 0:
        return []
    # Seeded from a string rather than a tuple: 3.14 accepts only None, int,
    # float, str, bytes and bytearray, and a tuple raises at the first draw.
    draw = random.Random(f"{seed}:{epoch}")
    if per_epoch >= len(shards):
        picks = list(shards)
        draw.shuffle(picks)
        return picks
    return draw.sample(list(shards), per_epoch)


def steps_per_shard(total: int, count: int) -> list[int]:
    """``total`` training steps divided as evenly as possible over ``count``.

    A step is one batch's forward and backward; under ``--grad-accum`` several
    of them make one optimizer step.

    Spread rather than piled onto the last shard: every shard in the epoch
    should contribute comparably, whatever the remainder.
    """
    if count <= 0:
        return []
    base, remainder = divmod(max(total, 0), count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


class EmptyHoldoutError(RuntimeError):
    """The card-disjoint stratum holds no records."""


def check_holdout(
    *,
    card_disjoint_records: int,
    unique_text_records: int,
    minimum: int = MIN_HOLDOUT_RECORDS,
) -> str | None:
    """Stop on an empty card-disjoint stratum; warn on a thin one (FR-088b).

    Empty is a failure rather than a warning because the run would otherwise
    write nothing at all: validation over no records is ``nan``, ``nan`` never
    compares below the running best, so no epoch is ever recorded as best and
    the checkpoint is never saved.

    Thin is the more expensive case and can only be a warning, since the floor
    is a judgement rather than a rule. A few hundred records produce a real
    number, and best-checkpoint selection, early stopping and gate 1's three
    margins all read it.

    Returns:
        A warning to log, or None when the stratum is big enough.

    Raises:
        EmptyHoldoutError: When the stratum holds no records.
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


# ── configuration ───────────────────────────────────────────────────────


@dataclass
class TrainEffectModelConfig:
    """The trainer's flag surface (contracts/cli.md § train-effect-model).

    ``help`` metadata is the flag's help, so a default lives here alone.
    """

    #: A curated dataset directory built by ``build-corpus``, and the trainer's
    #: only input (FR-125 withdrawn): its manifest already decides the split,
    #: the holdout, the validation samples and the rarity table.
    corpus: str | None = None
    cards_folders: tuple[Path, ...] = tuple(
        Path(folder) for folder in default_paths.DEFAULT_CARDS_FOLDERS
    )
    variant_scripts: Path | None = None
    #: ``--holdout-unit``: None takes the manifest's recorded unit (FR-041).
    holdout_unit: str | None = None
    vocab_path: Path = field(
        default_factory=lambda: Path(default_paths.DEFAULT_VOCAB_PATH),
    )
    #: ``None`` draws one from the OS and logs it, because a pinned default
    #: quietly makes every run sample the same shards in the same order — one
    #: arrangement of the corpus measured many times. The drawn value is
    #: recorded on the checkpoint, so a run stays reproducible after the fact.
    seed: int | None = field(default=None, metadata={"help": (
        "Seeds weight init, batch planning and each epoch's shard draw; drawn "
        "from the OS and logged when omitted, so any run can be repeated"
    )})
    keyword_definitions: Path = field(
        default_factory=lambda: Path(default_paths.DEFAULT_KEYWORD_DEFINITIONS),
    )
    model_output: Path | None = None
    variant: str = Baseline.FULL
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
    #: The script-API head's weight. Its term sums ~1,245 parameter-key BCEs per
    #: text (one reduction per row, as every head's), which at initialization is
    #: two orders of magnitude above the value term; this weight starts it on the
    #: scale of the value and MLM terms rather than beside the per-entity loss.
    api_weight: float = 0.003
    #: An epoch rather than a step so validation before and after the switch
    #: scores different field sets at an epoch boundary, never inside one: a
    #: switch mid-epoch validates the new fields' untrained heads at the
    #: epoch's end, which reads as a loss spike.
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
    #: Where a step is prepared never changes what it holds, so this is not
    #: recorded among the settings.
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
        # Refused before anything is read (FR-062): the encoder's attention
        # splits its width across a fixed head count. Imported here because
        # the encoder module loads torch, which this module does not.
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

    def resolved_model_output(self) -> Path:
        from effects.infrastructure.effect_model_store import model_output_for

        if self.model_output is not None:
            return Path(self.model_output)
        return model_output_for(self.variant)


@dataclass(frozen=True)
class CorpusFacts:
    """What a run reads off its curated corpus's manifest, once, at startup.

    The trainer derives none of it (FR-125 withdrawn): ``build-corpus``
    decided the split, the holdout and the rarity table, and a run reads them
    here and stamps them onto every checkpoint it saves.
    """

    #: The split, enumerated: the host half keeps its validation games out of
    #: training, and the checkpoint records it.
    split: CorpusSplit = field(default_factory=CorpusSplit)
    #: The holdout rule the corpus was composed against. The trainer has no
    #: flags for it: a depleted collection run was generated against these,
    #: and re-deriving them here could only disagree.
    holdout_permille: int = 0
    holdout_max_carriers: int = 0
    holdout_unit: str = "text"
    #: Records in the manifest's gate-1 slice — resolution records whose
    #: acting text is on no training card. Read rather than counted (FR-088b):
    #: the build already sized the slice.
    gate_one_records: int = 0
    #: The corpus-wide ``text -> games`` table (FR-146), threaded into every
    #: ``sample_weights`` call the run makes.
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
            holdout_permille=manifest.holdout_permille,
            holdout_max_carriers=manifest.holdout_max_carriers,
            holdout_unit=manifest.holdout_unit,
            gate_one_records=manifest.per_stratum.get(Stratum.GATE_ONE, 0),
            rarity=manifest.rarity,
            legality_mode_counts=manifest.legality_mode_counts or None,
            digest=manifest.digest(),
        )

    def provenance(self, config: TrainEffectModelConfig):
        """What a checkpoint of this run records about the data it saw.

        Both strata are the corpus's own, so every run against one dataset
        records the same split whatever it read or how early it stopped. That
        is what the evaluator needs: it scores the games the checkpoint names,
        and a boundary recomputed against a grown corpus would not be the one
        the model trained against.
        """
        from effects.infrastructure.effect_model_store import (
            SplitProvenance,
            content_hash,
        )

        return SplitProvenance(
            held_out_cards=self.split.held_out_cards,
            card_disjoint_games=tuple(sorted(self.split.card_disjoint_games)),
            game_disjoint_games=tuple(sorted(self.split.game_disjoint_games)),
            vocab_path=str(config.vocab_path),
            keyword_definitions_path=str(config.keyword_definitions),
            vocab_hash=content_hash(Path(config.vocab_path)),
            keyword_definitions_hash=content_hash(Path(config.keyword_definitions)),
            withheld_keyword=config.withhold_keyword,
            holdout_permille=self.holdout_permille,
            holdout_max_carriers=self.holdout_max_carriers,
            corpus_path=config.corpus or "",
            corpus_digest=self.digest,
            holdout_unit=self.holdout_unit,
        )


class SurfaceMismatchError(ValueError):
    """A curated dataset's rarity table was keyed on the other surface."""


def require_matching_surface(
    *, manifest_surface: str, vocab_path: Path, corpus: str,
) -> None:
    """Refuse a vocabulary whose encoding surface is not the dataset's (FR-141).

    The manifest's rarity table keys every ability text by the surface
    ``build-corpus`` resolved texts on, which ``--vocab-path`` decided there
    and decides again here. Read with a vocabulary on the other surface every
    lookup misses, and ``sample_weights`` reads a miss as "a shard collected
    after the dataset was built" and falls back to the resident shard's own
    count — silently reverting to exactly the per-shard weighting FR-141
    exists to replace, with nothing logged and every number still looking
    valid.
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


def rarity_coverage(
    texts: Iterable[str | None], rarity: Mapping[str, int],
) -> tuple[int, int]:
    """How many of the distinct ability texts present the table names.

    Distinct texts rather than records, because that is the unit the table is
    keyed by: a shard whose one uncovered text carries half its records is
    still one miss.
    """
    distinct = {text for text in texts if text is not None}
    return sum(1 for text in distinct if text in rarity), len(distinct)


# Hardcoded, not flags (FR-095).
LEARNING_RATE = 1e-4
WARMUP_FRACTION = 0.05
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01


def warmup_steps(*, epochs: int, steps_per_epoch: int) -> int:
    """Linear warmup over the first 5% of the scheduled steps."""
    return max(1, int(epochs * steps_per_epoch * WARMUP_FRACTION))


def fields_for_epoch(config: TrainEffectModelConfig, *, present: frozenset[str], epoch: int):
    """The field set every step of ``epoch`` trains with, and validation scores.

    The sparse group is on from the first step of ``--curriculum-epoch``; at 1
    or below, from the run's first step.
    """
    return active_fields(present, sparse=epoch >= config.curriculum_epoch)


def learning_rate_at(step: int, *, warmup: int) -> float:
    """Linear warmup, then constant.

    Constant rather than decayed because early stopping, not the schedule, is
    what ends the run — a decayed rate would make the patience counter measure
    the schedule instead of the model.
    """
    if step >= warmup:
        return LEARNING_RATE
    return LEARNING_RATE * (step + 1) / warmup


# ── the epoch loop ──────────────────────────────────────────────────────


@dataclass
class EpochResult:
    """One epoch's validation, on both strata."""

    epoch: int
    train_loss: float
    card_disjoint_loss: float
    game_disjoint_loss: float


class EarlyStopper:
    """Stops after ``patience`` epochs with no new card-disjoint best.

    Card-disjoint rather than game-disjoint because that is the number the model
    ships on: it stands in for deployment to a set the encoder has never seen,
    while the game-disjoint stratum measures in-distribution fit.
    """

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.since_best = 0

    def update(self, card_disjoint_loss: float) -> bool:
        """Record an epoch; return True when it set a new best."""
        if card_disjoint_loss < self.best:
            self.best = card_disjoint_loss
            self.since_best = 0
            return True
        self.since_best += 1
        return False

    def state_dict(self) -> dict:
        return {"best": self.best, "since_best": self.since_best}

    def load_state_dict(self, state: dict) -> None:
        self.best = state["best"]
        self.since_best = state["since_best"]

    def reset(self) -> None:
        """Forget the best: the objective changed, so no earlier number compares."""
        self.best = float("inf")
        self.since_best = 0

    @property
    def should_stop(self) -> bool:
        """Never with ``patience`` 0: that run trains every epoch it is given."""
        return self.patience > 0 and self.since_best >= self.patience


# ── the run ─────────────────────────────────────────────────────────────


def load_shard(path: Path) -> list[EffectRecord]:
    """One shard's records, the largest unit the trainer ever holds.

    A median shard costs about a gigabyte of resident memory once parsed, and
    the corpus as a whole would cost hundreds — which is why nothing here ever
    holds more than one shard at a time.
    """
    return list(read_shard(Path(path)))


def corpus_shards(records_dir: Path) -> list[Path]:
    return iter_shards(Path(records_dir))


class HoldoutMismatchError(ValueError):
    """The trainer's own holdout disagrees with the one its corpus recorded."""


def check_manifest_holdout(manifest, config: TrainEffectModelConfig) -> None:
    """Refuse a holdout unit or a held-out text set the manifest does not record.

    FR-041: ``--holdout-unit`` defaults to the manifest's unit and is refused
    when it names the other one. The holdout is then recomputed here, with the
    same function ``holdout-cards`` and ``build-corpus`` call, over this run's
    sidecar roots; a run whose sidecars hold out different texts than the
    corpus was built against would train on texts it believes are held out,
    so it is refused before any step rather than discovered at evaluation.
    """
    unit = manifest.holdout_unit
    if config.holdout_unit is not None and config.holdout_unit != unit:
        raise HoldoutMismatchError(
            f"--holdout-unit {config.holdout_unit} differs from the unit the "
            f"corpus at {config.corpus} was built under ({unit}). The split is "
            "the corpus's; rebuild it with build-corpus --holdout-unit "
            f"{config.holdout_unit}, or drop the flag."
        )
    trees = ConvertedTrees.from_folders(config.cards_folders)
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
    recomputed = trees.holdout(
        permille=manifest.holdout_permille,
        max_carriers=manifest.holdout_max_carriers,
        unit=unit,
    )
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


def run(config: TrainEffectModelConfig) -> int:
    """Train the encoder and effect head jointly. Returns an exit code.

    Torch is imported here rather than at module scope, so
    ``python -m effects --help`` and the unit suite stay fast. The corpus is
    never loaded whole: the shard list is read here and the records behind it
    one shard at a time, inside the loop.
    """
    if config.corpus is None:
        logger.error(
            "train-effect-model reads a curated corpus: pass --corpus DIR, built "
            "with `python -m effects build-corpus`. Training straight from raw "
            "shards was withdrawn with FR-125."
        )
        return 1
    store =CorpusStore(Path(config.corpus))
    manifest = store.load()
    require_matching_surface(
        manifest_surface=manifest.surface, vocab_path=Path(config.vocab_path),
        corpus=config.corpus,
    )
    training_shards = corpus_shards(store.training_dir)
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
    try:
        # Before any record is read (FR-033).
        refuse_mixed_generations(iter_shards(Path(config.corpus)))
        check_manifest_holdout(manifest, config)
    except (HoldoutMismatchError, MixedGenerationsError) as exc:
        logger.error("%s", exc)
        return 1
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
