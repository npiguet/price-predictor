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
weighted shuffle *without replacement* over the resident shard (FR-086): the
per-batch class draw is gone, because a sixteen-record shard resampled with
replacement was replayed 640 times.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from effects.domain.ability_tokenizer import keyword_token
from effects.domain.records import (
    EffectRecord,
    acting_keys,
)
from effects.domain.split import (
    CorpusSplit,
    HeldOutCards,
    text_keyed_holdout,
)
from effects.domain.variants import Variant

if TYPE_CHECKING:
    from effects.infrastructure.sidecar_io import SidecarCache

logger = logging.getLogger(__name__)


#: Unique-text resolution records the card-disjoint stratum should hold
#: before gate 1's margins mean anything. A floor, not a rule: below it
#: the run warns, at zero it stops.
MIN_HOLDOUT_RECORDS = 2000
RANDOM_SEED = 42

#: Shards an epoch reads.
#:
#: Sized so an epoch is a representative sample of the corpus rather than a
#: pass over it: ``--steps-per-epoch`` fixes the wall clock, and this fixes how
#: many different shards those steps are spread across. Raising it costs only
#: the per-shard load — about a third of a second against a 45-minute epoch —
#: so the ceiling is how thinly a shard can be sampled and still be worth
#: opening, not time.
#:
#: The previous value of 18 was correct for the corpus it was written against
#: (701 raw shards, covered once by 40 epochs) and silently wrong for the one
#: ``build-corpus`` writes, which is 3,194. Eighteen an epoch reads 0.6% of
#: that, and a run that early-stops at epoch 9 has seen 5% of it.
DEFAULT_SHARDS_PER_EPOCH = 256

#: Worker processes preparing training steps (``--prefetch-workers``). Three,
#: because a step's host half costs about as much CPU as its device half does
#: wall time, and a worker also spends a shard's first seconds parsing it: two
#: workers keep up with no margin, three keep up with one parsing. More buy
#: nothing once the device is the bottleneck and cost each a shard of memory.
DEFAULT_PREFETCH_WORKERS = 3


def load_card_texts(
    card_files: Mapping[str, str], sidecars: SidecarCache,
) -> dict[str, list[str]]:
    """``card name -> the script texts of its sidecar's lines``.

    The I/O half of the holdout, kept apart from the selection so the rule can
    be exercised without a converted tree on disk.
    """
    out: dict[str, list[str]] = {}
    for name, script_file in card_files.items():
        sidecar = sidecars.get(script_file)
        if sidecar is None:
            continue
        out[name] = [
            line.script_text for line in sidecar.lines if line.script_text
        ]
    return out


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
    """``total`` optimizer steps divided as evenly as possible over ``count``.

    Spread rather than piled onto the last shard: every shard in the epoch
    should contribute comparably, whatever the remainder.
    """
    if count <= 0:
        return []
    base, remainder = divmod(max(total, 0), count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


def load_card_files(cards_folder: Path) -> dict[str, str]:
    """``card name -> tree-prefixed script file`` for one converted tree.

    Names come from each file's ``name:`` line, which is authoritative — the
    filename is a sanitized derivative and several cards share a stem.
    """
    from price_predictor.domain.tokenizer import extract_card_name

    cards_folder = Path(cards_folder)
    out: dict[str, str] = {}
    for path in sorted(cards_folder.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        name = extract_card_name(text)
        if not name:
            continue
        relative = path.relative_to(cards_folder).as_posix()
        out[name] = f"cardsfolder/{relative}"
    return out


def load_first_printings(printings_path: Path) -> dict[str, str]:
    """``card name -> earliest release date`` from an MTGJSON dump.

    Faces are indexed under their own names as well as the combined one, so a
    record naming either resolves.
    """
    import json

    data = json.loads(Path(printings_path).read_text(encoding="utf-8"))
    earliest: dict[str, str] = {}
    for set_info in data.get("data", {}).values():
        cards = set_info if isinstance(set_info, list) else set_info.get("cards", [])
        release = (
            set_info.get("releaseDate", "") if isinstance(set_info, dict) else ""
        )
        for card in cards:
            for key in {card.get("name"), card.get("faceName")} - {None, ""}:
                previous = earliest.get(key)
                if previous is None or (release and release < previous):
                    earliest[key] = release
    return earliest


class MissingSplitError(RuntimeError):
    """A variant run was started without ``--split-from``."""


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


def require_split_from(
    variant: str, split_from: Path | None, *, corpus: str | None = None,
) -> None:
    """A variant run must inherit the split it is a baseline for (FR-091).

    Without it the run would silently compute its own split and the comparison
    would be between two models that saw different games — which looks like a
    result and is not one. A shared ``--corpus`` satisfies this at least as
    firmly as ``--split-from``: the manifest enumerates the split directly, so
    every run reading the same curated dataset trains against the same games
    (FR-146). ``--corpus`` is required on every run, so ``--split-from`` is
    accepted alongside it only for a variant run's compatibility with an
    older invocation; the split it would have computed is never read.
    """
    if variant != Variant.FULL and split_from is None and corpus is None:
        raise MissingSplitError(
            f"--variant {variant} requires --split-from PATH or --corpus DIR: "
            "a baseline must inherit the split, vocabulary and "
            "keyword-definition paths of the full run it baselines (via "
            "--split-from), or read the same curated dataset (via --corpus) — "
            "otherwise the comparison is between two models that saw "
            "different games."
        )


# ── configuration ───────────────────────────────────────────────────────


@dataclass
class TrainEffectModelConfig:
    """The trainer's flag surface (contracts/cli.md § train-effect-model)."""

    #: A curated dataset directory built by ``build-corpus``, and the trainer's
    #: only input (FR-125 withdrawn): its manifest already decides the split,
    #: the holdout, the validation samples and the rarity table.
    corpus: str | None = None
    cards_folders: tuple[Path, ...] = (
        Path("output/cardsfolder/"), Path("output/tokenscripts/"),
    )
    variant_scripts: Path | None = None
    split_from: Path | None = None
    #: ``--holdout-unit``: None takes the manifest's recorded unit (FR-041).
    holdout_unit: str | None = None
    vocab_path: Path = field(default_factory=lambda: Path("models/effects/vocab.txt"))
    #: Seeds weight init, batch planning and each epoch's shard draw.
    #:
    #: ``None`` draws one from the OS and logs it, because these runs are not
    #: meant to be repeatable and a pinned default quietly makes every run
    #: sample the same shards in the same order — which looks like a stable
    #: result and is one arrangement of the corpus measured many times. The
    #: drawn value is reported at startup and recorded on the checkpoint, so a
    #: run stays reproducible after the fact by passing it back.
    seed: int | None = None
    keyword_definitions: Path = field(
        default_factory=lambda: Path("output/effects/keyword-definitions.json"),
    )
    model_output: Path | None = None
    variant: str = Variant.FULL
    e_dim: int = 64
    #: The noise ratio ``r`` on ``e``, relative to the running covariance of
    #: ``e`` itself (FR-056) — no longer a fixed σ.
    e_noise: float = 0.1
    #: The value head's loss weight (FR-058).
    value_weight: float = 0.05
    #: The ability encoder's size (FR-061); the checkpoint records both.
    encoder_layers: int = 4
    encoder_d_model: int = 256
    keyword_expand_p: float = 0.25
    context_dropout: float = 0.15
    #: Counterfactual copies of each qualifying damage or -X/-X resolution a
    #: training step makes, labelled by the rules (FR-063e); 0 makes none.
    lethality_copies: int = 0
    #: A factor on the learning rate at every step (FR-063f). Not a resume
    #: setting: a resumed run may continue at another rate.
    lr_scale: float = 1.0
    #: Also keep each epoch's checkpoint under ``<model-output>/epochs/``
    #: (FR-063f), beside the best-only ones, so any epoch can be probed.
    save_every_epoch: bool = False
    mlm_weight: float = 0.1
    mlm_mask_prob: float = 0.15
    api_weight: float = 0.05
    #: The epoch whose first step enables the sparse field group (FR-082).
    #: An epoch rather than a step so validation before and after the switch
    #: scores different field sets at an epoch boundary, never inside one:
    #: the first run enabled the group at shard 244 of epoch 2 and validated
    #: twenty-five untrained heads at that epoch's end, a 134-loss spike.
    curriculum_epoch: int = 3
    batch_size: int = 32
    grad_accum: int = 1
    context_cache: bool = False
    cache_refresh: int = 500
    steps_per_epoch: int = 5_000
    shards_per_epoch: int = DEFAULT_SHARDS_PER_EPOCH
    #: Worker processes preparing each step's host half (``--prefetch-workers``);
    #: 0 prepares it in the training process. Where a step is prepared never
    #: changes what it holds, so this is not recorded among the settings.
    prefetch_workers: int = DEFAULT_PREFETCH_WORKERS
    epochs: int = 40
    patience: int = 5
    withhold_keyword: str | None = None
    #: ``--resume``: continue from the ``resume.pt`` the run left in
    #: ``--model-output`` after its last completed epoch.
    resume: bool = False

    @property
    def curriculum_step(self) -> int:
        """The optimizer step the sparse group enables at; 0 for epoch 1."""
        return max(0, self.curriculum_epoch - 1) * self.steps_per_epoch

    def resolved_model_output(self) -> Path:
        from effects.infrastructure.effect_model_store import model_output_for

        if self.model_output is not None:
            return Path(self.model_output)
        return model_output_for(self.variant)


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
    """The field set every step of ``epoch`` trains with, and validation scores."""
    from effects.domain.effect_model import active_fields

    return active_fields(
        present_classes=present,
        step=(epoch - 1) * config.steps_per_epoch,
        curriculum_step=config.curriculum_step,
    )


def learning_rate_at(step: int, *, warmup: int) -> float:
    """Linear warmup, then constant.

    Constant rather than decayed because early stopping, not the schedule, is
    what ends the run — a decayed rate would make the patience counter measure
    the schedule instead of the model.
    """
    if step >= warmup:
        return LEARNING_RATE
    return LEARNING_RATE * (step + 1) / warmup


# ── keyword withholding (FR-117) ────────────────────────────────────────


def withheld_keyword_tokens(withheld: str | None) -> frozenset[str]:
    """The token form of the keyword ``--withhold-keyword`` names, if any.

    Withholding is what gives the zero-shot check something to measure: the
    model never sees the keyword's *token*, only the sentence it stands for —
    the batcher force-expands it to its definition — so the check asks whether
    a keyword it has only ever read as prose lands near the right neighbours.
    """
    return frozenset({keyword_token(withheld)}) if withheld else frozenset()


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

    def reset(self) -> None:
        """Forget the best: the objective changed, so no earlier number compares."""
        self.best = float("inf")
        self.since_best = 0

    @property
    def should_stop(self) -> bool:
        """Never with ``patience`` 0: that run trains every epoch it is given."""
        return self.patience > 0 and self.since_best >= self.patience


# ── the run ─────────────────────────────────────────────────────────────


def text_for_key(key, sidecars, surface: str) -> str | None:
    """One provenance key's text on ``surface``, or None where it has none.

    The single key-level definition. ``build-corpus`` surveys by provenance key
    and folds to text afterwards, and a second spelling of this would key the
    rarity table differently from every lookup the trainer makes against it.

    Falls back to the key itself where the sidecar carries neither surface, so
    a line that cannot be read is still a distinct unit for rarity weighting
    rather than collapsing into every other unreadable line.
    """
    from effects.domain.ability_encoder import encoding_text
    from effects.infrastructure.sidecar_io import UnconfiguredTree

    try:
        line = sidecars.line_for(key)
    except UnconfiguredTree:
        return None
    if line is None:
        return None
    text = encoding_text(line, sidecars.prose_for(key), surface)
    return text or f"{key.script_file}:{key.trait_kind}:{key.index_within_kind}"


def ability_text_of(
    record: EffectRecord, sidecars, surface: str = "prose",
) -> str | None:
    """The acting line's text on ``surface``, or None where no line acts.

    ``combat`` and the ``attackers``/``blockers`` subkinds have none, which is
    what makes them sample uniformly rather than by rarity.
    """
    for key in acting_keys(record):
        text = text_for_key(key, sidecars, surface)
        if text is not None:
            return text
    return None


def load_shard(path: Path) -> list[EffectRecord]:
    """One shard's records, the largest unit the trainer ever holds.

    A median shard costs about a gigabyte of resident memory once parsed, and
    the corpus as a whole would cost hundreds — which is why nothing here ever
    holds more than one shard at a time.
    """
    from effects.infrastructure.record_io import read_shard

    return list(read_shard(Path(path)))


def corpus_shards(records_dir: Path) -> list[Path]:
    from effects.infrastructure.record_io import iter_shards

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
    from effects.domain.text_holdout import normalize_script_text
    from effects.infrastructure.sidecar_io import SidecarCache, sidecar_roots

    unit = manifest.holdout_unit
    if config.holdout_unit is not None and config.holdout_unit != unit:
        raise HoldoutMismatchError(
            f"--holdout-unit {config.holdout_unit} differs from the unit the "
            f"corpus at {config.corpus} was built under ({unit}). The split is "
            "the corpus's; rebuild it with build-corpus --holdout-unit "
            f"{config.holdout_unit}, or drop the flag."
        )
    roots = sidecar_roots(config.cards_folders)
    cards_folder = roots.get("cardsfolder")
    if cards_folder is None or not cards_folder.is_dir():
        raise HoldoutMismatchError(
            "the holdout cannot be recomputed: no converted card tree among "
            f"--cards-folder {', '.join(str(f) for f in config.cards_folders)}"
        )
    import time

    started = time.monotonic()
    card_files = load_card_files(cards_folder)
    logger.info(
        "Recomputing the %s holdout over %d cards' sidecars under %s.",
        unit, len(card_files), cards_folder,
    )
    recomputed = text_keyed_holdout(
        card_files, load_card_texts(card_files, SidecarCache(roots)),
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
    require_split_from(config.variant, config.split_from, corpus=config.corpus)
    if config.split_from is not None:
        logger.warning(
            "--split-from %s is accepted for compatibility with an older "
            "invocation, but nothing is read from this path: the manifest of "
            "--corpus is authoritative for the split.", config.split_from,
        )
    if config.context_cache:
        logger.warning(
            "--context-cache is accepted but not wired into the batcher in "
            "this trainer; context abilities are re-encoded live."
        )

    if config.corpus is None:
        logger.error(
            "train-effect-model reads a curated corpus: pass --corpus DIR, built "
            "with `python -m effects build-corpus`. Training straight from raw "
            "shards was withdrawn with FR-125."
        )
        return 1
    from effects.infrastructure.corpus_store import CorpusStore

    store = CorpusStore(Path(config.corpus))
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
        stratum: store.sample_path(stratum) for stratum in ("card-disjoint", "game-disjoint")
    }
    missing = [str(p) for p in validation_samples.values() if not p.exists()]
    if missing:
        logger.error(
            "The corpus at %s has no validation samples (%s). Rebuild it with a "
            "build-corpus that writes validation/samples/.", config.corpus, ", ".join(missing),
        )
        return 1
    from effects.infrastructure.record_io import (
        MixedGenerationsError,
        iter_shards,
        refuse_mixed_generations,
    )

    try:
        # Before any record is read (FR-033).
        refuse_mixed_generations(iter_shards(Path(config.corpus)))
        check_manifest_holdout(manifest, config)
    except (HoldoutMismatchError, MixedGenerationsError) as exc:
        logger.error("%s", exc)
        return 1
    # Copied into the checkpoint's split (FR-042), so evaluation knows the
    # unit without re-reading the manifest.
    config.holdout_unit = manifest.holdout_unit
    held_out = HeldOutCards(
        names=frozenset(manifest.held_out_cards), script_files=frozenset(),
        texts=frozenset(manifest.held_out_texts),
    )
    inherited = CorpusSplit(
        held_out_cards=manifest.held_out_cards,
        card_disjoint_games=frozenset(manifest.card_disjoint_games),
        game_disjoint_games=frozenset(manifest.game_disjoint_games),
    )
    logger.info(
        "Curated corpus at %s: %d training shards, %d gate-one record(s), "
        "%d held-out ability text(s) on %d card(s).",
        config.corpus, len(training_shards), manifest.per_stratum.get("gate-one", 0),
        len(manifest.held_out_texts), len(manifest.held_out_cards),
    )

    from effects.application.training_loop import TrainingLoop

    return TrainingLoop(
        config,
        held_out=held_out,
        inherited=inherited,
        training_shards=training_shards,
        validation_samples=validation_samples,
        holdout_permille=manifest.holdout_permille,
        holdout_max_carriers=manifest.holdout_max_carriers,
        gate_one_records=manifest.per_stratum.get("gate-one", 0),
        rarity=manifest.rarity,
        legality_mode_counts=manifest.legality_mode_counts or None,
        corpus_digest=manifest.digest(),
    ).execute()
