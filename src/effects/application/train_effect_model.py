"""``train-effect-model``: the split, the sampling mixture, and the training loop.

Three things here decide whether the numbers the evaluator reports mean anything.

**The split excludes whole games, not held-out rows.** Records from one game
share a board, so a held-out card sitting in the *context* of a training record
leaks through its context role even when it is not the acting ability — and the
context role is exactly how the design trains each embedding from both
directions. Excluding the row and keeping the game would make gate 1, the
shipping gate, score partly on cards the encoder had already seen.

**The mixture renormalizes over the classes present.** A stage-one corpus has
three of the eight classes. Renormalizing lets the same trainer run against it
without the absent classes silently starving the present ones.

**Rarity weighting counts games, not records.** One long game can produce
hundreds of records of one ability; weighting by records would make that ability
look well-observed when it has been seen in one board state.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from effects.domain.effect_model import (
    CLASS_COMBAT,
    CLASS_CONTINUOUS,
    CLASS_PLAYABILITY_DECISION,
    CLASS_PLAYABILITY_LEGALITY,
    CLASS_RESOLUTION_COST,
    CLASS_RESOLUTION_EFFECT,
    CLASS_REWRITE,
    CLASS_TRIGGER,
    SAMPLING_CLASSES,
)
from effects.domain.records import EffectRecord, Moment, PlayabilitySubkind, RecordKind

logger = logging.getLogger(__name__)

#: The eight-class mixture (root spec § Training). Shares are renormalized over
#: whichever classes the corpus actually holds.
DEFAULT_KIND_MIX: dict[str, float] = {
    CLASS_RESOLUTION_EFFECT: 0.30,
    CLASS_COMBAT: 0.20,
    CLASS_CONTINUOUS: 0.12,
    CLASS_PLAYABILITY_DECISION: 0.10,
    CLASS_RESOLUTION_COST: 0.08,
    CLASS_TRIGGER: 0.08,
    CLASS_REWRITE: 0.07,
    CLASS_PLAYABILITY_LEGALITY: 0.05,
}

#: Classes whose records have no acting ability text, so rarity weighting has
#: nothing to key on and they sample uniformly instead (FR-087).
UNIFORM_CLASSES: frozenset[str] = frozenset({
    CLASS_COMBAT, CLASS_PLAYABILITY_LEGALITY,
})

#: Fraction of the card corpus held out card-disjointly (FR-088).
CARD_HOLDOUT_FRACTION = 0.08
#: Rarity weight ceiling, as a multiple of the most-observed text's weight.
RARITY_CAP = 20.0
RANDOM_SEED = 42

#: Shards an epoch reads. The corpus is 701 shards and a full run is 40 epochs,
#: so this covers it once rather than re-reading its opening slice.
DEFAULT_SHARDS_PER_EPOCH = 18
#: Shards held back for validation and never trained on. Reserving whole shards
#: rather than freezing a record count is what makes the validation sample
#: representative: a shard carries ~90 games, drawn from one stretch of
#: collection rather than from whichever shards the first epoch happened to read.
RESERVED_VALIDATION_SHARDS = 4
#: Validation records cached per stratum from the reserved shards. A full shard
#: is ~20k records, and pushing all of them through the model every epoch would
#: cost ~625 batches against the 8 that per-epoch validation costs today.
VALIDATION_RECORDS_PER_STRATUM = 2_048


# ── sampling classes ────────────────────────────────────────────────────


def sampling_class(record: EffectRecord) -> str:
    """The mixture class a record belongs to."""
    match record.kind:
        case RecordKind.RESOLUTION:
            return (
                CLASS_RESOLUTION_COST
                if record.moment is Moment.ACTIVATION
                else CLASS_RESOLUTION_EFFECT
            )
        case RecordKind.COMBAT:
            return CLASS_COMBAT
        case RecordKind.CONTINUOUS:
            return CLASS_CONTINUOUS
        case RecordKind.TRIGGER:
            return CLASS_TRIGGER
        case RecordKind.REWRITE:
            return CLASS_REWRITE
        case RecordKind.PLAYABILITY:
            return (
                CLASS_PLAYABILITY_DECISION
                if record.subkind is PlayabilitySubkind.DECISION
                else CLASS_PLAYABILITY_LEGALITY
            )
    raise ValueError(f"no sampling class for record kind {record.kind}")


def renormalize_mix(
    mix: dict[str, float], present: Iterable[str],
) -> dict[str, float]:
    """Restrict ``mix`` to the classes present and rescale it to sum to 1.

    A stage-one corpus holds three of the eight classes, so the trainer has to
    run without the other five rather than sampling zero-sized slices of them.

    Raises:
        ValueError: If no present class has a share — there is nothing to draw.
    """
    present = set(present)
    kept = {name: share for name, share in mix.items() if name in present and share > 0}
    total = sum(kept.values())
    if total <= 0:
        raise ValueError(
            f"no sampling class in the corpus has a share; present={sorted(present)}, "
            f"mix={sorted(mix)}"
        )
    return {name: share / total for name, share in kept.items()}


def parse_kind_mix(value: str | None) -> dict[str, float]:
    """``--kind-mix`` as ``class=share,…``; the default mixture when absent."""
    if not value:
        return dict(DEFAULT_KIND_MIX)
    mix: dict[str, float] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        name, _, share = part.partition("=")
        name = name.strip()
        if name not in SAMPLING_CLASSES:
            raise ValueError(
                f"unknown sampling class {name!r}; known: {list(SAMPLING_CLASSES)}"
            )
        mix[name] = float(share)
    return mix


# ── rarity weighting ────────────────────────────────────────────────────


def effective_games(
    records: Iterable[EffectRecord], text_of,
) -> dict[str, int]:
    """Distinct games contributing a record of each unique ability text.

    Games rather than records: one long game can produce hundreds of records of
    one ability, and weighting by records would call that ability well-observed
    when it has been seen in a single board state.
    """
    games: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        text = text_of(record)
        if text is not None:
            games[text].add(record.game_id)
    return {text: len(ids) for text, ids in games.items()}


def rarity_weights(
    games_per_text: dict[str, int], *, cap: float = RARITY_CAP,
) -> dict[str, float]:
    """``effective_games ** -0.5``, capped at ``cap`` times the smallest weight.

    The exponent flattens the corpus's long tail without erasing it: an ability
    seen in one game gets more weight than one seen in a hundred, but not a
    hundred times more. The cap stops a single-game ability from dominating a
    batch, and is expressed against the *most-observed* text — the one with the
    smallest weight — so it scales with the corpus rather than being absolute.
    """
    if not games_per_text:
        return {}
    raw = {
        text: float(count) ** -0.5 if count > 0 else 0.0
        for text, count in games_per_text.items()
    }
    positive = [w for w in raw.values() if w > 0]
    if not positive:
        return raw
    ceiling = min(positive) * cap
    return {text: min(weight, ceiling) for text, weight in raw.items()}


def sample_weights(
    records: Sequence[EffectRecord], text_of,
) -> list[float]:
    """Per-record sampling weight within its class.

    A record with no acting ability text weighs 1: ``combat`` and
    ``playability``/``attackers``/``blockers`` sample uniformly, because there
    is no ability text for rarity to key on (FR-087).
    """
    weights = rarity_weights(effective_games(records, text_of))
    out: list[float] = []
    for record in records:
        text = text_of(record)
        out.append(1.0 if text is None else weights.get(text, 1.0))
    return out


# ── the split ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HeldOutCards:
    """The card-disjoint holdout, addressable both ways a record names a card.

    A record names a card by the entity names in its snapshot and by the script
    files its provenance keys point at; matching on either catches both.
    """

    names: frozenset[str]
    script_files: frozenset[str]

    def __len__(self) -> int:
        return len(self.script_files or self.names)


@dataclass(frozen=True, slots=True)
class CorpusSplit:
    """The split a checkpoint records and the evaluator reads back.

    Both strata are stored as explicit ``game_id`` sets rather than as a rule,
    because the corpus is append-only and grows between runs: a recomputed
    split would not be the trained-against one, and the gates would score partly
    on games the model had trained on.
    """

    held_out_cards: tuple[str, ...]
    card_disjoint_games: frozenset[str]
    game_disjoint_games: frozenset[str]

    @property
    def validation_games(self) -> frozenset[str]:
        return self.card_disjoint_games | self.game_disjoint_games

    def is_training_game(self, game_id: str) -> bool:
        return game_id not in self.validation_games


def newest_first_holdout(
    card_files: dict[str, str],
    first_printing: dict[str, str],
    *,
    fraction: float = CARD_HOLDOUT_FRACTION,
) -> HeldOutCards:
    """Hold out the newest-printed cards until they cover ``fraction``.

    Args:
        card_files: ``card name -> script file``, one entry per converted card
            under ``output/cardsfolder/``. Token scripts and variant scripts are
            not in the printing order and are **not** in this denominator.
        first_printing: ``card name -> first-printing release date`` (ISO), from
            ``--printings-path``.

    Newest-first is the point: the holdout stands in for deployment to a set the
    model has never seen, and taking a random 8% would hold out cards whose
    mechanics are all over the training corpus already.

    A card with no printing date sorts as oldest, so an unrecognized name never
    ends up in the holdout by accident.
    """
    if not card_files:
        return HeldOutCards(frozenset(), frozenset())
    target = math.ceil(len(card_files) * fraction)
    ordered = sorted(
        card_files,
        key=lambda name: (first_printing.get(name) or "", name),
        reverse=True,
    )
    chosen = ordered[:target]
    return HeldOutCards(
        names=frozenset(chosen),
        script_files=frozenset(card_files[name] for name in chosen),
    )


def record_names_held_out_card(
    record: EffectRecord, held_out: HeldOutCards,
) -> bool:
    """Whether this record names a held-out card, acting or in context."""
    for entity in record.state.entities:
        if entity.name in held_out.names:
            return True
    for key in record.ability or ():
        if key.script_file in held_out.script_files:
            return True
    for entity in record.state.entities:
        for key in (*entity.printed, *entity.granted_attached,
                    *entity.granted_temporary.abilities):
            if key.script_file in held_out.script_files:
                return True
    return False


# ── reading the corpus one shard at a time ──────────────────────────────


def reserved_shard_indices(count: int, *, reserved: int) -> frozenset[int]:
    """``reserved`` shard positions spread evenly across ``count`` shards.

    Evenly rather than the first few: consecutive shards come from one worker
    over one stretch of collection, so the opening shards would sample one
    worker's first games rather than the corpus.
    """
    if reserved <= 0 or count <= 0:
        return frozenset()
    reserved = min(reserved, count)
    step = count / reserved
    return frozenset(
        min(count - 1, int(index * step + step / 2)) for index in range(reserved)
    )


def shard_games(
    records: Iterable[EffectRecord], held_out: HeldOutCards,
) -> tuple[frozenset[str], frozenset[str]]:
    """One shard's games, split by whether any record names a held-out card.

    Decidable from the shard alone because a game never spans two shards: the
    collector gives every shard its own game-id namespace. Returns
    ``(tainted, clean)``, and a game appears in exactly one of them.
    """
    tainted: set[str] = set()
    seen: set[str] = set()
    for record in records:
        seen.add(record.game_id)
        if record.game_id not in tainted and record_names_held_out_card(
            record, held_out
        ):
            tainted.add(record.game_id)
    return frozenset(tainted), frozenset(seen - tainted)


class SplitAccumulator:
    """The split, built up shard by shard instead of derived in one pass.

    A game naming a held-out card joins the card-disjoint stratum wherever it is
    found, because training on it would let the shipping gate score on cards the
    encoder had already seen. A clean game joins the game-disjoint stratum only
    when it comes from a reserved shard, which is what makes the validation
    records the same ones every epoch.

    A ``--split-from`` run inherits a fixed split instead: the corpus grows
    between runs, and re-deriving the boundary would move it.
    """

    def __init__(
        self,
        *,
        held_out_cards: Sequence[str] = (),
        inherited: CorpusSplit | None = None,
    ) -> None:
        self._held_out_cards = tuple(held_out_cards)
        self._inherited = inherited
        self._card_disjoint: set[str] = set()
        self._game_disjoint: set[str] = set()

    @classmethod
    def inheriting(cls, split: CorpusSplit) -> SplitAccumulator:
        return cls(held_out_cards=split.held_out_cards, inherited=split)

    @property
    def inherited(self) -> bool:
        return self._inherited is not None

    def note_shard(
        self,
        records: Iterable[EffectRecord],
        held_out: HeldOutCards,
        *,
        reserved: bool,
    ) -> tuple[frozenset[str], frozenset[str]]:
        """Record one shard's games and return its ``(tainted, clean)`` sets."""
        tainted, clean = shard_games(records, held_out)
        if self._inherited is None:
            self._card_disjoint |= tainted
            if reserved:
                self._game_disjoint |= clean
        return tainted, clean

    def split(self) -> CorpusSplit:
        if self._inherited is not None:
            return self._inherited
        return CorpusSplit(
            held_out_cards=self._held_out_cards,
            card_disjoint_games=frozenset(self._card_disjoint),
            game_disjoint_games=frozenset(self._game_disjoint),
        )


def epoch_shards(
    shards: Sequence[Path], *, epoch: int, per_epoch: int,
) -> list[Path]:
    """The shards epoch ``epoch`` (1-based) reads, wrapping at the end.

    Epochs advance through the list rather than re-reading the same shards, so a
    long run covers the corpus instead of overfitting its opening slice.
    """
    if not shards or per_epoch <= 0:
        return []
    start = ((epoch - 1) * per_epoch) % len(shards)
    return [shards[(start + offset) % len(shards)] for offset in range(per_epoch)]


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


# ── batch assembly ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Which records a batch draws, grouped by the game they came from.

    Grouping matters for cost, not correctness: records from one game share a
    board, so their context entities share ability texts, and one encode of a
    text then serves many records instead of one.
    """

    by_game: dict[str, list[EffectRecord]] = field(default_factory=dict)

    @property
    def records(self) -> list[EffectRecord]:
        return [r for group in self.by_game.values() for r in group]

    def unique_ability_texts(self, text_of) -> set[str]:
        return {
            text for record in self.records
            if (text := text_of(record)) is not None
        }


def plan_batch(
    pools: dict[str, Sequence[EffectRecord]],
    weights: dict[str, Sequence[float]],
    mix: dict[str, float],
    *,
    batch_size: int,
    rng: random.Random,
) -> BatchPlan:
    """Draw ``batch_size`` records under the renormalized mixture.

    Args:
        pools: candidate records per sampling class.
        weights: per-record weights parallel to ``pools``.
        mix: shares already renormalized over the classes present.
    """
    plan: dict[str, list[EffectRecord]] = defaultdict(list)
    if not mix:
        return BatchPlan({})
    classes = list(mix)
    shares = [mix[name] for name in classes]
    drawn = 0
    while drawn < batch_size:
        name = rng.choices(classes, weights=shares, k=1)[0]
        pool = pools.get(name) or ()
        if not pool:
            continue
        pool_weights = weights.get(name)
        record = rng.choices(
            list(pool),
            weights=list(pool_weights) if pool_weights else None,
            k=1,
        )[0]
        plan[record.game_id].append(record)
        drawn += 1
    return BatchPlan(dict(plan))


def class_counts(records: Iterable[EffectRecord]) -> Counter[str]:
    """How many records of each sampling class a corpus holds."""
    return Counter(sampling_class(record) for record in records)


# ── variants (FR-094) ───────────────────────────────────────────────────


VARIANT_FULL = "full"
VARIANT_IDENTITY = "identity"
VARIANT_STATE_ONLY = "state-only"
VARIANT_NO_STATE = "no-state"
VARIANT_TAXONOMY = "taxonomy"

VARIANTS: tuple[str, ...] = (
    VARIANT_FULL, VARIANT_IDENTITY, VARIANT_STATE_ONLY, VARIANT_NO_STATE,
    VARIANT_TAXONOMY,
)

#: What each baseline removes, in one line each. Gate 1 is defined against
#: ``identity``, so its definition is contract rather than commentary.
VARIANT_DESCRIPTIONS: dict[str, str] = {
    VARIANT_FULL: "the shipping model: encoder plus effect head",
    VARIANT_IDENTITY:
        "a free embedding per unique ability text replaces the encoder — it can "
        "memorize what each seen text does and knows nothing about an unseen one",
    VARIANT_STATE_ONLY:
        "all e inputs zeroed: the floor every record kind is reported against",
    VARIANT_NO_STATE:
        "every input zeroed except [ACT] and the record-kind flag, entity "
        "ability e tokens included — the average-effect control",
    VARIANT_TAXONOMY:
        "e replaced by an embedding of the sidecar's API type and parameter keys",
}


@dataclass(frozen=True, slots=True)
class VariantMasks:
    """Which parts of the input a variant zeroes.

    Expressed as masks rather than as four training paths, so every baseline
    runs the identical pipeline and a difference in the numbers is a difference
    in the inputs rather than in the code (FR-094).
    """

    #: Zero the acting line's and every entity ability token's ``e``.
    zero_e: bool = False
    #: Zero the ``[PLAYER]`` and ``[CARD]`` features and their ability tokens.
    zero_state: bool = False
    #: Replace the encoder with a free per-text embedding table.
    identity_embedding: bool = False
    #: Replace ``e`` with an embedding of the sidecar's API type and param keys.
    taxonomy_embedding: bool = False


def variant_masks(variant: str) -> VariantMasks:
    """The masks a variant applies.

    ``no-state`` zeroes the entity ability ``e`` tokens **as well as** the state
    features. Leaving them live would let it read the board through the
    abilities on it, and it would no longer be the average-effect control.
    """
    match variant:
        case "full":
            return VariantMasks()
        case "identity":
            return VariantMasks(identity_embedding=True)
        case "state-only":
            return VariantMasks(zero_e=True)
        case "no-state":
            return VariantMasks(zero_state=True, zero_e=True)
        case "taxonomy":
            return VariantMasks(taxonomy_embedding=True)
    raise ValueError(f"unknown --variant {variant!r}; known: {list(VARIANTS)}")


class MissingSplitError(RuntimeError):
    """A variant run was started without ``--split-from``."""


def require_split_from(variant: str, split_from: Path | None) -> None:
    """A variant run must inherit the split it is a baseline for (FR-091).

    Without it the run would silently compute its own split and the comparison
    would be between two models that saw different games — which looks like a
    result and is not one.
    """
    if variant != VARIANT_FULL and split_from is None:
        raise MissingSplitError(
            f"--variant {variant} requires --split-from PATH: a baseline must "
            "inherit the split, vocabulary and keyword-definition paths of the "
            "full run it baselines, or the comparison is between two models "
            "that saw different games."
        )


# ── configuration ───────────────────────────────────────────────────────


@dataclass
class TrainEffectModelConfig:
    """The trainer's flag surface (contracts/cli.md § train-effect-model)."""

    records_dir: Path = field(default_factory=lambda: Path("output/effects/records/"))
    cards_folders: tuple[Path, ...] = (
        Path("output/cardsfolder/"), Path("output/tokenscripts/"),
    )
    variant_scripts: Path | None = None
    split_from: Path | None = None
    vocab_path: Path = field(default_factory=lambda: Path("models/effects/vocab.txt"))
    printings_path: Path = field(
        default_factory=lambda: Path("resources/AllPrintings.json"),
    )
    keyword_definitions: Path = field(
        default_factory=lambda: Path("output/effects/keyword-definitions.json"),
    )
    model_output: Path | None = None
    variant: str = VARIANT_FULL
    e_dim: int = 64
    e_noise: float = 0.05
    keyword_expand_p: float = 0.25
    context_dropout: float = 0.15
    mlm_weight: float = 0.1
    mlm_mask_prob: float = 0.15
    api_weight: float = 0.05
    curriculum_step: int = 10_000
    batch_size: int = 32
    grad_accum: int = 1
    kind_mix: str | None = None
    context_cache: bool = False
    cache_refresh: int = 500
    steps_per_epoch: int = 5_000
    shards_per_epoch: int = DEFAULT_SHARDS_PER_EPOCH
    reserved_shards: int = RESERVED_VALIDATION_SHARDS
    epochs: int = 40
    patience: int = 5
    withhold_keyword: str | None = None

    def resolved_model_output(self) -> Path:
        from effects.infrastructure.effect_model_store import model_output_for

        if self.model_output is not None:
            return Path(self.model_output)
        return model_output_for(self.variant)


# Hardcoded, not flags (FR-095).
LEARNING_RATE = 1e-4
WARMUP_FRACTION = 0.05
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01


def warmup_steps(*, epochs: int, steps_per_epoch: int) -> int:
    """Linear warmup over the first 5% of the scheduled steps."""
    return max(1, int(epochs * steps_per_epoch * WARMUP_FRACTION))


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


def withheld_keyword_rules(withheld: str | None) -> tuple[frozenset[str], bool]:
    """Which keyword tokens to hide, and whether to force their expansion.

    Withholding is what gives the zero-shot check something to measure: the
    model never sees the keyword's *token*, only the sentence it stands for, so
    the check asks whether a keyword it has only ever read as prose lands near
    the right neighbours.
    """
    if not withheld:
        return frozenset(), False
    return frozenset({withheld.lower().replace(" ", "_")}), True


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

    @property
    def should_stop(self) -> bool:
        return self.since_best >= self.patience


class ContextCache:
    """Stop-gradient momentum cache for context ability vectors (FR-084).

    The documented fallback when live context re-encoding exceeds the 8 GB GPU
    budget: context abilities read a cached ``e`` refreshed every
    ``--cache-refresh`` batches instead of being re-encoded with gradient. The
    acting ability is always live — it is the one under study.

    Momentum rather than a hard overwrite so a refresh moves the cache toward
    the current encoder rather than snapping to one batch's view of it.
    """

    def __init__(self, refresh_every: int = 500, momentum: float = 0.9) -> None:
        self.refresh_every = refresh_every
        self.momentum = momentum
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._batches_since_refresh = 0

    def get(self, text: str) -> tuple[float, ...] | None:
        return self._vectors.get(text)

    def due_for_refresh(self) -> bool:
        return self._batches_since_refresh >= self.refresh_every

    def note_batch(self) -> None:
        self._batches_since_refresh += 1

    def refresh(self, fresh: dict[str, tuple[float, ...]]) -> None:
        """Blend freshly-encoded vectors into the cache and reset the counter."""
        for text, vector in fresh.items():
            previous = self._vectors.get(text)
            if previous is None:
                self._vectors[text] = vector
            else:
                self._vectors[text] = tuple(
                    self.momentum * old + (1.0 - self.momentum) * new
                    for old, new in zip(previous, vector)
                )
        self._batches_since_refresh = 0

    def __len__(self) -> int:
        return len(self._vectors)


# ── the run ─────────────────────────────────────────────────────────────


def ability_text_of(
    record: EffectRecord, sidecars, surface: str = "prose",
) -> str | None:
    """The acting line's text on ``surface``, or None where no line acts.

    ``combat`` and the ``attackers``/``blockers`` subkinds have none, which is
    what makes them sample uniformly rather than by rarity.

    Falls back to the key itself where the sidecar carries neither surface, so a
    line that cannot be read is still a distinct unit for rarity weighting
    rather than collapsing into every other unreadable line.
    """
    from effects.domain.ability_encoder import encoding_text

    if not record.ability:
        return None
    for key in record.ability:
        try:
            line = sidecars.line_for(key)
        except KeyError:
            continue
        if line is None:
            continue
        text = encoding_text(line, sidecars.prose_for(key), surface)
        return text or f"{key.script_file}:{key.trait_kind}:{key.index_within_kind}"
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


def resolve_holdout(
    config: TrainEffectModelConfig,
) -> tuple[HeldOutCards, CorpusSplit | None]:
    """The held-out cards, and an inherited split where one applies (FR-091).

    A ``--split-from`` run returns the checkpoint's own split alongside its
    cards: the corpus grows between runs, and re-deriving the boundary against a
    grown corpus would not be the one the source model trained against.
    """
    if config.split_from is not None:
        from effects.infrastructure.effect_model_store import EffectModelStore

        source = Path(config.split_from)
        provenance = EffectModelStore(source.parent).load(source).provenance
        held_out = HeldOutCards(
            names=frozenset(provenance.held_out_cards), script_files=frozenset(),
        )
        return held_out, CorpusSplit(
            held_out_cards=provenance.held_out_cards,
            card_disjoint_games=frozenset(provenance.card_disjoint_games),
            game_disjoint_games=frozenset(provenance.game_disjoint_games),
        )

    cards_folder = next(
        (Path(f) for f in config.cards_folders if Path(f).name == "cardsfolder"),
        Path(config.cards_folders[0]),
    )
    card_files = load_card_files(cards_folder)
    printings = (
        load_first_printings(config.printings_path)
        if Path(config.printings_path).exists()
        else {}
    )
    return newest_first_holdout(card_files, printings), None


def run(config: TrainEffectModelConfig) -> int:
    """Train the encoder and effect head jointly. Returns an exit code.

    Torch is imported here rather than at module scope, so
    ``python -m effects --help`` and the unit suite stay fast. The corpus is
    never loaded whole: the shard list is read here and the records behind it
    one shard at a time, inside the loop.
    """
    require_split_from(config.variant, config.split_from)

    shards = corpus_shards(config.records_dir)
    if not shards:
        logger.error(
            "No effect records under %s. Collect some first with "
            "'python -m sealed match-outcomes --effect-records %s'.",
            config.records_dir, config.records_dir,
        )
        return 1

    reserved = reserved_shard_indices(len(shards), reserved=config.reserved_shards)
    validation_shards = [shards[index] for index in sorted(reserved)]
    training_shards = [
        shard for index, shard in enumerate(shards) if index not in reserved
    ]
    if not training_shards:
        logger.error(
            "All %d shards under %s are reserved for validation; lower "
            "--reserved-shards.", len(shards), config.records_dir,
        )
        return 1

    on_disk = sum(shard.stat().st_size for shard in shards)
    logger.info(
        "Corpus: %d shards, %.1f GB on disk, under %s.",
        len(shards), on_disk / 1e9, config.records_dir,
    )
    logger.info(
        "%d shards reserved for validation, %d for training; an epoch reads "
        "%d of them, so %d epochs cover the corpus %.1f times.",
        len(validation_shards), len(training_shards), config.shards_per_epoch,
        config.epochs,
        config.epochs * config.shards_per_epoch / max(len(training_shards), 1),
    )

    from effects.application.training_loop import TrainingLoop

    held_out, inherited = resolve_holdout(config)
    return TrainingLoop(
        config,
        held_out=held_out,
        inherited=inherited,
        validation_shards=validation_shards,
        training_shards=training_shards,
    ).execute()
