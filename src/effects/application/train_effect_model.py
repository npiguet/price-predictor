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
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from effects.infrastructure.sidecar_io import SidecarCache

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

#: An eligible ability text is held out when crc32(text) % 1000 falls below
#: this. Keyed on the text rather than the card because the model never reads a
#: card's name, and stable per text so a depleted corpus stays valid (FR-088).
HOLDOUT_PERMILLE = 20
#: A text more cards than this carry is never eligible. `Flying` is on thousands
#: of cards and depleting all of them would empty the training corpus.
HOLDOUT_MAX_CARRIERS = 8
#: Unique-text resolution records the card-disjoint stratum should hold
#: before gate 1's margins mean anything. A floor, not a rule: below it
#: the run warns, at zero it stops.
MIN_HOLDOUT_RECORDS = 2000
#: Rarity weight ceiling, as a multiple of the most-observed text's weight.
RARITY_CAP = 20.0
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
    records: Sequence[EffectRecord], text_of, *,
    rarity: Mapping[str, int] | None = None,
) -> list[float]:
    """Per-record sampling weight within its class.

    A record with no acting ability text weighs 1: ``combat`` and
    ``playability``/``attackers``/``blockers`` sample uniformly, because there
    is no ability text for rarity to key on (FR-087).

    Args:
        rarity: a corpus-wide ``text -> games`` table from a curated dataset's
            manifest, read in preference to the resident shard's own count. A
            text the table does not name falls back to the shard's count for
            it rather than to zero, so a shard collected after the dataset was
            built still weights sanely instead of dominating every batch.
    """
    shard_counts = effective_games(records, text_of)
    games_per_text = (
        shard_counts if rarity is None
        else {text: rarity.get(text, count) for text, count in shard_counts.items()}
    )
    weights = rarity_weights(games_per_text)
    out: list[float] = []
    for record in records:
        text = text_of(record)
        out.append(1.0 if text is None else weights.get(text, 1.0))
    return out


# ── the split ───────────────────────────────────────────────────────────


def fold_card_name(name: str) -> str:
    """A card name in the one spelling both sides of the boundary agree on.

    Converted card text is lowercased, so every name read out of
    ``output/cardsfolder/`` arrives as ``soul echo``. Forge's own
    ``getName()`` — and so every record's ``EntityState.name`` and every
    booster's cards — carries printed case, ``Soul Echo``. Comparing the two
    directly never matches, and nothing says so: a collection run told to
    deplete its pools depletes nothing and writes an ordinary corpus.

    ``lower()`` rather than ``casefold()`` to agree with the Java side's
    ``toLowerCase(Locale.ROOT)``; the two differ on characters no card name
    has, and a holdout the two languages disagree about is the same bug again.
    """
    return name.lower()


@dataclass(frozen=True, slots=True)
class HeldOutCards:
    """The card-disjoint holdout, addressable both ways a record names a card.

    A record names a card by the entity names in its snapshot and by the script
    files its provenance keys point at; matching on either catches both.

    ``names`` is folded on construction (:func:`fold_card_name`), because the
    two sides of this comparison spell a card differently and a caller that
    passed printed case would produce a holdout matching nothing.
    """

    names: frozenset[str]
    script_files: frozenset[str]
    #: The held-out ability texts themselves. Under depletion no training card
    #: carries one, so this is also gate 1's unique-text slice — sizing the
    #: stratum needs no scan of the training corpus.
    texts: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        folded = frozenset(fold_card_name(name) for name in self.names)
        if folded != self.names:
            object.__setattr__(self, "names", folded)

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


def text_keyed_holdout(
    card_files: dict[str, str],
    texts_by_card: Mapping[str, Iterable[str]],
    *,
    permille: int = HOLDOUT_PERMILLE,
    max_carriers: int = HOLDOUT_MAX_CARRIERS,
) -> HeldOutCards:
    """Hold out ability texts, and with them every card carrying one (FR-088).

    Args:
        card_files: ``card name -> script file``, one entry per converted card
            under ``output/cardsfolder/``. Token scripts and variant scripts are
            not in this denominator.
        texts_by_card: ``card name -> its sidecar's script texts``, from
            ``load_card_texts``.

    The text rather than the card, because the model never reads a card's name
    and functional reprints compile to the same script: a name-keyed holdout
    trains on Lightning Strike while holding out Searing Spear, and gate 1 then
    drops those records for having text a training card carries.
    """
    from effects.domain.text_holdout import select_holdout

    chosen = select_holdout(
        texts_by_card, permille=permille, max_carriers=max_carriers,
    )
    names = frozenset(chosen.cards & card_files.keys())
    return HeldOutCards(
        names=names,
        script_files=frozenset(card_files[name] for name in names),
        texts=chosen.texts,
    )


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


def record_names_held_out_card(
    record: EffectRecord, held_out: HeldOutCards,
) -> bool:
    """Whether this record names a held-out card, acting or in context."""
    for entity in record.state.entities:
        if fold_card_name(entity.name) in held_out.names:
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


#: Records the full-strength probe reads before calling a shard depleted. One
#: gzip member is 256 records and several games, and a full-strength pool holds
#: held-out cards throughout, so a shard that shows none in its opening records
#: almost certainly has none.
FULL_STRENGTH_PROBE_RECORDS = 512


def records_name_held_out(
    records: Iterable[EffectRecord],
    held_out: HeldOutCards,
    *,
    limit: int = FULL_STRENGTH_PROBE_RECORDS,
) -> bool:
    """Whether a shard's opening records name a held-out card.

    A probe rather than a proof, and deliberately so: parsing all 701 shards to
    decide reservation would read the corpus twice over before the first epoch.
    It stops at the first held-out card, and gives up after ``limit`` records.

    A shard it misses stays a training shard and loses nothing that matters:
    ``shard_games`` still routes every one of its games to the card-disjoint
    stratum when an epoch reads it, so training purity does not depend on this.
    What a miss costs is that shard's validation records.
    """
    from itertools import islice

    return any(
        record_names_held_out_card(record, held_out)
        for record in islice(records, limit)
    )


def reserve_shards(
    shards: Sequence[Path],
    *,
    reserved: int,
    holds_held_out_card,
) -> frozenset[int]:
    """Which shard positions are held back from training (FR-125).

    Every shard holding a held-out card is reserved, whatever else is chosen:
    those are a full-strength collection run's shards, and they are the
    card-disjoint stratum. Nothing names them — a depleted run's pools could not
    have produced one — so the corpus says which it is and no flag has to.

    ``reserved`` further shards are then spread evenly over the depleted ones,
    for the game-disjoint stratum. Taking them from the full-strength shards
    instead would leave the two strata sharing games.
    """
    full_strength = frozenset(
        index for index, shard in enumerate(shards)
        if holds_held_out_card(shard)
    )
    depleted = [
        index for index in range(len(shards)) if index not in full_strength
    ]
    spread = reserved_shard_indices(len(depleted), reserved=reserved)
    return full_strength | frozenset(depleted[position] for position in spread)


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

    def note_games(
        self, *, card_disjoint: Iterable[str], game_disjoint: Iterable[str],
    ) -> None:
        """Record games already routed, without the records that routed them.

        The parallel validation sweep decides a shard's strata in the worker
        that read it — which it can, because a game never spans two shards — and
        returns the game ids rather than the records. This is how those reach
        the accumulator, and it is the same accumulation ``note_shard`` does
        with the shard in hand.
        """
        if self._inherited is not None:
            return
        self._card_disjoint |= set(card_disjoint)
        self._game_disjoint |= set(game_disjoint)

    def split(self) -> CorpusSplit:
        if self._inherited is not None:
            return self._inherited
        return CorpusSplit(
            held_out_cards=self._held_out_cards,
            card_disjoint_games=frozenset(self._card_disjoint),
            game_disjoint_games=frozenset(self._game_disjoint),
        )


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


def weighted_order(weights: Sequence[float], rng: random.Random) -> list[int]:
    """Indices in a weighted shuffle: each drawn once, heavier ones earlier.

    Efraimidis–Spirakis: key ``u ** (1 / w)`` per item, sorted descending, is a
    sample without replacement in proportion to ``w``. One shuffle per pass is
    what lets a shard be seen whole rather than resampled from with
    replacement, which is how a sixteen-record shard was replayed 640 times.
    """
    keys = [
        rng.random() ** (1.0 / max(float(weight), 1e-9)) for weight in weights
    ]
    return sorted(range(len(keys)), key=keys.__getitem__, reverse=True)


def batches_without_replacement(
    records: Sequence[EffectRecord], weights: Sequence[float], *,
    batch_size: int, rng: random.Random,
) -> Iterator[BatchPlan]:
    """Endless full batches over ``records``: one weighted shuffle per pass.

    A batch smaller than ``batch_size`` is yielded only when the whole shard
    is smaller; otherwise the ragged tail is dropped and the next pass
    starts, so every step trains on a full batch.
    """
    if not records:
        return
    size = min(batch_size, len(records))
    while True:
        order = weighted_order(weights, rng)
        for start in range(0, len(order) - size + 1, size):
            plan: dict[str, list[EffectRecord]] = defaultdict(list)
            for index in order[start:start + size]:
                plan[records[index].game_id].append(records[index])
            yield BatchPlan(dict(plan))


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


def unique_text_resolution_records(
    records: Iterable[EffectRecord],
    held_out_texts: frozenset[str],
    *,
    text_of,
) -> int:
    """Resolution records in the stratum whose acting text is held out.

    Args:
        text_of: a provenance key to the text it is encoded from — the
            batcher's own, so this counts what gate 1 will actually read.

    This is gate 1's slice. Under depletion no training game holds a card
    carrying a held-out text, so "acting text appears on no training card" and
    "acting text is held out" name the same records, and the stratum can be
    sized from the holdout without scanning the training corpus.
    """
    count = 0
    for record in records:
        if record.kind is not RecordKind.RESOLUTION:
            continue
        for key in record.ability or ():
            text = text_of(key)
            if text is not None and text in held_out_texts:
                count += 1
                break
    return count


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
    (FR-146). ``validate_corpus_flags`` already refuses the two together, so a
    caller never has both to offer at once.
    """
    if variant != VARIANT_FULL and split_from is None and corpus is None:
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

    records_dir: Path = field(default_factory=lambda: Path("output/effects/records/"))
    #: A curated dataset directory built by ``build-corpus``. Mutually
    #: exclusive with every flag in ``CORPUS_EXCLUSIVE_FLAGS``: its manifest
    #: already decides the split, the holdout and the rarity table.
    corpus: str | None = None
    cards_folders: tuple[Path, ...] = (
        Path("output/cardsfolder/"), Path("output/tokenscripts/"),
    )
    variant_scripts: Path | None = None
    split_from: Path | None = None
    vocab_path: Path = field(default_factory=lambda: Path("models/effects/vocab.txt"))
    printings_path: Path = field(
        default_factory=lambda: Path("resources/AllPrintings.json"),
    )
    #: An eligible ability text is held out when crc32(text) % 1000 falls below
    #: this; a text more than ``holdout_max_carriers`` cards carry is never
    #: eligible. Pinned for the life of a corpus: a depleted collection run
    #: composed its pools against these values (FR-134).
    holdout_permille: int = HOLDOUT_PERMILLE
    holdout_max_carriers: int = HOLDOUT_MAX_CARRIERS
    #: Unique-text resolution records the card-disjoint stratum warns below.
    min_holdout_records: int = MIN_HOLDOUT_RECORDS
    #: Processes the pre-training validation sweep runs across. 0 takes the
    #: CPU count. Processes rather than threads because the cost is
    #: ``json.loads`` and record construction, both of which hold the GIL.
    workers: int = 0
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


#: Flags a curated manifest already records. Passing one beside `--corpus` is
#: two spellings of one decision, and a disagreement nothing would report.
CORPUS_EXCLUSIVE_FLAGS: tuple[str, ...] = (
    "records_dir", "reserved_shards", "split_from",
    "holdout_permille", "holdout_max_carriers",
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


def validate_corpus_flags(config: TrainEffectModelConfig) -> None:
    """Refuse a flag the curated manifest decides (FR-146)."""
    if config.corpus is None:
        return
    defaults = TrainEffectModelConfig(corpus=config.corpus)
    for name in CORPUS_EXCLUSIVE_FLAGS:
        if getattr(config, name) != getattr(defaults, name):
            spelled = "--" + name.replace("_", "-")
            raise ValueError(
                f"{spelled} cannot be passed with --corpus: the dataset's "
                f"manifest at {config.corpus} already records it."
            )


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

    try:
        line = sidecars.line_for(key)
    except KeyError:
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
    if not record.ability:
        return None
    for key in record.ability:
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
    from effects.infrastructure.sidecar_io import SidecarCache

    card_files = load_card_files(cards_folder)
    sidecars = SidecarCache({"cardsfolder": cards_folder})
    held_out = text_keyed_holdout(
        card_files,
        load_card_texts(card_files, sidecars),
        permille=config.holdout_permille,
        max_carriers=config.holdout_max_carriers,
    )
    logger.info(
        "Holdout: %d ability texts, carried by %d of %d cards (%.1f%% of the "
        "corpus). Every one of them is absent from a depleted training pool.",
        len(held_out.texts), len(held_out.names), len(card_files),
        100.0 * len(held_out.names) / max(len(card_files), 1),
    )
    return held_out, None


def run(config: TrainEffectModelConfig) -> int:
    """Train the encoder and effect head jointly. Returns an exit code.

    Torch is imported here rather than at module scope, so
    ``python -m effects --help`` and the unit suite stay fast. The corpus is
    never loaded whole: the shard list is read here and the records behind it
    one shard at a time, inside the loop.
    """
    validate_corpus_flags(config)
    require_split_from(config.variant, config.split_from, corpus=config.corpus)

    rarity: dict[str, int] | None = None
    corpus_digest = ""
    if config.corpus is not None:
        # A curated dataset (FR-146): every decision below was already made by
        # `build-corpus` and recorded in its manifest, so it is read rather
        # than derived — the same reason a `--split-from` run inherits its
        # split instead of recomputing it. `reserve_shards` never runs here:
        # `build-corpus` already sorted every shard into `training/` or one of
        # the two validation strata.
        from effects.infrastructure.corpus_store import CorpusStore

        store = CorpusStore(Path(config.corpus))
        manifest = store.load()
        require_matching_surface(
            manifest_surface=manifest.surface,
            vocab_path=Path(config.vocab_path),
            corpus=config.corpus,
        )
        training_shards = corpus_shards(store.training_dir)
        validation_shards = (
            corpus_shards(store.card_disjoint_dir)
            + corpus_shards(store.game_disjoint_dir)
        )
        if not training_shards:
            logger.error("No training shards under %s.", store.training_dir)
            return 1

        held_out = HeldOutCards(
            names=frozenset(manifest.held_out_cards), script_files=frozenset(),
            texts=frozenset(manifest.held_out_texts),
        )
        inherited = CorpusSplit(
            held_out_cards=manifest.held_out_cards,
            card_disjoint_games=frozenset(manifest.card_disjoint_games),
            game_disjoint_games=frozenset(manifest.game_disjoint_games),
        )
        rarity = manifest.rarity
        # Recorded onto the checkpoint's provenance below (FR-147), so
        # `evaluate-effect-model` can refuse a dataset rebuilt since this run
        # read it — computed once here rather than re-hashed at every save.
        corpus_digest = manifest.digest()
        logger.info(
            "Curated corpus at %s: %d training shards, %d validation shards "
            "(%d card-disjoint texts held out).",
            config.corpus, len(training_shards), len(validation_shards),
            len(manifest.held_out_cards),
        )
    else:
        shards = corpus_shards(config.records_dir)
        if not shards:
            logger.error(
                "No effect records under %s. Collect some first with "
                "'python -m sealed match-outcomes --effect-records %s'.",
                config.records_dir, config.records_dir,
            )
            return 1

        held_out, inherited = resolve_holdout(config)

        # Before the even spread, because a shard holding a held-out card came
        # from a full-strength collection run and *is* the card-disjoint
        # stratum. The probe reads each shard's opening records rather than
        # all of it.
        def _full_strength(shard: Path) -> bool:
            from effects.infrastructure.record_io import read_shard

            return records_name_held_out(read_shard(shard), held_out)

        reserved = reserve_shards(
            shards, reserved=config.reserved_shards,
            holds_held_out_card=_full_strength,
        )
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
            "%d shards reserved for validation, %d for training; an epoch "
            "reads %d of them, so %d epochs cover the corpus %.1f times.",
            len(validation_shards), len(training_shards), config.shards_per_epoch,
            config.epochs,
            config.epochs * config.shards_per_epoch / max(len(training_shards), 1),
        )

    from effects.application.training_loop import TrainingLoop

    return TrainingLoop(
        config,
        held_out=held_out,
        inherited=inherited,
        validation_shards=validation_shards,
        training_shards=training_shards,
        rarity=rarity,
        corpus_digest=corpus_digest,
    ).execute()
