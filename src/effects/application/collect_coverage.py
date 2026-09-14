"""``collect-coverage``: put every card into a game, not just the popular ones.

Sealed self-play only ever plays what sealed pools contain, which is a small and
badly skewed slice of the corpus — a card that appears in no sealed-legal set
never appears in a record at all. This command builds decks over the *whole*
converted corpus, weighted toward the cards with the fewest records, and plays
them until every card is either satisfied or retired.

Three rules keep it honest:

- **The castability consult only ranks.** It never drops a card from deck
  building, because being in a game is the precondition a stage-three
  intervention forks from: a card the consult judges uncastable is exactly the
  card an intervention has to force into play.
- **Presence in a snapshot does not count.** A card is satisfied when it has
  been the acting line's host, an event subject, or a referenced ref — sitting
  on the battlefield while something else happens teaches the model nothing
  about that card.
- **A card with no progress retires.** Without it the run would never end: some
  cards cannot be cast by Forge's AI at all, and waiting for them is waiting
  forever. They fall to stage-three interventions instead, which is what the
  two residues report.

Coverage matches write effect records **only**. Their decks are built for
coverage rather than as a fair self-play sample, so mixing them into
``match-outcomes.txt`` would corrupt what the scorer trains on. The guard is in
the Java worker's records-only mode, not here, because that is where the writers
are constructed.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from effects.domain.collection_caps import CollectionCaps

logger = logging.getLogger(__name__)

DEFAULT_RECORDS_DIR = Path("output/effects/records/")
DEFAULT_CARDS_FOLDERS: tuple[Path, ...] = (
    Path("output/cardsfolder/"), Path("output/tokenscripts/"),
)
DEFAULT_TARGET_RECORDS = 50
DEFAULT_DECKS_PER_ROUND = 500
DEFAULT_NO_PROGRESS_ROUNDS = 3

#: A 40-card limited deck: 23 nonlands plus basics.
NONLANDS_PER_DECK = 23
DECK_SIZE = 40

#: Seeds the per-round deck sampler so which cards fill a round is reproducible.
RANDOM_SEED = 42


@dataclass
class CollectCoverageConfig:
    effect_records: Path = field(default_factory=lambda: DEFAULT_RECORDS_DIR)
    cards_folders: tuple[Path, ...] = DEFAULT_CARDS_FOLDERS
    split_from: Path | None = None
    exclude_cards: Path | None = None
    target_records: int = DEFAULT_TARGET_RECORDS
    decks_per_round: int = DEFAULT_DECKS_PER_ROUND
    no_progress_rounds: int = DEFAULT_NO_PROGRESS_ROUNDS
    workers: int = 12
    caps: CollectionCaps = field(default_factory=CollectionCaps)

    def coverage_folder(self) -> Path:
        """The one tree deck candidates and the coverage unit come from.

        ``output/cardsfolder/`` alone (FR-046, FR-048): a token script is not a
        deckable card and could never be satisfied or retired, so counting one
        would guarantee the run never terminates.
        """
        for folder in self.cards_folders:
            if Path(folder).name == "cardsfolder":
                return Path(folder)
        return Path(self.cards_folders[0])


class ConsultVerdict:
    """What the castability consult said about a card."""

    CASTABLE = "castable"
    UNCASTABLE = "uncastable"
    UNKNOWN = "unknown"


@dataclass
class CardCoverage:
    """One card's progress toward its record target."""

    name: str
    records: int = 0
    rounds_without_progress: int = 0
    retired: bool = False
    verdict: str = ConsultVerdict.UNKNOWN

    def satisfied(self, target: int) -> bool:
        return self.records >= target

    def done(self, target: int) -> bool:
        return self.retired or self.satisfied(target)


def qualifying_cards(record) -> set[str]:
    """Cards this record counts toward (FR-049).

    The acting line's host, every event subject, and every referenced ref —
    but **not** an entity that merely sat in the snapshot. A card on the
    battlefield while something else resolves teaches the model nothing about
    that card, and counting it would let a popular card satisfy every card
    beside it.
    """
    from effects.domain.effect_targets import events_of

    by_id: dict[str, str] = {
        entity.id: entity.name for entity in record.state.entities
    }
    names: set[str] = set()

    source = record.state.refs.source
    if source in by_id:
        names.add(by_id[source])
    for target in record.state.refs.targets:
        if target in by_id:
            names.add(by_id[target])
    for event in events_of(record):
        for subject in event.subjects:
            if subject in by_id:
                names.add(by_id[subject])
    return names


def count_coverage(records: Iterable) -> Counter[str]:
    """How many qualifying records each card has across the whole corpus."""
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(qualifying_cards(record))
    return counts


class CoverageCounter:
    """Counts each shard once across a whole run, not once per round.

    The recount used to re-read every shard under ``--effect-records`` after
    every round. On a 1,502-shard corpus that is about 66 minutes against the
    ~35 minutes a round spends playing, so most of a coverage run's wall clock
    went to re-deriving numbers that had not changed.

    Counting incrementally is sound because a shard is final once counted. A
    round's workers are all terminated before ``play_round`` returns —
    ``ForgeWorkerPool._supervisor_loop`` calls ``_terminate_all`` when the stop
    condition fires — and a shard file is never reopened afterwards, because
    its name carries the worker's JVM lifetime and each round's workers are new
    JVMs. So a shard this has read cannot gain records later.

    The consequence for callers: ``CardCoverage.records`` is now accumulated
    rather than assigned, so nothing else may overwrite it.
    """

    def __init__(self) -> None:
        self._counted: set[Path] = set()

    def update(
        self, coverage: dict[str, CardCoverage], records_dir: Path,
    ) -> int:
        """Add the counts of shards not yet read. Returns how many it read."""
        from effects.infrastructure.record_io import iter_shards, read_shard

        fresh = [
            shard for shard in iter_shards(Path(records_dir))
            if shard not in self._counted
        ]
        for shard in fresh:
            for record in read_shard(shard):
                for name in qualifying_cards(record):
                    card = coverage.get(name)
                    if card is not None:
                        card.records += 1
            self._counted.add(shard)
        return len(fresh)


def deck_weights(
    coverage: dict[str, CardCoverage], target: int,
) -> dict[str, float]:
    """Weight each card by how far it is from its target (FR-046).

    A card with no records weighs most; one already satisfied weighs nothing.
    Linear in the shortfall rather than inverse in the count, so a card at 49
    of 50 still gets drawn occasionally and the last few records do not take
    unboundedly many rounds.
    """
    weights: dict[str, float] = {}
    for name, card in coverage.items():
        if card.done(target):
            continue
        weights[name] = float(target - card.records)
    return weights


def rank_by_consult(
    weights: dict[str, float], verdicts: dict[str, str],
) -> dict[str, float]:
    """Apply the consult as a **ranking** input, never as a filter (FR-047).

    A card the consult judges uncastable keeps a weight, because being in a
    game is the precondition an intervention forks from — dropping it here
    would mean stage three had nothing to fork.
    """
    ranked: dict[str, float] = {}
    for name, weight in weights.items():
        verdict = verdicts.get(name, ConsultVerdict.UNKNOWN)
        multiplier = 0.25 if verdict == ConsultVerdict.UNCASTABLE else 1.0
        ranked[name] = weight * multiplier
    return ranked


@dataclass(frozen=True)
class _NonlandText:
    """Adapts a raw converted-text string to what ``compute_basic_lands`` reads.

    FR-002 declares only ``compute_basic_lands`` on the allowed surface of
    ``sealed.domain.manabase``, not the ``ConvertedCardText`` wrapper its
    signature is typed against — so rather than reaching past the declared
    surface for that wrapper, this mirrors the single method the manabase
    heuristic actually calls (``mana_cost_line``) over the plain string
    ``build_coverage_decks`` already has on hand.
    """

    text: str

    def mana_cost_line(self) -> str | None:
        for line in self.text.splitlines():
            if line.strip().lower().startswith("mana cost:"):
                return line.split(":", 1)[1].strip() or None
        return None


def build_coverage_decks(
    weights: dict[str, float],
    texts: dict[str, str],
    count: int,
    *,
    rng: random.Random,
) -> list[list[str]]:
    """``count`` 40-card decks drawn from ``weights`` (FR-046).

    23 nonlands sampled with replacement in proportion to weight, then basics
    from ``compute_basic_lands`` over the chosen cards' converted text. With
    replacement because a weight of 500 against 1 should be able to fill a deck
    with one card: the point is to get that card into games, not to build a
    deck anyone would play.

    A weighted card with no converted text is skipped — the manabase heuristic
    reads the text, and a card without one cannot be placed.
    """
    from sealed.domain.manabase import compute_basic_lands

    pool = [(name, w) for name, w in weights.items() if name in texts and w > 0]
    if not pool:
        return []
    names = [name for name, _ in pool]
    ws = [w for _, w in pool]

    decks: list[list[str]] = []
    for _ in range(count):
        nonlands = rng.choices(names, weights=ws, k=NONLANDS_PER_DECK)
        basics = compute_basic_lands([_NonlandText(texts[n]) for n in nonlands])
        deck = list(nonlands)
        for land, n in basics.items():
            deck.extend([land] * n)
        decks.append(deck)
    return decks


def retire_stalled(
    coverage: dict[str, CardCoverage],
    previous: dict[str, int],
    *,
    no_progress_rounds: int,
    decked: Collection[str],
) -> list[str]:
    """Retire cards that gained no qualifying record this round (FR-050).

    Without this the run never ends: some cards Forge's AI cannot cast at all,
    and a round that makes no progress on them will make none on the next.

    ``decked`` is the names this round's decks actually held, and only those
    can stall: a card no deck contained was never attempted, so counting the
    round against it retires a card the run has not yet tried once. That is
    not a rare edge — at the defaults a round draws 500 x 23 slots against
    33,680 candidates, so a given card sits out a whole round with probability
    about 0.71 and roughly a third of the corpus would retire after three
    rounds having never been in a game. The run still terminates, because a
    live card keeps its weight and is drawn eventually; it just stops
    terminating with almost nothing collected and a residue list the size of
    the corpus.

    Required rather than defaulted to "everything": a default would restore
    the exact silent behaviour this parameter exists to remove.
    """
    decked_names = frozenset(decked)
    retired: list[str] = []
    for name, card in coverage.items():
        if card.retired:
            continue
        if card.records > previous.get(name, 0):
            card.rounds_without_progress = 0
            continue
        if name not in decked_names:
            continue
        card.rounds_without_progress += 1
        if card.rounds_without_progress >= no_progress_rounds:
            card.retired = True
            retired.append(name)
    return retired


@dataclass
class CoverageResidues:
    """What the run could not reach, split by why (FR-051).

    Both fall to stage-three interventions, and the report existing is what
    SC-006 turns on: an operator has to be able to see which cards need forcing
    and which merely need more games.
    """

    uncastable: tuple[str, ...] = ()
    castable_but_short: tuple[str, ...] = ()

    def render(self, target: int) -> str:
        return (
            f"Residues: {len(self.uncastable)} cards the consult judged "
            f"uncastable, {len(self.castable_but_short)} castable cards short "
            f"of {target} records. Both fall to stage-three interventions."
        )


def residues(
    coverage: dict[str, CardCoverage], target: int,
) -> CoverageResidues:
    """Each unsatisfied card counted under its consult verdict."""
    uncastable: list[str] = []
    short: list[str] = []
    for name, card in sorted(coverage.items()):
        if card.satisfied(target):
            continue
        if card.verdict == ConsultVerdict.UNCASTABLE:
            uncastable.append(name)
        else:
            short.append(name)
    return CoverageResidues(tuple(uncastable), tuple(short))


def load_exclusions(
    *, split_from: Path | None, exclude_cards: Path | None,
) -> frozenset[str]:
    """Cards to keep out of every deck, from whichever source names them.

    ``--exclude-cards`` is the ordinary one: the holdout is derived from the
    converted tree and two flags, so it needs no trained model and this command
    no longer waits on a training run that is supposed to follow it.
    ``--split-from`` remains for collecting more coverage against a checkpoint
    that already exists, whose recorded split is then the authority.

    Both at once is refused rather than merged. Two spellings of one holdout is
    precisely the failure this feature has already had once, and a merge would
    hide a disagreement instead of reporting it.
    """
    from effects.application.train_effect_model import fold_card_name

    if split_from is not None and exclude_cards is not None:
        raise ValueError(
            "pass one source of held-out cards, not both: --exclude-cards "
            "names the list this corpus is being depleted against, "
            "--split-from inherits a checkpoint's recorded split, and a "
            "disagreement between them is silent"
        )
    if exclude_cards is not None:
        return frozenset(
            fold_card_name(line.strip())
            for line in Path(exclude_cards).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return load_held_out(split_from)


def load_held_out(split_from: Path | None) -> frozenset[str]:
    """Cards to keep out of every deck (FR-045).

    A coverage run that put held-out cards into games would contaminate the
    card-disjoint split of the checkpoint it inherited from — the split is what
    makes gate 1 mean "deployment to an unseen set".
    """
    if split_from is None:
        return frozenset()
    from effects.infrastructure.effect_model_store import EffectModelStore

    path = Path(split_from)
    provenance = EffectModelStore(path.parent).load(path).provenance
    return frozenset(provenance.held_out_cards)


def build_coverage_units(
    card_files: dict[str, str], held_out: frozenset[str],
) -> dict[str, CardCoverage]:
    """Every deckable card, minus the holdout.

    Takes the already-loaded ``name -> script`` mapping rather than the folder,
    because ``run`` needs the same mapping to read each card's converted text
    and ``load_card_files`` is a full rglob plus a read of some 33,000 files.
    """
    return {
        name: CardCoverage(name=name)
        for name in card_files
        if name not in held_out
    }


def round_summary(
    round_number: int, coverage: dict[str, CardCoverage], target: int,
    retired: list[str],
) -> str:
    satisfied = sum(1 for c in coverage.values() if c.satisfied(target))
    remaining = sum(1 for c in coverage.values() if not c.done(target))
    return (
        f"round {round_number}: {satisfied} satisfied, {remaining} remaining, "
        f"{len(retired)} retired this round"
    )


def is_complete(coverage: dict[str, CardCoverage], target: int) -> bool:
    """The run ends when every card is satisfied or retired (FR-050)."""
    return all(card.done(target) for card in coverage.values())


def run(config: CollectCoverageConfig) -> int:
    """Play rounds until every card is satisfied or retired."""
    from effects.infrastructure.collector_connector import CollectorSupervisor
    from effects.infrastructure.deck_file import COVERAGE_SET_CODE, write_deck_file

    cards_folder = config.coverage_folder()
    if not cards_folder.is_dir():
        logger.error(
            "No converted cards under %s. Run 'python -m price_predictor "
            "convert' first.", cards_folder,
        )
        return 1

    held_out = load_exclusions(
        split_from=config.split_from, exclude_cards=config.exclude_cards,
    )
    from effects.application.train_effect_model import load_card_files

    card_files = load_card_files(cards_folder)
    coverage = build_coverage_units(card_files, held_out)
    if not coverage:
        logger.error("No deckable cards under %s", cards_folder)
        return 1
    logger.info(
        "Coverage unit: %d cards (%d held out), target %d records each",
        len(coverage), len(held_out), config.target_records,
    )

    # The existing corpus is counted once, here, rather than again after every
    # round. On 1,502 shards that is the difference between paying ~66 minutes
    # once and paying it per round.
    counter = CoverageCounter()
    seeded = counter.update(coverage, Path(config.effect_records))
    logger.info(
        "Counted %d existing shard(s); %d of %d cards already at target",
        seeded,
        sum(1 for c in coverage.values() if c.satisfied(config.target_records)),
        len(coverage),
    )

    verdicts = consult_castability(sorted(coverage), cards_folder)
    for name, verdict in verdicts.items():
        if name in coverage:
            coverage[name].verdict = verdict

    texts = {
        name: (cards_folder.parent / script).read_text(encoding="utf-8")
        for name, script in card_files.items()
        if (cards_folder.parent / script).exists()
    }
    decks_file = Path(config.effect_records) / "coverage-decks.txt"
    rng = random.Random(RANDOM_SEED)

    supervisor = CollectorSupervisor(
        worker_count=config.workers, effect_records=config.effect_records,
        caps=config.caps,
    )
    interrupted = False
    try:
        round_number = 0
        while not is_complete(coverage, config.target_records):
            round_number += 1
            previous = {name: card.records for name, card in coverage.items()}
            weights = rank_by_consult(
                deck_weights(coverage, config.target_records), verdicts,
            )
            if not weights:
                break
            decks = build_coverage_decks(
                weights, texts, config.decks_per_round, rng=rng,
            )
            if not decks:
                logger.error(
                    "No card with a weight has converted text; nothing to deck."
                )
                break
            write_deck_file(
                decks, decks_file,
                label="coverage", set_code=COVERAGE_SET_CODE,
            )
            decked = {name for deck in decks for name in deck}
            supervisor.play_round(decks_file, matches=config.decks_per_round)
            read = counter.update(coverage, Path(config.effect_records))
            logger.info("Counted %d new shard(s) from round %d", read, round_number)
            retired = retire_stalled(
                coverage, previous, no_progress_rounds=config.no_progress_rounds,
                decked=decked,
            )
            logger.info("%s", round_summary(
                round_number, coverage, config.target_records, retired,
            ))
            if supervisor.interrupted:
                # ForgeWorkerPool swallows SIGINT -- it sets its own event and
                # returns normally -- so without this the loop builds a fresh
                # pool and Ctrl-C means "skip this round, start another".
                interrupted = True
                logger.warning(
                    "Interrupted after round %d; stopping instead of starting "
                    "another round", round_number,
                )
                break
    finally:
        # Closes the worker log files and shuts the pool down (final-fix-3.md
        # item 5). Without this, the only latched effect-record failure
        # reporters this run has (ApiEvents.reportEmitterFailure and its
        # siblings, F4) never surface: their handles stay open, unflushed
        # and unclosed, until the interpreter exits on its own -- a
        # collection run that gets killed or piped never sees them at all.
        supervisor.stop()

    report = residues(coverage, config.target_records)
    logger.info("%s", report.render(config.target_records))
    # The residues are printed either way -- what was collected before the
    # interrupt is still what an operator needs to see -- but an interrupted
    # run did not finish, and reporting success would be the same lie the
    # swallowed SIGINT already told.
    return 130 if interrupted else 0


def consult_castability(
    names: list[str], cards_folder: Path,
) -> dict[str, str]:
    """Ask a live Forge whether each card can be cast at all (FR-047).

    Needs a game, so it runs in the JVM. A failure to consult leaves every card
    ``unknown``, which ranks them as castable — the safe direction, since the
    consult only ranks and an under-ranked card merely takes longer.
    """
    from effects.infrastructure.castability_connector import CastabilityConnector

    try:
        return CastabilityConnector().consult(names, cards_folder)
    except (FileNotFoundError, OSError) as exc:
        logger.warning(
            "Castability consult unavailable (%s); every card ranks as "
            "castable, which only slows the run down", exc,
        )
        return {}


def group_by_verdict(verdicts: dict[str, str]) -> dict[str, list[str]]:
    """Cards per consult verdict, for the residue report."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, verdict in verdicts.items():
        grouped[verdict].append(name)
    return {verdict: sorted(names) for verdict, names in grouped.items()}
