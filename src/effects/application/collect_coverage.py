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
import os
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from effects.domain.collection_caps import CollectionCaps
from effects.domain.coverage import (
    CardCoverage,
    ConsultVerdict,
    CoverageUnit,
    DeckCandidate,
    TextCoverage,
    awaited_texts,
    deck_weights,
    held_out_texts_of,
    is_complete,
    is_deckable,
    qualifying_cards,
    rank_by_consult,
    retire_stalled,
    text_deck_weights,
)

if TYPE_CHECKING:
    from effects.infrastructure.collector_connector import CollectorSupervisor

logger = logging.getLogger(__name__)

DEFAULT_RECORDS_DIR = Path("output/effects/records/")
DEFAULT_CARDS_FOLDERS: tuple[Path, ...] = (
    Path("output/cardsfolder/"), Path("output/tokenscripts/"),
)
DEFAULT_TARGET_RECORDS = 50
DEFAULT_DECKS_PER_ROUND = 500
DEFAULT_NO_PROGRESS_ROUNDS = 3

#: Shards between progress lines while counting. The first pass reads the whole
#: existing corpus -- on 801 shards about 35 minutes -- and silence that long is
#: indistinguishable from the hang this command kept producing. Ten keeps the
#: line moving often enough to read as progress rather than as a stall.
PROGRESS_EVERY_SHARDS = 10

#: A 40-card limited deck: 23 nonlands plus basics.
NONLANDS_PER_DECK = 23
DECK_SIZE = 40

#: Seeds the per-round deck sampler so which cards fill a round is reproducible.
RANDOM_SEED = 42

#: FR-036's floor: distinct games a held-out text must act in.
DEFAULT_MIN_TEXT_GAMES = 5


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
    #: ``--only-cards``: the ``holdout-cards`` list. Set, the run is the
    #: held-out coverage round (FR-035): decks from these cards alone, and the
    #: unit is the held-out ability text rather than the card.
    only_cards: Path | None = None
    #: Distinct games a held-out text must act in to be satisfied (FR-036).
    min_text_games: int = DEFAULT_MIN_TEXT_GAMES

    def coverage_folder(self) -> Path:
        """The one tree deck candidates and the coverage unit come from.

        ``output/cardsfolder/`` alone (FR-046, FR-048): a token script is not a
        deckable card and could never be satisfied or retired, so counting one
        would guarantee the run never terminates.
        """
        from effects.infrastructure.sidecar_io import tree_of_folder

        for folder in self.cards_folders:
            if tree_of_folder(Path(folder)) == "cardsfolder":
                return Path(folder)
        return Path(self.cards_folders[0])


def count_shard(shard: Path) -> Counter[str]:
    """One shard's qualifying records, keyed by folded card name.

    Top-level rather than a method so a process pool can pickle it. Returns
    folded names because that is what a coverage unit is keyed by: the unit
    comes from converted card text (lowercase) while a record's entity name is
    Forge's own (printed case), and an unfolded key matches nothing.
    """
    from effects.domain.card_names import fold_card_name
    from effects.infrastructure.record_io import read_shard

    counts: Counter[str] = Counter()
    for record in read_shard(shard):
        for name in qualifying_cards(record):
            counts[fold_card_name(name)] += 1
    return counts


class _ShardCounter:
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
    """

    def __init__(self, progress_every: int = PROGRESS_EVERY_SHARDS) -> None:
        self._counted: set[Path] = set()
        self._progress_every = progress_every

    def _count_fresh[T](
        self,
        records_dir: Path,
        count: Callable[[Path], T],
        apply: Callable[[T], None],
        *,
        workers: int | None,
    ) -> int:
        """``count`` each shard not yet read and ``apply`` it. Returns how many.

        Shards are independent and each contributes only its own counts, so
        the work maps: count shards in parallel, apply each as it arrives.
        Applying adds, which is associative, so the answer cannot depend on
        how the shards were divided or in what order they finished.

        ``workers`` defaults to the machine's usable cores. One means in-process
        — what the tests use for the serial reference, and what a handful of
        shards wants, since a process pool costs more to start than it saves.
        ``count`` must pickle, since a pool ships it to every process.
        """
        from effects.infrastructure.record_io import iter_shards

        fresh = [
            shard for shard in iter_shards(Path(records_dir))
            if shard not in self._counted
        ]
        if not fresh:
            return 0

        workers = workers or (os.process_cpu_count() or 1)
        workers = max(1, min(workers, len(fresh)))
        if workers == 1:
            for done, shard in enumerate(fresh, start=1):
                apply(count(shard))
                self._note_progress(done, len(fresh))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                parts = pool.map(count, fresh, chunksize=1)
                for done, part in enumerate(parts, start=1):
                    apply(part)
                    self._note_progress(done, len(fresh))

        self._counted.update(fresh)
        return len(fresh)

    def _note_progress(self, done: int, total: int) -> None:
        if not self._progress_every or total <= self._progress_every:
            return
        if done % self._progress_every == 0 or done == total:
            logger.info(
                "Counted %d of %d shard(s) (%.0f%%)",
                done, total, 100.0 * done / total,
            )


class CoverageCounter(_ShardCounter):
    """Each card's qualifying records, each shard counted once across the run.

    The consequence for callers: ``CardCoverage.records`` is accumulated
    rather than assigned, so nothing else may overwrite it.
    """

    def update(
        self,
        coverage: Mapping[str, CardCoverage],
        records_dir: Path,
        *,
        workers: int | None = None,
    ) -> int:
        """Add the counts of shards not yet read. Returns how many it read."""

        def add(counts: Counter[str]) -> None:
            for name, records in counts.items():
                card = coverage.get(name)
                if card is not None:
                    card.records += records

        return self._count_fresh(records_dir, count_shard, add, workers=workers)


def build_coverage_decks(
    weights: dict[str, float],
    candidates: Mapping[str, DeckCandidate],
    count: int,
    *,
    rng: random.Random,
) -> list[list[str]]:
    """``count`` 40-card decks drawn from ``weights`` (FR-046).

    23 nonlands sampled with replacement in proportion to weight, then basics
    from ``compute_basic_lands`` over the chosen cards' mana costs. With
    replacement because a weight of 500 against 1 should be able to fill a deck
    with one card: the point is to get that card into games, not to build a
    deck anyone would play.

    A weighted card that is no candidate is skipped — the manabase heuristic
    reads its cost, and a card the tree has no text for cannot be placed.
    """
    from sealed.domain.manabase import compute_basic_lands

    pool = [(name, w) for name, w in weights.items() if name in candidates and w > 0]
    if not pool:
        return []
    names = [name for name, _ in pool]
    ws = [w for _, w in pool]

    decks: list[list[str]] = []
    for _ in range(count):
        nonlands = rng.choices(names, weights=ws, k=NONLANDS_PER_DECK)
        basics = compute_basic_lands([candidates[n] for n in nonlands])
        deck = list(nonlands)
        for land, n in basics.items():
            deck.extend([land] * n)
        decks.append(deck)
    return decks


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


TWO_HOLDOUT_SOURCES = (
    "pass one source of held-out cards, not both: --exclude-cards "
    "names the list this corpus is being depleted against, "
    "--split-from inherits a checkpoint's recorded split, and a "
    "disagreement between them is silent"
)


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
    if split_from is not None and exclude_cards is not None:
        raise ValueError(TWO_HOLDOUT_SOURCES)
    if exclude_cards is not None:
        return read_name_list(exclude_cards)
    return load_held_out(split_from)


def read_name_list(path: Path) -> frozenset[str]:
    """The card names a one-per-line list holds, folded; blank lines skipped.

    Folded because a list ``holdout-cards`` writes carries Forge's printed
    case while coverage decks are built from converted names, which are
    lowercase.
    """
    from effects.domain.card_names import fold_card_name

    return frozenset(
        fold_card_name(line.strip())
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def load_held_out(split_from: Path | None) -> frozenset[str]:
    """Cards to keep out of every deck (FR-045).

    A coverage run that put held-out cards into games would contaminate the
    card-disjoint split of the checkpoint it inherited from — the split is what
    makes gate 1 mean "deployment to an unseen set".
    """
    if split_from is None:
        return frozenset()
    from effects.infrastructure.effect_model_store import load_effect_checkpoint

    provenance = load_effect_checkpoint(Path(split_from)).provenance
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


def read_converted_texts(
    cards_folder: Path, card_files: Mapping[str, str],
) -> dict[str, str]:
    """``name -> converted text`` for every card whose file exists."""
    return {
        name: (cards_folder.parent / script).read_text(encoding="utf-8")
        for name, script in card_files.items()
        if (cards_folder.parent / script).exists()
    }


def non_deck_names(texts: Mapping[str, str]) -> set[str]:
    """The cards whose converted text names a type no deck holds."""
    return {name for name, text in texts.items() if not is_deckable(text)}


@contextmanager
def collector_supervisor(
    *,
    workers: int,
    effect_records: Path,
    caps: CollectionCaps,
    variant_scripts: Path | None = None,
) -> Iterator[CollectorSupervisor]:
    """A records-only supervisor that is stopped however its rounds end.

    ``stop()`` closes the worker log files and shuts the pool down. Without it
    the latched effect-record failure reporters (``ApiEvents.
    reportEmitterFailure`` and its siblings) never surface: their handles stay
    open, unflushed and unclosed, until the interpreter exits on its own -- a
    collection run that gets killed or piped never sees them at all.
    """
    from effects.infrastructure.collector_connector import CollectorSupervisor

    supervisor = CollectorSupervisor(
        worker_count=workers, effect_records=effect_records, caps=caps,
        variant_scripts=variant_scripts,
    )
    try:
        yield supervisor
    finally:
        supervisor.stop()


def play_decks(
    supervisor: CollectorSupervisor,
    decks: list[list[str]],
    decks_file: Path,
    *,
    label: str,
    matches: int,
) -> None:
    """Play one round of ``decks``, which a records-only round reads from a file."""
    from effects.infrastructure.deck_file import COVERAGE_SET_CODE, write_deck_file

    write_deck_file(decks, decks_file, label=label, set_code=COVERAGE_SET_CODE)
    supervisor.play_round(decks_file, matches=matches)


@dataclass(frozen=True)
class CoverageRounds:
    """What the round loop needs from one kind of coverage unit.

    The card run and the held-out round differ in what a unit is, how its
    cards are weighted and which units have to finish; a round itself -- deck,
    play, count, retire, stop on Ctrl-C -- is the same for both.
    """

    units: Mapping[str, CoverageUnit]
    #: The units whose completion ends the run: every card, but only the
    #: held-out texts some castable card carries (FR-037).
    awaited: Mapping[str, CoverageUnit]
    target: int
    candidates: Mapping[str, DeckCandidate]
    #: The next round's per-card deck weights, the consult applied.
    weights: Callable[[], dict[str, float]]
    #: The units a round attempted, from the cards its decks held.
    units_decked: Callable[[set[str]], set[str]]
    #: Counts the shards the last round wrote into ``units``; returns how many.
    count_new_shards: Callable[[], int]
    #: Logged when the weighted cards yield no deck at all.
    no_decks: str


def round_summary(
    round_number: int, rounds: CoverageRounds, retired: list[str],
) -> str:
    satisfied = sum(
        1 for unit in rounds.units.values() if unit.satisfied(rounds.target)
    )
    remaining = sum(
        1 for unit in rounds.awaited.values() if not unit.done(rounds.target)
    )
    return (
        f"round {round_number}: {satisfied} satisfied, {remaining} remaining, "
        f"{len(retired)} retired this round"
    )


def play_rounds(
    rounds: CoverageRounds,
    supervisor: CollectorSupervisor,
    config: CollectCoverageConfig,
) -> bool:
    """Play rounds until every awaited unit is done. True when Ctrl-C ended them."""
    decks_file = Path(config.effect_records) / "coverage-decks.txt"
    rng = random.Random(RANDOM_SEED)
    round_number = 0
    while not is_complete(rounds.awaited, rounds.target):
        round_number += 1
        previous = {key: unit.records for key, unit in rounds.units.items()}
        weights = rounds.weights()
        if not weights:
            break
        decks = build_coverage_decks(
            weights, rounds.candidates, config.decks_per_round, rng=rng,
        )
        if not decks:
            logger.error("%s", rounds.no_decks)
            break
        decked = rounds.units_decked({name for deck in decks for name in deck})
        play_decks(
            supervisor, decks, decks_file,
            label="coverage", matches=config.decks_per_round,
        )
        read = rounds.count_new_shards()
        logger.info("Counted %d new shard(s) from round %d", read, round_number)
        retired = retire_stalled(
            rounds.units, previous, no_progress_rounds=config.no_progress_rounds,
            decked=decked,
        )
        logger.info("%s", round_summary(round_number, rounds, retired))
        if supervisor.interrupted:
            # ForgeWorkerPool swallows SIGINT -- it sets its own event and
            # returns normally -- so without this the loop builds a fresh
            # pool and Ctrl-C means "skip this round, start another".
            logger.warning(
                "Interrupted after round %d; stopping instead of starting "
                "another round", round_number,
            )
            return True
    return False


def run(config: CollectCoverageConfig) -> int:
    """Play rounds until every card is satisfied or retired."""
    if config.only_cards is not None:
        return run_text_coverage(config)

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
    from effects.infrastructure.card_sources import load_card_files

    card_files = load_card_files(cards_folder)
    texts = read_converted_texts(cards_folder, card_files)
    non_deck = non_deck_names(texts)
    coverage = build_coverage_units(
        {name: script for name, script in card_files.items() if name not in non_deck},
        held_out,
    )
    if not coverage:
        logger.error("No deckable cards under %s", cards_folder)
        return 1
    logger.info(
        "Coverage unit: %d cards (%d held out, %d of a non-deck type), target "
        "%d records each",
        len(coverage), len(held_out), len(non_deck), config.target_records,
    )

    records_dir = Path(config.effect_records)
    counter = CoverageCounter()
    seeded = counter.update(coverage, records_dir)
    logger.info(
        "Counted %d existing shard(s); %d of %d cards already at target",
        seeded,
        sum(1 for c in coverage.values() if c.satisfied(config.target_records)),
        len(coverage),
    )

    verdicts = consult_castability(sorted(coverage))
    for name, verdict in verdicts.items():
        if name in coverage:
            coverage[name].verdict = verdict

    rounds = CoverageRounds(
        units=coverage,
        awaited=coverage,
        target=config.target_records,
        candidates={
            name: DeckCandidate.from_converted_text(name, text)
            for name, text in texts.items()
        },
        weights=lambda: rank_by_consult(
            deck_weights(coverage, config.target_records), verdicts,
        ),
        units_decked=lambda cards: cards,
        count_new_shards=lambda: counter.update(coverage, records_dir),
        no_decks="No card with a weight has converted text; nothing to deck.",
    )
    with collector_supervisor(
        workers=config.workers, effect_records=config.effect_records,
        caps=config.caps,
    ) as supervisor:
        interrupted = play_rounds(rounds, supervisor, config)

    report = residues(coverage, config.target_records)
    logger.info("%s", report.render(config.target_records))
    # The residues are printed either way -- what was collected before the
    # interrupt is still what an operator needs to see -- but an interrupted
    # run did not finish, and reporting success would be the same lie the
    # swallowed SIGINT already told.
    return 130 if interrupted else 0


def consult_castability(names: list[str]) -> dict[str, ConsultVerdict]:
    """Ask a live Forge whether each card can be cast at all (FR-047).

    Needs a game, so it runs in the JVM. A failure to consult leaves every card
    ``unknown``, which ranks them as castable — the safe direction, since the
    consult only ranks and an under-ranked card merely takes longer.
    """
    from effects.infrastructure.castability_connector import CastabilityConnector

    try:
        verdicts = CastabilityConnector().consult(names)
    except (FileNotFoundError, OSError) as exc:
        logger.warning(
            "Castability consult unavailable (%s); every card ranks as "
            "castable, which only slows the run down", exc,
        )
        return {}
    return {name: ConsultVerdict.parse(raw) for name, raw in verdicts.items()}


# ── the held-out coverage round (FR-035–FR-038) ─────────────────────────


def count_text_shard(
    shard: Path, *, roots: dict[str, str], texts: frozenset[str],
) -> dict[str, set[str]]:
    """``held-out text -> the game ids it acted in``, over one shard.

    A text acts in a record when the record's acting line resolves to it.
    Top-level so a process pool can pickle it; each process builds its own
    sidecar cache, because one does not pickle.
    """
    from effects.domain.text_holdout import normalize_script_text
    from effects.infrastructure.record_io import read_shard
    from effects.infrastructure.sidecar_io import SidecarCache

    sidecars = SidecarCache({tree: Path(root) for tree, root in roots.items()})
    text_by_key: dict = {}
    games: dict[str, set[str]] = defaultdict(set)
    for record in read_shard(shard):
        for key in record.ability or ():
            if key not in text_by_key:
                line = sidecars.line_or_none(key)
                text = (
                    normalize_script_text(line.script_text)
                    if line is not None and line.script_text else None
                )
                text_by_key[key] = text if text in texts else None
            text = text_by_key[key]
            if text is not None:
                games[text].add(record.game_id)
    return dict(games)


class TextCoverageCounter(_ShardCounter):
    """Each held-out text's games, each shard counted once across the run."""

    def __init__(
        self, roots: dict[str, str], progress_every: int = PROGRESS_EVERY_SHARDS,
    ) -> None:
        super().__init__(progress_every)
        self._roots = dict(roots)

    def update(
        self, coverage: Mapping[str, TextCoverage], records_dir: Path,
        *, workers: int | None = None,
    ) -> int:
        """Add the games of shards not yet read. Returns how many it read."""

        def add(games_by_text: dict[str, set[str]]) -> None:
            for text, games in games_by_text.items():
                coverage[text].games |= games

        count = partial(count_text_shard, roots=self._roots, texts=frozenset(coverage))
        return self._count_fresh(records_dir, count, add, workers=workers)


@dataclass
class TextResidues:
    """What the held-out round could not reach (FR-037, SC-004)."""

    under_floor: dict[str, int] = field(default_factory=dict)
    uncastable: tuple[str, ...] = ()

    def render(self, floor: int) -> str:
        lines = [
            f"{len(self.under_floor)} held-out text(s) with a castable carrier "
            f"under {floor} games; {len(self.uncastable)} with no castable "
            "carrier, never counted as satisfied."
        ]
        lines += [f"  under floor ({games} games): {text}"
                  for text, games in sorted(self.under_floor.items())]
        lines += [f"  uncastable: {text}" for text in self.uncastable]
        return "\n".join(lines)


def text_residues(coverage: dict[str, TextCoverage], floor: int) -> TextResidues:
    under = {
        text: len(unit.games) for text, unit in coverage.items()
        if unit.castable and not unit.satisfied(floor)
    }
    uncastable = tuple(sorted(
        text for text, unit in coverage.items() if not unit.castable
    ))
    return TextResidues(under, uncastable)


def held_out_round_caps(caps: CollectionCaps) -> CollectionCaps:
    """Full strength (FR-035): every record the held-out cards produce is kept,
    because these are the only games that will ever see them.
    """
    return replace(caps, playability_rate=1.0)


def run_text_coverage(config: CollectCoverageConfig) -> int:
    """The held-out coverage round: listed cards only, keyed by held-out text."""

    from effects.domain.card_names import fold_card_name
    from effects.infrastructure.card_sources import (
        load_card_files,
        load_card_texts,
    )
    from effects.infrastructure.sidecar_io import SidecarCache, sidecar_roots

    cards_folder = config.coverage_folder()
    if not cards_folder.is_dir():
        logger.error("No converted cards under %s.", cards_folder)
        return 1
    listed = read_name_list(config.only_cards)
    card_files = load_card_files(cards_folder)
    roots = sidecar_roots(config.cards_folders)
    texts_by_card = load_card_texts(card_files, SidecarCache(roots))
    listed_files = {
        name: script for name, script in card_files.items()
        if fold_card_name(name) in listed
    }
    texts = read_converted_texts(cards_folder, listed_files)
    # Which texts are held out is decided over every listed card, non-deck
    # ones included: left out of ``listed``, they would read as unlisted
    # carriers and take the texts they share out of the round. Only then do
    # they leave as carriers, since no deck can hold one, and a text no
    # deckable card carries leaves the unit rather than retiring after rounds
    # of decks that never load.
    non_deck = non_deck_names(texts)
    present = frozenset(name for name in listed_files if name not in non_deck)
    carriers = {
        text: cards - non_deck
        for text, cards in held_out_texts_of(
            frozenset(listed_files), texts_by_card,
        ).items()
        if cards - non_deck
    }
    if non_deck:
        logger.info(
            "Held-out round: %d listed card(s) of a non-deck type left out",
            len(non_deck),
        )
    if not carriers:
        logger.error(
            "None of the %d card(s) in %s carries a text only listed cards "
            "carry; nothing to cover.", len(listed), config.only_cards,
        )
        return 1

    verdicts = consult_castability(sorted(present))
    coverage = {
        text: TextCoverage(text=text, castable=any(
            verdicts.get(card, ConsultVerdict.UNKNOWN) != ConsultVerdict.UNCASTABLE
            for card in cards
        ))
        for text, cards in carriers.items()
    }
    floor = config.min_text_games
    records_dir = Path(config.effect_records)
    counter = TextCoverageCounter({tree: str(root) for tree, root in roots.items()})
    counter.update(coverage, records_dir)
    logger.info(
        "Held-out round: %d text(s) on %d card(s); %d already at %d games; %d "
        "with no castable carrier",
        len(coverage), len(present),
        sum(1 for unit in coverage.values() if unit.satisfied(floor)), floor,
        sum(1 for unit in coverage.values() if not unit.castable),
    )

    rounds = CoverageRounds(
        units=coverage,
        awaited=awaited_texts(coverage),
        target=floor,
        candidates={
            name: DeckCandidate.from_converted_text(name, text)
            for name, text in texts.items() if name in present
        },
        weights=lambda: rank_by_consult(
            text_deck_weights(coverage, carriers, floor), verdicts,
        ),
        units_decked=lambda cards: {
            text for text, carrying in carriers.items() if carrying & cards
        },
        count_new_shards=lambda: counter.update(coverage, records_dir),
        no_decks="No unfinished held-out text has a card with converted text.",
    )
    with collector_supervisor(
        workers=config.workers, effect_records=config.effect_records,
        caps=held_out_round_caps(config.caps),
    ) as supervisor:
        interrupted = play_rounds(rounds, supervisor, config)

    logger.info("%s", text_residues(coverage, floor).render(floor))
    return 130 if interrupted else 0
