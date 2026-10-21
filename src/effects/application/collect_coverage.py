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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from effects.domain.collection_caps import CollectionCaps
from effects.domain.coverage import (
    CardCoverage,
    ConsultVerdict,
    TextCoverage,
    deck_weights,
    held_out_texts_of,
    is_complete,
    is_deckable,
    is_text_round_complete,
    qualifying_cards,
    rank_by_consult,
    retire_stalled,
    text_deck_weights,
)

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

    def __init__(self, progress_every: int = PROGRESS_EVERY_SHARDS) -> None:
        self._counted: set[Path] = set()
        self._progress_every = progress_every

    def update(
        self,
        coverage: dict[str, CardCoverage],
        records_dir: Path,
        *,
        workers: int | None = None,
    ) -> int:
        """Add the counts of shards not yet read. Returns how many it read.

        Shards are independent and each contributes only a per-card count, so
        the work maps: count shards in parallel, sum the counters, apply once.
        Summation is associative, so the answer cannot depend on how the shards
        were divided or in what order they finished.

        ``workers`` defaults to the machine's usable cores. One means in-process
        — what the tests use for the serial reference, and what a handful of
        shards wants, since a process pool costs more to start than it saves.
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
        totals: Counter[str] = Counter()
        if workers == 1:
            for done, shard in enumerate(fresh, start=1):
                totals += count_shard(shard)
                self._note_progress(done, len(fresh))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                done = 0
                for counted in pool.map(count_shard, fresh, chunksize=1):
                    totals += counted
                    done += 1
                    self._note_progress(done, len(fresh))

        for name, count in totals.items():
            card = coverage.get(name)
            if card is not None:
                card.records += count
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
    from effects.domain.card_names import fold_card_name

    if split_from is not None and exclude_cards is not None:
        raise ValueError(TWO_HOLDOUT_SOURCES)
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


def run(config: CollectCoverageConfig) -> int:
    """Play rounds until every card is satisfied or retired."""
    if config.only_cards is not None:
        return run_text_coverage(config)
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
    texts = {
        name: (cards_folder.parent / script).read_text(encoding="utf-8")
        for name, script in card_files.items()
        if (cards_folder.parent / script).exists()
    }
    non_deck = {name for name, text in texts.items() if not is_deckable(text)}
    card_files = {
        name: script for name, script in card_files.items() if name not in non_deck
    }
    coverage = build_coverage_units(card_files, held_out)
    if not coverage:
        logger.error("No deckable cards under %s", cards_folder)
        return 1
    logger.info(
        "Coverage unit: %d cards (%d held out, %d of a non-deck type), target "
        "%d records each",
        len(coverage), len(held_out), len(non_deck), config.target_records,
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
) -> dict[str, ConsultVerdict]:
    """Ask a live Forge whether each card can be cast at all (FR-047).

    Needs a game, so it runs in the JVM. A failure to consult leaves every card
    ``unknown``, which ranks them as castable — the safe direction, since the
    consult only ranks and an under-ranked card merely takes longer.
    """
    from effects.infrastructure.castability_connector import CastabilityConnector

    try:
        verdicts = CastabilityConnector().consult(names, cards_folder)
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
                try:
                    line = sidecars.line_for(key)
                except KeyError:
                    line = None
                text = (
                    normalize_script_text(line.script_text)
                    if line is not None and line.script_text else None
                )
                text_by_key[key] = text if text in texts else None
            text = text_by_key[key]
            if text is not None:
                games[text].add(record.game_id)
    return dict(games)


class TextCoverageCounter:
    """Each shard counted once across the run, as ``CoverageCounter`` does."""

    def __init__(self, roots: dict[str, str]) -> None:
        self._roots = dict(roots)
        self._counted: set[Path] = set()

    def update(
        self, coverage: dict[str, TextCoverage], records_dir: Path,
        *, workers: int | None = None,
    ) -> int:
        from functools import partial

        from effects.infrastructure.record_io import iter_shards

        fresh = [s for s in iter_shards(Path(records_dir)) if s not in self._counted]
        if not fresh:
            return 0
        count = partial(count_text_shard, roots=self._roots, texts=frozenset(coverage))
        workers = max(1, min(workers or (os.process_cpu_count() or 1), len(fresh)))
        if workers == 1:
            parts = map(count, fresh)
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            parts = pool.map(count, fresh, chunksize=1)
        try:
            for part in parts:
                for text, games in part.items():
                    coverage[text].games |= games
        finally:
            if workers > 1:
                pool.shutdown()
        self._counted.update(fresh)
        return len(fresh)


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

    What-if legality queries stay at the run's rate: they are the AI's
    planning, not observations of the card, and real declarations are
    written whatever the rate.
    """
    return replace(caps, playability_rate=1.0)


def run_text_coverage(config: CollectCoverageConfig) -> int:
    """The held-out coverage round: listed cards only, keyed by held-out text."""

    from effects.application.train_effect_model import (
        load_card_files,
        load_card_texts,
    )
    from effects.domain.card_names import fold_card_name
    from effects.infrastructure.collector_connector import CollectorSupervisor
    from effects.infrastructure.deck_file import COVERAGE_SET_CODE, write_deck_file
    from effects.infrastructure.sidecar_io import SidecarCache, sidecar_roots

    cards_folder = config.coverage_folder()
    if not cards_folder.is_dir():
        logger.error("No converted cards under %s.", cards_folder)
        return 1
    listed = frozenset(
        fold_card_name(line.strip())
        for line in Path(config.only_cards).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    card_files = load_card_files(cards_folder)
    roots = sidecar_roots(config.cards_folders)
    texts_by_card = load_card_texts(card_files, SidecarCache(roots))
    listed_files = {
        name: script for name, script in card_files.items()
        if fold_card_name(name) in listed
    }
    texts = {
        name: (cards_folder.parent / script).read_text(encoding="utf-8")
        for name, script in listed_files.items()
        if (cards_folder.parent / script).exists()
    }
    # Which texts are held out is decided over every listed card, non-deck
    # ones included: left out of ``listed``, they would read as unlisted
    # carriers and take the texts they share out of the round. Only then do
    # they leave as carriers, since no deck can hold one, and a text no
    # deckable card carries leaves the unit rather than retiring after rounds
    # of decks that never load.
    non_deck = {name for name, text in texts.items() if not is_deckable(text)}
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

    verdicts = consult_castability(sorted(present), cards_folder)
    coverage = {
        text: TextCoverage(text=text, castable=any(
            verdicts.get(card, ConsultVerdict.UNKNOWN) != ConsultVerdict.UNCASTABLE
            for card in cards
        ))
        for text, cards in carriers.items()
    }
    floor = config.min_text_games
    counter = TextCoverageCounter({tree: str(root) for tree, root in roots.items()})
    counter.update(coverage, Path(config.effect_records))
    logger.info(
        "Held-out round: %d text(s) on %d card(s); %d already at %d games; %d "
        "with no castable carrier",
        len(coverage), len(present),
        sum(1 for unit in coverage.values() if unit.satisfied(floor)), floor,
        sum(1 for unit in coverage.values() if not unit.castable),
    )

    texts = {name: text for name, text in texts.items() if name in present}
    decks_file = Path(config.effect_records) / "coverage-decks.txt"
    rng = random.Random(RANDOM_SEED)
    caps = held_out_round_caps(config.caps)
    supervisor = CollectorSupervisor(
        worker_count=config.workers, effect_records=config.effect_records, caps=caps,
    )
    interrupted = False
    try:
        round_number = 0
        while not is_text_round_complete(coverage, floor):
            round_number += 1
            previous = {text: len(unit.games) for text, unit in coverage.items()}
            weights = rank_by_consult(text_deck_weights(coverage, carriers, floor), verdicts)
            decks = build_coverage_decks(weights, texts, config.decks_per_round, rng=rng)
            if not decks:
                logger.error("No unfinished held-out text has a card with converted text.")
                break
            write_deck_file(decks, decks_file, label="coverage", set_code=COVERAGE_SET_CODE)
            decked_cards = {name for deck in decks for name in deck}
            decked = {
                text for text, cards in carriers.items() if cards & decked_cards
            }
            supervisor.play_round(decks_file, matches=config.decks_per_round)
            counter.update(coverage, Path(config.effect_records))
            retired = retire_stalled(
                coverage, previous, no_progress_rounds=config.no_progress_rounds,
                decked=decked,
            )
            logger.info(
                "round %d: %d satisfied, %d remaining, %d retired this round",
                round_number,
                sum(1 for unit in coverage.values() if unit.satisfied(floor)),
                sum(1 for unit in coverage.values()
                    if unit.castable and not unit.done(floor)),
                len(retired),
            )
            if supervisor.interrupted:
                interrupted = True
                logger.warning("Interrupted after round %d.", round_number)
                break
    finally:
        supervisor.stop()

    logger.info("%s", text_residues(coverage, floor).render(floor))
    return 130 if interrupted else 0
