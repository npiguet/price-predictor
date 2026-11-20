"""What both collectors share: where they write, the cards they keep out of
every deck, how a deck is drawn from weights, and the records-only supervisor
that plays it.

``collect-coverage`` and ``collect-variants`` differ in what they deck and why;
everything here is the plumbing beneath both, so neither imports the other.
"""

from __future__ import annotations

import argparse
import random
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from effects.domain import default_paths
from effects.domain.card_names import fold_card_name
from effects.domain.collection_caps import CollectionCaps
from effects.domain.split import HoldoutRule
from price_predictor.domain.worker_status import ALLOW_DEGRADED_HELP
from price_predictor.infrastructure.forge_worker_pool import MAX_WORKERS, worker_count
from price_predictor.infrastructure.worker_stops import (
    DEFAULT_RECYCLE_SECONDS,
    DEFAULT_STOP_TIMEOUT,
    RECYCLE_SECONDS_HELP,
    STOP_TIMEOUT_HELP,
    WorkerStopSettings,
    recycle_seconds,
    stop_timeout,
)
from sealed.domain.manabase import compute_basic_lands

if TYPE_CHECKING:
    from effects.domain.coverage import DeckCandidate
    from effects.infrastructure.collector_connector import CollectorSupervisor
    from effects.infrastructure.converted_trees import SourceTrees

#: Java worker processes, for both collectors; the most a run may take is
#: ``forge_worker_pool.MAX_WORKERS``, which its shard counters also run.
DEFAULT_WORKERS = 12
DEFAULT_DECKS_PER_ROUND = 500


def positive_int(text: str) -> int:
    """A count of at least 1: a round of no decks, a target of no records or
    no rounds before retiring is no run (V4 #9)."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def non_negative_int(text: str) -> int:
    """A count of at least 0."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0, got {value}")
    return value



#: Seeds both collectors' deck draws, and the variant collector's perturbations.
RANDOM_SEED = 42

#: A 40-card limited deck: 23 nonlands plus basics.
NONLANDS_PER_DECK = 23


@dataclass(frozen=True, slots=True)
class DeckHoldout:
    """The held-out cards a collector's decks are drawn against: one list kept
    out of every deck, ``--exclude-cards`` (needs no trained model) or
    ``--split-from`` (an existing checkpoint's split), or, for the held-out
    coverage round, ``--only-cards``, the list decked alone.

    Two sources at once are refused rather than merged, since a merge would
    hide a disagreement, and ``--only-cards`` beside a list to keep out is
    refused as the opposite instruction.
    """

    exclude_cards: Path | None = None
    split_from: Path | None = None
    only_cards: Path | None = None

    def __post_init__(self) -> None:
        if self.only_cards is not None:
            _refuse_holdout_beside_only_cards(
                exclude_cards=self.exclude_cards, split_from=self.split_from,
            )
        if self.split_from is not None and self.exclude_cards is not None:
            raise ValueError(TWO_HOLDOUT_SOURCES)

    def exclusions(self) -> frozenset[str]:
        """Cards to keep out of every deck, folded, from whichever source names them."""
        if self.exclude_cards is not None:
            return read_name_list(self.exclude_cards)
        return load_held_out(self.split_from)


@dataclass
class CollectorConfig:
    """What both collectors take: the corpus they write to, the converted
    trees, the holdout they keep out of their decks, and how they play."""

    effect_records: Path = field(kw_only=True)
    cards_folders: tuple[Path, ...] = default_paths.DEFAULT_CARDS_FOLDER_PATHS
    deck_holdout: DeckHoldout = DeckHoldout()
    decks_per_round: int = field(default=DEFAULT_DECKS_PER_ROUND, metadata={
        "type": positive_int,
        "help": "Decks played as matches per round (default: %(default)s)",
    })
    workers: int = field(default=DEFAULT_WORKERS, metadata={
        "type": worker_count,
        "help": (
            f"Parallel Java worker processes, and shard-counting processes, 1 to "
            f"{MAX_WORKERS} (default: %(default)s)"
        ),
    })
    stop_timeout: int = field(default=DEFAULT_STOP_TIMEOUT, metadata={
        "type": stop_timeout,
        "help": STOP_TIMEOUT_HELP,
    })
    recycle_seconds: int = field(default=DEFAULT_RECYCLE_SECONDS, metadata={
        "type": recycle_seconds,
        "help": RECYCLE_SECONDS_HELP,
    })
    allow_degraded: bool = field(default=False, metadata={"help": ALLOW_DEGRADED_HELP})
    #: Every record a collector writes is aimed (FR-009a): at the run's
    #: ``--directed-fork-rules``, and at no rule, ``[]``, where it names none.
    caps: CollectionCaps = CollectionCaps(aimed_at=())
    seed: int = field(default=RANDOM_SEED, metadata={
        "help": (
            "Seeds the per-round deck draw, so which cards fill a round is "
            "reproducible (default: %(default)s)"
        ),
    })
    #: The holdout ``build-corpus`` chooses, over ``cards_folders``: a variant
    #: printing a held-out text is dropped under it, and the held-out round
    #: credits a game to a held-out text by it. Given the flags
    #: ``build-corpus`` is, so the three agree.
    holdout: HoldoutRule = HoldoutRule()

    @property
    def source_trees(self) -> SourceTrees:
        """The converted trees the decks and the holdout read: ``--cards-folder``
        alone, since a collector's own variant tree is one it writes."""
        from effects.infrastructure.converted_trees import SourceTrees

        return SourceTrees.of(self.cards_folders)

    @property
    def worker_stop(self) -> WorkerStopSettings:
        """How the run's workers are stopped and recycled."""
        return WorkerStopSettings(timeout=self.stop_timeout, recycle_seconds=self.recycle_seconds)


TWO_HOLDOUT_SOURCES = (
    "pass one source of held-out cards, not both: --exclude-cards "
    "names the list this corpus is being depleted against, "
    "--split-from inherits a checkpoint's recorded split, and a "
    "disagreement between them is silent"
)


#: The name a depletion list takes inside the corpus it depleted.
HOLDOUT_LIST_NAME = "holdout-cards.txt"


@dataclass(frozen=True)
class CollectionFlags:
    """A collector's corpus and holdout flags as given, each None when unset."""

    effect_records: Path | None
    exclude_cards: Path | None
    split_from: Path | None
    training_corpus: Path | None = None
    only_cards: Path | None = None


def collection_target(flags: CollectionFlags) -> tuple[Path, DeckHoldout]:
    """Where a collector's flags write and the holdout they name, refused
    before any worker starts.

    ``--training-corpus DIR`` stands for ``--effect-records DIR`` with
    ``DIR/holdout-cards.txt`` as the depletion list, so a corpus and the list
    that depleted it cannot be named apart; an unset ``--effect-records`` is
    the default directory.

    Raises ``ValueError`` when ``--only-cards`` comes with a holdout to keep
    out, when ``--training-corpus`` comes with a flag it implies (an explicit
    default ``--effect-records`` included) or its directory holds no depletion
    list, or when two sources name the held-out cards.
    """
    if flags.only_cards is not None:
        _refuse_holdout_beside_only_cards(
            exclude_cards=flags.exclude_cards, training_corpus=flags.training_corpus,
            split_from=flags.split_from,
        )
    if flags.training_corpus is None:
        if flags.effect_records is None:
            raise ValueError(
                "name the corpus: pass --effect-records DIR, or --training-corpus DIR"
            )
        return flags.effect_records, DeckHoldout(
            exclude_cards=flags.exclude_cards, split_from=flags.split_from,
            only_cards=flags.only_cards,
        )
    if flags.effect_records is not None or flags.exclude_cards is not None:
        raise ValueError(
            "pass one source of the corpus location, not both: "
            "--training-corpus DIR means --effect-records DIR with its own "
            f"{HOLDOUT_LIST_NAME}"
        )
    return flags.training_corpus, DeckHoldout(
        exclude_cards=_depletion_list(flags.training_corpus), split_from=flags.split_from,
    )


def _refuse_holdout_beside_only_cards(
    *, exclude_cards: Path | None, split_from: Path | None,
    training_corpus: Path | None = None,
) -> None:
    clashing = [
        flag for flag, value in (
            ("--exclude-cards", exclude_cards),
            ("--training-corpus", training_corpus),
            ("--split-from", split_from),
        ) if value is not None
    ]
    if clashing:
        # FR-038: opposite instructions, so one of them is a mistake.
        raise ValueError(
            f"--only-cards decks the held-out cards; {', '.join(clashing)} "
            "keeps a holdout out of every deck. Pass one or the other."
        )


def _depletion_list(training_corpus: Path) -> Path:
    """The list ``training_corpus`` was depleted against, which it must hold."""
    listed = training_corpus / HOLDOUT_LIST_NAME
    if not listed.is_file():
        raise ValueError(
            f"{training_corpus} holds no {HOLDOUT_LIST_NAME}. A training "
            "corpus is depleted against a list, and without it this run "
            "would deck the held-out cards and contaminate the split. Write "
            "one with 'python -m effects holdout-cards', or pass "
            "--effect-records and --exclude-cards separately if they "
            "genuinely live apart."
        )
    return listed


def read_name_list(path: Path) -> frozenset[str]:
    """The card names a one-per-line list holds, folded to match converted
    names; blank lines skipped.

    A split card's combined name (``Fire // Ice``, as ``holdout-cards`` writes
    it for Forge) is read as its first face too, the name the converted tree
    gives the card.
    """
    names: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        name = fold_card_name(line.strip())
        if not name:
            continue
        names.add(name)
        if " // " in name:
            names.add(name.split(" // ", 1)[0])
    return frozenset(names)


def load_held_out(split_from: Path | None) -> frozenset[str]:
    """Cards to keep out of every deck (FR-045): the held-out cards of the
    checkpoint whose card-disjoint split this run must not contaminate."""
    if split_from is None:
        return frozenset()
    from effects.infrastructure.effect_model_store import load_effect_checkpoint

    provenance = load_effect_checkpoint(Path(split_from)).provenance
    return frozenset(provenance.held_out_cards)



def build_coverage_decks(
    weights: dict[str, float],
    candidates: Mapping[str, DeckCandidate],
    count: int,
    *,
    rng: random.Random,
) -> list[list[str]]:
    """``count`` 40-card decks drawn from ``weights`` (FR-046).

    23 nonlands sampled with replacement in proportion to weight, since the
    point is to get a card into games rather than a deck anyone would play,
    then basics from ``compute_basic_lands``. A weighted card that is no
    candidate has no cost to read and is skipped.
    """
    pool = _drawable(weights, candidates)
    if not pool:
        return []
    names = [name for name, _ in pool]
    ws = [w for _, w in pool]
    return [
        _with_basics(rng.choices(names, weights=ws, k=NONLANDS_PER_DECK), candidates)
        for _ in range(count)
    ]


def _drawable(
    weights: dict[str, float], candidates: Mapping[str, DeckCandidate],
) -> list[tuple[str, float]]:
    """The weighted cards a deck can hold: a candidate with a positive weight."""
    return [(name, w) for name, w in weights.items() if name in candidates and w > 0]


def _with_basics(nonlands: list[str], candidates: Mapping[str, DeckCandidate]) -> list[str]:
    """``nonlands`` and the basics ``compute_basic_lands`` gives them."""
    basics = compute_basic_lands([candidates[n] for n in nonlands])
    deck = list(nonlands)
    for land, n in basics.items():
        deck.extend([land] * n)
    return deck



@contextmanager
def collector_supervisor(
    config: CollectorConfig, *, variant_scripts: Path | None = None,
) -> Iterator[CollectorSupervisor]:
    """A records-only supervisor of ``config``'s workers, writing to its
    corpus under its caps and a fresh run id, stopped however its rounds end:
    without ``stop()`` the worker logs holding the latched failure reports
    stay unflushed until the interpreter exits. ``--allow-degraded`` lets the
    run go on over an engine that is not whole; ``variant_scripts`` is the
    variant tree the workers stage."""
    from effects.infrastructure.collector_connector import CollectorRun, CollectorSupervisor

    supervisor = CollectorSupervisor(CollectorRun(
        worker_count=config.workers, effect_records=config.effect_records,
        run_id=str(uuid.uuid4()), caps=config.caps, variant_scripts=variant_scripts,
        stop=config.worker_stop, allow_degraded=config.allow_degraded,
    ))
    try:
        yield supervisor
    finally:
        supervisor.stop()


def record_damaged_shards(supervisor, damaged: list[str]) -> None:
    """Keep every damaged shard the run has read so far, each its read's
    message, in ``run.json`` as ``corrupt_shards``; nothing when there is none."""
    if damaged:
        supervisor.record(corrupt_shards=sorted(damaged))
