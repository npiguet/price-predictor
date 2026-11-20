"""``collect-coverage``'s counting: what each shard adds to the card run's
cards and to the held-out round's texts, each shard counted once across the
run through the shard ledger.

Only the records build-corpus keeps count (python D1): no ``turn_action`` (a
draw or an untap is no record of the card), none it refuses, and none of a
game its shard holds no game-end marker for (a cut game, which build-corpus
drops whole; credits are held per game until the shard is read).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from effects.application.shard_ledger import ShardLedger, ShardPass, ShardRead, read_records
from effects.domain.card_names import fold_card_name
from effects.domain.coverage import (
    CardCoverage,
    CardOfScript,
    TextCoverage,
    cards_shown,
    qualifying_cards,
)
from effects.domain.provenance import SidecarLookup, SourceTree, split_script_file
from effects.domain.record_quality import quality_defect, record_refusal
from effects.domain.records import EffectRecord, RecordKind, acting_keys

if TYPE_CHECKING:
    from effects.domain.split import HeldOutCards

logger = logging.getLogger(__name__)

#: Shards between progress lines while counting: the first pass reads the whole
#: existing corpus, and silence that long looks like a hang.
PROGRESS_EVERY_SHARDS = 10


@dataclass(frozen=True)
class ShardCoverage:
    """What one shard adds to the card run: each card's credited records,
    every card its games showed (a round charges a stalled round to those
    alone), and how many records it holds."""

    credits: Counter[str]
    seen: frozenset[str]
    records: int = 0
    #: What ended the shard's read early (a damaged member or line), its
    #: records before the damage counted; empty for a whole shard.
    damaged: tuple[str, ...] = ()
    #: Games without a game-end marker, whose records credit nothing.
    cut_games: int = 0


@dataclass(frozen=True)
class TextShardCoverage:
    """What one shard adds to the held-out round: each held-out text's games,
    every card the shard's games showed, and how many records it holds."""

    games: dict[str, set[str]]
    seen: frozenset[str]
    records: int = 0
    #: As :attr:`ShardCoverage.damaged`.
    damaged: tuple[str, ...] = ()
    #: As :attr:`ShardCoverage.cut_games`.
    cut_games: int = 0


def _card_of_script(sidecars: SidecarLookup | None) -> CardOfScript | None:
    """``script file -> converted card name`` (a sidecar's ``card``), for
    card-tree scripts only; None without sidecars."""
    if sidecars is None:
        return None

    def card_of(script_file: str) -> str | None:
        if split_script_file(script_file)[0] != SourceTree.CARDSFOLDER:
            return None
        try:
            return sidecars.get(script_file).card
        except (KeyError, OSError, ValueError):
            return None

    return card_of


def _kept(record: EffectRecord, sidecars: SidecarLookup | None) -> bool:
    """Whether build-corpus would keep the record: not a ``turn_action``, and
    not refused (its quality rule alone without sidecars)."""
    if record.kind is RecordKind.TURN_ACTION:
        return False
    if sidecars is None:
        return quality_defect(record) is None
    return record_refusal(record, sidecars.resolution_of) is None


@dataclass
class _CardCredits:
    """One shard's per-record visitor for the card run: each game's credits,
    held until the shard's read says which games were cut."""

    card_of: CardOfScript | None
    sidecars: SidecarLookup | None
    by_game: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    seen: set[str] = field(default_factory=set)
    records: int = 0

    def add(self, record: EffectRecord) -> None:
        self.records += 1
        self.seen.update(fold_card_name(name) for name in cards_shown(record, self.card_of))
        if not _kept(record, self.sidecars):
            return
        for name in qualifying_cards(record, self.card_of):
            self.by_game[record.game_id][fold_card_name(name)] += 1

    def coverage(self, read: ShardRead) -> ShardCoverage:
        counts: Counter[str] = Counter()
        for game, credits in self.by_game.items():
            if game not in read.cut:
                counts.update(credits)
        return ShardCoverage(
            counts, frozenset(self.seen), self.records, read.damaged, len(read.cut),
        )


def count_shard(shard: Path, *, roots: Mapping[str, Path] | None = None) -> ShardCoverage:
    """One shard's qualifying records, keyed by folded card name to match the
    converted (lowercase) unit names. Top-level so a process pool can pickle it.

    A record of an older generation stops the count with
    :class:`~effects.domain.shard_generation.OlderGenerationError`, as it
    stops every reader. Under ``roots`` an entity counts as the converted card
    its printed script is (python D6), and a static's host through its script
    (python D5). The cards shown are every game's, a cut one's too: its games
    showed them.
    """
    from effects.infrastructure.converted_trees import process_sidecars
    from effects.infrastructure.shard_io import read_shard

    sidecars = process_sidecars(roots) if roots else None
    credits = _CardCredits(_card_of_script(sidecars), sidecars)
    return credits.coverage(read_records(read_shard(shard), credits.add))


@dataclass
class _TextGames:
    """One shard's per-record visitor for the held-out round: each held-out
    text's games, from the resolution records the refusal rule keeps."""

    sidecars: Any
    card_of: CardOfScript | None
    held_out: HeldOutCards
    games: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    seen: set[str] = field(default_factory=set)
    records: int = 0

    def every_record(self, head) -> bool:
        """Count every record; parse only the resolutions and the turn
        actions, among which the game-end markers."""
        self.records += 1
        return head.kind in (RecordKind.RESOLUTION.value, RecordKind.TURN_ACTION.value)

    def add(self, record: EffectRecord) -> None:
        if record.kind is not RecordKind.RESOLUTION:
            return
        self.seen.update(fold_card_name(name) for name in cards_shown(record, self.card_of))
        if record_refusal(record, self.sidecars.resolution_of) is not None:
            return
        text = self.held_out.text_acted_through(acting_keys(record), self.sidecars.script_text)
        if text is not None:
            self.games[text].add(record.game_id)

    def coverage(self, read: ShardRead) -> TextShardCoverage:
        kept = {text: ids - read.cut for text, ids in self.games.items() if ids - read.cut}
        return TextShardCoverage(
            kept, frozenset(self.seen), self.records, read.damaged, len(read.cut),
        )


def count_text_shard(
    shard: Path, *, roots: dict[str, Path], held_out: HeldOutCards,
) -> TextShardCoverage:
    """``held-out text -> the game ids it resolved in``, over one shard.

    Counted as build-corpus counts a held-out text's games (FR-053): from the
    resolution records its refusal rule keeps, through
    :meth:`HeldOutCards.text_acted_through`. A random-seat game counts: it
    goes to the card-disjoint stratum like any game naming a held-out card
    (FR-059). A game its shard holds no game-end marker for is cut and
    counts toward no text, as build-corpus drops it. The cards shown are read
    off the same resolution records, whose snapshots hold every hand,
    graveyard and permanent. Top-level so a
    process pool can pickle it; each process resolves keys through its own
    :func:`~effects.infrastructure.converted_trees.process_sidecars`, because a
    sidecar cache does not pickle.
    """
    from effects.infrastructure.converted_trees import process_sidecars
    from effects.infrastructure.shard_io import iter_shard_matching

    sidecars = process_sidecars(roots)
    games = _TextGames(sidecars, _card_of_script(sidecars), held_out)
    return games.coverage(read_records(
        iter_shard_matching(shard, where=games.every_record), games.add,
    ))


def _count_fresh(
    ledger: ShardLedger, records_dir: Path, how: ShardPass,
    add: Callable[[Any], None],
) -> int:
    """``add`` the count of each shard under ``records_dir`` the ledger has
    not read, warning of each damaged one; returns how many. Adding sums, so
    the order shards finish in cannot matter."""
    fresh = ledger.fresh(records_dir)

    def take(part: ShardCoverage | TextShardCoverage) -> None:
        add(part)
        for message in part.damaged:
            logger.warning("%s; its %d records before the damage are counted, and "
                           "the shard is not read again", message, part.records)

    ledger.count(fresh, how, take)
    return len(fresh)


class CoverageCounter(ShardLedger):
    """Each card's qualifying records, each shard counted once across the run.
    ``CardCoverage.records`` is accumulated, so nothing else may overwrite it.

    ``roots`` (the converted trees' sidecar roots) apply build-corpus's whole
    refusal rule and credit each entity as its converted card; without them
    only the quality rule applies and an entity counts by its name.
    """

    def __init__(
        self, roots: Mapping[str, Path] | None = None,
        progress_every: int = PROGRESS_EVERY_SHARDS,
    ) -> None:
        super().__init__()
        self._roots = dict(roots) if roots else None
        self._progress_every = progress_every
        #: Every card the shards of the last ``update`` showed.
        self.seen: set[str] = set()

    def update(
        self,
        coverage: Mapping[str, CardCoverage],
        records_dir: Path,
        *,
        workers: int | None = None,
    ) -> int:
        """Add the counts of shards not yet read. Returns how many it read."""
        self.seen = set()

        def add(part: ShardCoverage) -> None:
            self.seen |= part.seen
            for name, records in part.credits.items():
                card = coverage.get(name)
                if card is not None:
                    card.records += records

        how = ShardPass(partial(count_shard, roots=self._roots), workers, self._progress_every)
        return _count_fresh(self, records_dir, how, add)


class TextCoverageCounter(ShardLedger):
    """Each held-out text's games, each shard counted once across the run.

    A game credited to a held-out text that is no unit of the round, such as
    one only a token script prints, counts toward none: build-corpus credits
    it to that text too, not to a later acting key's.
    """

    def __init__(
        self, roots: dict[str, Path], held_out: HeldOutCards,
        progress_every: int = PROGRESS_EVERY_SHARDS,
    ) -> None:
        super().__init__()
        self._roots = dict(roots)
        self._held_out = held_out
        self._progress_every = progress_every
        #: Every card the shards of the last ``update`` showed.
        self.seen: set[str] = set()

    def update(
        self, coverage: Mapping[str, TextCoverage], records_dir: Path,
        *, workers: int | None = None,
    ) -> int:
        """Add the games of shards not yet read. Returns how many it read."""
        self.seen = set()

        def add(part: TextShardCoverage) -> None:
            self.seen |= part.seen
            for text, games in part.games.items():
                if text in coverage:
                    coverage[text].games |= games

        count = partial(count_text_shard, roots=self._roots, held_out=self._held_out)
        return _count_fresh(self, records_dir, ShardPass(count, workers, self._progress_every), add)
