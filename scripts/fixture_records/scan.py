"""The scan that fills a fixture's categories from real shards, shard by
shard, with the records each chosen one brings along: a chosen cost half its
effect half, a chosen fired trigger the resolutions after it in its game, a
chosen burn kill the dies triggers it fired, and (gen 3) every record sharing
a ``link_id`` with a chosen one, mirrored by one, or ending its game."""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import orjson

from effects.infrastructure.shard_io import iter_shard_lines
from fixture_records.acceptance import burn_kill
from fixture_records.shards import Partners, ended_games, joined, tag_line

#: The category a game whose seats the Forge AI played both of fills.
FORGE_AI = "forge-ai"
#: Records that follow a fired trigger in its game, kept so an interaction join
#: (trigger -> the resolution it put on the stack) has its second half.
_FOLLOWERS_PER_TRIGGER = 6
#: Trigger records kept per game to find the ones a chosen kill fired.
_RECENT_TRIGGERS = 12
_BRINGS_FOLLOWERS = ("trigger-fired", "a7-burn-kill")
_BURN_KILL = "a7-burn-kill"


@dataclass
class Selection:
    """What the scan has chosen so far, across shards. ``chosen`` lists its
    categories in the order the scan first looked at each, which is the order
    the summary prints them in."""

    categories: dict[str, Callable[[dict], bool]]
    per_category: int
    forge_ai_wanted: bool
    chosen: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    partners: Partners = field(default_factory=Partners)
    #: Records that came in beside a chosen one, by record id.
    extras: dict[str, dict] = field(default_factory=dict)

    def is_full(self, name: str) -> bool:
        return len(self.chosen[name]) >= self.per_category

    def category_of(self, record: dict) -> str | None:
        """The first category still open that ``record`` fits."""
        for name, test in self.categories.items():
            if not self.is_full(name) and test(record):
                return name
        return None

    def complete(self) -> bool:
        return all(self.is_full(name) for name in self.categories) and (
            not self.forge_ai_wanted or self.is_full(FORGE_AI))

    def filled(self) -> int:
        return sum(self.is_full(name) for name in self.categories)

    def missing(self) -> list[str]:
        names = [*self.categories, *([FORGE_AI] if self.forge_ai_wanted else [])]
        return [name for name in names if not self.chosen[name]]

    def records(self) -> dict[str, dict]:
        """The chosen records, then those that came in beside them."""
        records = {record["record_id"]: record
                   for name in self.chosen for record in self.chosen[name]}
        for record_id, record in self.extras.items():
            records.setdefault(record_id, record)
        return records

    def print_counts(self) -> None:
        for name in self.chosen:
            print(f"{name:30s} {len(self.chosen[name])}", file=sys.stderr)


def _parsed(line: str) -> dict | None:
    try:
        # orjson, as the codec reads: a gen-3 shard is tens of gigabytes of JSON.
        return orjson.loads(line)
    except orjson.JSONDecodeError:
        return None


def _event_subjects(trigger: dict) -> set[str]:
    event = trigger["payload"].get("event") or {}
    return set(event.get("subjects") or ())


class ShardScan:
    """One shard's read: what lives only while the shard is read. Games never
    span shards, so a game is known to be Forge AI on both seats, and every
    partner of a chosen record is known, once its shard has been read through."""

    def __init__(self, selection: Selection, shard: Path, shard_no: int,
                 ended: set[str] | None) -> None:
        self.selection = selection
        self.shard = shard
        self.shard_no = shard_no
        #: Gen 3: the games that ended, as every reader keeps only those.
        self.ended = ended
        self.games: set[str] = set()
        self.followers: dict[str, int] = {}
        #: The trigger records just written in each game: a dies trigger's
        #: record precedes the held effect half of the kill that fired it.
        self.recent_triggers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=_RECENT_TRIGGERS))
        self.seat_games: set[str] = set()
        self.plain: dict[str, list[dict]] = defaultdict(list)

    def read(self) -> None:
        self._read_lines()
        if self.selection.forge_ai_wanted:
            self._choose_forge_ai()
        if self.ended is not None and self.games:
            self._join_partners()

    def _read_lines(self) -> None:
        for number, line in enumerate(iter_shard_lines(self.shard)):
            record = _parsed(line)
            if record is not None and self._admit(record, line, number):
                self._read(record, record["game_id"])

    def _join_partners(self) -> None:
        """Gen 3: bring every partner of the records chosen from this shard."""
        found = joined(self.shard, self.shard_no, self.games, self.selection.partners)
        for record_id, record in found.items():
            self.selection.extras.setdefault(record_id, record)

    def _admit(self, record: dict, line: str, number: int) -> bool:
        if self.ended is None:
            return True
        if record["game_id"] not in self.ended:
            return False
        tag_line(record, line, self.shard_no, number)
        return True

    def _read(self, record: dict, game: str) -> None:
        self._note_seat(record, game)
        self._follow(record, game)
        self._keep_partner(record)
        if record["kind"] == "trigger":
            self.recent_triggers[game].append(record)
        name = self.selection.category_of(record)
        if name is not None:
            self._choose(name, record, game)

    def _note_seat(self, record: dict, game: str) -> None:
        if record.get("random_seat"):
            self.seat_games.add(game)
        elif self._may_be_forge_ai(record, game):
            self.plain[game].append(record)

    def _may_be_forge_ai(self, record: dict, game: str) -> bool:
        return self.selection.forge_ai_wanted and not record["fork"] and len(self.plain[game]) < 2

    def _follow(self, record: dict, game: str) -> None:
        """Keep a resolution following a chosen fired trigger in its game."""
        if self.followers.get(game) and record["kind"] == "resolution":
            self.selection.extras[record["record_id"]] = record
            self.games.add(game)
            self.followers[game] -= 1

    def _keep_partner(self, record: dict) -> None:
        record_id = record["record_id"]
        partners = self.selection.partners
        if record.get("link_id") in partners.links:
            self.selection.extras.setdefault(record_id, record)
        if record_id in partners.mirrors:
            self.selection.extras[record_id] = record

    def _choose(self, name: str, record: dict, game: str) -> None:
        self.selection.chosen[name].append(record)
        self.games.add(game)
        self.selection.partners.note(record)
        if name in _BRINGS_FOLLOWERS:
            self.followers[game] = _FOLLOWERS_PER_TRIGGER
        if name == _BURN_KILL:
            self._bring_dies_triggers(record, game)

    def _bring_dies_triggers(self, record: dict, game: str) -> None:
        dead = burn_kill(record)
        for trigger in self.recent_triggers[game]:
            if _event_subjects(trigger) & dead:
                self.selection.extras[trigger["record_id"]] = trigger

    def _choose_forge_ai(self) -> None:
        for game, kept in self.plain.items():
            if game not in self.seat_games and not self.selection.is_full(FORGE_AI):
                self.selection.chosen[FORGE_AI].append(kept[0])
                self.games.add(game)


def scan(shards: list[Path], selection: Selection, *, gen3: bool) -> None:
    """Read ``shards`` in order until every category is full."""
    for shard_no, shard in enumerate(shards):
        ended = ended_games(shard) if gen3 else None
        ShardScan(selection, shard, shard_no, ended).read()
        if selection.complete():
            break
        print(f"{shard_no + 1} shards read, {selection.filled()}/{len(selection.categories)} "
              "categories filled", file=sys.stderr, flush=True)
