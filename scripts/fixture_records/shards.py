"""Reading shards for a fixture and writing it: the interleaved shard order,
a gen-3 shard's ended games and the records a chosen one joins with, and the
fixture file itself."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import orjson

from effects.infrastructure.shard_io import iter_shard_lines, iter_shards
from fixture_records.shapes import is_game_end


@dataclass
class Partners:
    """The ``link_id`` and mirrored record ids the chosen records name: every
    record sharing one comes into the fixture beside them."""

    links: set[str] = field(default_factory=set)
    mirrors: set[str] = field(default_factory=set)

    def note(self, record: dict) -> None:
        if record.get("link_id"):
            self.links.add(record["link_id"])
        if record.get("mirror_of"):
            self.mirrors.add(record["mirror_of"])

    def joins(self, record: dict) -> bool:
        """Whether ``record`` is one end of a wanted link, a mirrored record or
        a game-end marker."""
        return (record.get("link_id") in self.links or record["record_id"] in self.mirrors
                or is_game_end(record))


def tag_line(record: dict, line: str, shard_no: int, number: int) -> None:
    """Keep the line as the collector wrote it, and where it sits (gen 3)."""
    record["_line"], record["_shard"], record["_number"] = line, shard_no, number


def interleaved(records_dir: Path) -> list[Path]:
    """Every seventh shard first, then the rest, so a category only another
    collection run reaches is found early."""
    shards = iter_shards(records_dir)
    return shards[::7] + [shard for i, shard in enumerate(shards) if i % 7]


def ended_games(shard: Path) -> set[str]:
    """The games of ``shard`` that wrote a game-end marker (gen 3): the rest
    were cut, and every reader drops them."""
    ended: set[str] = set()
    for line in iter_shard_lines(shard):
        if '"game_end"' not in line:
            continue
        record = orjson.loads(line)
        if is_game_end(record):
            ended.add(record["game_id"])
    return ended


def joined(shard: Path, shard_no: int, games: set[str], partners: Partners) -> dict[str, dict]:
    """The records of ``shard`` a chosen record joins with (gen 3): each one
    sharing a wanted ``link_id``, each mirrored record and each chosen game's
    marker, wherever it sits in its game, a cost half before its effect half
    included. A game never spans shards, so its partners are all here."""
    found: dict[str, dict] = {}
    needles = [f'"game_id":"{game}"' for game in games]
    for number, line in enumerate(iter_shard_lines(shard)):
        if not any(needle in line for needle in needles):
            continue
        record = orjson.loads(line)
        if partners.joins(record):
            tag_line(record, line, shard_no, number)
            found[record["record_id"]] = record
    return found


def read_fixture(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {r["record_id"]: r for r in (json.loads(line) for line in handle if line.strip())}


def write_fixture(output: Path, records: dict[str, dict]) -> None:
    """Each record's line as the collector wrote it, in shard order."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8", newline="\n") as out:
        for record in sorted(records.values(), key=lambda r: (r["_shard"], r["_number"])):
            out.write(record["_line"] + "\n")
