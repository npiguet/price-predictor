"""JSONL shard IO for the effect-record corpus.

The Java collector writes raw shards; Python reads them and writes the
curated shards ``build-corpus`` produces. A line's JSON form is
``record_codec``'s.
"""

from __future__ import annotations

import gzip
import logging
import math
import random
import re
import shutil
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, NamedTuple

from effects.domain.corpus_curation import record_hash
from effects.domain.corpus_manifest import SourceShard
from effects.domain.records import EffectRecord, RecordKind
from effects.domain.shard_generation import UnacceptedGenerationError, require_gen3
from effects.domain.wire_closed_lists import off_list_values
from effects.infrastructure.record_codec import (
    format_record_line,
    parse_record_line,
    record_from_dict,
)
from price_predictor.infrastructure.append_only import (
    count_complete_lines,
    iter_complete_lines,
)

logger = logging.getLogger(__name__)


def _write_line(handle: IO[str], record: EffectRecord) -> None:
    handle.write(format_record_line(record))
    handle.write("\n")


def write_shard(path: Path, records: Iterable[EffectRecord]) -> int:
    """Write a whole shard at once, gzipped; return the record count.

    One gzip member: per-block members guard a worker killed mid-run, and a
    curated shard is written by a process that finishes or leaves nothing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            _write_line(handle, record)
            written += 1
    return written


#: How many group files the first deal holds open: a large stratum has more
#: shards than one process may hold files open.
_GROUPS = 256


def repack_shards(
    parts: Sequence[Path], out_dir: Path, *, shard_records: int,
    seed: int = 0, records: int | None = None, prefix: str = "shard",
) -> list[Path]:
    """Deal ``parts``' games into shards of about ``shard_records`` records each.

    Whole games are dealt by a seeded hash of ``game_id``, so a game never
    spans two shards and each shard mixes games from across the stratum
    rather than one run's consecutive lifetimes (the trainer shuffles only
    inside the shard it holds). Each shard is then shuffled under ``seed``.

    Streamed through at most :data:`_GROUPS` compressed group files beside
    ``out_dir``, since a stratum does not fit in memory. ``records`` is the
    parts' record count when the caller knows it. ``shard_records <= 0``
    writes one shard per part, unshuffled. Parts are not deleted.
    """
    parts = [Path(p) for p in parts]
    if not parts:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if shard_records <= 0:
        return _copy_parts_unshuffled(parts, out_dir, prefix)

    if records is None:
        records = sum(1 for part in parts for _ in iter_shard_lines(part))
    count = max(1, math.ceil(records / shard_records))
    scratch = out_dir.with_name(f".{out_dir.name}.buckets")
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    try:
        filled = _deal_into_buckets(parts, scratch, seed=seed, count=count)
        # Named before writing: groups write shards out of index order.
        names = {
            index: out_dir / f"{prefix}-{number:05d}.jsonl.gz"
            for number, index in enumerate(sorted(filled), start=1)
        }
        _write_shuffled_buckets(scratch, names, seed=seed, count=count)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return list(names.values())


def _copy_parts_unshuffled(parts: list[Path], out_dir: Path, prefix: str) -> list[Path]:
    written = []
    for number, part in enumerate(parts, start=1):
        path = out_dir / f"{prefix}-{number:05d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for line in iter_shard_lines(part):
                handle.write(line)
                handle.write("\n")
        written.append(path)
    return written


def _bucket_of(line: str, *, seed: int, count: int) -> int:
    """The index of the output shard a line's game is dealt to."""
    return record_hash(_game_of_line(line), seed=seed) % count


#: Between a group line's shard index and its record: no digit, so the first
#: one on the line is always this.
_INDEX_SEPARATOR = ":"


def _group_path(scratch: Path, group: int) -> Path:
    return scratch / f"{group:05d}.gz"


def _deal_into_buckets(
    parts: list[Path], scratch: Path, *, seed: int, count: int,
) -> set[int]:
    """Deal every line into the group of its shard's index; return the indices.

    A group keeps the parts' order, so each shuffle sees a fixed input order.
    Each group line carries its shard's index ahead of the record (``index:``),
    so the split that follows does not hash the line a second time.
    """
    groups = min(count, _GROUPS)
    filled: set[int] = set()
    handles = [
        gzip.open(_group_path(scratch, group), "wt", encoding="utf-8", compresslevel=1)
        for group in range(groups)
    ]
    try:
        for part in parts:
            for line in iter_shard_lines(part):
                index = _bucket_of(line, seed=seed, count=count)
                filled.add(index)
                group = handles[index % groups]
                group.write(f"{index}{_INDEX_SEPARATOR}{line}\n")
    finally:
        for handle in handles:
            handle.close()
    return filled


def _write_shuffled_buckets(
    scratch: Path, names: dict[int, Path], *, seed: int, count: int,
) -> None:
    """Split each group into its shards by the index each line carries,
    shuffle each under ``seed``, write it.

    A group is held whole, which is about ``count / _GROUPS`` shards' lines.
    """
    for group in range(min(count, _GROUPS)):
        buckets: dict[int, list[str]] = defaultdict(list)
        for tagged in iter_shard_lines(_group_path(scratch, group)):
            index, _, line = tagged.partition(_INDEX_SEPARATOR)
            buckets[int(index)].append(line)
        for index, lines in buckets.items():
            random.Random(f"{seed}:{index}").shuffle(lines)
            with gzip.open(names[index], "wt", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(line)
                    handle.write("\n")


def _game_of_line(line: str) -> str:
    """A shard line's ``game_id``, read off the envelope head without a parse."""
    head = peek_envelope(line)
    if head is not None:
        return head.game_id
    return parse_record_line(line)["game_id"]


#: Both shard spellings: plain `.jsonl` predates compression.
SHARD_GLOBS: tuple[str, ...] = ("*.jsonl.gz", "*.jsonl")


class CorruptShardError(ValueError):
    """A gzip member that does not decompress, with a member after it.

    A killed worker can only cut the last member short; damage before the
    last would silently drop every member after it, so the read stops loudly.
    """


class RecordDecodeError(ValueError):
    """A shard line that does not parse into a record, located by shard and line.

    Pickles with its location, so a pool or prefetch worker hands it over whole.
    """

    def __init__(self, message: str, path: Path | None = None, line: int | None = None) -> None:
        super().__init__(message)
        self.path = None if path is None else Path(path)
        self.line = line

    @classmethod
    def at(cls, path: Path, line: int, cause: Exception) -> RecordDecodeError:
        return cls(
            f"{path}, line {line}: the record does not decode "
            f"({type(cause).__name__}: {cause})", path, line,
        )

    def __reduce__(self):
        return type(self), (self.args[0], self.path, self.line)


#: What ends one shard's read early while leaving the records before it
#: usable: a reader that must not stop on one damaged shard catches these.
SHARD_DAMAGE: tuple[type[ValueError], ...] = (CorruptShardError, RecordDecodeError)


@dataclass
class DecodeTally:
    """What reading shards met without stopping: each value outside its
    member's closed list (``(member, repr of the value) -> times read``) and
    each shard whose last gzip member was damaged, which ends its read.

    One per process (:func:`take_decode_tally`); a caller whose workers read
    shards sums what each hands back (:meth:`merge`).
    """

    off_list: Counter = field(default_factory=Counter)
    damaged_tails: list[str] = field(default_factory=list)

    def merge(self, other: DecodeTally) -> None:
        self.off_list.update(other.off_list)
        self.damaged_tails.extend(other.damaged_tails)

    def __bool__(self) -> bool:
        return bool(self.off_list or self.damaged_tails)

    def report_lines(self) -> list[str]:
        """The tally as printed lines; none when it is empty."""
        if not self:
            return []
        lines = [f"values outside the wire's closed lists: {sum(self.off_list.values())}"]
        lines += [f"  {member} {value}: {count}"
                  for (member, value), count in sorted(self.off_list.items())]
        if self.damaged_tails:
            lines.append("shards whose damaged last gzip member ended the read: "
                         f"{len(self.damaged_tails)}")
            lines += [f"  {where}" for where in self.damaged_tails]
        return lines


#: This process's tally since it was last taken.
_TALLY = DecodeTally()
#: ``(member, repr of the value)`` this process has warned of, so each is
#: logged once whether or not its tally has been taken since.
_WARNED: set[tuple[str, str]] = set()


def decode_tally() -> DecodeTally:
    """A copy of what this process's reads have tallied since the last take."""
    return DecodeTally(Counter(_TALLY.off_list), list(_TALLY.damaged_tails))


def take_decode_tally() -> DecodeTally:
    """This process's tally, emptied: a pool worker returns it with each
    shard's result, and the caller sums them (:meth:`DecodeTally.merge`)."""
    taken = decode_tally()
    reset_decode_tally()
    return taken


def absorb_decode_tally(tally: DecodeTally) -> None:
    """Add a tally a worker handed back to this process's."""
    _TALLY.merge(tally)


def reset_decode_tally() -> None:
    _TALLY.off_list.clear()
    _TALLY.damaged_tails.clear()


#: Compressed bytes read at a time.
_GZIP_CHUNK = 1 << 20
#: A gzip member's first bytes: its magic and the deflate method.
_GZIP_MAGIC = b"\x1f\x8b\x08"
#: Compressed bytes tried at each place a member may start.
_MEMBER_PROBE = 1 << 16


def _member_follows(path: Path, after: int) -> bool:
    """Whether a gzip member that decompresses starts past byte ``after``:
    every place its magic recurs is tried, since deflate data can hold the
    three bytes by chance."""
    with Path(path).open("rb") as handle:
        handle.seek(after + 1)
        rest = handle.read()
    at = rest.find(_GZIP_MAGIC)
    while at >= 0:
        probe = zlib.decompressobj(31)
        try:
            out = probe.decompress(rest[at:at + _MEMBER_PROBE])
        except zlib.error:
            out = None
        if out is not None and (out or probe.eof):
            return True
        at = rest.find(_GZIP_MAGIC, at + 1)
    return False


def _gzip_numbered_lines(path: Path, *, tally: bool = True) -> Iterator[tuple[int, str]]:
    """Each newline-terminated line of a gzip file of concatenated members,
    with its 1-based line number, terminator stripped.

    A last member cut short (a killed worker) ends the stream quietly with
    the lines it completed. A member that does not decompress, or whose
    check sum fails, ends it too, with a warning and a tally entry, when no
    member follows it (a damaged tail: the lines it completed are kept);
    with one after it, it raises :class:`CorruptShardError` naming the byte
    it starts at, since every later member would be lost unsaid. A line that
    is no UTF-8 raises :class:`RecordDecodeError`. ``tally`` false notes a
    damaged tail neither in the tally nor in the log: a read beside the one
    that reports the shard (:func:`ended_games`).
    """
    number = 0
    pending = b""
    member_start = 0
    consumed = 0
    decoder = zlib.decompressobj(31)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_GZIP_CHUNK):
            data = chunk
            while data:
                if decoder is None:
                    # Between members: the gzip module also skips zero padding.
                    stripped = data.lstrip(b"\x00")
                    consumed += len(data) - len(stripped)
                    data = stripped
                    if not data:
                        break
                    decoder = zlib.decompressobj(31)
                    member_start = consumed
                try:
                    out = decoder.decompress(data)
                except zlib.error as exc:
                    where = (f"{path}: the gzip member starting at byte {member_start} "
                             f"is corrupt ({exc})")
                    if _member_follows(path, member_start):
                        raise CorruptShardError(
                            f"{where}; a member follows it, so the shard is read no "
                            f"further, past line {number}") from exc
                    if tally:
                        logger.warning("%s; it is the shard's last, so its read ends "
                                       "there, past line %d", where, number)
                        _TALLY.damaged_tails.append(f"{path}, byte {member_start}")
                    return
                rest = decoder.unused_data if decoder.eof else b""
                consumed += len(data) - len(rest)
                data = rest
                if decoder.eof:
                    decoder = None
                pending += out
                *lines, pending = pending.split(b"\n")
                for line in lines:
                    number += 1
                    try:
                        text = line.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise RecordDecodeError.at(path, number, exc) from exc
                    yield number, text.removesuffix("\r")


def _plain_numbered_lines(path: Path) -> Iterator[tuple[int, str]]:
    number = 0
    try:
        for number, line in enumerate(iter_complete_lines(path), start=1):  # noqa: B007
            yield number, line
    except UnicodeDecodeError as exc:
        # Read in blocks: the bad byte is on the line after the last read, or later.
        raise RecordDecodeError.at(path, number + 1, exc) from exc


def _numbered_lines(path: Path, *, tally: bool = True) -> Iterator[tuple[int, str]]:
    """Every complete, non-blank line of a shard with its 1-based number in
    the file, blank lines counted; ``tally`` as :func:`_gzip_numbered_lines`."""
    path = Path(path)
    lines = (_gzip_numbered_lines(path, tally=tally) if path.suffix == ".gz"
             else _plain_numbered_lines(path))
    for number, line in lines:
        if line.strip():
            yield number, line


def iter_shard_lines(path: Path) -> Iterator[str]:
    """Every complete, non-blank line of a shard, its terminator stripped.

    One shape for both spellings, so a caller that writes lines back adds the
    newline itself whichever spelling it read. Blank lines are skipped: every
    reader ignores them, and a copy that kept them would be the one place
    they survived. A truncated final gzip member, a damaged one with no
    member after it, or a trailing partial line, ends the stream; a damaged
    member before the last raises :class:`CorruptShardError`.
    """
    for _number, line in _numbered_lines(path):
        yield line


def _decode[T, R](path: Path, number: int, value: T, parse: Callable[[T], R]) -> R:
    """``parse(value)``, any failure located at ``path`` and line ``number``.

    A refused generation keeps its type, which each command maps to its exit
    status, and gains the location in its message.
    """
    try:
        return parse(value)
    except UnacceptedGenerationError as exc:
        raise type(exc)(f"{path}, line {number}: {exc}") from exc
    except Exception as exc:
        raise RecordDecodeError.at(path, number, exc) from exc


def _parse_object(line: str) -> dict:
    """One shard line as the JSON object it must be."""
    data = parse_record_line(line)
    if not isinstance(data, dict):
        raise TypeError(f"a record is a JSON object, not {type(data).__name__}")
    return data


def _check_closed_lists(path: Path, number: int, data: dict) -> None:
    """Count each value of one raw record outside its closed list, warning
    the first time this process reads it; the record still decodes."""
    for member, value in off_list_values(data):
        seen = (member, repr(value))
        if seen not in _WARNED:
            _WARNED.add(seen)
            logger.warning(
                "%s, line %d: %s %r is outside the wire's closed list; read as "
                "written, its later sightings only counted", path, number, member, value,
            )
        _TALLY.off_list[seen] += 1


def _checked(path: Path, number: int, line: str) -> dict:
    """One line parsed and checked against the closed lists, any failure
    (no JSON object, a malformed member) located."""
    data = _decode(path, number, line, _parse_object)
    _decode(path, number, data, lambda raw: _check_closed_lists(path, number, raw))
    return data


def _record_at(path: Path, number: int, line: str) -> EffectRecord:
    return _decode(path, number, _checked(path, number, line), record_from_dict)


def iter_record_dicts(path: Path) -> Iterator[dict]:
    """Every complete record of one shard, as the dict its line parses to.

    For a caller that needs the raw JSON; a record of any generation reads.
    """
    for number, line in _numbered_lines(path):
        yield _decode(path, number, line, _parse_object)



def read_window(
    directory: Path, limit: int = 0,
    observe: Callable[[str, dict], None] | None = None,
    *, shards: Sequence[SourceShard] | None = None,
) -> Iterator[EffectRecord]:
    """The first ``limit`` records under ``directory``; 0 reads all of them.

    ``observe``, when given, sees each record's raw dict, with its shard's
    path relative to ``directory``, before it is parsed: a parsed record
    cannot say whether a field was on the wire. ``shards`` is the directory's
    :func:`list_shards` when the caller already holds it.
    """
    root = Path(directory)
    seen = 0
    for shard in list_shards(root) if shards is None else shards:
        path = shard.path(root)
        for number, line in _numbered_lines(path):
            data = _checked(path, number, line)
            if observe is not None:
                _decode(path, number, data, lambda raw: observe(shard.name, raw))
            yield _decode(path, number, data, record_from_dict)
            seen += 1
            if limit and seen >= limit:
                return


def read_shard(path: Path) -> Iterator[EffectRecord]:
    """Yield every complete record in one shard; skip a trailing partial line.

    A line that does not decode raises :class:`RecordDecodeError` naming the
    shard and the line.
    """
    for number, line in _numbered_lines(path):
        yield _record_at(path, number, line)


def load_shard(path: Path) -> list[EffectRecord]:
    """One shard's records, the largest unit the trainer ever holds."""
    return list(read_shard(Path(path)))


def iter_shards(directory: Path) -> list[Path]:
    """Every shard at or under ``directory``, in path order, either spelling.

    Recursive, so collection runs kept in subdirectories read as one corpus.
    Path order rather than name order, so every reader walks a corpus in the
    same order on every run.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        path for pattern in SHARD_GLOBS for path in directory.rglob(pattern)
    )


def list_shards(directory: Path) -> tuple[SourceShard, ...]:
    """Every shard :func:`iter_shards` finds under ``directory``, in its order,
    by posix path relative to ``directory`` and size."""
    root = Path(directory)
    return tuple(
        SourceShard(name=shard.relative_to(root).as_posix(), size=shard.stat().st_size)
        for shard in iter_shards(root)
    )


class NoShardsError(ValueError):
    """A reader was pointed at a directory holding no shard."""


def require_shards(directory: Path) -> tuple[SourceShard, ...]:
    """:func:`list_shards`, refusing an empty listing: for a command whose
    whole job is reading the directory, where nothing to read is a mistake."""
    shards = list_shards(directory)
    if not shards:
        raise NoShardsError(f"no shards under {directory}")
    return shards


def read_records(directory: Path) -> Iterator[EffectRecord]:
    """Stream every record under ``directory``, shard by shard in path order.

    A missing directory yields nothing: an empty corpus, not an error.
    """
    for shard in iter_shards(directory):
        yield from read_shard(shard)


#: The envelope's head as both writers lay it out: the Java
#: ``EffectRecord.toJson`` and ``record_codec.record_to_dict`` emit ``record_id``,
#: ``run_id``, ``timestamp``, ``game_id`` and ``kind`` first, in that order.
_ENVELOPE_HEAD = re.compile(
    r'"record_id":"([^"]*)",.*?"game_id":"([^"]*)","kind":"([a-z_]+)"'
)

#: How far into a line the head is looked for. The five leading fields are a
#: few hundred characters; the state snapshot that follows is the bulk.
_ENVELOPE_HEAD_SPAN = 512


class EnvelopeHead(NamedTuple):
    """A shard line's leading envelope fields, read without parsing it."""

    record_id: str
    game_id: str
    kind: str


def peek_envelope(line: str) -> EnvelopeHead | None:
    """The head of one shard line, or None where it has another shape, which
    the caller then parses: a reordered envelope costs speed, never records."""
    head = _ENVELOPE_HEAD.search(line, 0, _ENVELOPE_HEAD_SPAN)
    return None if head is None else EnvelopeHead(*head.groups())


#: The game-end marker's subkind as a shard line spells it, near its head.
_GAME_END = re.compile(r'"subkind"\s*:\s*"game_end"')


def ended_games(path: Path) -> frozenset[str]:
    """The games of one shard that wrote their game-end marker
    (``turn_action`` / ``game_end``, WP7): a game of the shard missing from
    them was cut. Reads each line's head, parsing only a turn action's
    or a line whose head is in another shape. A damaged last member ends the
    read as it ends every read, but goes unreported here: the read of the
    shard's records reports it, once."""
    ended: set[str] = set()
    for number, line in _numbered_lines(path, tally=False):
        head = peek_envelope(line)
        if head is None:
            data = _decode(path, number, line, _parse_object)
            if data.get("kind") == "turn_action" and data.get("subkind") == "game_end":
                ended.add(data["game_id"])
        elif head.kind == "turn_action" and _GAME_END.search(line, 0, 2 * _ENVELOPE_HEAD_SPAN):
            ended.add(head.game_id)
    return frozenset(ended)


def read_records_matching(
    directory: Path,
    *,
    game_ids: Collection[str] | None = None,
    kinds: Collection[RecordKind] | None = None,
) -> Iterator[EffectRecord]:
    """The records under ``directory`` of the named games and kinds.

    The same records a filter over :func:`read_records` keeps, without
    parsing the rest; a line :func:`peek_envelope` cannot read is parsed and
    filtered the slow way.
    """
    shards = iter_shards(directory)
    for done, shard in enumerate(shards):
        if done and done % 250 == 0:
            logger.info("scanned %d of %d shards under %s", done, len(shards), directory)
        yield from iter_shard_matching(shard, game_ids=game_ids, kinds=kinds)


def iter_shard_matching(
    shard: Path,
    *,
    game_ids: Collection[str] | None = None,
    kinds: Collection[RecordKind] | None = None,
    where: Callable[[EnvelopeHead], bool] | None = None,
) -> Iterator[EffectRecord]:
    """The records of one shard of the named games and kinds that ``where``
    keeps, the rest skipped unparsed; :func:`read_records_matching` for one
    shard.

    ``where`` sees each line's head once, and only once the two filters pass,
    so it may keep state, such as the record ids it has already let through.
    """
    for number, line in _numbered_lines(shard):
        head = peek_envelope(line)
        if head is not None and not _wanted(head, game_ids, kinds, where):
            continue
        record = _record_at(shard, number, line)
        if head is None and not _wanted(
            EnvelopeHead(record.record_id, record.game_id, record.kind.value),
            game_ids, kinds, where,
        ):
            continue
        yield record


def _wanted(
    head: EnvelopeHead, game_ids: Collection[str] | None,
    kinds: Collection[RecordKind] | None, where: Callable[[EnvelopeHead], bool] | None,
) -> bool:
    """Whether a record with this head passes every filter: a
    :class:`RecordKind` hashes as its value, so the envelope's string finds it."""
    return (
        (game_ids is None or head.game_id in game_ids)
        and (kinds is None or head.kind in kinds)
        and (where is None or where(head))
    )


def count_records(directory: Path, *, ceiling: int = 0) -> int:
    """Complete records under ``directory``, for progress and resume.

    Counts complete lines without parsing them. A non-zero ``ceiling`` stops
    the count there and returns it, since decompressing a full corpus takes
    minutes.
    """
    total = 0
    for shard in iter_shards(directory):
        total += count_shard_records(shard)
        if ceiling and total >= ceiling:
            return ceiling
    return total


def count_shard_records(shard: Path) -> int:
    """Complete records in one shard, counted as :func:`count_records` does:
    blank lines are not records, in either spelling."""
    if shard.suffix == ".gz":
        return sum(1 for _ in iter_shard_lines(shard))
    return count_complete_lines(shard, skip_blank=True)


# ── shard generation (FR-033) ───────────────────────────────────────────

def require_generation(paths: Iterable[Path]) -> None:
    """Refuse a records set before reading it whole if a shard's first
    record is not gen-3, naming the shard, the line and the generation.

    Every reader of a records set calls this first (FR-020, FR-033): one
    record speaks for its shard, and an older record later in a shard is
    refused where it is decoded (``record_codec.record_from_dict``).
    """
    for path in paths:
        for number, line in _numbered_lines(path):
            data = _decode(path, number, line, _parse_object)
            _decode(path, number, data, require_gen3)
            break
