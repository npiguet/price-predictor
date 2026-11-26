"""What the validator needs from the infrastructure layer, as ports the CLI fills.

``application`` imports no ``infrastructure``: the shard reader and the record
renderer the checks use arrive as these values, built by the command that runs
the validator (``cli.run_validate_corpus``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from effects.domain.corpus_manifest import SourceShard
from effects.domain.records import EffectRecord

#: A record's JSON-serializable dict form, as the collector's writer renders it.
RecordRenderer = Callable[[EffectRecord], dict]


class DecodeTallyReading(Protocol):
    """What a shard read met without stopping (``shard_io.DecodeTally``)."""

    @property
    def off_list(self) -> Mapping[tuple[str, str], int]: ...

    @property
    def damaged_tails(self) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class ShardSource:
    """The shard directory's reader: every call ``validate_directory`` makes
    into ``shard_io``."""

    require_shards: Callable[[Path], tuple[SourceShard, ...]]
    require_generation: Callable[[Iterable[Path]], None]
    #: What ``read_window`` meets without stopping: the tally it fills, one
    #: per source, so one source serves one validation.
    decode_tally: DecodeTallyReading
    #: ``read_window(root, limit, observe, shards=...)``, filling :attr:`decode_tally`.
    read_window: Callable[..., Iterator[EffectRecord]]
    #: The errors that end one shard's read.
    damage: tuple[type[ValueError], ...]
