"""Files the effects tests write as a run would: raw shards, validation
samples, a vocabulary, a sidecar; and the environment a child interpreter
needs to import the tree under test."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path

from effects.domain.records import EffectRecord
from tests.unit.effects.record_builders import with_game_end_markers


def write_ended_shard(path: Path, records) -> int:
    """``write_shard`` of ``records`` with every game ended by its marker."""
    from effects.infrastructure.shard_io import write_shard

    return write_shard(path, with_game_end_markers(records))


#: Records per gzip member, as the Java collector blocks them.
RAW_BLOCK_RECORDS = 256


@dataclass(eq=False)
class RawShardWriter:
    """One worker lifetime's raw shard, laid out as the Java collector writes it.

    ``{run_id}.{worker}-{lifetime}.jsonl.gz``, a concatenation of gzip members
    of :data:`RAW_BLOCK_RECORDS` records each, appended a block at a time.
    ``compressed=False`` writes the plain ``.jsonl`` shards collected before
    compression, which every reader still loads. Python never writes a raw
    shard outside the tests, so the writer lives here rather than beside the
    readers.
    """

    directory: str | Path
    run_id: str
    worker: int
    lifetime: str
    _: KW_ONLY
    compressed: bool = True
    path: Path = field(init=False)
    _block: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        suffix = ".jsonl.gz" if self.compressed else ".jsonl"
        self.path = Path(self.directory) / f"{self.run_id}.{self.worker}-{self.lifetime}{suffix}"

    def __enter__(self) -> RawShardWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def write(self, record: EffectRecord) -> None:
        from effects.infrastructure.record_codec import format_record_line

        self._block.append(format_record_line(record) + "\n")
        if len(self._block) >= RAW_BLOCK_RECORDS:
            self._flush()

    def _flush(self) -> None:
        import gzip

        if not self._block:
            return
        data = "".join(self._block).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(gzip.compress(data) if self.compressed else data)
        self._block = []

    def close(self) -> None:
        self._flush()


@dataclass(frozen=True)
class PlainShard:
    """Which worker's shard :func:`write_plain_shard` writes, and whether its
    games ended."""

    run_id: str = "run"
    worker: int = 0
    #: False for a worker killed mid-game: no game-end markers.
    ended: bool = True


def write_plain_shard(directory, lifetime: str, records, shard: PlainShard = PlainShard()) -> Path:
    """A plain ``{run_id}.{worker}-{lifetime}.jsonl`` raw shard holding
    ``records``, written through :class:`RawShardWriter`: each gen-3 game
    ended by its game-end marker, as a worker that finished it writes it,
    unless ``shard.ended`` is False (a worker killed mid-game)."""
    with RawShardWriter(
            directory, shard.run_id, shard.worker, lifetime, compressed=False) as writer:
        for record in with_game_end_markers(records) if shard.ended else records:
            writer.write(record)
    return writer.path


def write_validation_samples(directory, records) -> dict:
    """Both validation strata's samples under ``directory``, each ``records``."""
    from effects.infrastructure.shard_io import write_shard

    samples = {}
    for stratum in ("card-disjoint", "game-disjoint"):
        samples[stratum] = directory / f"{stratum}.jsonl.gz"
        write_shard(samples[stratum], records)
    return samples


def write_vocabulary(path: Path, surface: str = "prose", extra: tuple[str, ...] = ()) -> Path:
    """An effects vocabulary file of the specials and ``extra``, its header
    recording ``surface``: what a command reading only the surface needs."""
    from effects.domain.ability_tokenizer import SPECIAL_TOKENS
    from effects.infrastructure.vocabulary_file import save_vocabulary

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = dict.fromkeys([*SPECIAL_TOKENS, *extra])
    save_vocabulary({token: index for index, token in enumerate(tokens)}, surface, path)
    return path


def write_sidecar(sidecar, path: Path) -> None:
    """Write one sidecar where ``convert`` would, so a test can produce one
    without Forge; the inverse of ``sidecar_io.read_sidecar``."""
    import json

    from effects.infrastructure.sidecar_io import sidecar_to_dict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sidecar_to_dict(sidecar), separators=(",", ":")), encoding="utf-8",
    )


def fresh_interpreter_env(**overrides: str) -> dict[str, str]:
    """The environment for a ``sys.executable`` child that imports the tree
    under test, which an installed copy of the package would otherwise shadow."""
    import os
    from pathlib import Path

    import effects

    source = Path(effects.__file__).parents[1]
    return {**os.environ, "PYTHONPATH": str(source), **overrides}
