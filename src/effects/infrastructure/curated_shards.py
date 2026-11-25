"""The shards a curated corpus lists, against the shards on disk (finding 11.4).

The manifest names each output stratum's shards with the records the write
pass put in them. A reader that walked the stratum directory instead would
train on a stale shard of an earlier build, which may hold games now placed in
validation; this module is the one place that compares the two.
"""

from __future__ import annotations

from pathlib import Path

from effects.domain.corpus_manifest import CorpusManifest
from effects.domain.strata import OUTPUT_STRATA
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import iter_shard_lines, list_shards


class CorpusShardsError(ValueError):
    """The shards under a corpus are not the ones its manifest lists."""


def count_records(path: Path) -> int:
    """The records a shard holds: its complete, non-blank lines, counted as
    every reader counts them but never decoded."""
    return sum(1 for _ in iter_shard_lines(path))


def require_listed_shards(
    store: CorpusStore, manifest: CorpusManifest,
) -> dict[str, tuple[Path, ...]]:
    """Each output stratum's listed shards as paths, in the manifest's order,
    once the shards on disk are the listed ones: raises
    :class:`CorpusShardsError` naming every extra, missing and resized shard.

    Records are counted by reading every listed shard's lines, undecoded.
    """
    if not manifest.shards:
        raise CorpusShardsError(
            f"The manifest {store.manifest_path} lists no shards, so the shards "
            "under the corpus cannot be told from a stale build's. Rebuild it "
            "with `python -m effects build-corpus`."
        )
    problems: list[str] = []
    paths: dict[str, tuple[Path, ...]] = {}
    for stratum in OUTPUT_STRATA:
        directory = store.output_dir(stratum)
        listed = {shard.name: shard.records for shard in manifest.shards.get(str(stratum), ())}
        problems += _stratum_problems(str(stratum), directory, listed)
        paths[str(stratum)] = tuple(directory / name for name in listed)
    if problems:
        raise CorpusShardsError(
            f"The shards under {store.directory} are not the ones its manifest lists "
            "(a stale shard of an earlier build, or a shard changed since): "
            + "; ".join(problems)
        )
    return paths


def _stratum_problems(stratum: str, directory: Path, listed: dict[str, int]) -> list[str]:
    """What differs between ``directory``'s shards and ``listed`` (name to records)."""
    found = {shard.name for shard in list_shards(directory)}
    problems = [f"{stratum}: {name} is not listed" for name in sorted(found - listed.keys())]
    problems += [
        f"{stratum}: {name} is listed but absent" for name in sorted(listed.keys() - found)
    ]
    problems += [
        f"{stratum}: {name} holds {count} record(s), the manifest {records}"
        for name, records in listed.items()
        if name in found and (count := count_records(directory / name)) != records
    ]
    return problems
