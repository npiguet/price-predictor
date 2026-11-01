"""Real gen-2 records for the detector families."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.infrastructure.shard_io import read_shard

GEN2_RECORDS = Path(__file__).parents[4] / "fixtures" / "effects" / "gen2-records.jsonl.gz"


@pytest.fixture(scope="session")
def gen2_records():
    return list(read_shard(GEN2_RECORDS))


@pytest.fixture(scope="session")
def gen2_sidecars():
    from effects.infrastructure.converted_trees import SourceTrees

    root = GEN2_RECORDS.parent / "gen2-sidecars"
    return SourceTrees.of([root / "cardsfolder", root / "tokenscripts"]).sidecar_cache()


@pytest.fixture(scope="session")
def by_kind(gen2_records):
    """The fixture's records of one kind (and subkind), in fixture order."""
    def pick(kind: str, subkind: str | None = None):
        return [r for r in gen2_records if r.kind.value == kind
                and (subkind is None or (r.subkind and r.subkind.value == subkind))]
    return pick
