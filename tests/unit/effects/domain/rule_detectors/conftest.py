"""Real gen-3 records for the detector families."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.infrastructure.shard_io import read_shard

GEN3_RECORDS = Path(__file__).parents[4] / "fixtures" / "effects" / "gen3-records.jsonl.gz"


@pytest.fixture(scope="session")
def gen3_records():
    return list(read_shard(GEN3_RECORDS))


@pytest.fixture(scope="session")
def gen3_sidecars():
    from effects.infrastructure.converted_trees import SourceTrees

    root = GEN3_RECORDS.parent / "gen3-sidecars"
    return SourceTrees.of([root / "cardsfolder", root / "tokenscripts"]).sidecar_cache()


@pytest.fixture(scope="session")
def by_kind(gen3_records):
    """The fixture's records of one kind (and subkind), in fixture order."""
    def pick(kind: str, subkind: str | None = None):
        return [r for r in gen3_records if r.kind.value == kind
                and (subkind is None or (r.subkind and r.subkind.value == subkind))]
    return pick
