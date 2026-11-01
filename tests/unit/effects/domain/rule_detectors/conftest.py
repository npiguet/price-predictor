"""Real gen-2 records for the detector families."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.infrastructure.shard_io import read_shard

GEN2_RECORDS = Path(__file__).parents[4] / "fixtures" / "effects" / "gen2-records.jsonl.gz"


@pytest.fixture(scope="session")
def gen2_records():
    return list(read_shard(GEN2_RECORDS))
