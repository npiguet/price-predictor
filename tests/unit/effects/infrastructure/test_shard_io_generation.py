"""Shard IO: the generation check (FR-033).
"""

from __future__ import annotations

import pytest

from tests.unit.effects.builders import GEN3_RECORDS
from tests.unit.effects.infrastructure.shard_io_support import (
    _EARLIER,
    _older_shard,
)


class TestTheGenerationCheck:
    """FR-033: a records set of gen-3 shards passes; a shard whose first record
    is older is refused naming it, its shard and its line."""

    def test_an_empty_shard_passes(self, tmp_path):
        from effects.infrastructure.shard_io import require_generation

        empty = tmp_path / "e.jsonl"
        empty.write_text("", encoding="utf-8")
        require_generation([empty])
        require_generation([])

    def test_a_gen3_shard_passes(self):
        from effects.infrastructure.shard_io import require_generation

        require_generation([GEN3_RECORDS, GEN3_RECORDS])

    @pytest.mark.parametrize("generation", sorted(_EARLIER))
    def test_an_older_shard_beside_a_gen3_one_is_refused_naming_it(self, tmp_path, generation):
        from effects.domain.shard_generation import OlderGenerationError
        from effects.infrastructure.shard_io import require_generation

        older = _older_shard(tmp_path / "old.jsonl.gz", generation)
        with pytest.raises(OlderGenerationError,
                           match=rf"old\.jsonl\.gz, line 1: the record is {generation} "):
            require_generation([GEN3_RECORDS, older])
