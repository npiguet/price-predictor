"""What each evaluation baseline masks (FR-094)."""

from __future__ import annotations

import pytest

from effects.domain.baselines import (
    BASELINES,
    baseline_masks,
)


class TestBaselineMasks:
    def test_an_unknown_variant_is_rejected(self):
        with pytest.raises(ValueError, match="unknown --variant"):
            baseline_masks("oracle-text")


class TestEveryBaselineStagesEveryText:
    """Staging reads each text through its sidecar line; a keyword line has
    none, and no variant may crash on one (finding 13.8)."""

    @staticmethod
    def _batcher(masks):
        from pathlib import Path

        from effects.domain.ability_tokenizer import AbilityTokenizer
        from effects.infrastructure.sidecar_io import SidecarCache
        from effects.infrastructure.tokenizer_loading import load_keyword_definitions
        from tests.unit.effects.builders import make_batcher

        fixtures = Path(__file__).parents[3] / "fixtures"
        words = ["[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]"]
        tokenizer = AbilityTokenizer(
            {word: index for index, word in enumerate(words)},
            load_keyword_definitions(fixtures / "effects" / "keyword-definitions.json"),
        )
        sidecars = SidecarCache({
            "cardsfolder": fixtures / "effects" / "gen3-sidecars" / "cardsfolder",
            "tokenscripts": fixtures / "effects" / "gen3-sidecars" / "tokenscripts",
        })
        return make_batcher(tokenizer=tokenizer, sidecars=sidecars, masks=masks)

    @pytest.mark.parametrize("name", list(BASELINES))
    def test_staging_the_whole_fixture_raises_nothing(self, name):
        from pathlib import Path

        from effects.infrastructure.shard_io import read_shard

        batcher = self._batcher(baseline_masks(name))
        shard = Path(__file__).parents[3] / "fixtures" / "effects" / "gen3-records.jsonl.gz"
        texts = batcher.batch_texts(list(read_shard(shard)))
        assert any(line is None for line in texts.values()), "no keyword line to stage"
        assert batcher.stage(texts).rows.keys() == texts.keys()
