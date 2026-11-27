"""The cache layout and the card representation it feeds.

Row alignment is the whole interface: a consumer holds the cache file and the
sidecar and needs nothing else. A misaligned cache is the worst failure
available here — every vector loads, every shape checks out, and each ability
reads as its neighbour's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from effects.domain.ability_cache_layout import (
    ARRAY_KEY,
    cache_path_for,
    pooled_card_vector,
    validate_alignment,
)
from effects.domain.baselines import Baseline
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine

_SCRIPT = "cardsfolder/s/serra_angel.txt"
ROOT = Path("output/effects/gen3/cache/2026-11-20-e256-rule-mask")


def _sidecar(n_lines: int = 3) -> ProvenanceSidecar:
    return ProvenanceSidecar(
        card="Serra Angel",
        script_file=_SCRIPT,
        lines=tuple(
            SidecarLine(
                line_index=4 + i, line_kind="static",
                provenance=(ProvenanceKey(_SCRIPT, 0, "static", i),), prose_offset=0,
            )
            for i in range(n_lines)
        ),
    )


class TestCachePaths:
    def test_the_cache_mirrors_its_source_trees_layout(self):
        assert cache_path_for(_SCRIPT, root=ROOT) == (
            ROOT / "cardsfolder" / "s" / "serra_angel.npz"
        )

    def test_the_token_tree_gets_its_own_flat_subtree(self):
        assert cache_path_for("tokenscripts/soldier.txt", root=ROOT) == (
            ROOT / "tokenscripts" / "soldier.npz"
        )

    def test_a_variant_writes_beside_the_shipping_cache_not_over_it(self):
        shipping = cache_path_for(_SCRIPT, root=ROOT)
        variant = cache_path_for(_SCRIPT, root=ROOT, variant=Baseline.NO_STATE)
        assert variant != shipping
        assert variant.name == "serra_angel.no-state.npz"
        assert variant.parent == shipping.parent

    def test_the_root_has_no_default(self):
        """The cache run's directory is named by the command (artifact-layout rule 2)."""
        with pytest.raises(TypeError):
            cache_path_for(_SCRIPT)

    def test_the_root_is_configurable(self, tmp_path):
        assert cache_path_for(_SCRIPT, root=tmp_path).is_relative_to(tmp_path)


class TestRowAlignment:
    def test_one_row_per_sidecar_line_validates(self):
        validate_alignment(np.zeros((3, 8), dtype=np.float32), _sidecar(3))

    def test_too_few_rows_fails_loudly(self):
        with pytest.raises(ValueError, match="reads as its neighbour"):
            validate_alignment(np.zeros((2, 8), dtype=np.float32), _sidecar(3))

    def test_too_many_rows_fails_loudly(self):
        with pytest.raises(ValueError, match="4 rows"):
            validate_alignment(np.zeros((4, 8), dtype=np.float32), _sidecar(3))

    def test_a_one_dimensional_array_is_rejected(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            validate_alignment(np.zeros(8, dtype=np.float32), _sidecar(3))

    def test_a_card_with_no_ability_lines_aligns_at_zero_rows(self):
        validate_alignment(np.zeros((0, 8), dtype=np.float32), _sidecar(0))

    def test_the_array_key_is_stable(self):
        assert ARRAY_KEY == "e"


class TestVariantOfACacheFile:
    def test_each_variant_reads_back_from_the_path_it_was_written_to(self):
        from effects.domain.ability_cache_layout import variant_of_cache_file
        from effects.domain.baselines import BASELINES

        for variant in BASELINES:
            path = cache_path_for(_SCRIPT, root=ROOT, variant=variant)
            assert variant_of_cache_file(path) == variant

    def test_a_dot_in_a_sources_own_name_is_still_the_shipping_cache(self):
        """The suffix is read against the variant names, not counted in dots."""
        from effects.domain.ability_cache_layout import variant_of_cache_file

        path = cache_path_for("cardsfolder/a/a.d._decoy.txt", root=ROOT)
        assert variant_of_cache_file(path) == "full"
        assert variant_of_cache_file(path.with_suffix(".txt")) is None


class TestPooling:
    def test_the_pooled_vector_is_mean_then_max(self):
        matrix = np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        pooled = pooled_card_vector(matrix)
        np.testing.assert_allclose(pooled, [2.0, 2.0, 3.0, 4.0])

    def test_pooling_doubles_the_width(self):
        assert pooled_card_vector(np.zeros((5, 64), dtype=np.float32)).shape == (128,)

    def test_a_vanilla_creature_pools_to_zeros_of_the_right_width(self):
        """No ability lines is a card, not an error."""
        pooled = pooled_card_vector(np.zeros((0, 64), dtype=np.float32))
        assert pooled.shape == (128,)
        assert not pooled.any()

    def test_the_pooled_vector_is_float32(self):
        assert pooled_card_vector(np.zeros((2, 4))).dtype == np.float32


class TestTheStamp:
    """The stamp ties a cache to the checkpoint, surface, vocabulary and
    converted trees it was encoded from."""

    _STAMP = dict(checkpoint_hash="c" * 64, surface="prose", vocab_hash="v" * 64,
                  sidecar_digest="s" * 64)

    def test_each_variant_has_a_stamp_file_of_its_own(self):
        from effects.domain.ability_cache_layout import stamp_path
        from effects.domain.baselines import BASELINES

        paths = {stamp_path(ROOT, variant) for variant in BASELINES}
        assert len(paths) == len(BASELINES)
        assert stamp_path(ROOT, Baseline.FULL) == ROOT / "ability-cache.stamp.json"
        assert stamp_path(ROOT, Baseline.TAXONOMY).name == "ability-cache.taxonomy.stamp.json"

    def test_a_stamp_reads_back_from_what_it_writes(self):
        from effects.domain.ability_cache_layout import CacheStamp

        stamp = CacheStamp(**self._STAMP)
        assert CacheStamp.from_dict(stamp.as_dict()) == stamp

    def test_a_stamp_missing_a_member_is_refused_by_its_name(self):
        from effects.domain.ability_cache_layout import CacheStamp, CacheStampError

        recorded = CacheStamp(**self._STAMP).as_dict()
        del recorded["sidecar_digest"]
        with pytest.raises(CacheStampError, match="sidecar_digest"):
            CacheStamp.from_dict(recorded)

    def test_each_difference_names_both_values(self):
        from effects.domain.ability_cache_layout import CacheStamp

        cache = CacheStamp(**self._STAMP)
        wanted = CacheStamp(**{**self._STAMP, "checkpoint_hash": "d" * 64, "surface": "script"})
        assert cache.differences(wanted) == [
            f"checkpoint_hash: cache {'c' * 64}, wanted {'d' * 64}",
            "surface: cache prose, wanted script",
        ]

    def test_a_member_wanted_as_none_is_not_compared(self):
        """A taxonomy cache read without its checkpoint: no weights enter it."""
        from effects.domain.ability_cache_layout import CacheStamp

        cache = CacheStamp(**self._STAMP)
        assert cache.differences(cache.without_checkpoint()) == []
