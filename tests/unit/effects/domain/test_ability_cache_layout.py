"""The cache layout and the card representation it feeds (T087).

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
    DEFAULT_CACHE_ROOT,
    cache_path_for,
    pooled_card_vector,
    validate_alignment,
)
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine

_SCRIPT = "cardsfolder/s/serra_angel.txt"


def _sidecar(n_lines: int = 3) -> ProvenanceSidecar:
    return ProvenanceSidecar(
        card="Serra Angel",
        script_file=_SCRIPT,
        lines=tuple(
            SidecarLine(
                line_index=4 + i, line_kind="static",
                provenance=(ProvenanceKey(_SCRIPT, 0, "static", i),),
            )
            for i in range(n_lines)
        ),
    )


class TestCachePaths:
    def test_the_cache_mirrors_its_source_trees_layout(self):
        assert cache_path_for(_SCRIPT) == (
            DEFAULT_CACHE_ROOT / "cardsfolder" / "s" / "serra_angel.npz"
        )

    def test_the_token_tree_gets_its_own_flat_subtree(self):
        assert cache_path_for("tokenscripts/soldier.txt") == (
            DEFAULT_CACHE_ROOT / "tokenscripts" / "soldier.npz"
        )

    def test_a_variant_writes_beside_the_shipping_cache_not_over_it(self):
        shipping = cache_path_for(_SCRIPT)
        variant = cache_path_for(_SCRIPT, variant="no-state")
        assert variant != shipping
        assert variant.name == "serra_angel.no-state.npz"
        assert variant.parent == shipping.parent

    def test_the_cache_lives_outside_the_converted_tree(self):
        """encode-cards --clean deletes every .npz under output/cardsfolder/."""
        assert "cardsfolder" not in DEFAULT_CACHE_ROOT.parts[:1]
        assert DEFAULT_CACHE_ROOT == Path("output/effects/abilities")

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
        from effects.domain.variants import VARIANTS

        for variant in VARIANTS:
            assert variant_of_cache_file(cache_path_for(_SCRIPT, variant=variant)) == variant

    def test_a_dot_in_a_sources_own_name_is_still_the_shipping_cache(self):
        """The suffix is read against the variant names, not counted in dots."""
        from effects.domain.ability_cache_layout import variant_of_cache_file

        path = cache_path_for("cardsfolder/a/a.d._decoy.txt")
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
