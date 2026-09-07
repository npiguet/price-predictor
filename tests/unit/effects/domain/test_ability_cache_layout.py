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
    CardSlotKind,
    cache_path_for,
    card_slots,
    pooled_card_vector,
    read_layout_line,
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


class TestCardRepresentation:
    def test_a_single_face_card_is_a_card_token_then_its_rows(self):
        slots = card_slots([2])
        assert [s.kind for s in slots] == [
            CardSlotKind.CARD, CardSlotKind.ABILITY, CardSlotKind.ABILITY,
        ]

    def test_positions_reset_at_the_card_token(self):
        slots = card_slots([2])
        assert [s.position for s in slots] == [0, 1, 2]

    def test_faces_are_separated_by_an_alternate_token(self):
        slots = card_slots([1, 1], layout="transform")
        assert [s.kind for s in slots] == [
            CardSlotKind.CARD, CardSlotKind.ABILITY,
            CardSlotKind.ALTERNATE,
            CardSlotKind.CARD, CardSlotKind.ABILITY,
        ]

    def test_positions_continue_across_the_alternate_token(self):
        """Only [CARD] resets them; a face boundary is a tagged separator."""
        slots = card_slots([2, 1], layout="transform")
        alternate = next(s for s in slots if s.kind is CardSlotKind.ALTERNATE)
        assert alternate.position == 3  # continues from the first face

    def test_the_second_face_restarts_at_zero(self):
        slots = card_slots([2, 1], layout="transform")
        cards = [s for s in slots if s.kind is CardSlotKind.CARD]
        assert [s.position for s in cards] == [0, 0]

    def test_the_alternate_token_carries_the_layout_tag(self):
        slots = card_slots([1, 1], layout="modal_dfc")
        alternate = next(s for s in slots if s.kind is CardSlotKind.ALTERNATE)
        assert alternate.layout == "modal_dfc"

    def test_rows_number_continuously_across_faces(self):
        """The cache is one matrix per file, faces included."""
        slots = card_slots([2, 2], layout="split")
        rows = [s.row for s in slots if s.kind is CardSlotKind.ABILITY]
        assert rows == [0, 1, 2, 3]

    def test_each_slot_records_the_face_it_belongs_to(self):
        slots = card_slots([1, 1], layout="transform")
        faces = [s.face for s in slots if s.kind is CardSlotKind.ABILITY]
        assert faces == [0, 1]

    def test_a_face_with_no_abilities_still_gets_its_card_token(self):
        slots = card_slots([0, 1], layout="transform")
        assert sum(1 for s in slots if s.kind is CardSlotKind.CARD) == 2


class TestLayoutLine:
    def test_the_layout_line_is_read_from_the_converted_text(self):
        text = "layout: transform\nname: front\ntypes: creature\n"
        assert read_layout_line(text) == "transform"

    def test_a_single_face_card_has_no_layout_line(self):
        assert read_layout_line("name: serra angel\ntypes: creature angel\n") is None

    def test_an_empty_file_has_none(self):
        assert read_layout_line("") is None

    def test_the_converted_file_is_the_authority_on_the_vocabulary(self):
        """Whatever the converter wrote is the tag; no allowlist here."""
        assert read_layout_line("layout: aftermath\nname: x\n") == "aftermath"
