"""The ``taxonomy`` baseline's stand-in for ``e`` (FR-102)."""

from __future__ import annotations

import numpy as np

from effects.domain.baselines import taxonomy_matrix, taxonomy_vector
from effects.domain.provenance import ProvenanceSidecar, SidecarLine

E_DIM = 8


class TestTaxonomyBaseline:
    def test_it_emits_into_the_same_row_layout_without_an_encoder(self):
        """Every e-geometry check reads one file shape (FR-102)."""
        sidecar = ProvenanceSidecar(
            card="serra angel", script_file="cardsfolder/s/serra_angel.txt",
            lines=tuple(
                SidecarLine(line_index=i, line_kind="static", provenance=())
                for i in range(3)
            ),
        )
        matrix = taxonomy_matrix(sidecar, E_DIM)
        assert matrix.shape == (3, E_DIM)

    def test_the_vector_is_a_function_of_the_api_type_and_params(self):
        first = SidecarLine(
            line_index=0, line_kind="static", provenance=(),
            script_api_type="Pump", script_param_keys=("Defined",),
        )
        same = SidecarLine(
            line_index=9, line_kind="triggered", provenance=(),
            script_api_type="Pump", script_param_keys=("Defined",),
        )
        different = SidecarLine(
            line_index=0, line_kind="static", provenance=(),
            script_api_type="Destroy", script_param_keys=("Defined",),
        )
        np.testing.assert_allclose(
            taxonomy_vector(first, E_DIM), taxonomy_vector(same, E_DIM),
        )
        assert not np.allclose(
            taxonomy_vector(first, E_DIM), taxonomy_vector(different, E_DIM),
        )

    def test_a_line_with_no_script_facts_still_yields_a_vector(self):
        bare = SidecarLine(line_index=0, line_kind="text", provenance=())
        assert taxonomy_vector(bare, E_DIM).shape == (E_DIM,)

    def test_a_source_with_no_lines_yields_an_empty_matrix(self):
        empty = ProvenanceSidecar(card="x", script_file="cardsfolder/x.txt", lines=())
        assert taxonomy_matrix(empty, E_DIM).shape == (0, E_DIM)
