"""What each evaluation baseline masks (FR-094)."""

from __future__ import annotations

import pytest

from effects.domain.baselines import (
    baseline_masks,
)


class TestBaselineMasks:
    def test_full_masks_nothing(self):
        masks = baseline_masks("full")
        assert not (masks.zero_act_e or masks.zero_context_e or masks.zero_state)
        assert not (masks.identity_embedding or masks.taxonomy_embedding)

    def test_identity_replaces_the_encoder(self):
        assert baseline_masks("identity").identity_embedding

    def test_state_only_zeroes_every_e(self):
        masks = baseline_masks("state-only")
        assert masks.zero_act_e and masks.zero_context_e
        assert not masks.zero_state

    def test_no_state_zeroes_the_ability_tokens_too(self):
        """Otherwise it would read the board through the abilities on it."""
        masks = baseline_masks("no-state")
        assert masks.zero_state
        assert masks.zero_context_e

    def test_no_state_keeps_the_acting_lines_e(self):
        """FR-094: every input zeroed except [ACT] and the record-kind flag."""
        assert not baseline_masks("no-state").zero_act_e

    def test_taxonomy_swaps_e_for_the_sidecars_api_fields(self):
        assert baseline_masks("taxonomy").taxonomy_embedding

    def test_an_unknown_variant_is_rejected(self):
        with pytest.raises(ValueError, match="unknown --variant"):
            baseline_masks("oracle-text")
