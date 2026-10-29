"""What each evaluation baseline masks (FR-094)."""

from __future__ import annotations

import pytest

from effects.domain.baselines import (
    baseline_masks,
)


class TestBaselineMasks:
    def test_an_unknown_variant_is_rejected(self):
        with pytest.raises(ValueError, match="unknown --variant"):
            baseline_masks("oracle-text")
