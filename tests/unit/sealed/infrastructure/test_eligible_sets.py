"""Unit tests for eligible_sealed_sets()."""

from __future__ import annotations

import json
from pathlib import Path

from sealed.infrastructure.eligible_sets import eligible_sealed_sets


def _draft_booster(card_count: int) -> dict:
    """Build a MTGJSON-shaped draft-booster entry whose contents sum to card_count."""
    return {
        "boosters": [
            {"contents": {"common": card_count}, "weight": 1},
        ],
    }


def _write_printings(tmp_path: Path, sets: dict) -> Path:
    payload = {"meta": {"date": "2026-04-21"}, "data": sets}
    path = tmp_path / "AllPrintings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestEligibleSealedSets:
    def test_includes_set_with_draft_booster(self, tmp_path):
        path = _write_printings(tmp_path, {
            "MH3": {"type": "expansion", "booster": {"draft": _draft_booster(15)}},
        })
        assert eligible_sealed_sets(path) == ["MH3"]

    def test_excludes_funny_sets(self, tmp_path):
        path = _write_printings(tmp_path, {
            "UNH": {"type": "funny", "booster": {"draft": _draft_booster(15)}},
            "MH3": {"type": "expansion", "booster": {"draft": _draft_booster(15)}},
        })
        assert eligible_sealed_sets(path) == ["MH3"]

    def test_excludes_sets_without_draft_booster(self, tmp_path):
        path = _write_printings(tmp_path, {
            "C20": {"type": "commander", "booster": {"set": {}}},
            "PRM": {"type": "promo"},  # no booster key
            "RVR": {"type": "expansion", "booster": {"draft": _draft_booster(15)}},
        })
        assert eligible_sealed_sets(path) == ["RVR"]

    def test_excludes_sets_with_empty_booster_dict(self, tmp_path):
        path = _write_printings(tmp_path, {
            "EMP": {"type": "expansion", "booster": {}},
            "RVR": {"type": "expansion", "booster": {"draft": _draft_booster(15)}},
        })
        assert eligible_sealed_sets(path) == ["RVR"]

    def test_excludes_small_booster_sets(self, tmp_path):
        """DRK/FEM-era sets with 8-card boosters are too small for real sealed play."""
        path = _write_printings(tmp_path, {
            "DRK": {"type": "expansion", "booster": {"draft": _draft_booster(8)}},
            "FEM": {"type": "expansion", "booster": {"draft": _draft_booster(8)}},
            "RVR": {"type": "expansion", "booster": {"draft": _draft_booster(15)}},
        })
        assert eligible_sealed_sets(path) == ["RVR"]

    def test_boundary_exactly_twelve_cards_is_eligible(self, tmp_path):
        path = _write_printings(tmp_path, {
            "SML": {"type": "expansion", "booster": {"draft": _draft_booster(12)}},
        })
        assert eligible_sealed_sets(path) == ["SML"]

    def test_boundary_eleven_cards_is_excluded(self, tmp_path):
        path = _write_printings(tmp_path, {
            "XS": {"type": "expansion", "booster": {"draft": _draft_booster(11)}},
        })
        assert eligible_sealed_sets(path) == []

    def test_malformed_draft_booster_excluded(self, tmp_path):
        """A draft booster without a valid ``boosters`` list is treated as size 0."""
        path = _write_printings(tmp_path, {
            "BAD": {"type": "expansion", "booster": {"draft": {}}},
            "RVR": {"type": "expansion", "booster": {"draft": _draft_booster(15)}},
        })
        assert eligible_sealed_sets(path) == ["RVR"]

    def test_preserves_source_ordering(self, tmp_path):
        path = _write_printings(tmp_path, {
            "10E": {"type": "core", "booster": {"draft": _draft_booster(15)}},
            "MH3": {"type": "expansion", "booster": {"draft": _draft_booster(15)}},
            "BLB": {"type": "expansion", "booster": {"draft": _draft_booster(15)}},
        })
        assert eligible_sealed_sets(path) == ["10E", "MH3", "BLB"]

    def test_missing_file_raises(self, tmp_path):
        import pytest
        with pytest.raises(FileNotFoundError):
            eligible_sealed_sets(tmp_path / "nope.json")

    def test_returns_empty_when_no_data_key(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"meta": {}}), encoding="utf-8")
        assert eligible_sealed_sets(path) == []
