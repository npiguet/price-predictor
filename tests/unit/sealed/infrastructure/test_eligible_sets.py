"""Unit tests for eligible_sealed_sets()."""

from __future__ import annotations

import json
from pathlib import Path

from sealed.infrastructure.eligible_sets import eligible_sealed_sets


def _write_printings(tmp_path: Path, sets: dict) -> Path:
    payload = {"meta": {"date": "2026-04-21"}, "data": sets}
    path = tmp_path / "AllPrintings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestEligibleSealedSets:
    def test_includes_set_with_draft_booster(self, tmp_path):
        path = _write_printings(tmp_path, {
            "MH3": {"type": "expansion", "booster": {"draft": {}}},
        })
        assert eligible_sealed_sets(path) == ["MH3"]

    def test_excludes_funny_sets(self, tmp_path):
        path = _write_printings(tmp_path, {
            "UNH": {"type": "funny", "booster": {"draft": {}}},
            "MH3": {"type": "expansion", "booster": {"draft": {}}},
        })
        assert eligible_sealed_sets(path) == ["MH3"]

    def test_excludes_sets_without_draft_booster(self, tmp_path):
        path = _write_printings(tmp_path, {
            "C20": {"type": "commander", "booster": {"set": {}}},
            "PRM": {"type": "promo"},  # no booster key
            "RVR": {"type": "expansion", "booster": {"draft": {}}},
        })
        assert eligible_sealed_sets(path) == ["RVR"]

    def test_excludes_sets_with_empty_booster_dict(self, tmp_path):
        path = _write_printings(tmp_path, {
            "EMP": {"type": "expansion", "booster": {}},
            "RVR": {"type": "expansion", "booster": {"draft": {}}},
        })
        assert eligible_sealed_sets(path) == ["RVR"]

    def test_preserves_source_ordering(self, tmp_path):
        path = _write_printings(tmp_path, {
            "10E": {"type": "core", "booster": {"draft": {}}},
            "MH3": {"type": "expansion", "booster": {"draft": {}}},
            "BLB": {"type": "expansion", "booster": {"draft": {}}},
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
