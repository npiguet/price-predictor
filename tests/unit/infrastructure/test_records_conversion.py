"""The conversions a records directory records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from price_predictor.domain.converted_provenance import ConversionRefused
from price_predictor.infrastructure.converted_provenance import read_conversion
from price_predictor.infrastructure.records_conversion import (
    records_conversions,
    require_collectable,
)
from tests.unit.conversion_fixtures import OTHER_COMMIT, conversion_member, write_conversion


def _records(root: Path, *invocations: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "run.json").write_text(json.dumps({"invocations": list(invocations)}), encoding="utf-8")


def test_every_invocations_conversion_is_read_and_one_recording_none_is_passed_over(tmp_path):
    _records(tmp_path / "records", {"conversion": conversion_member()}, {"command": "x"})
    _records(tmp_path / "records" / "sub", {"conversion": conversion_member(OTHER_COMMIT)})

    found = records_conversions(tmp_path / "records")

    assert {path.parent.name: [c.forge_commit[0] for c in cs] for path, cs in found.items()} == {
        "records": ["a"], "sub": ["b"],
    }


def test_one_differing_invocation_among_agreeing_ones_refuses(tmp_path):
    wanted = read_conversion(write_conversion(tmp_path / "converted") / "cardsfolder")
    _records(
        tmp_path / "records",
        {"conversion": conversion_member()}, {"conversion": conversion_member(OTHER_COMMIT)},
    )

    with pytest.raises(ConversionRefused, match="holds one conversion"):
        require_collectable(tmp_path / "records", wanted)


def test_a_directory_recording_nothing_collects_against_any_tree(tmp_path):
    wanted = read_conversion(write_conversion(tmp_path / "converted") / "cardsfolder")
    _records(tmp_path / "records", {"command": "collect-coverage"})

    require_collectable(tmp_path / "records", wanted)
