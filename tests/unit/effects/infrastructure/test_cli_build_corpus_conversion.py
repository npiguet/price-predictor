"""``build-corpus`` joins only against the conversion its records were
collected against: each ``run.json`` under
``--records-dir`` records the conversion its collection ran against, and the
build exits 1, both conversions named, when one differs from the
``--cards-folder`` trees', or when a shard's directory records none."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from effects.infrastructure import cli
from price_predictor.infrastructure.converted_provenance import read_conversion
from price_predictor.infrastructure.records_conversion import record_collection
from price_predictor.infrastructure.run_manifest import RunCommand, run_recorded
from tests.unit.conversion_fixtures import (
    DIGEST,
    FORGE_COMMIT,
    OTHER_COMMIT,
    conversion_member,
    write_conversion,
)


def _collected_against(records: Path, converted: Path) -> None:
    """``records`` as a collection leaves it: a shard and the ``run.json``
    its collector wrote, through the collectors' own recording."""
    records.mkdir(parents=True, exist_ok=True)
    (records / "run1.shard.jsonl.gz").write_bytes(b"")
    conversion = read_conversion(converted / "cardsfolder")
    run_recorded(
        records, lambda: record_collection(records, conversion),
        RunCommand(command="collect-coverage", create=True),
    )


def _build(monkeypatch, tmp_path: Path, converted: Path, records: Path) -> int:
    monkeypatch.setattr(sys, "argv", [
        "effects", "build-corpus", "--records-dir", str(records),
        "--output", str(tmp_path / "corpus"), "--vocab-path", "vocab.txt",
        "--keyword-definitions", "keywords.json",
        "--cards-folder", str(converted / "cardsfolder"),
        "--cards-folder", str(converted / "tokenscripts"),
    ])
    with patch("effects.application.build_corpus.build", return_value=0), \
            pytest.raises(SystemExit) as exit_:
        cli.main()
    return exit_.value.code


def test_records_of_a_conversion_build_against_a_tree_of_it(monkeypatch, tmp_path):
    converted = write_conversion(tmp_path / "converted")
    _collected_against(tmp_path / "records", converted)

    assert _build(monkeypatch, tmp_path, converted, tmp_path / "records") == 0


def test_records_of_one_conversion_refuse_a_tree_of_another_naming_both(
    monkeypatch, tmp_path, caplog,
):
    _collected_against(tmp_path / "records", write_conversion(tmp_path / "x"))
    other = write_conversion(tmp_path / "y", conversion_member(OTHER_COMMIT))

    assert _build(monkeypatch, tmp_path, other, tmp_path / "records") == 1
    err = caplog.text
    assert FORGE_COMMIT in err
    assert OTHER_COMMIT in err
    assert DIGEST in err


def test_a_second_collection_of_another_conversion_in_a_subdirectory_is_refused(
    monkeypatch, tmp_path, caplog,
):
    converted = write_conversion(tmp_path / "x")
    _collected_against(tmp_path / "records", converted)
    other = write_conversion(tmp_path / "y", conversion_member(OTHER_COMMIT))
    _collected_against(tmp_path / "records" / "later", other)

    assert _build(monkeypatch, tmp_path, converted, tmp_path / "records") == 1
    assert OTHER_COMMIT in caplog.text


def test_shards_whose_directory_records_no_conversion_are_refused_saying_so(
    monkeypatch, tmp_path, caplog,
):
    converted = write_conversion(tmp_path / "converted")
    records = tmp_path / "records"
    _collected_against(records, converted)
    (records / "old").mkdir()
    (records / "old" / "run0.shard.jsonl.gz").write_bytes(b"")
    (records / "old" / "run.json").write_text(
        json.dumps({"invocations": [{"command": "collect-coverage"}]}), encoding="utf-8",
    )

    assert _build(monkeypatch, tmp_path, converted, records) == 1
    err = caplog.text
    assert "name no conversion" in err
    assert str(records / "old") in err
