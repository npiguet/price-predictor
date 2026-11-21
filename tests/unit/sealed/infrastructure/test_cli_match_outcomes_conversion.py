"""``match-outcomes --effect-records`` is tied to the conversion its keys join
against (finding 11.3, task K03).

The collection names its converted tree, ``--cards-folder``, and refuses to
start (exit 1, the reason in ``run.json``'s outcome) when the connector JAR
was built at another Forge commit than the tree's conversion, names none, or
the tree names no conversion; it keeps the conversion's commit and digest in
its ``run.json``. A plain run names no tree and checks nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import sealed.application.match_outcomes as supervisor_module
from price_predictor.infrastructure import forge_classpath, run_manifest
from sealed.infrastructure import cli
from tests.unit.conversion_fixtures import (
    DIGEST,
    FORGE_COMMIT,
    OTHER_COMMIT,
    conversion_member,
    stand_in_checkout,
    write_conversion,
    write_jar,
)


@pytest.fixture
def started(monkeypatch, tmp_path) -> list:
    """The supervisors the command built; the JAR the check reads is a
    stand-in whose manifest names :data:`FORGE_COMMIT`."""
    built: list = []

    def supervisor(**kwargs):
        built.append(kwargs)
        return MagicMock(collected_nothing=False)

    monkeypatch.setattr(supervisor_module, "MatchOutcomeSupervisor", supervisor)
    monkeypatch.setattr(forge_classpath, "check_worker_prerequisites", lambda **_kw: None)
    jar = write_jar(tmp_path / "forge-connector.jar", FORGE_COMMIT)
    monkeypatch.setattr(forge_classpath, "resolve_connector_jar", lambda: jar)
    stand_in_checkout(monkeypatch)
    monkeypatch.setattr(
        run_manifest, "read_commits",
        lambda *_args, **_kw: {"repo": "abc1234", "forge": "d44cf28", "dirty": False},
    )
    return built


def _run(monkeypatch, tmp_path: Path, *argv: str) -> tuple[int, dict]:
    """``match-outcomes`` through ``main`` into an existing records directory:
    its exit code and its entry in ``run.json``."""
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    monkeypatch.setattr(sys, "argv", [
        "sealed", "match-outcomes", "--workers", "1", "--effect-records", str(records), *argv,
    ])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    document = json.loads((records / "run.json").read_text(encoding="utf-8"))
    return exit_.value.code, document["invocations"][-1]


def test_a_tree_converted_at_another_commit_is_refused_with_both_commits(
    started, monkeypatch, tmp_path, capsys,
):
    converted = write_conversion(tmp_path / "converted", conversion_member(OTHER_COMMIT))

    code, entry = _run(monkeypatch, tmp_path, "--cards-folder", str(converted / "cardsfolder"))

    assert (code, started) == (1, [])
    assert FORGE_COMMIT in entry["outcome"]
    assert OTHER_COMMIT in entry["outcome"]
    err = capsys.readouterr().err
    assert FORGE_COMMIT in err
    assert OTHER_COMMIT in err


def test_the_run_keeps_the_conversions_commit_and_digest(started, monkeypatch, tmp_path):
    converted = write_conversion(tmp_path / "converted")

    code, entry = _run(monkeypatch, tmp_path, "--cards-folder", str(converted / "cardsfolder"))

    assert (code, len(started)) == (0, 1)
    assert entry["conversion"] == {
        "run": converted.as_posix(), "forge_commit": FORGE_COMMIT,
        "forge_dirty": False, "sidecar_digest": DIGEST, "checkout_dirty": False,
    }


def test_a_connector_naming_no_commit_is_refused_naming_the_jar(
    started, monkeypatch, tmp_path,
):
    converted = write_conversion(tmp_path / "converted")
    jar = write_jar(tmp_path / "old-connector.jar", None)
    monkeypatch.setattr(forge_classpath, "resolve_connector_jar", lambda: jar)

    code, entry = _run(monkeypatch, tmp_path, "--cards-folder", str(converted / "cardsfolder"))

    assert (code, started) == (1, [])
    assert str(jar) in entry["outcome"]
    assert "rebuild the connector" in entry["outcome"]


def test_a_tree_from_before_the_record_is_refused_naming_its_path(
    started, monkeypatch, tmp_path,
):
    converted = write_conversion(tmp_path / "converted", omit=True)

    code, entry = _run(monkeypatch, tmp_path, "--cards-folder", str(converted / "cardsfolder"))

    assert (code, started) == (1, [])
    assert str(converted / "cardsfolder") in entry["outcome"]


def _usage_error(*argv: str) -> str | None:
    from sealed.infrastructure.collection_conversion import cards_folder_error

    return cards_folder_error(cli.build_parser().parse_args(["match-outcomes", *argv]))


def test_a_collection_names_its_tree_and_a_plain_run_names_none():
    assert "needs --cards-folder" in _usage_error("--effect-records", "records")
    assert "needs --effect-records" in _usage_error("--cards-folder", "converted/cardsfolder")
    assert _usage_error() is None
    assert _usage_error("--effect-records", "records", "--cards-folder", "c/cardsfolder") is None


def test_a_plain_run_reads_no_conversion(started, monkeypatch, tmp_path):
    outcomes = tmp_path / "outcomes"
    monkeypatch.setattr(sys, "argv", [
        "sealed", "match-outcomes", "--workers", "1", "--outcomes-dir", str(outcomes)])
    monkeypatch.setattr(forge_classpath, "resolve_connector_jar", lambda: tmp_path / "absent.jar")

    with pytest.raises(SystemExit) as exit_:
        cli.main()

    assert (exit_.value.code, len(started)) == (0, 1)


def test_a_directory_recording_another_conversion_is_refused_with_both(
    started, monkeypatch, tmp_path,
):
    """The records directory already holds a run of conversion X (KR4)."""
    first = write_conversion(tmp_path / "x")
    assert _run(monkeypatch, tmp_path, "--cards-folder", str(first / "cardsfolder"))[0] == 0
    jar = write_jar(tmp_path / "forge-connector.jar", OTHER_COMMIT)
    monkeypatch.setattr(forge_classpath, "resolve_connector_jar", lambda: jar)
    stand_in_checkout(monkeypatch, OTHER_COMMIT)
    other = write_conversion(tmp_path / "y", conversion_member(OTHER_COMMIT))

    code, entry = _run(monkeypatch, tmp_path, "--cards-folder", str(other / "cardsfolder"))

    assert (code, len(started)) == (1, 1)
    assert FORGE_COMMIT in entry["outcome"]
    assert OTHER_COMMIT in entry["outcome"]
    assert DIGEST in entry["outcome"]
