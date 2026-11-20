"""The effects collectors are tied to the conversion their keys join against
(finding 11.3, task K03).

Before, nothing tied a shard to the converted tree its provenance keys were
computed against: a tree reconverted at another Forge commit joined keys to
other lines silently. Now each collector reads the conversion's record from
the ``--cards-folder`` trees, refuses to start (exit 1, the reason in
``run.json``'s outcome) when the connector JAR was built at another Forge
commit, names no commit, or a tree names no conversion, and keeps the
conversion's commit and digest in its own ``run.json``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from effects.infrastructure import cli
from price_predictor.infrastructure import forge_jvm, run_manifest
from tests.unit.conversion_fixtures import (
    DIGEST,
    FORGE_COMMIT,
    OTHER_COMMIT,
    conversion_member,
    stand_in_checkout,
    write_conversion,
    write_jar,
)

_COLLECTORS = [
    ("collect-coverage", "effects.application.collect_coverage.run"),
    ("collect-variants", "effects.application.collect_variants.run"),
]


@pytest.fixture(autouse=True)
def _no_real_checkouts(monkeypatch):
    monkeypatch.setattr(
        run_manifest, "read_commits",
        lambda *_args, **_kw: {"repo": "abc1234", "forge": "d44cf28", "dirty": False},
    )
    monkeypatch.setattr(forge_jvm, "check_worker_prerequisites", lambda **_kw: None)


@dataclass
class Collector:
    """One collector run through ``main`` into an existing records directory,
    over a converted tree, with a stand-in connector JAR."""

    command: str
    module: str
    monkeypatch: pytest.MonkeyPatch
    tmp_path: Path

    def jar(self, commit: str | None = FORGE_COMMIT) -> Path:
        jar = write_jar(self.tmp_path / "forge-connector.jar", commit)
        self.monkeypatch.setattr(forge_jvm, "resolve_connector_jar", lambda: jar)
        stand_in_checkout(self.monkeypatch)
        return jar

    def converted(self, *members: dict | None, omit: bool = False) -> Path:
        return write_conversion(self.tmp_path / "converted", *members, omit=omit)

    def run(self, converted: Path) -> tuple[int, dict, bool]:
        """Its exit code, its entry in ``run.json`` and whether the collection
        itself was reached."""
        records = self.tmp_path / "records"
        records.mkdir(exist_ok=True)
        holdout = self.tmp_path / "holdout.txt"
        holdout.write_text("", encoding="utf-8")
        argv = [
            self.command, "--effect-records", str(records), "--exclude-cards", str(holdout),
            "--cards-folder", str(converted / "cardsfolder"),
            "--cards-folder", str(converted / "tokenscripts"),
            *(["--variant-scripts", str(self.tmp_path / "variants")]
              if self.command == "collect-variants" else []),
        ]
        self.monkeypatch.setattr(sys, "argv", ["effects", *argv])
        with patch(self.module, return_value=0) as collect, pytest.raises(SystemExit) as exit_:
            cli.main()
        document = json.loads((records / "run.json").read_text(encoding="utf-8"))
        return exit_.value.code, document["invocations"][-1], collect.called


@pytest.fixture(params=_COLLECTORS, ids=[command for command, _ in _COLLECTORS])
def collector(request, monkeypatch, tmp_path) -> Collector:
    command, module = request.param
    return Collector(command, module, monkeypatch, tmp_path)


def test_a_tree_converted_at_another_commit_is_refused_with_both_commits(collector, caplog):
    converted = collector.converted(conversion_member(OTHER_COMMIT))
    collector.jar(FORGE_COMMIT)

    code, entry, collected = collector.run(converted)

    assert (code, collected) == (1, False)
    assert FORGE_COMMIT in entry["outcome"]
    assert OTHER_COMMIT in entry["outcome"]
    assert FORGE_COMMIT in caplog.text
    assert OTHER_COMMIT in caplog.text


def test_the_run_keeps_the_conversions_commit_and_digest(collector):
    converted = collector.converted()
    collector.jar()

    code, entry, collected = collector.run(converted)

    assert (code, collected) == (0, True)
    assert entry["conversion"] == {
        "run": converted.as_posix(), "forge_commit": FORGE_COMMIT,
        "forge_dirty": False, "sidecar_digest": DIGEST, "checkout_dirty": False,
    }


def test_a_connector_naming_no_commit_is_refused_naming_the_jar(collector):
    converted = collector.converted()
    jar = collector.jar(None)

    code, entry, collected = collector.run(converted)

    assert (code, collected) == (1, False)
    assert str(jar) in entry["outcome"]
    assert "rebuild the connector" in entry["outcome"]


def test_a_tree_from_before_the_record_is_refused_naming_its_path(collector):
    converted = collector.converted(omit=True)
    collector.jar()

    code, entry, collected = collector.run(converted)

    assert (code, collected) == (1, False)
    assert str(converted / "cardsfolder") in entry["outcome"]


def test_a_dirty_conversion_is_recorded_and_warned_about_not_refused(collector, caplog):
    converted = collector.converted(conversion_member(dirty=True))
    collector.jar()

    code, entry, collected = collector.run(converted)

    assert (code, collected) == (0, True)
    assert entry["conversion"]["forge_dirty"] is True
    assert "uncommitted changes" in caplog.text
