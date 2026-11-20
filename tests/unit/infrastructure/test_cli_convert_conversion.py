"""``convert`` records the conversion it leaves in its ``run.json`` (finding
11.3, task K03): the Forge checkout's full commit, whether it was dirty, and
its sidecars' digest once it finished whole; ``null`` once it may have
changed the trees without finishing; nothing when ``ConvertMain`` refused
before touching them (exit 2). The JVM is stood in by a function writing the
sidecars a conversion would."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from price_predictor.infrastructure import (
    cli,
    converted_provenance,
    forge_classpath,
    jvm_processes,
    run_manifest,
)
from price_predictor.infrastructure.cli_run_manifest import dispatch, tag_subcommands
from price_predictor.infrastructure.converted_provenance import sidecar_digest

COMMIT = "c" * 40


@pytest.fixture
def jvm(monkeypatch) -> dict:
    """The stand-in JVM: writes one sidecar per tree, then exits with ``code``
    (or raises ``raise``)."""
    behaviour: dict = {"code": 0}

    def run(cmd, check):
        arguments = dict(zip(cmd[2::2], cmd[3::2], strict=False))
        for flag in ("--output-path", "--tokens-output-path"):
            sidecar = Path(arguments[flag]) / "a" / "card.provenance.json"
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text("{}", encoding="utf-8")
        if "raise" in behaviour:
            raise behaviour["raise"]
        return subprocess.CompletedProcess(cmd, behaviour["code"])

    monkeypatch.setattr(forge_classpath, "build_forge_classpath", lambda **_kw: "forge.jar")
    monkeypatch.setattr(
        jvm_processes, "build_jvm_command",
        lambda **kw: ["java", kw["main_class"], *kw["main_args"]],
    )
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(converted_provenance, "forge_checkout", lambda _dir: (COMMIT, True))
    monkeypatch.setattr(
        run_manifest, "read_commits",
        lambda *_args, **_kw: {"repo": "abc1234", "forge": "ccccccc", "dirty": True},
    )
    return behaviour


def _convert(root: Path) -> tuple[int, list[dict]]:
    """``convert`` into ``root``'s two trees, inside its manifest: the exit
    code and the manifest's invocations."""
    argv = [
        "convert", "--cards-path", "forge/cardsfolder", "--tokens-path", "forge/tokenscripts",
        "--output-path", str(root / "cardsfolder"),
        "--tokens-output-path", str(root / "tokenscripts"),
    ]
    parser = cli.build_parser()
    tag_subcommands(parser)
    code = dispatch(parser.parse_args(argv), cli.RUN_MANIFESTS, module="price_predictor",
                    argv=argv)
    document = json.loads((root / "run.json").read_text(encoding="utf-8"))
    return code, document["invocations"]


def test_a_whole_conversion_records_its_commit_and_digest(jvm, tmp_path):
    root = tmp_path / "converted"

    code, invocations = _convert(root)

    assert code == 0
    assert invocations[-1]["conversion"] == {
        "forge_commit": COMMIT, "forge_dirty": True,
        "sidecar_digest": sidecar_digest({
            "cardsfolder": root / "cardsfolder", "tokenscripts": root / "tokenscripts",
        }),
    }


def test_a_failed_conversion_records_none(jvm, tmp_path):
    jvm["code"] = 1

    code, invocations = _convert(tmp_path / "converted")

    assert code == 1
    assert invocations[-1]["conversion"] is None


def test_a_refusal_before_the_trees_records_nothing(jvm, tmp_path):
    root = tmp_path / "converted"
    _convert(root)
    jvm["code"] = 2

    code, invocations = _convert(root)

    assert code == 2
    assert "conversion" not in invocations[-1]
    assert invocations[0]["conversion"]["forge_commit"] == COMMIT


def test_an_interrupted_conversion_records_none(jvm, tmp_path):
    root = tmp_path / "converted"
    _convert(root)
    jvm["raise"] = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _convert(root)

    invocations = json.loads((root / "run.json").read_text(encoding="utf-8"))["invocations"]
    assert invocations[-1]["conversion"] is None
