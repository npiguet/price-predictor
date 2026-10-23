"""CastabilityConnector: the arguments it hands CastabilityMain and how it reads back."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from effects.infrastructure import castability_connector
from effects.infrastructure.castability_connector import CastabilityConnector


def _run_with(answer: dict[str, str] | None, returncode: int = 0):
    """Patch the JVM launch: record the main args, write ``answer`` as the output."""
    seen: dict[str, list[str]] = {}

    def build(main_class, classpath, main_args):
        seen["args"] = list(main_args)
        return ["java", main_class, *main_args]

    def run(cmd, check):
        if answer is not None:
            output = Path(cmd[cmd.index("--output") + 1])
            output.write_text(json.dumps(answer), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode)

    patches = (
        patch.object(castability_connector, "build_jvm_command", side_effect=build),
        patch.object(castability_connector, "build_forge_classpath", return_value="cp"),
        patch.object(castability_connector.subprocess, "run", side_effect=run),
    )
    return seen, patches


def test_the_worker_is_given_only_the_card_list_and_the_output_path():
    seen, patches = _run_with({"Forest": "uncastable"})
    with patches[0], patches[1], patches[2]:
        verdicts = CastabilityConnector().consult(["Forest"])

    assert verdicts == {"Forest": "uncastable"}
    flags = [arg for arg in seen["args"] if arg.startswith("--")]
    assert flags == ["--cards", "--output"]


def test_a_failed_worker_yields_no_verdicts():
    _, patches = _run_with(None, returncode=1)
    with patches[0], patches[1], patches[2]:
        assert CastabilityConnector().consult(["Forest"]) == {}
