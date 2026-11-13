"""The one-shot Java mains: the arguments each is handed and how the consult reads back."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from effects.infrastructure import one_shot_workers


def _run_with(answer: dict[str, str] | str | None, returncode: int = 0):
    """Patch the JVM launch: record the call, write ``answer`` as the output."""
    seen: dict[str, object] = {}

    def run(main_class, *, main_args, include_dependency_glob, timeout):
        seen.update(
            main_class=main_class, args=list(main_args),
            glob=include_dependency_glob, timeout=timeout,
        )
        if answer is not None:
            output = Path(main_args[main_args.index("--output") + 1])
            text = answer if isinstance(answer, str) else json.dumps(answer)
            output.write_text(text, encoding="utf-8")
        return subprocess.CompletedProcess([main_class], returncode)

    return seen, patch.object(one_shot_workers, "run_forge_worker", side_effect=run)


def test_the_consult_is_given_only_the_card_list_and_the_output_path():
    seen, launch = _run_with({"Forest": "uncastable"})
    with launch:
        verdicts = one_shot_workers.castability_verdicts(["Forest"])

    assert verdicts == {"Forest": "uncastable"}
    flags = [arg for arg in seen["args"] if arg.startswith("--")]
    assert flags == ["--cards", "--output"]


def test_a_failed_consult_yields_no_verdicts():
    _, launch = _run_with(None, returncode=1)
    with launch:
        assert one_shot_workers.castability_verdicts(["Forest"]) == {}


@pytest.mark.parametrize(("write", "main", "flag"), [
    (one_shot_workers.write_keyword_definitions, "KeywordDefinitionMain", "--output"),
    (one_shot_workers.write_variant_sidecars, "VariantSidecarMain", "--variants-path"),
])
def test_each_main_runs_on_the_full_runtime_and_returns_its_exit_code(
    tmp_path, write, main, flag,
):
    seen, launch = _run_with(None, returncode=3)
    with launch:
        assert write(tmp_path) == 3

    assert seen["main_class"] == f"com.pricepredictor.connector.{main}"
    assert seen["args"] == [flag, str(tmp_path)]
    assert seen["glob"] is True


def test_an_unreadable_answer_yields_no_verdicts(caplog):
    """P1 #12: a truncated or garbled verdicts file raised JSONDecodeError out
    of the consult, which only ranks and must degrade instead."""
    _, launch = _run_with('{"Forest": "unc')
    with launch:
        assert one_shot_workers.castability_verdicts(["Forest"]) == {}
    assert "unreadable" in caplog.text


@pytest.mark.parametrize("call", [
    lambda path: one_shot_workers.castability_verdicts(["Forest"]),
    one_shot_workers.write_keyword_definitions,
    one_shot_workers.write_variant_sidecars,
])
def test_every_main_runs_under_a_timeout(tmp_path, call):
    """P1 #12: a one-shot JVM had none, so a hung Forge startup hung the command."""
    seen, launch = _run_with(None)
    with launch:
        call(tmp_path)

    assert seen["timeout"] == one_shot_workers.ONE_SHOT_TIMEOUT_SECONDS
    assert 0 < one_shot_workers.ONE_SHOT_TIMEOUT_SECONDS < float("inf")
