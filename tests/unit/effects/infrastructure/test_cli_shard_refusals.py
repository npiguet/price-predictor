"""Every command that reads shards exits 1 on a damaged shard, a line that
does not decode, an unknown generation or a record of an older one, naming
where (FR-033).

One damaged shard used to be truncated silently; now the read refuses it,
and the command maps the refusal to exit 1 with the located message rather
than a traceback.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from effects.domain.shard_generation import OlderGenerationError, UnknownGenerationError
from effects.infrastructure import cli
from effects.infrastructure.shard_io import CorruptShardError, RecordDecodeError
from tests.unit.effects.cli_args import parse_cli

FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"


@pytest.mark.parametrize("handler", [
    cli.run_rule_counts, cli.run_field_coverage,
    cli.run_train_effect_model, cli.run_evaluate_effect_model,
])
@pytest.mark.parametrize("error", [
    CorruptShardError("s.jsonl.gz: the gzip member starting at byte 9 is corrupt"),
    RecordDecodeError.at(Path("s.jsonl.gz"), 3, KeyError("kind")),
    UnknownGenerationError("s.jsonl.gz, line 1: generation 4"),
    OlderGenerationError("s.jsonl.gz, line 1: the record is gen-2"),
])
def test_each_reader_exits_1_on_a_refused_shard(handler, error):
    assert cli._exit_code(handler.exit_codes, error) == 1


def _gen3_lines() -> list[str]:
    with gzip.open(FIXTURES / "gen3-records.jsonl.gz", "rt", encoding="utf-8") as source:
        return [line.rstrip("\n") for line in source if line.strip()]


def _field_coverage(records: Path) -> int:
    args = parse_cli(["field-coverage", "--effect-records", str(records)])
    return args.func(args)


def test_field_coverage_exits_1_naming_the_line_that_does_not_decode(tmp_path, caplog):
    lines = _gen3_lines()
    broken = json.loads(lines[1])
    broken["kind"] = "no_such_kind"
    (tmp_path / "a.jsonl").write_text(f"{lines[0]}\n{json.dumps(broken)}\n", encoding="utf-8")
    with caplog.at_level("ERROR"):
        assert _field_coverage(tmp_path) == 1
    assert any("a.jsonl, line 2" in r.getMessage() for r in caplog.records)


def test_field_coverage_prints_the_values_outside_the_closed_lists(tmp_path, capsys):
    odd = json.loads(_gen3_lines()[0])
    odd["state"]["global"]["phase"] = "teatime"
    (tmp_path / "a.jsonl").write_text(json.dumps(odd) + "\n", encoding="utf-8")
    assert _field_coverage(tmp_path) == 0
    assert "global.phase 'teatime': 1" in capsys.readouterr().out


def test_validate_corpus_exits_1_naming_the_line_that_does_not_decode(tmp_path, caplog):
    lines = _gen3_lines()
    (tmp_path / "a.jsonl").write_text(f"{lines[0]}\n[1, 2]\n", encoding="utf-8")
    args = parse_cli([
        "validate-corpus", "--effect-records", str(tmp_path),
        "--cards-folder", str(tmp_path / "none")])
    with caplog.at_level("ERROR"):
        assert args.func(args) == 1
    assert any("a.jsonl, line 2" in r.getMessage() for r in caplog.records)


def _older_line(*dropped: str) -> str:
    """A real gen-3 line as an older generation wrote it: ``generation``
    dropped is gen-2, ``random_seat`` too gen-1."""
    data = json.loads(_gen3_lines()[0])
    for member in dropped:
        del data[member]
    return json.dumps(data)


@pytest.mark.parametrize("older", [
    (("generation",), "gen-2"),
    (("generation", "random_seat"), "gen-1"),
], ids=["gen-2", "gen-1"])
@pytest.mark.parametrize("command", ["field-coverage", "validate-corpus"])
def test_a_record_of_an_older_generation_exits_1_naming_it(tmp_path, caplog, command, older):
    dropped, named = older
    lines = _gen3_lines()
    (tmp_path / "a.jsonl").write_text(f"{lines[0]}\n{_older_line(*dropped)}\n",
                                      encoding="utf-8")
    argv = [command, "--effect-records", str(tmp_path)]
    if command == "validate-corpus":
        argv += ["--cards-folder", str(tmp_path / "none")]
    args = parse_cli(argv)
    with caplog.at_level("ERROR"):
        assert args.func(args) == 1
    assert any("a.jsonl" in r.getMessage() and f"the record is {named} " in r.getMessage()
               for r in caplog.records)
