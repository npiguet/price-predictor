"""validate-corpus on the real gen-3 fixture, broken in one way each.

Each case edits the records of ``gen3-records.jsonl.gz`` the way a collector
regression would and asserts the check that owns the defect fails (V5 #11.9):
a check that passes a corpus it exists to refuse is worse than none.
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from effects.application.corpus_validation.findings import Finding
from effects.application.validate_corpus import validate_directory
from effects.domain.shard_generation import OlderGenerationError, UnknownGenerationError

GEN3 = Path(__file__).parents[3] / "fixtures" / "effects" / "gen3-records.jsonl.gz"


def _fixture() -> list[dict]:
    with gzip.open(GEN3, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(root: Path, records: list[dict], name: str = "s.jsonl.gz") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    with gzip.open(root / name, "wt", encoding="utf-8") as handle:
        for raw in records:
            handle.write(json.dumps(raw, separators=(",", ":")) + "\n")
    return root


def _named(findings, fragment: str) -> Finding:
    return next(f for f in findings if fragment in f.name)


def _is_effect_half(raw: dict) -> bool:
    return raw["kind"] == "resolution" and raw["moment"] == "resolution"


def test_every_effect_half_with_a_null_outcome_fails(tmp_path):
    records = _fixture()
    for raw in records:
        if _is_effect_half(raw):
            raw["payload"]["outcome"] = None
    report = validate_directory(_write(tmp_path / "c", records))
    finding = _named(report.findings, "outcome")
    assert not finding.ok, finding.measured
    assert "100.0% of the" in finding.measured
    assert "effect halves carry no outcome" in finding.measured


def test_a_null_outcome_among_varied_ones_is_reported_not_failed(tmp_path):
    records = _fixture()
    halves = [raw for raw in records if _is_effect_half(raw) and raw["link_id"]]
    halves[0]["payload"]["outcome"] = "fizzled"
    halves[1]["payload"]["outcome"] = None
    finding = _named(validate_directory(_write(tmp_path / "c", records)).findings, "outcome")
    assert finding.ok, finding.measured
    assert "carry no outcome" in finding.measured


def test_dangling_mirrors_fail_when_the_whole_directory_was_read(tmp_path):
    records = _fixture()
    for raw in records:
        if raw.get("mirror_of"):
            raw["mirror_of"] += "999"
    report = validate_directory(_write(tmp_path / "c", records))
    finding = _named(report.findings, "mirrors")
    assert not finding.ok, finding.measured
    assert "5 mirrors" in finding.measured


def test_dangling_mirrors_under_a_limit_are_watched_never_passed(tmp_path):
    records = _fixture()
    for raw in records:
        if raw.get("mirror_of"):
            raw["mirror_of"] += "999"
    report = validate_directory(_write(tmp_path / "c", records), limit=len(records) - 1)
    finding = _named(report.findings, "mirrors")
    assert finding.watched and not finding.lines()[0].startswith("[PASS]")
    assert "no comparison" in finding.measured


def _drop_markers(records: list[dict], games: set[str]) -> list[dict]:
    return [raw for raw in records
            if not (raw.get("subkind") == "game_end" and raw["game_id"] in games)]


def _games_with_a_later_game(records: list[dict]) -> list[str]:
    games = sorted({raw["game_id"] for raw in records})
    return [g for g in games if g.rpartition(".")[0] and g != games[-1]
            and any(o.rpartition(".")[0] == g.rpartition(".")[0]
                    and int(o.rpartition(".")[2]) > int(g.rpartition(".")[2])
                    for o in games)]


def test_games_a_later_game_follows_that_lack_a_marker_are_their_own_failed_figure(tmp_path):
    records = _fixture()
    closed = _games_with_a_later_game(records)
    assert len(closed) > 4
    cut = _drop_markers(records, set(closed[: len(closed) // 2]))
    report = validate_directory(_write(tmp_path / "c", cut))
    finding = _named(report.findings, "closed game")
    assert not finding.ok, finding.measured
    assert "have no game-end marker" in finding.measured


def test_every_closed_game_marked_passes_the_figure(tmp_path):
    report = validate_directory(_write(tmp_path / "c", _fixture()))
    finding = _named(report.findings, "closed game")
    assert finding.ok and "0 of" in finding.measured


def test_an_unknown_generation_names_the_shard_and_the_line(tmp_path):
    records = _fixture()
    records[5]["generation"] = 4
    with pytest.raises(UnknownGenerationError, match=r"s\.jsonl\.gz, line 6: "):
        validate_directory(_write(tmp_path / "c", records))


def test_one_shard_mixing_generations_names_the_shard_and_the_line(tmp_path):
    records = [copy.deepcopy(raw) for raw in _fixture()]
    del records[2]["generation"]
    with pytest.raises(OlderGenerationError, match=r"s\.jsonl\.gz, line 3: "):
        validate_directory(_write(tmp_path / "c", records))


def test_validate_directory_and_field_coverage_refuse_a_mixed_directory_first(
    tmp_path, monkeypatch,
):
    """Both commands call ``require_generation`` before reading, as every
    other reader of a records set does (record-format.md, Generations)."""
    from effects.infrastructure import cli, shard_io

    records = _fixture()
    root = _write(tmp_path / "c", records)
    older = [copy.deepcopy(raw) for raw in records[:3]]
    for raw in older:
        del raw["generation"]
    _write(root, older, "z.jsonl.gz")
    called = []
    real = shard_io.require_generation
    monkeypatch.setattr(shard_io, "require_generation",
                        lambda paths: (called.append(1), real(paths))[1])
    with pytest.raises(OlderGenerationError, match=r"z\.jsonl\.gz, line 1: "):
        validate_directory(root)
    args = type("A", (), {"effect_records": root, "all": False})()
    assert cli.run_field_coverage(args) == 1
    assert len(called) == 2
