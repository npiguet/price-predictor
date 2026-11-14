"""``rule-counts``' command line (T035, contracts/cli.md).

Flags and defaults as the contract lists them; the counts file and the
printed table say the same thing; an empty directory and a mix of
generations each exit 1.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest

from effects.domain.default_paths import DEFAULT_CARDS_FOLDER_PATHS, DEFAULT_VARIANT_SCRIPTS
from effects.infrastructure.cli import build_parser

FIXTURES = Path(__file__).parents[2] / "fixtures" / "effects"
SIDECARS = FIXTURES / "gen2-sidecars"


def parse(*argv: str):
    return build_parser().parse_args(["rule-counts", *argv])


def test_records_dir_is_required():
    with pytest.raises(SystemExit):
        parse()


def test_the_defaults_are_the_contract_s():
    args = parse("--records-dir", "r")
    assert args.output == Path("output/effects/rule-counts.json")
    assert tuple(args.cards_folders) == tuple(DEFAULT_CARDS_FOLDER_PATHS)
    assert str(args.variant_scripts) == str(Path(DEFAULT_VARIANT_SCRIPTS))
    assert args.rules_dir == Path("rules")
    assert args.workers == 12


@pytest.mark.parametrize("workers", ["0", "13"])
def test_the_shard_readers_are_1_to_12(workers):
    """The machine runs at most twelve CPU-busy processes: an explicit,
    bounded count, never the CPU count."""
    with pytest.raises(SystemExit):
        parse("--records-dir", "r", "--workers", workers)


def test_cards_folder_repeats():
    args = parse("--records-dir", "r", "--cards-folder", "a", "--cards-folder", "b")
    assert [str(p) for p in args.cards_folders] == ["a", "b"]


def _rules(tmp_path) -> Path:
    scenarios = tmp_path / "rules" / "scenarios"
    scenarios.mkdir(parents=True)
    scenario = {"id": "704.5f-1", "rules": ["704.5f"], "record_kind": "resolution",
                "detector": {"family": "sba_zone_move", "record_kind": "resolution",
                             "outcome": "o", "near_miss": "n",
                             "params": {"check": "toughness_zero"}},
                "variants": [{"label": "b", "polarity": "fires", "delta": "", "expected": ""}]}
    (scenarios / "704.json").write_text(
        json.dumps({"unit": "704", "scenarios": [scenario]}), encoding="utf-8")
    return scenarios.parent


def _run(tmp_path, records: Path, workers: int = 1):
    args = parse("--records-dir", str(records), "--output", str(tmp_path / "counts.json"),
                 "--cards-folder", str(SIDECARS / "cardsfolder"),
                 "--rules-dir", str(_rules(tmp_path)), "--workers", str(workers))
    return args.func(args)


def _gen2_lines() -> list[str]:
    with gzip.open(FIXTURES / "gen2-records.jsonl.gz", "rt", encoding="utf-8") as source:
        return [line.rstrip("\n") for line in source if line.strip()]


@pytest.mark.parametrize("workers", [1, 2])
def test_a_line_that_does_not_decode_exits_1_naming_shard_and_line(tmp_path, caplog, workers):
    """Raised in a pool worker too: the location crosses the process boundary."""
    records = tmp_path / "records"
    records.mkdir()
    lines = _gen2_lines()
    shutil.copyfile(FIXTURES / "gen2-records.jsonl.gz", records / "a.jsonl.gz")
    broken = json.loads(lines[1])
    broken["kind"] = "no_such_kind"
    (records / "b.jsonl").write_text(
        "\n".join([lines[0], json.dumps(broken)]) + "\n", encoding="utf-8")
    with caplog.at_level("ERROR"):
        assert _run(tmp_path, records, workers=workers) == 1
    assert any("b.jsonl, line 2" in r.getMessage() and "no_such_kind" in r.getMessage()
               for r in caplog.records)


def test_a_shard_damaged_before_its_last_member_exits_1(tmp_path, caplog):
    records = tmp_path / "records"
    records.mkdir()
    first, second = (gzip.compress((line + "\n").encode("utf-8")) for line in _gen2_lines()[:2])
    (records / "c.jsonl.gz").write_bytes(first + b"XX" + first[2:] + second)
    with caplog.at_level("ERROR"):
        assert _run(tmp_path, records, workers=2) == 1
    assert any("c.jsonl.gz" in r.getMessage() and f"byte {len(first)}" in r.getMessage()
               for r in caplog.records)


def test_the_values_outside_the_closed_lists_are_printed_summed(tmp_path, capsys):
    """Counted in each worker, handed back with its shard's counts, printed once."""
    records = tmp_path / "records"
    records.mkdir()
    lines = _gen2_lines()
    for name in ("a", "b"):
        odd = json.loads(lines[0])
        odd["state"]["global"]["phase"] = "teatime"
        (records / f"{name}.jsonl").write_text(json.dumps(odd) + "\n", encoding="utf-8")
    assert _run(tmp_path, records, workers=2) == 0
    out = capsys.readouterr().out
    assert "global.phase 'teatime': 2" in out


def test_the_file_and_the_table_agree(tmp_path, capsys):
    records = tmp_path / "records"
    records.mkdir()
    shutil.copyfile(FIXTURES / "gen2-records.jsonl.gz", records / "gen2.jsonl.gz")
    assert _run(tmp_path, records) == 0
    document = json.loads((tmp_path / "counts.json").read_text(encoding="utf-8"))
    sides = document["rules"]["704.5f"]
    printed = next(line for line in capsys.readouterr().out.splitlines()
                   if line.startswith("704.5f"))
    for polarity in ("fires", "does_not_fire"):
        split = "/".join(str(sides.get(polarity, {}).get(s, 0))
                         for s in ("random_seat", "forge_ai", "aimed"))
        assert split in printed


def test_an_empty_directory_exits_1(tmp_path):
    empty = tmp_path / "records"
    empty.mkdir()
    assert _run(tmp_path, empty) == 1


def test_a_mix_of_generations_exits_1(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    shutil.copyfile(FIXTURES / "gen2-records.jsonl.gz", records / "gen2.jsonl.gz")
    with gzip.open(FIXTURES / "gen2-records.jsonl.gz", "rt", encoding="utf-8") as source, \
            gzip.open(records / "gen3.jsonl.gz", "wt", encoding="utf-8") as out:
        for line in source:
            data = json.loads(line)
            out.write(json.dumps({"record_id": data["record_id"], "generation": 3, **data}) + "\n")
    assert _run(tmp_path, records) == 1


def test_a_gen1_set_exits_1(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    shutil.copyfile(FIXTURES / "gen1-records.jsonl.gz", records / "gen1.jsonl.gz")
    assert _run(tmp_path, records) == 1


def test_the_counts_file_lists_each_rule_s_carriers(tmp_path):
    """FR-058f, contracts/cli.md: per rule, the script files of its tags'
    condition entities' cards, each with its record count and the lines its
    tags act through, each with its own."""
    records = tmp_path / "records"
    records.mkdir()
    shutil.copyfile(FIXTURES / "gen2-records.jsonl.gz", records / "gen2.jsonl.gz")
    assert _run(tmp_path, records) == 0
    document = json.loads((tmp_path / "counts.json").read_text(encoding="utf-8"))
    carriers = document["carriers"]
    assert set(carriers) <= set(document["rules"]) and carriers["704.5f"]
    for found in carriers.values():
        for carrier in found:
            assert set(carrier) == {"script_file", "records", "lines"}
            assert carrier["script_file"].split("/")[0] in {"cardsfolder", "tokenscripts",
                                                            "variant-scripts"}
            for line in carrier["lines"]:
                assert set(line) == {"key", "records"}
                assert {"face", "trait_kind", "index_within_kind"} <= set(line["key"])


def test_the_counts_file_names_its_records_generation(tmp_path):
    """The steered variant round refuses counts of gen-2 records (FR-058g)."""
    records = tmp_path / "records"
    records.mkdir()
    shutil.copyfile(FIXTURES / "gen2-records.jsonl.gz", records / "gen2.jsonl.gz")
    assert _run(tmp_path, records) == 0
    document = json.loads((tmp_path / "counts.json").read_text(encoding="utf-8"))
    assert document["generation"] == "gen-2"


def test_cut_games_are_left_out_and_reported(tmp_path, capsys):
    """contracts/cli.md: a gen-3 game whose shard holds no game-end marker is
    counted toward no rule, and the file and the report say how many."""
    records = tmp_path / "records"
    records.mkdir()
    shutil.copyfile(FIXTURES / "gen3-records.jsonl.gz", records / "gen3.jsonl.gz")
    assert _run(tmp_path, records) == 0
    document = json.loads((tmp_path / "counts.json").read_text(encoding="utf-8"))
    assert document["cut_games"] == 58
    assert document["rules"] == {}
    assert "58 gen-3 game(s) without a game-end marker (cut) left out" in capsys.readouterr().out
