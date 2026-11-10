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
    assert args.workers is None


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


def _run(tmp_path, records: Path):
    args = parse("--records-dir", str(records), "--output", str(tmp_path / "counts.json"),
                 "--cards-folder", str(SIDECARS / "cardsfolder"),
                 "--rules-dir", str(_rules(tmp_path)), "--workers", "1")
    return args.func(args)


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
