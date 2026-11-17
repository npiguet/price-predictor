"""``scripts/rule_breadth.py`` over the real gen-3 fixture (T036)."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import rule_breadth

FIXTURES = Path(__file__).parents[2] / "fixtures" / "effects"
SIDECARS = FIXTURES / "gen3-sidecars"


def _rules(tmp_path: Path) -> Path:
    scenarios = tmp_path / "rules" / "scenarios"
    scenarios.mkdir(parents=True)

    def scenario(sid, params):
        return {"id": sid, "rules": [sid.split("-")[0]], "record_kind": "resolution",
                "detector": {"family": "outcome_event", "record_kind": "resolution",
                             "outcome": "o", "near_miss": "n", "params": params},
                "variants": [{"label": "b", "polarity": "fires", "delta": "", "expected": ""}]}

    (scenarios / "1.json").write_text(json.dumps({"unit": "1", "scenarios": [
        scenario("1.1-1", {"event": "zone_change"}),
        scenario("1.2-1", {"event": "zone_change", "to_zone": "ante"}),
    ]}), encoding="utf-8")
    return scenarios.parent


def test_a_detector_cut_to_nothing_of_a_large_pool_is_flagged(tmp_path, capsys):
    records = tmp_path / "records"
    records.mkdir()
    shutil.copyfile(FIXTURES / "gen3-records.jsonl.gz", records / "gen3.jsonl.gz")
    out = tmp_path / "breadth.json"
    assert rule_breadth.main([
        "--records-dir", str(records), "--rules-dir", str(_rules(tmp_path)),
        "--cards-folder", str(SIDECARS / "cardsfolder"), "--workers", "1",
        "--min-pool", "5", "--output", str(out)]) == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    rows = {row["scenario"]: row for row in document["scenarios"]}
    assert rows["1.1-1"]["pool"] == rows["1.1-1"]["tagged"] > 5
    assert rows["1.2-1"]["pool"] == rows["1.1-1"]["pool"] and rows["1.2-1"]["tagged"] == 0
    assert document["flagged"] == ["1.2-1"]
    assert "1.2-1" in capsys.readouterr().out


def test_a_set_of_an_older_generation_exits_1_naming_it(tmp_path, caplog):
    records = tmp_path / "records"
    records.mkdir()
    with gzip.open(FIXTURES / "gen3-records.jsonl.gz", "rt", encoding="utf-8") as source:
        data = json.loads(next(source))
    del data["generation"]
    (records / "gen2.jsonl").write_text(json.dumps(data) + "\n", encoding="utf-8")
    with caplog.at_level("ERROR"):
        assert rule_breadth.main([
            "--records-dir", str(records), "--rules-dir", str(_rules(tmp_path)),
            "--cards-folder", str(SIDECARS / "cardsfolder"), "--workers", "1",
            "--output", str(tmp_path / "breadth.json")]) == 1
    assert any("holds gen-2 records; this command reads gen-3 only" in r.getMessage()
               for r in caplog.records)
