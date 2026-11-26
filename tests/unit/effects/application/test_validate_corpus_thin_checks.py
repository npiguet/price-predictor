"""validate-corpus on the real gen-3 fixture, broken so that a thin check gives
the wrong verdict (V5 #11.9, task K05b): the repeat-event check, the
attribution share, the copy-key count and the per-kind duplicate floor.

Each case edits records of ``gen3-records.jsonl.gz`` and asserts the finding
that owns the defect.
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

from effects.application.corpus_validation.findings import Finding, Thresholds
from tests.unit.effects.application.validate_corpus_ports import validate_directory

GEN3 = Path(__file__).parents[3] / "fixtures" / "effects" / "gen3-records.jsonl.gz"
REPEATS = "no record repeats an event inside itself"
UNIT_REPEATS = "one-event-per-unit types"
COPIES = "a copy keys its keyword lines"
_REST_TYPES = {"tapped", "mana_produced", "damage_dealt"}


def _fixture() -> list[dict]:
    with gzip.open(GEN3, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _findings(tmp_path, records, thresholds=Thresholds()):
    root = tmp_path / "c"
    root.mkdir(parents=True, exist_ok=True)
    with gzip.open(root / "s.jsonl.gz", "wt", encoding="utf-8") as handle:
        for raw in records:
            handle.write(json.dumps(raw, separators=(",", ":")) + "\n")
    return validate_directory(root, thresholds=thresholds).findings


def _named(findings, fragment: str) -> Finding:
    return next(f for f in findings if fragment in f.name)


def _events_with(records: list[dict], event_type: str):
    for raw in records:
        events = raw["payload"].get("events") or []
        for event in events:
            if event["type"] == event_type:
                return events, event
    raise AssertionError(f"the fixture holds no {event_type} event")


# --- the repeat-event check ---------------------------------------------


def test_a_mill_or_draw_of_many_cards_does_not_fail_the_run(tmp_path):
    """A draw of N is N identical events by design: it is measured apart."""
    records = _fixture()
    events, drawn = _events_with(records, "card_drawn")
    events.extend(copy.deepcopy(drawn) for _ in range(40))
    findings = _findings(tmp_path, records)
    assert _named(findings, REPEATS).ok, _named(findings, REPEATS).measured
    unit = _named(findings, UNIT_REPEATS)
    assert unit.watched and unit.ok, unit.measured
    assert "card_drawn" in " ".join(unit.detail)


def test_the_unit_types_are_judged_once_given_a_ceiling(tmp_path):
    records = _fixture()
    events, drawn = _events_with(records, "card_drawn")
    events.extend(copy.deepcopy(drawn) for _ in range(40))
    findings = _findings(tmp_path, records, Thresholds(max_unit_event_repeat_rate=0.05))
    assert not _named(findings, UNIT_REPEATS).ok


def test_the_rest_of_the_types_repeating_fail_with_their_own_rate(tmp_path):
    """A damage written twice is a double write; the unit repeats do not
    dilute it."""
    records = _fixture()
    doubled = 0
    for raw in records:
        events = raw["payload"].get("events") or []
        for event in [e for e in events if e["type"] in _REST_TYPES]:
            events.append(copy.deepcopy(event))
            doubled += 1
    assert doubled >= 12
    finding = _named(_findings(tmp_path, records), REPEATS)
    assert not finding.ok, finding.measured
    assert "tapped" in " ".join(finding.detail) or "damage_dealt" in " ".join(finding.detail)


# --- the attribution share ----------------------------------------------


def test_every_resolution_event_of_a_current_writer_names_a_clause(tmp_path):
    findings = _findings(tmp_path, _fixture(), Thresholds(min_attributed_rate=0.99))
    finding = _named(findings, "resolution events name")
    assert finding.ok and "291/291" in finding.measured, finding.measured


def test_a_writer_that_stops_stamping_resolution_events_fails(tmp_path):
    records = _fixture()
    for raw in records:
        if raw["kind"] == "resolution":
            for event in raw["payload"].get("events") or []:
                event["attributed_to"] = None
    findings = _findings(tmp_path, records, Thresholds(min_attributed_rate=0.99))
    finding = _named(findings, "resolution events name")
    assert not finding.ok, finding.measured
    assert "0/291" in finding.measured


# --- the copy-key count --------------------------------------------------


def _keys_elsewhere(entity: dict) -> bool:
    """A card other than E32 with a line keyed outside celestial_regulator.txt."""
    return entity["id"] != "E32" and any(
        line["keys"] and "celestial_regulator" not in line["keys"][0]["script_file"]
        for line in entity.get("lines") or ())


def _copy_of_a_card(records: list[dict], indexes=(164, 165)) -> None:
    """E32 (Celestial Regulator, its lines keyed into celestial_regulator.txt)
    is made a copy of one other real card of the board, the same in each of
    these records."""
    source_id = next(e["id"] for e in records[indexes[0]]["state"]["entities"]
                     if _keys_elsewhere(e))
    for index in indexes:
        entities = records[index]["state"]["entities"]
        source = next(e for e in entities if e["id"] == source_id)
        next(e for e in entities if e["id"] == "E32")["copy_source"] = source_id
    return source


def test_one_miskeyed_copy_seen_in_several_records_is_one_copy(tmp_path):
    records = _fixture()
    _copy_of_a_card(records)
    finding = _named(_findings(tmp_path, records), COPIES)
    assert "1/2 copies" in finding.measured, finding.measured


def test_a_copy_keeping_the_copied_cards_keys_and_one_file_of_its_own_is_accepted(tmp_path):
    """A clone's keyword lines key into the copied file; the one keyword the
    copy effect adds (Omni-Changeling's ``AddKeywords$ Changeling``) keys into
    the copying card's own file."""
    records = _fixture()
    source = _copy_of_a_card(records)
    copied_file = next(
        line["keys"][0]["script_file"] for line in source["lines"]
        if line["keys"] and "celestial_regulator" not in line["keys"][0]["script_file"])
    for index in (164, 165):
        copy_entity = next(e for e in records[index]["state"]["entities"] if e["id"] == "E32")
        kept = copy.deepcopy(copy_entity["lines"][-1])
        kept["keys"][0]["script_file"] = copied_file
        copy_entity["lines"].append(kept)
    finding = _named(_findings(tmp_path, records), COPIES)
    assert "0/2 copies" in finding.measured, finding.measured


# --- the per-kind duplicate floor ----------------------------------------


def _duplicate(records: list[dict], kind: str, subkind: str | None, times: int) -> None:
    pool = [r for r in records if r["kind"] == kind and r.get("subkind") == subkind]
    for number, raw in enumerate(pool[:times]):
        again = copy.deepcopy(raw)
        again["record_id"] += f".dup{number}"
        records.append(again)


def test_a_kind_with_a_few_records_is_not_judged_on_one_repeat(tmp_path):
    records = _fixture()
    _duplicate(records, "turn_action", "lore_counter", 1)
    finding = _named(_findings(tmp_path, records), "exact duplicates")
    assert finding.ok, finding.measured
    assert "lore_counter: 1/3" in " ".join(finding.detail)


def test_a_kind_with_enough_records_is_judged(tmp_path):
    records = _fixture()
    _duplicate(records, "playability", "decision", 10)
    finding = _named(_findings(tmp_path, records), "exact duplicates")
    assert not finding.ok, finding.measured
