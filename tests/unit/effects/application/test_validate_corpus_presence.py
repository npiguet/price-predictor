"""validate-corpus's presence checks: every record kind is present, blockers
records carry ``what_if``, and the gen-3 fields each record owes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from effects.application.corpus_validation.findings import Findings
from effects.application.corpus_validation.presence import FieldPresence
from effects.domain.gen3_members import ALWAYS, Site
from effects.domain.records import PlayabilitySubkind, RecordKind
from effects.infrastructure.shard_io import read_window
from tests.unit.effects.application.validate_corpus_ports import validate_corpus
from tests.unit.effects.application.validate_corpus_support import (
    _healthy,
    _named,
    _resolution,
)
from tests.unit.effects.builders import make_record


class TestEveryRecordKindIsPresent:
    """A kind no record of a long gen-3 window holds is not reported ``ok``
    ("no records of this kind") by a per-kind check."""

    @staticmethod
    def _window(records: int):
        return [_resolution(f"run.0-L1.{index}") for index in range(records)]

    def test_a_long_window_missing_kinds_every_game_writes_fails(self):
        finding = _named(validate_corpus(self._window(1000), playability_rate=0.01),
                         "record kind")
        assert not finding.ok
        for label in ("resolution/activation", "trigger", "combat", "playability/decision",
                      "playability/attackers", "playability/blockers", "turn_action/draw"):
            assert label in finding.measured
        assert "rewrite" in " ".join(finding.detail)  # watched, named, not judged

    def test_decision_records_are_judged_only_where_the_run_samples_them(self):
        """A run collected with --playability-rate 0 writes no decision record;
        without a run manifest saying the rate, their absence is watched."""
        for rate in (0.0, None):
            finding = _named(validate_corpus(self._window(1000), playability_rate=rate),
                             "record kind")
            assert "playability/decision" not in finding.measured
            assert "playability/decision" in " ".join(finding.detail)

    def test_the_rate_is_read_off_the_run_manifest(self, tmp_path):
        from effects.application.validate_corpus import playability_rate_of

        assert playability_rate_of(tmp_path) is None
        (tmp_path / "run.json").write_text(json.dumps({"invocations": [
            {"command": "x", "caps": {"playability_rate": 0.01}},
            {"command": "y", "caps": {"playability_rate": 0.0}}]}), encoding="utf-8")
        assert playability_rate_of(tmp_path) == 0.0

    def test_a_short_window_is_not_judged(self):
        finding = _named(validate_corpus(self._window(10)), "record kind")
        assert finding.ok and "under the 1000" in finding.measured
        assert _named(validate_corpus(_healthy()), "record kind").ok


def _read_noting_damage(root: Path, presence: FieldPresence):
    """The records under ``root`` as ``validate-corpus`` reads them: a line
    missing a member the decoder requires ends its shard, after the presence
    tally has seen it."""
    from effects.infrastructure.shard_io import list_shards
    from tests.unit.effects.application.validate_corpus_ports import ShardDamage

    return ShardDamage().read(root, list_shards(root), 0, presence.observe)


class TestWhatIfPresence:
    """FR-033: every blockers record of a shard carries ``what_if``, and no
    other record does."""

    def _write(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _rows(self, *, what_if: bool | None) -> list[dict]:
        from effects.infrastructure.record_codec import record_to_dict

        rows = [record_to_dict(_resolution("run.0-L1.0")),
                record_to_dict(make_record(
                    RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.BLOCKERS,
                    record_id="run.0-L1.1", game_id="run.0-L1.0", what_if=what_if))]
        if what_if is None:
            rows[1].pop("what_if", None)
        return rows

    def _finding(self, tmp_path: Path, name: str = "what_if is on every blockers record"):
        presence = FieldPresence()
        findings = validate_corpus(_read_noting_damage(tmp_path, presence), presence=presence)
        return _named(findings, name)

    def test_a_shard_with_what_if_on_its_blockers_passes(self, tmp_path):
        self._write(tmp_path / "a.jsonl", self._rows(what_if=False))
        finding = self._finding(tmp_path)
        assert finding.ok and finding.measured.startswith("1 shard(s): 0 with"), finding.measured

    def test_a_blockers_record_without_what_if_fails(self, tmp_path):
        self._write(tmp_path / "a.jsonl", self._rows(what_if=None))
        finding = self._finding(tmp_path)
        assert not finding.ok
        assert "1 with what_if on some blockers records only" in finding.measured, finding.measured

    def test_a_record_without_random_seat_is_short(self, tmp_path):
        rows = self._rows(what_if=False)
        rows[0].pop("random_seat")
        self._write(tmp_path / "a.jsonl", rows)
        finding = self._finding(tmp_path, "gen-3 fields")
        assert not finding.ok and "record.random_seat 1" in finding.detail[0]

    def test_what_if_on_another_kind_is_refused_by_the_reader(self, tmp_path):
        """The record model rejects it outright, so it never reaches a tally."""
        rows = self._rows(what_if=True)
        rows[0]["what_if"] = True
        self._write(tmp_path / "a.jsonl", rows)
        with pytest.raises(ValueError, match="what_if"):
            list(read_window(tmp_path))


class TestGen3FieldPresence:
    """Root § 8.13 on the raw lines: what every gen-3 record owes, what a
    gen-3 attackers record no longer carries, and the curation member."""

    def _write(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _findings(self, tmp_path: Path, rows: list[dict]) -> Findings:
        self._write(tmp_path / "a.jsonl", rows)
        presence = FieldPresence()
        return validate_corpus(_read_noting_damage(tmp_path, presence), presence=presence)

    @staticmethod
    def _golden() -> list[dict]:
        from tests.unit.effects.builders import java_golden_records

        return [json.loads(json.dumps(raw)) for raw in java_golden_records()]

    def test_the_collector_s_records_owe_nothing_they_lack(self, tmp_path):
        findings = self._findings(tmp_path, self._golden())
        gen3 = _named(findings, "gen-3 fields")
        assert gen3.ok, gen3.detail
        assert gen3.measured.startswith(f"0/{len(self._golden())} gen-3 records")
        assert _named(findings, "what_if is on every blockers record").ok
        assert _named(findings, "curation").ok

    def test_an_effect_half_without_conditions_met_is_short(self, tmp_path):
        rows = self._golden()
        half = next(row for row in rows if row["kind"] == "resolution"
                    and row["moment"] == "resolution")
        del half["payload"]["conditions_met"]
        finding = _named(self._findings(tmp_path, rows), "gen-3 fields")
        assert not finding.ok
        assert "payload<resolution/resolution>.conditions_met 1" in finding.detail[0]
        assert finding.detail[1] == (
            f"{half['record_id']}: no payload<resolution/resolution>.conditions_met")

    def test_a_game_end_marker_without_its_withheld_count_is_short(self, tmp_path):
        """The marker says how many of its game's records the cap withheld."""
        rows = self._golden()
        marker = next(row for row in rows if row.get("subkind") == "game_end")
        del marker["payload"]["records_withheld"]
        finding = _named(self._findings(tmp_path, rows), "gen-3 fields")
        assert not finding.ok
        assert finding.detail[1] == (
            f"{marker['record_id']}: no payload<turn_action/game_end>.records_withheld")

    @pytest.mark.parametrize(("site", "edit"), [
        ("state.links", lambda row: row["state"].pop("links")),
        ("state.global.day_night", lambda row: row["state"]["global"].pop("day_night")),
        ("state.players[].turn_counts",
         lambda row: row["state"]["players"][0].pop("turn_counts")),
        ("state.players[].max_land_plays",
         lambda row: row["state"]["players"][0].pop("max_land_plays")),
        ("state.entities[].owner", lambda row: row["state"]["entities"][0].pop("owner")),
        ("state.refs.random", lambda row: row["state"]["refs"].pop("random")),
    ])
    def test_a_snapshot_member_missing_anywhere_is_named(self, tmp_path, site, edit):
        rows = self._golden()
        edit(rows[0])
        finding = _named(self._findings(tmp_path, rows), "gen-3 fields")
        assert not finding.ok and f"{site} 1" in finding.detail[0]

    @pytest.mark.parametrize(("site", "member"), [
        (site.value, member)
        for site in (Site.STATE, Site.GLOBAL, Site.PLAYER, Site.ENTITY, Site.REFS)
        for member in ALWAYS[site]
    ])
    def test_every_owed_snapshot_member_missing_is_named(self, tmp_path, site, member):
        """Each member of every owed list, not one per site: dropped from the
        first golden record that carries it."""
        rows = self._golden()
        holder = {
            "state": lambda row: [row["state"]],
            "state.global": lambda row: [row["state"]["global"]],
            "state.players[]": lambda row: row["state"]["players"],
            "state.entities[]": lambda row: row["state"]["entities"],
            "state.refs": lambda row: [row["state"]["refs"]],
        }[site]
        target = next(part for row in rows for part in holder(row) if member in part)
        del target[member]
        finding = _named(self._findings(tmp_path, rows), "gen-3 fields")
        assert not finding.ok and f"{site}.{member} 1" in finding.detail[0], finding.detail

    def test_a_static_line_without_active_is_short(self, tmp_path):
        rows = self._golden()
        line = next(line for row in rows for entity in row["state"]["entities"]
                    for line in entity["lines"] if line["kind"] == "static")
        del line["active"]
        finding = _named(self._findings(tmp_path, rows), "gen-3 fields")
        assert "state.entities[].lines[static].active 1" in finding.detail[0]

    @pytest.mark.parametrize(("subkind", "member"), [
        ("attackers", "legal_defenders"), ("blockers", "must_block")])
    def test_a_legality_payload_member_missing_is_named(self, tmp_path, subkind, member):
        rows = self._golden()
        row = next(row for row in rows if row.get("subkind") == subkind)
        del row["payload"][member]
        finding = _named(self._findings(tmp_path, rows), "gen-3 fields")
        assert f"payload<playability/{subkind}>.{member} 1" in finding.detail[0]

    def test_a_candidate_without_timing_or_failed_checks_is_short(self, tmp_path):
        rows = self._golden()
        candidate = next(row for row in rows
                         if row.get("subkind") == "decision")["payload"]["candidates"][0]
        del candidate["verdict"]["timing"], candidate["failed_checks"]
        finding = _named(self._findings(tmp_path, rows), "gen-3 fields")
        decision = "payload<playability/decision>"
        assert f"{decision}.candidates[].verdict.timing 1" in finding.detail[0]
        assert f"{decision}.candidates[].failed_checks 1" in finding.detail[0]

    def test_a_candidate_without_land_play_is_short(self, tmp_path):
        rows = self._golden()
        candidate = next(row for row in rows
                         if row.get("subkind") == "decision")["payload"]["candidates"][0]
        del candidate["land_play"]
        finding = _named(self._findings(tmp_path, rows), "gen-3 fields")
        assert not finding.ok and (
            "payload<playability/decision>.candidates[].land_play 1" in finding.detail[0])

    def test_an_unlimited_land_play_count_is_carried_not_missing(self, tmp_path):
        """Fastbond's null is the member written, not the member absent."""
        rows = self._golden()
        rows[0]["state"]["players"][0]["max_land_plays"] = None
        assert _named(self._findings(tmp_path, rows), "gen-3 fields").ok

    def test_an_attackers_record_carrying_what_if_fails(self, tmp_path):
        rows = self._golden()
        next(row for row in rows if row.get("subkind") == "attackers")["what_if"] = False
        finding = _named(self._findings(tmp_path, rows), "what_if is on every blockers record")
        assert not finding.ok and "another kind than blockers" in " ".join(finding.detail)

    def test_a_blockers_record_without_what_if_fails(self, tmp_path):
        rows = self._golden()
        next(row for row in rows if row.get("subkind") == "blockers").pop("what_if")
        assert not _named(self._findings(tmp_path, rows),
                          "what_if is on every blockers record").ok

    def test_a_curation_member_is_refused(self, tmp_path):
        rows = self._golden()
        rows[3]["curation"] = {"natural_share": 1, "copies": 1, "drawn_rule": None,
                               "drawn_rules": [], "condition_entities": [],
                               "outcome_entities": [], "class_weight": 1.0}
        finding = _named(self._findings(tmp_path, rows), "curation")
        assert not finding.ok and finding.measured.startswith("1 record(s) in 1 of 1")

    @pytest.mark.parametrize(("dropped", "named"), [
        (("generation",), "gen-2"),
        (("generation", "random_seat"), "gen-1"),
    ])
    def test_a_record_of_an_older_generation_is_refused_naming_it(
        self, tmp_path, dropped, named,
    ):
        from effects.domain.shard_generation import OlderGenerationError

        rows = self._golden()
        for member in dropped:
            del rows[1][member]
        with pytest.raises(OlderGenerationError,
                           match=rf"a\.jsonl, line 2: the record is {named} "):
            self._findings(tmp_path, rows)
