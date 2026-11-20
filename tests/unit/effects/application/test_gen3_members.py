"""The one table of the members a gen-3 record writes, and its readers.

The presence check and ``field-coverage`` each kept a hand list of the gen-3
members, and the two had drifted apart: neither held a contribution's
``abilities`` or a combat payload's ``blocks``, so a writer dropping one
passed ``validate-corpus``. Both lists now derive from
:mod:`effects.domain.gen3_members`; the lists as they stood are pinned here
so the table is shown to cover them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from effects.application.corpus_validation.presence import FieldPresence
from effects.application.field_coverage import field_coverage, report_lines
from effects.domain.gen3_members import ALWAYS, MEMBERS, WALK_PATHS, Site, site_blocks
from tests.unit.effects.builders import gen3_fixture_records, java_golden_records

#: The presence check's owed lists at 6ea9ab68, by the site each now sits on.
_OLD_OWED: tuple[tuple[Site, str], ...] = (
    (Site.RECORD, "random_seat"), (Site.STATE, "links"),
    *((Site.GLOBAL, name) for name in (
        "day_night", "floating_effects", "delayed_triggers", "dungeons")),
    *((Site.PLAYER, name) for name in (
        "max_land_plays", "speed", "city_blessing", "rad", "monarch", "initiative",
        "turn_counts")),
    *((Site.ENTITY, name) for name in (
        "owner", "face_state", "mana_cost", "base_pt", "pt_boosts", "printed_loyalty",
        "printed_defense", "real_face", "designations", "x_paid", "chosen", "costs_paid",
        "history", "order_rank", "lines", "role")),
    (Site.REFS, "random"), (Site.STATIC_LINE, "active"),
    *((Site.ATTACKERS, name) for name in ("anchor_attacker", "legal_defenders", "must_attack")),
    (Site.BLOCKERS, "must_block"),
    (Site.CANDIDATE, "failed_checks"), (Site.CANDIDATE, "land_play"), (Site.VERDICT, "timing"),
    (Site.EFFECT_HALF, "conditions_met"), (Site.GAME_END, "records_withheld"),
)

_ENTITY = "record.state.entities[]"
_DECISION = "record.payload<PlayabilityDecisionPayload>.candidates[]"
_ATTACKERS = "record.payload<PlayabilityAttackersPayload>"
#: ``field_coverage.GEN3_FIELDS`` at 6ea9ab68.
_OLD_GEN3_FIELDS: tuple[str, ...] = (
    "record.generation", "record.aimed_at", "record.directed", "record.copy_dropped",
    *(f"{_ENTITY}.{name}" for name in (
        "owner", "face_state", "mana_cost", "printed_loyalty", "printed_defense",
        "real_face", "designations", "x_paid", "chosen", "costs_paid", "history",
        "order_rank", "role")),
    f"{_ENTITY}.lines[].active", f"{_ENTITY}.lines[].affected",
    *(f"record.state.players[].{name}" for name in (
        "max_land_plays", "land_plays_unlimited", "speed", "city_blessing", "rad",
        "monarch", "initiative", "turn_counts")),
    *(f"record.state.global_.{name}" for name in (
        "day_night", "floating_effects", "delayed_triggers", "dungeons")),
    "record.state.refs.random", "record.state.links",
    "record.payload<ResolutionPayload>.conditions_met",
    "record.payload<ResolutionPayload>.outcome", "record.payload<ActivationPayload>.events",
    f"{_DECISION}.timing", f"{_DECISION}.failed_checks", f"{_DECISION}.land_play",
    *(f"{_ATTACKERS}.{name}" for name in (
        "anchor_attacker", "legal_defenders", "must_attack", "tapped_by_attack")),
    "record.payload<PlayabilityBlockersPayload>.must_block",
    "record.payload<TurnActionPayload>.events",
    "record.payload<TurnActionPayload>.records_withheld",
    *(f"record.payload<RewritePayload>.{name}" for name in (
        "incoming_unknown", "buffered", "buffered_ran", "buffered_events")),
    "record.payload<PlayabilityBlockersPayload>.forbidden[].required_elsewhere",
    *(f"record.payload<{payload}>.forbidden[].{name}"
      for payload in ("PlayabilityAttackersPayload", "PlayabilityBlockersPayload")
      for name in ("responsible_own", "responsible_unresolved")),
)

#: A line of one kind is a line too: its members are the line's and its own.
_REFINES = {Site.STATIC_LINE: Site.LINE, Site.KEYWORD_LINE: Site.LINE}


def _decoded(rows: list[dict]):
    from effects.infrastructure.record_codec import record_from_dict

    return [record_from_dict(row) for row in rows]


class TestTheTable:
    def test_it_owes_every_member_the_presence_check_owed(self):
        always = {(site, name) for site, names in ALWAYS.items() for name in names}
        assert [pair for pair in _OLD_OWED if pair not in always] == []

    def test_it_rows_every_path_field_coverage_rowed(self):
        assert [path for path in _OLD_GEN3_FIELDS if path not in WALK_PATHS] == []

    def test_no_member_is_listed_twice_on_one_site(self):
        rows = [(row.site, row.name) for row in MEMBERS]
        assert len(rows) == len(set(rows))

    @pytest.mark.parametrize("source", [gen3_fixture_records, java_golden_records])
    def test_every_always_member_is_on_every_instance_the_collector_wrote(self, source):
        missing = defaultdict(int)
        for row in source():
            for site, block in site_blocks(row):
                for name in ALWAYS[site]:
                    missing[f"{site.value}.{name}"] += name not in block
        assert {site: count for site, count in missing.items() if count} == {}

    @pytest.mark.parametrize("source", [gen3_fixture_records, java_golden_records])
    def test_every_member_the_collector_wrote_has_a_row(self, source):
        tabled = defaultdict(set)
        for row in MEMBERS:
            tabled[row.site].add(row.name)
        untabled = set()
        for row in source():
            for site, block in site_blocks(row):
                known = tabled[site] | tabled[_REFINES.get(site, site)]
                untabled |= {f"{site.value}.{name}" for name in block if name not in known}
        assert untabled == set()

    def test_the_fixture_holds_an_instance_of_every_site(self):
        seen = {site for row in gen3_fixture_records() for site, _ in site_blocks(row)}
        assert set(Site) - seen == set()

    def test_the_python_writer_writes_every_always_member(self):
        """A curated shard is written by the codec, not the collector: it
        owes the same members."""
        from effects.infrastructure.record_codec import record_to_dict

        rows = [record_to_dict(record) for record in _decoded(gen3_fixture_records())]
        missing = {f"{site.value}.{name}" for row in rows for site, block in site_blocks(row)
                   for name in ALWAYS[site] if name not in block}
        assert missing == set()


def _write(directory: Path, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "a.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _gen3_finding(tmp_path: Path, rows: list[dict]):
    from effects.application.validate_corpus import ShardDamage, validate_corpus
    from effects.infrastructure.shard_io import list_shards

    _write(tmp_path, rows)
    presence = FieldPresence()
    records = ShardDamage().read(tmp_path, list_shards(tmp_path), 0, presence.observe)
    findings = validate_corpus(records, presence=presence)
    return next(finding for finding in findings if finding.name.startswith("gen-3 fields"))


def _drop_from_first(rows: list[dict], site: Site, member: str) -> str:
    """Drop ``member`` from the first instance of ``site`` that carries it,
    in a real fixture record; the record's id."""
    for row in rows:
        for found, block in site_blocks(row):
            if found is site and member in block:
                del block[member]
                return row["record_id"]
    raise AssertionError(f"no {site.value} carries {member}")


class TestThePresenceCheck:
    def test_the_fixture_owes_nothing_it_lacks(self, tmp_path):
        assert _gen3_finding(tmp_path, gen3_fixture_records()).ok

    @pytest.mark.parametrize(("site", "member"), [
        (Site.RECORD, "fork"),
        (Site.COMBAT, "blocks"),
        (Site.CONTRIBUTION, "abilities"),
        (Site.COMBAT, "assignment_choices"),
        (Site.CONTRIBUTION, "hidden_keywords"),
        (Site.RECORD, "interventional"),
        (Site.EVENT, "attributed_to"),
        (Site.CANDIDATE, "cost_after_adjustment"),
    ])
    def test_a_member_dropped_from_one_instance_is_named(self, tmp_path, site, member):
        rows = gen3_fixture_records()
        record_id = _drop_from_first(rows, site, member)
        finding = _gen3_finding(tmp_path, rows)
        assert not finding.ok
        assert f"{site.value}.{member} 1" in finding.detail[0], finding.detail
        assert f"{record_id}: no {site.value}.{member}" in finding.detail

    @pytest.mark.parametrize("member", ["fork", "random_seat", "mode"])
    def test_a_whole_shard_without_an_envelope_member_fails(self, tmp_path, member):
        rows = gen3_fixture_records()
        for row in rows:
            del row[member]
        finding = _gen3_finding(tmp_path, rows)
        assert not finding.ok and f"record.{member}" in finding.detail[0], finding.detail


class TestFieldCoverageTags:
    def test_a_member_constant_by_construction_is_tagged_fixed_not_new(self):
        lines = report_lines(field_coverage(_decoded(gen3_fixture_records())))
        tagged = {line.split("]", 1)[1].split()[0]: line[3:9] for line in lines
                  if line.startswith("  [") and "] record." in line}
        assert tagged["record.generation"] == "fixed "
        assert tagged["record.state.tiers"] == "fixed "
        assert "NEW" not in tagged.get("record.variant_of", "")
