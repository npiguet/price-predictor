"""The cost half's ``costs`` members beyond mana, tapped, life, sacrificed,
discarded and exiled (contract ``record-schema-delta.md``): each
decodes into its field and writes back where the collector wrote it.

Starts from a Java golden's cost half, edited here.
"""

from __future__ import annotations

import json

from effects.domain.records import Costs
from effects.infrastructure.record_codec import format_record_line, record_from_dict
from tests.unit.effects.builders import java_golden_records

_NEW_MEMBERS = (
    "returned", "revealed", "exerted", "untapped", "convoked", "delved", "counters",
)


def _cost_half() -> dict:
    return next(
        raw for raw in java_golden_records()
        if raw.get("moment") == "activation" and "costs" in raw["payload"]
    )


def _with_costs(**members) -> dict:
    raw = _cost_half()
    return {**raw, "payload": {**raw["payload"], "costs": {**raw["payload"]["costs"], **members}}}


class TestTheCostMembers:
    def test_the_golden_carries_every_new_member_and_decodes_it_empty(self):
        raw = _cost_half()
        assert all(member in raw["payload"]["costs"] for member in _NEW_MEMBERS)
        costs = record_from_dict(raw).payload.costs
        assert (costs.counters, costs.returned, costs.delved) == ({}, (), ())

    def test_a_loyalty_cost_reads_as_a_signed_count_per_type(self):
        raw = _with_costs(counters={"LOYALTY": -2, "P1P1": 1})
        assert record_from_dict(raw).payload.costs.counters == {"LOYALTY": -2, "P1P1": 1}

    def test_each_entity_channel_reads_into_its_own_field(self):
        raw = _with_costs(
            returned=["E1"], revealed=["E2"], exerted=["E3"],
            untapped=["E4"], convoked=["E5", "E6"], delved=["E7"],
        )
        costs = record_from_dict(raw).payload.costs
        assert (costs.returned, costs.revealed, costs.exerted) == (("E1",), ("E2",), ("E3",))
        assert (costs.untapped, costs.convoked, costs.delved) == (
            ("E4",), ("E5", "E6"), ("E7",))
        assert costs.tapped == tuple(raw["payload"]["costs"]["tapped"])

    def test_the_members_write_back_as_the_collector_wrote_them(self):
        raw = _with_costs(counters={"LOYALTY": 1}, convoked=["E5"])
        written = json.loads(format_record_line(record_from_dict(raw)))["payload"]["costs"]
        assert written == raw["payload"]["costs"]
        assert list(written) == list(raw["payload"]["costs"])

    def test_costs_built_by_hand_default_every_new_member_to_empty(self):
        costs = Costs(life=2)
        assert (costs.counters, costs.returned, costs.untapped) == ({}, (), ())
