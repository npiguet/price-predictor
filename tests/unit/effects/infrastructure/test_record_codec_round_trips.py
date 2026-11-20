"""Record codec round-trips (T027).

The corpus cannot be rebuilt cheaply, so a serializer bug is unrecoverable
rather than inconvenient. Every record kind round-trips here, including the
stage-two-to-four kinds no collector writes yet — the schema is frozen before
collection precisely so those can be checked now.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from effects.domain.records import (
    ActivationPayload,
    Costs,
    Moment,
    ResolutionOutcome,
)
from effects.domain.state_snapshot import (
    EntityState,
)
from effects.infrastructure.record_codec import (
    format_record_line,
    record_from_dict,
    record_to_dict,
)
from effects.infrastructure.shard_io import (
    read_shard,
    write_shard,
)
from tests.unit.effects.infrastructure.record_codec_support import (
    ALL_KINDS,
    _record,
    _rich_snapshot,
)


class TestRoundTrip:
    @pytest.mark.parametrize("name", sorted(ALL_KINDS))
    def test_every_record_kind_round_trips(self, name: str):
        record = _record(**ALL_KINDS[name])
        assert record_from_dict(record_to_dict(record)) == record

    def test_a_line_holds_no_newline_of_its_own(self):
        assert "\n" not in format_record_line(_record())

    def test_a_non_ascii_name_is_written_as_utf8_and_reads_back(self, tmp_path):
        """The writer is ``orjson``, which does not escape; the reader decodes
        a ``\\u`` escape and the raw character alike."""
        record = _record(state=replace(_rich_snapshot(), entities=(
            EntityState(id="E1", name="Lim-Dûl's Vault", zone="hand", controller="P0",
                        lines=()),
        )))
        line = format_record_line(record)
        assert "Lim-Dûl's Vault" in line
        assert record_from_dict(json.loads(line)) == record
        shard = tmp_path / "run.0-a.jsonl.gz"
        write_shard(shard, [record])
        assert list(read_shard(shard)) == [record]

    def test_collected_tiers_are_written_explicitly(self):
        """Without them, an empty entity list and an uncollected tier look alike."""
        data = record_to_dict(_record())
        assert data["state"]["tiers"] == [1, 2]

class TestRecordMoments:
    """The record-moment payloads (root spec § 8.14, FR-042a, FR-042b) keep
    their bytes through a read and a write."""

    @staticmethod
    def _payload(record_payload) -> dict:
        return record_to_dict(_record(**record_payload))["payload"]

    def test_a_gen3_cost_half_is_its_events_then_its_costs(self):
        data = self._payload({
            "moment": Moment.ACTIVATION,
            "payload": ActivationPayload(costs=Costs(), events=()),
        })
        assert list(data) == ["events", "costs"]
        assert record_from_dict(record_to_dict(_record(
            moment=Moment.ACTIVATION, payload=ActivationPayload(costs=Costs(), events=()),
        ))).payload == ActivationPayload(costs=Costs(), events=())

    def test_an_effect_halfs_outcome_and_a_written_null_survive(self):
        for outcome in (ResolutionOutcome.FIZZLED, None):
            data = record_to_dict(_record())
            data["payload"]["conditions_met"] = []
            data["payload"]["outcome"] = outcome.value if outcome else None
            restored = record_from_dict(data)
            assert restored.payload.outcome == outcome
            assert record_to_dict(restored)["payload"] == data["payload"]

class TestForwardCompatibility:
    def test_an_unknown_envelope_field_is_preserved_on_read(self):
        data = record_to_dict(_record())
        data["collector_build"] = "2027.1"
        restored = record_from_dict(data)
        assert restored.extra_fields == {"collector_build": "2027.1"}

    def test_an_unknown_field_survives_a_rewrite(self):
        data = record_to_dict(_record())
        data["collector_build"] = "2027.1"
        assert record_to_dict(record_from_dict(data))["collector_build"] == "2027.1"

    def test_an_unknown_field_never_shadows_a_known_one(self):
        """A kept field spelled like a known one is not written over it, and on
        a read a known key always goes to its field, never into the extras."""
        record = replace(_record(), extra_fields={"run_id": "shadow",
                                                  "collector_build": "2027.1"})
        data = record_to_dict(record)
        assert (data["run_id"], data["collector_build"]) == (record.run_id, "2027.1")
        restored = record_from_dict(data)
        assert restored.run_id == record.run_id
        assert restored.extra_fields == {"collector_build": "2027.1"}

    def test_ability_unresolved_round_trips(self):
        record = _record(ability=(), ability_unresolved="engine_effect")
        assert record_from_dict(record_to_dict(record)) == record
