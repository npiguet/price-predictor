"""Record codec: the gen-3 envelope members and the refusal of a record
of an older or unknown generation."""

from __future__ import annotations

import json

import pytest

from effects.domain.records import (
    RecordKind,
    TurnActionSubkind,
)
from effects.infrastructure.record_codec import (
    record_from_dict,
    record_to_dict,
)
from effects.infrastructure.shard_io import (
    read_shard,
)
from tests.unit.effects.builders import (
    gen3_fixture_records,
    make_record,
)
from tests.unit.effects.infrastructure.record_codec_support import (
    ALL_KINDS,
    _record,
)


class TestTheRandomSeatAndWhatIf:
    """``random_seat`` and ``what_if`` (FR-030, FR-031, FR-033)."""

    def test_what_if_round_trips_on_a_legality_record(self):
        for value in (True, False):
            record = _record(**ALL_KINDS["playability-blockers"], what_if=value)
            data = record_to_dict(record)
            assert data["what_if"] is value
            assert record_from_dict(data) == record

    def test_what_if_is_written_only_when_set(self):
        data = record_to_dict(_record(**ALL_KINDS["playability-blockers"]))
        assert "what_if" not in data
        assert data["random_seat"] is False

    def test_random_seat_follows_synthetic_in_the_envelope(self):
        keys = list(record_to_dict(_record(**ALL_KINDS["playability-blockers"],
                                           what_if=False)))
        assert keys.index("random_seat") == keys.index("synthetic") + 1
        assert keys.index("what_if") == keys.index("random_seat") + 1


class TestGen3EnvelopeFields:
    """``generation``, ``aimed_at``, ``directed`` and the ``turn_action`` kind (FR-018)."""

    def test_generation_is_first_after_the_record_id(self):
        keys = list(record_to_dict(_record()))
        assert keys[:2] == ["record_id", "generation"]

    def test_aimed_at_and_directed_are_written_only_when_set(self):
        data = record_to_dict(_record())
        assert not {"aimed_at", "directed"} & data.keys()

    def test_a_gen3_envelope_round_trips(self):
        from effects.domain.fork_operation import ForkOperation
        from effects.domain.records import DirectedTag

        record = make_record(
            RecordKind.COMBAT, fork=True, aimed_at=("704.5f",),
            directed=DirectedTag("tougher", ForkOperation.PT))
        data = record_to_dict(record)
        assert data["aimed_at"] == ["704.5f"]
        assert data["directed"] == {"variant": "tougher", "operation": "pt"}
        assert record_from_dict(json.loads(json.dumps(data))) == record

    def test_a_curated_record_s_curation_round_trips(self):
        """Written by ``build-corpus`` alone (research T1); absent elsewhere."""
        from effects.domain.records import Curation

        assert "curation" not in record_to_dict(_record())
        record = _record(curation=Curation(
            natural_share=1, copies=3, drawn_rule="704.5f",
            drawn_rules=("704.5f", "704.5g"), condition_entities=("E1",),
            outcome_entities=("E2",), class_weight=2.5))
        data = record_to_dict(record)
        assert data["curation"] == {
            "natural_share": 1, "copies": 3, "drawn_rule": "704.5f",
            "drawn_rules": ["704.5f", "704.5g"], "condition_entities": ["E1"],
            "outcome_entities": ["E2"], "class_weight": 2.5}
        assert record_from_dict(json.loads(json.dumps(data))) == record

    def test_an_empty_aimed_at_is_kept_apart_from_none(self):
        record = _record(aimed_at=())
        data = record_to_dict(record)
        assert data["aimed_at"] == []
        assert record_from_dict(data).aimed_at == ()

    @pytest.mark.parametrize("subkind", list(TurnActionSubkind))
    def test_a_turn_action_record_round_trips(self, subkind):
        record = make_record(RecordKind.TURN_ACTION, subkind=subkind)
        data = record_to_dict(record)
        assert (data["kind"], data["subkind"]) == ("turn_action", subkind.value)
        assert record_from_dict(data) == record

    @pytest.mark.parametrize("value", [4, "3", True, None])
    def test_a_record_naming_no_known_generation_is_refused(self, value):
        """``generation`` decides how several fields read, so a record whose
        value names none is refused rather than read as another generation."""
        from effects.domain.shard_generation import UnknownGenerationError

        data = record_to_dict(_record()) | {"generation": value}
        with pytest.raises(UnknownGenerationError):
            record_from_dict(data)

    def test_an_unknown_generation_in_a_shard_names_the_shard_and_line(self, tmp_path):
        from effects.domain.shard_generation import UnknownGenerationError
        from effects.infrastructure.shard_io import require_generation

        path = tmp_path / "g4.jsonl"
        path.write_text(json.dumps(record_to_dict(_record()) | {"generation": 4}) + "\n",
                        encoding="utf-8")
        for read in (lambda: list(read_shard(path)), lambda: require_generation([path])):
            with pytest.raises(UnknownGenerationError, match=r"g4\.jsonl, line 1: .*4"):
                read()


class TestAnOlderRecordIsRefused:
    """A record of an older generation, with no ``generation`` member, is
    refused where it is decoded, naming its generation (FR-033). Built from a
    real gen-3 record less what the older generation did not write."""

    @staticmethod
    def _older(*dropped: str) -> dict:
        raw = gen3_fixture_records()[0]
        for member in dropped:
            del raw[member]
        return raw

    @pytest.mark.parametrize(("dropped", "named"), [
        (("generation",), "gen-2"),
        (("generation", "random_seat"), "gen-1"),
    ])
    def test_its_decode_is_refused_naming_its_generation(self, dropped, named):
        from effects.domain.shard_generation import OlderGenerationError

        with pytest.raises(OlderGenerationError, match=f"the record is {named} "):
            record_from_dict(self._older(*dropped))

    @pytest.mark.parametrize(("dropped", "named"), [
        (("generation",), "gen-2"),
        (("generation", "random_seat"), "gen-1"),
    ])
    def test_in_a_shard_it_names_the_shard_and_line(self, tmp_path, dropped, named):
        from effects.domain.shard_generation import OlderGenerationError
        from effects.infrastructure.shard_io import require_generation

        path = tmp_path / "old.jsonl"
        path.write_text(json.dumps(self._older(*dropped)) + "\n", encoding="utf-8")
        for read in (lambda: list(read_shard(path)), lambda: require_generation([path])):
            with pytest.raises(OlderGenerationError,
                               match=rf"old\.jsonl, line 1: the record is {named} "):
                read()
