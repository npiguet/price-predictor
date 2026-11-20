"""Record codec: the pinned wire form and the interned provenance keys."""

from __future__ import annotations

from effects.domain.provenance import ProvenanceKey
from effects.infrastructure.record_codec import (
    format_record_line,
    parse_record_line,
    record_from_dict,
)
from effects.infrastructure.shard_io import (
    read_records,
    read_shard,
)
from tests.unit.effects.builders import (
    GEN3_RECORDS,
)
from tests.unit.effects.infrastructure.record_codec_support import (
    ALL_KINDS,
    _record,
)


class TestInternedProvenanceKeys:
    """A shard repeats the same few hundred keys tens of thousands of times."""

    def test_an_equal_key_parses_to_the_same_object(self, tmp_path):
        directory = tmp_path / "records"
        directory.mkdir()
        (directory / GEN3_RECORDS.name).write_bytes(GEN3_RECORDS.read_bytes())
        seen: dict[ProvenanceKey, ProvenanceKey] = {}
        repeats = 0
        for record in read_records(directory):
            for entity in record.state.entities:
                for key in entity.printed:
                    first = seen.setdefault(key, key)
                    assert first is key
                    repeats += 1
        assert repeats > len(seen) > 1

    def test_an_option_keeps_the_mode_key_apart_from_its_root(self):
        from effects.infrastructure.record_codec import _keys_from_json

        root = {"script_file": "cardsfolder/c/charm.txt", "face": 0,
                "trait_kind": "spell", "index_within_kind": 0}
        (a, b, c) = _keys_from_json([root, {**root, "option": 1}, dict(root)])
        assert a is c
        assert a != b and b.option == 1 and b.root == a

class TestTheWireFormIsPinned:
    """The bytes a record serializes to, pinned against a digest.

    A round trip alone cannot catch a codec change that alters encoder and
    decoder together — a renamed key, a reordered field — and the corpus is
    append-only, so a shard written under the changed form would no longer
    read the way the shards beside it do. The digest was taken when the codec
    came to write gen 3 alone.
    """

    @staticmethod
    def _digest(records) -> str:
        import hashlib

        digest = hashlib.sha256()
        for record in records:
            digest.update(format_record_line(record).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def test_every_kind_serializes_to_the_pinned_bytes(self):
        records = [_record(**ALL_KINDS[name]) for name in sorted(ALL_KINDS)]
        assert self._digest(records) == (
            "82b26b422aba06e10c63df317907335dc03e234fa16a458bf3169a4be9a77704"
        )

    def test_the_fixture_round_trips_through_its_own_wire_form(self):
        for record in read_shard(GEN3_RECORDS):
            line = format_record_line(record)
            again = record_from_dict(parse_record_line(line))
            assert again == record
            assert format_record_line(again) == line
