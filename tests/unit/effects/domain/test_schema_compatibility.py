"""The four rules that let later stages widen the corpus (T023).

There is no ``schema_version`` field — the root spec defines none — so these
tests are the enforcement, not a version stamp. plan.md's Principle III claim
rests on exactly that, which is why they check the rules rather than the code
that happens to satisfy them today.

The rules, from ``contracts/record-schema.md``:

1. A field may be added; existing fields may not change meaning or type.
2. A new ``kind`` or ``subkind`` value may be introduced; existing values may
   not be repurposed.
3. Snapshot tiers are additive; absence means uncollected.
4. Nothing listed as collection metadata may become a model input.
"""

from __future__ import annotations

from dataclasses import fields

from effects.domain.records import (
    COLLECTION_METADATA_FIELDS,
    PAYLOAD_TYPES,
    ActivationPayload,
    CollectionMode,
    EffectRecord,
    Moment,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
)
from effects.domain.state_snapshot import STAGE_ONE_TIERS, InclusionTier, StateSnapshot

# The envelope as frozen before stage-one collection. A field may be added to
# this set by a later stage; one may never be removed, renamed, or retyped.
_FROZEN_ENVELOPE: dict[str, str] = {
    "record_id": "str",
    "run_id": "str",
    "timestamp": "str",
    "game_id": "str",
    "kind": "RecordKind",
    "actor_player": "str",
    "state": "StateSnapshot",
    "payload": "Payload",
    "mode": "CollectionMode",
    "moment": "Moment | None",
    "subkind": "PlayabilitySubkind | None",
    "link_id": "str | None",
    "mirror_of": "str | None",
    "variant_of": "str | None",
    "interventional": "bool",
    "fork": "bool",
    "synthetic": "bool",
    "ability": "tuple[ProvenanceKey, ...] | None",
    "extra_fields": "dict",
}

# The enum values reachable from stage one onward. A later stage may append;
# nothing here may be reused for a different meaning.
_FROZEN_KINDS = {
    "resolution", "rewrite", "continuous", "combat", "trigger", "playability",
}
_FROZEN_MOMENTS = {"activation", "resolution"}
_FROZEN_SUBKINDS = {"decision", "attackers", "blockers"}
_FROZEN_MODES = {"patched", "degraded"}
_FROZEN_OUTCOMES = {
    "resolved", "fizzled", "partially_fizzled", "declined", "countered",
}


class TestRuleOneFieldsMayBeAddedNotRepurposed:
    def test_every_frozen_envelope_field_still_exists_with_its_type(self):
        actual = {f.name: f.type for f in fields(EffectRecord)}
        for name, declared in _FROZEN_ENVELOPE.items():
            assert name in actual, f"envelope field {name!r} was removed"
            assert actual[name] == declared, (
                f"envelope field {name!r} changed type from {declared!r} to "
                f"{actual[name]!r}; rule 1 allows additions, not redefinitions"
            )

    def test_an_added_field_is_the_only_permitted_difference(self):
        added = set(f.name for f in fields(EffectRecord)) - set(_FROZEN_ENVELOPE)
        assert not added, (
            "a field was added to the envelope; that is allowed, but this list "
            f"must be updated in the same change so the next one is caught: {added}"
        )


class TestRuleTwoEnumValuesMayBeAddedNotReused:
    def test_the_frozen_kinds_all_survive(self):
        assert _FROZEN_KINDS <= {k.value for k in RecordKind}

    def test_the_frozen_moments_and_subkinds_all_survive(self):
        assert _FROZEN_MOMENTS <= {m.value for m in Moment}
        assert _FROZEN_SUBKINDS <= {s.value for s in PlayabilitySubkind}

    def test_the_frozen_modes_and_outcomes_all_survive(self):
        assert _FROZEN_MODES <= {m.value for m in CollectionMode}
        assert _FROZEN_OUTCOMES <= {o.value for o in ResolutionOutcome}

    def test_every_kind_and_discriminator_pair_has_a_payload_type(self):
        expected = set()
        for kind in RecordKind:
            if kind is RecordKind.RESOLUTION:
                expected |= {(kind, m) for m in Moment}
            elif kind is RecordKind.PLAYABILITY:
                expected |= {(kind, s) for s in PlayabilitySubkind}
            else:
                expected.add((kind, None))
        assert set(PAYLOAD_TYPES) == expected, (
            "a new kind or subkind was introduced without a payload type; rule 2 "
            "permits the value, but the schema has to say what it carries"
        )

    def test_an_existing_value_keeps_its_payload_type(self):
        assert PAYLOAD_TYPES[(RecordKind.RESOLUTION, Moment.ACTIVATION)] is (
            ActivationPayload
        )


class TestRuleThreeTiersAreAdditive:
    def test_the_tiers_are_ordered_by_the_stage_that_adds_them(self):
        assert [t.value for t in InclusionTier] == [1, 2, 3, 4]

    def test_stage_one_collects_the_first_two_tiers(self):
        assert STAGE_ONE_TIERS == {InclusionTier.REFERENCED, InclusionTier.CORE}

    def test_a_later_stage_only_adds_tiers(self, snapshot):
        later = StateSnapshot(
            global_=snapshot.global_,
            players=snapshot.players,
            entities=snapshot.entities,
            tiers=STAGE_ONE_TIERS | {InclusionTier.UNREFERENCED_STACK},
        )
        assert STAGE_ONE_TIERS <= later.tiers

    def test_an_absent_tier_reads_as_uncollected_not_empty(self, snapshot):
        assert snapshot.entities == ()
        assert not snapshot.collected(InclusionTier.UNREFERENCED_STACK)
        assert snapshot.collected(InclusionTier.CORE)


class TestRuleFourMetadataNeverBecomesAModelInput:
    def test_the_metadata_set_is_the_one_the_contract_names(self):
        assert COLLECTION_METADATA_FIELDS == {
            "mode", "interventional", "fork", "synthetic",
        }

    def test_every_metadata_field_is_a_real_envelope_field(self):
        assert COLLECTION_METADATA_FIELDS <= {f.name for f in fields(EffectRecord)}

    def test_the_projection_excludes_them_for_every_kind(self, make_record):
        for kind in (RecordKind.RESOLUTION, RecordKind.COMBAT):
            record = make_record(kind=kind)
            leaked = COLLECTION_METADATA_FIELDS & set(record.model_input_fields())
            assert not leaked, f"{kind.value} leaks {sorted(leaked)} to the model"
