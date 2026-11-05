"""The four rules that let later stages widen the corpus (T023), pinned to
the gen-3 schema (T075, FR-019).

The rules hold within a generation: a record says its own (``generation``
on gen-3, ``random_seat`` on gen-2, neither on gen-1), and gen-3 may change
or drop a gen-2 field. These tests pin the gen-3 schema — the envelope, the
kinds and the card slot — so the next change within gen-3 is caught, and they
check the rules rather than the code that happens to satisfy them today.

The rules, from ``contracts/record-schema.md`` and feature 025's
``contracts/record-schema-delta.md``:

1. A field may be added; existing fields may not change meaning or type.
2. A new ``kind`` or ``subkind`` value may be introduced; existing values may
   not be repurposed.
3. Snapshot tiers are additive; absence means uncollected.
4. Nothing listed as collection metadata may become a model input.

What gen-3 changed relative to gen-2 is listed in ``_REMOVED_IN_GEN3`` and
``_REDEFINED_IN_GEN2`` beside the pins it would otherwise break.
"""

from __future__ import annotations

import random
from dataclasses import fields, replace

import pytest

from effects.domain.effect_head_input import SurfaceBuilder
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    PAYLOAD_TYPES,
    ActivationPayload,
    CollectionMode,
    Curation,
    EffectRecord,
    Moment,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    TurnActionSubkind,
    UnresolvedReason,
)
from effects.domain.state_snapshot import (
    CardLine,
    EntityState,
    InclusionTier,
    LineKind,
    PlayerState,
    StateSnapshot,
)
from tests.unit.effects.builders import (
    java_golden_entities,
    make_entity,
    make_record,
    make_snapshot,
)

# The envelope as gen-3 freezes it. A field may be added to this set within
# gen-3; one may never be removed, renamed, or retyped.
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
    # Widened by gen-3 to the turn-action subkinds; every gen-2 value keeps its meaning.
    "subkind": "PlayabilitySubkind | TurnActionSubkind | None",
    "link_id": "str | None",
    "mirror_of": "str | None",
    "variant_of": "str | None",
    "interventional": "bool",
    "fork": "bool",
    "synthetic": "bool",
    "ability": "tuple[ProvenanceKey, ...] | None",
    # Added after the first collected corpus, under rule 1: an empty `ability`
    # could not tell "no printed line exists" (an engine-built card) from "the
    # resolver failed", which is how a resolver bug survived a whole run. Typed
    # as the closed vocabulary it was always validated against; on the wire it
    # is still one of those strings or null.
    "ability_unresolved": "UnresolvedReason | None",
    # Gen-2 (feature 024), both under rule 1 and both collection metadata:
    # whether the acting player was the random seat, and whether a legality
    # record is a real declaration or a what-if query. Absent on gen-1 shards,
    # which both read as None in memory, so a rewritten record keeps its
    # generation; on the wire each is still a bool or absent.
    "random_seat": "bool | None",
    "what_if": "bool | None",
    # Gen-3 (feature 025), all three collection metadata: the generation marker
    # (FR-018), the rules a run or fork aimed at (FR-009a) and a directed
    # fork's change. Absent on gen-1 and gen-2 lines, read as None.
    "generation": "int | None",
    "aimed_at": "tuple[str, ...] | None",
    "directed": "DirectedTag | None",
    # Written by build-corpus on curated records only (research T1): how
    # curation copied the record, never collected.
    "curation": "Curation | None",
    "extra_fields": "dict",
}

#: The one field gen-2 redefines rather than adds, and so the one named
#: exception to rule 1 (FR-030a). On a gen-2 ``playability`` record
#: ``actor_player`` names the deciding player — the candidates' controller on
#: ``attackers``, the blockers' controller on ``blockers``, the candidate
#: ability's controller on ``decision`` — where feature 023 wrote the active
#: player (``attackers``, ``decision``) or the attacking player (``blockers``).
#: Its type is unchanged; its meaning is versioned by shard generation, the
#: presence of ``random_seat`` (FR-033). Listed here so the exception is
#: documented in the test that would otherwise forbid it, not skipped.
_REDEFINED_IN_GEN2: dict[str, set[str]] = {
    "actor_player": {"attackers", "blockers", "decision"},
}

# The enum values gen-3 reaches. A later stage may append; nothing here may be
# reused for a different meaning.
_FROZEN_KINDS = {
    "resolution", "rewrite", "continuous", "combat", "trigger", "playability",
    "turn_action",
}
_FROZEN_MOMENTS = {"activation", "resolution"}
_FROZEN_SUBKINDS = {"decision", "attackers", "blockers"}
_FROZEN_TURN_ACTION_SUBKINDS = {
    "draw", "cleanup_discard", "untap", "phasing", "day_night", "lore_counter",
}

#: A gen-3 card slot's members, in the order the collector writes them
#: (``FE/EntityJson.java``), and in the order the codec writes them back. A
#: member may be added within gen-3; none may be removed or repurposed.
_FROZEN_GEN3_ENTITY = (
    "id", "name", "zone", "owner", "controller", "face", "face_state", "copy_source",
    "token_script_id", "types", "subtypes", "supertypes", "colors", "mana_value",
    "mana_cost", "base_pt", "pt_boosts", "printed_loyalty", "printed_defense", "tapped",
    "sick", "damage", "counters", "combat", "attached_to", "face_down", "real_face",
    "designations", "x_paid", "chosen", "costs_paid", "history", "order_rank",
    "granted_temporary", "lines", "stack_extras", "role",
)
#: A line's members (``FE/CardLines.java``): the ones always written, and the
#: ones written only when set.
_FROZEN_LINE_MEMBERS = {"keys", "kind", "granted"}
_FROZEN_OPTIONAL_LINE_MEMBERS = {"active", "cast_with", "chosen", "fragment", "affected"}
_FROZEN_LINE_KINDS = {"spell", "activated", "trigger", "static", "replacement", "keyword"}

#: The gen-2 card-slot members gen-3 drops (FR-019): ``printed`` and
#: ``granted_attached`` are now ``lines``, ``pt`` is now ``base_pt`` and
#: ``pt_boosts``. A gen-2 shard still carries them, and still reads.
_REMOVED_IN_GEN3: frozenset[str] = frozenset({"printed", "granted_attached", "pt"})
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


class TestTheActorPlayerException:
    """FR-030a: the one redefinition, on ``playability`` records of gen-2 only.

    Its type and the generation marker ``random_seat`` are pinned with the rest
    of the envelope by rule 1's tests.
    """

    def test_the_exception_covers_exactly_the_playability_subkinds(self):
        assert _REDEFINED_IN_GEN2["actor_player"] == {
            s.value for s in PlayabilitySubkind
        }


class TestRuleTwoEnumValuesMayBeAddedNotReused:
    def test_the_frozen_kinds_all_survive(self):
        assert _FROZEN_KINDS <= {k.value for k in RecordKind}

    def test_the_frozen_moments_and_subkinds_all_survive(self):
        assert _FROZEN_MOMENTS <= {m.value for m in Moment}
        assert _FROZEN_SUBKINDS <= {s.value for s in PlayabilitySubkind}
        assert _FROZEN_TURN_ACTION_SUBKINDS <= {s.value for s in TurnActionSubkind}

    def test_the_frozen_line_kinds_all_survive(self):
        assert _FROZEN_LINE_KINDS <= {k.value for k in LineKind}

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
            elif kind is RecordKind.TURN_ACTION:
                expected |= {(kind, s) for s in TurnActionSubkind}
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


class TestTheGen3CardSlotIsPinned:
    """Rule 1 for the card slot, which gen-3 rewrote (FR-021, FR-025)."""

    def test_the_collector_writes_exactly_the_frozen_members(self):
        entities = java_golden_entities()
        assert entities
        for entity in entities:
            assert tuple(entity) == _FROZEN_GEN3_ENTITY
            for line in entity["lines"]:
                assert _FROZEN_LINE_MEMBERS <= line.keys()
                assert line.keys() <= _FROZEN_LINE_MEMBERS | _FROZEN_OPTIONAL_LINE_MEMBERS
                assert line["kind"] in _FROZEN_LINE_KINDS

    def test_the_reader_writes_a_gen3_entity_with_the_frozen_members(self):
        from effects.infrastructure.record_codec import record_to_dict

        entity = EntityState(id="E1", name="A", zone="battlefield", controller="P0",
                             lines=())
        record = make_record(state=make_snapshot(entities=(entity,)))
        written = record_to_dict(record)["state"]["entities"][0]
        assert tuple(written) == _FROZEN_GEN3_ENTITY
        assert not _REMOVED_IN_GEN3 & written.keys()


class TestRuleThreeTiersAreAdditive:
    def test_the_tiers_are_ordered_by_the_stage_that_adds_them(self):
        assert [t.value for t in InclusionTier] == [1, 2, 3, 4]

    def test_an_absent_tier_reads_as_uncollected_not_empty(self):
        snapshot = make_snapshot()
        assert InclusionTier.UNREFERENCED_STACK not in snapshot.tiers
        assert InclusionTier.CORE in snapshot.tiers

    def test_a_snapshot_naming_no_tier_claims_none_collected(self, snapshot):
        """Never inferred: a default is not evidence of a collector's depth."""
        assert snapshot.tiers == frozenset()


_ACTING = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 0)
_PRINTED = ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "keyword", 0)


class _EveryLine:
    """A :class:`LineLookup` where every key has a line of its own ``e``."""

    def e_for(self, key):
        return (float(key.index_within_kind), float(len(key.script_file)))

    def has_line(self, key):
        return True

    def keyword_of(self, key):
        return None

    def options_for(self, key):
        return ()


def _board() -> StateSnapshot:
    """A gen-3 board: card slots that list ``lines``."""
    return make_snapshot(
        players=(
            PlayerState(id="P0", life=14, hand=3, library=21, graveyard=9),
            PlayerState(id="P1", life=8, hand=1, library=30, graveyard=4),
        ),
        entities=(
            make_entity("E1", "Serra Angel", mana_value=5, lines=(CardLine(
                keys=(_PRINTED,), kind="keyword", granted=False, fragment="Flying"),)),
            make_entity("E2", "Grizzly Bears", controller="P1", mana_value=2, lines=()),
        ),
    )


def _surface(record: EffectRecord):
    return SurfaceBuilder(_EveryLine(), e_dim=2, rng=random.Random(0)).build(record)


class TestRuleFourMetadataNeverBecomesAModelInput:
    """Flipping a collection-metadata field leaves the model's input unchanged.

    The contract's metadata list is ``mode``, ``interventional``, ``fork``,
    ``synthetic``, ``ability_unresolved``, ``random_seat``, ``what_if`` and,
    from gen-3, ``generation``, ``aimed_at`` and ``directed``: a
    model that could see one would learn to predict differently on forks, or
    on the random seat, which is the confound the flags exist to keep out.
    A curated record's ``curation`` member says how curation copied it, never
    what the board was, so it is held to the same rule.
    """

    def test_the_board_is_a_gen3_board(self):
        assert all(entity.lines is not None for entity in _board().entities)
        assert _surface(make_record(state=_board(), ability=(_ACTING,))).slots

    @pytest.mark.parametrize("flip", [
        {"mode": CollectionMode.PATCHED},
        {"fork": True},
        {"synthetic": True},
        {"random_seat": True},
        {"random_seat": None},
        {"generation": None},
        {"aimed_at": ()},
        {"aimed_at": ("704.5f",)},
        {"curation": Curation(
            natural_share=1, copies=2, drawn_rule="704.5f",
            drawn_rules=("704.5f",), rule_entities=("E2",))},
    ])
    def test_a_resolution_s_surface_ignores_the_flag(self, flip):
        base = make_record(state=_board(), ability=(_ACTING,), generation=3)
        assert _surface(replace(base, **flip)) == _surface(base)

    def test_an_intervention_s_surface_is_its_fork_s(self):
        fork = make_record(state=_board(), ability=(_ACTING,), fork=True)
        assert _surface(replace(fork, interventional=True)) == _surface(fork)

    def test_a_directed_fork_s_surface_is_its_plain_fork_s(self):
        from effects.domain.fork_operation import ForkOperation
        from effects.domain.records import DirectedTag

        fork = make_record(state=_board(), ability=(_ACTING,), fork=True)
        directed = replace(fork, aimed_at=("704.5f",),
                           directed=DirectedTag("tougher", ForkOperation.PT))
        assert _surface(directed) == _surface(fork)

    def test_every_listed_metadata_field_is_flipped_here(self):
        from effects.domain.records import COLLECTION_METADATA_FIELDS

        assert COLLECTION_METADATA_FIELDS == {
            "mode", "interventional", "fork", "synthetic", "ability_unresolved",
            "random_seat", "what_if", "generation", "aimed_at", "directed",
        }

    def test_an_unresolved_reason_leaves_the_surface_alone(self):
        base = make_record(state=_board())
        flipped = replace(base, ability_unresolved=UnresolvedReason.ENGINE_EFFECT)
        assert _surface(flipped) == _surface(base)

    @pytest.mark.parametrize("what_if", [True, False])
    def test_a_legality_record_s_surface_ignores_what_if(self, what_if):
        base = make_record(
            RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.ATTACKERS, state=_board(),
        )
        assert _surface(replace(base, what_if=what_if)) == _surface(base)

