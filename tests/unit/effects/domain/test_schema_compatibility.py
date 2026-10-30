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

import random
from dataclasses import fields, replace

import pytest

from effects.domain.effect_head_input import SurfaceBuilder
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    PAYLOAD_TYPES,
    ActivationPayload,
    CollectionMode,
    EffectRecord,
    Moment,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    UnresolvedReason,
)
from effects.domain.state_snapshot import InclusionTier, PlayerState, StateSnapshot
from tests.unit.effects.builders import make_entity, make_record, make_snapshot

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
    return make_snapshot(
        players=(
            PlayerState(id="P0", life=14, hand=3, library=21, graveyard=9),
            PlayerState(id="P1", life=8, hand=1, library=30, graveyard=4),
        ),
        entities=(
            make_entity("E1", "Serra Angel", printed=(_PRINTED,), mana_value=5),
            make_entity("E2", "Grizzly Bears", controller="P1", mana_value=2),
        ),
    )


def _surface(record: EffectRecord):
    return SurfaceBuilder(_EveryLine(), e_dim=2, rng=random.Random(0)).build(record)


class TestRuleFourMetadataNeverBecomesAModelInput:
    """Flipping a collection-metadata field leaves the model's input unchanged.

    The contract's metadata list is ``mode``, ``interventional``, ``fork``,
    ``synthetic``, ``ability_unresolved``, ``random_seat`` and ``what_if``: a
    model that could see one would learn to predict differently on forks, or
    on the random seat, which is the confound the flags exist to keep out.
    """

    @pytest.mark.parametrize("flip", [
        {"mode": CollectionMode.PATCHED},
        {"fork": True},
        {"synthetic": True},
        {"random_seat": True},
        {"random_seat": None},
    ])
    def test_a_resolution_s_surface_ignores_the_flag(self, flip):
        base = make_record(state=_board(), ability=(_ACTING,))
        assert _surface(replace(base, **flip)) == _surface(base)

    def test_an_intervention_s_surface_is_its_fork_s(self):
        fork = make_record(state=_board(), ability=(_ACTING,), fork=True)
        assert _surface(replace(fork, interventional=True)) == _surface(fork)

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

