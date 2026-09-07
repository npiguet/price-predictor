"""The record envelope's invariants (T022).

The schema is frozen before collection, so these are contract rather than
implementation detail: a corpus written against a violated invariant cannot be
repaired without recollecting it.
"""

from __future__ import annotations

import pytest

from effects.domain.records import (
    COLLECTION_METADATA_FIELDS,
    ActivationPayload,
    CombatPayload,
    Costs,
    EffectRecord,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
)


class TestFlagSignatures:
    """The three signatures from the schema contract's flag table."""

    def test_an_ordinary_observation_carries_neither_flag(self, make_record):
        record = make_record()
        assert (record.interventional, record.fork) == (False, False)
        assert not record.is_probe

    def test_an_interventional_resolution_is_both(self, make_record):
        record = make_record(interventional=True, fork=True)
        assert record.kind is RecordKind.RESOLUTION
        assert (record.interventional, record.fork) == (True, True)
        assert not record.is_probe

    def test_a_damage_step_probe_is_a_fork_that_intervened_in_nothing(
        self, make_record,
    ):
        record = make_record(kind=RecordKind.COMBAT, fork=True)
        assert (record.interventional, record.fork) == (False, True)
        assert record.is_probe

    def test_an_intervention_must_be_a_resolution(self, make_record):
        with pytest.raises(ValueError, match="forced resolution"):
            make_record(kind=RecordKind.COMBAT, interventional=True, fork=True)

    def test_an_intervention_is_always_a_fork(self, make_record):
        with pytest.raises(ValueError, match="always a fork"):
            make_record(interventional=True, fork=False)

    def test_mirror_of_requires_the_fork_flag(self, make_record):
        with pytest.raises(ValueError, match="mirror_of"):
            make_record(kind=RecordKind.COMBAT, mirror_of="run.3.9")

    def test_variant_of_requires_the_synthetic_flag(self, make_record):
        with pytest.raises(ValueError, match="variant_of"):
            make_record(variant_of="Serra Angel")

    def test_a_fork_may_name_the_record_it_mirrors(self, make_record):
        record = make_record(kind=RecordKind.COMBAT, fork=True, mirror_of="run.3.9")
        assert record.mirror_of == "run.3.9"


class TestLinkId:
    """``link_id`` joins the two halves of a resolution pair, where there is one."""

    @pytest.mark.parametrize(
        "outcome",
        [
            ResolutionOutcome.FIZZLED,
            ResolutionOutcome.DECLINED,
            ResolutionOutcome.COUNTERED,
        ],
    )
    def test_a_partnerless_outcome_carries_no_link_id(self, make_record, outcome):
        with pytest.raises(ValueError, match="no linked effect half"):
            make_record(
                moment=Moment.ACTIVATION,
                payload=ActivationPayload(costs=Costs(), outcome=outcome),
                link_id="pair-1",
            )

    @pytest.mark.parametrize(
        "outcome",
        [ResolutionOutcome.RESOLVED, ResolutionOutcome.PARTIALLY_FIZZLED],
    )
    def test_a_resolving_outcome_may_carry_one(self, make_record, outcome):
        record = make_record(
            moment=Moment.ACTIVATION,
            payload=ActivationPayload(costs=Costs(), outcome=outcome),
            link_id="pair-1",
        )
        assert record.link_id == "pair-1"

    def test_an_interventional_effect_half_has_no_partner_to_link_to(
        self, make_record,
    ):
        with pytest.raises(ValueError, match="no activation partner"):
            make_record(interventional=True, fork=True, link_id="pair-1")

    def test_a_partnerless_outcome_without_a_link_id_is_fine(self, make_record):
        record = make_record(
            moment=Moment.ACTIVATION,
            payload=ActivationPayload(
                costs=Costs(), outcome=ResolutionOutcome.COUNTERED,
            ),
        )
        assert record.link_id is None
        assert not record.payload.has_partner


class TestActingAbility:
    """``ability`` names the acting line, where one line acts."""

    def test_combat_records_name_no_acting_line(self, make_record, ability_key):
        with pytest.raises(ValueError, match="no single acting line"):
            make_record(kind=RecordKind.COMBAT, ability=(ability_key,))

    def test_playability_records_name_no_acting_line(
        self, make_record, ability_key, snapshot,
    ):
        with pytest.raises(ValueError, match="no single acting line"):
            EffectRecord(
                record_id="run.3.11", run_id="run", timestamp="t",
                game_id="run.3.2", kind=RecordKind.PLAYABILITY,
                subkind=PlayabilitySubkind.ATTACKERS,
                actor_player="P0", state=snapshot,
                payload=PlayabilityAttackersPayload(),
                ability=(ability_key,),
            )

    def test_a_resolution_names_its_line(self, make_record, ability_key):
        assert make_record().ability == (ability_key,)

    def test_a_merged_line_carries_several_keys(self, make_record, ability_key):
        second = type(ability_key)(
            script_file=ability_key.script_file, face=0,
            trait_kind="static", index_within_kind=1,
        )
        record = make_record(ability=(ability_key, second))
        assert len(record.ability) == 2


class TestDiscriminators:
    def test_moment_belongs_to_the_resolution_kind(self, make_record):
        with pytest.raises(ValueError, match="moment is the resolution kind"):
            make_record(
                kind=RecordKind.COMBAT,
                moment=Moment.RESOLUTION,
                payload=CombatPayload(),
            )

    def test_a_resolution_without_a_moment_is_rejected(self, snapshot):
        with pytest.raises(ValueError, match="moment is the resolution kind"):
            EffectRecord(
                record_id="r.0.1", run_id="r", timestamp="t", game_id="r.0.1",
                kind=RecordKind.RESOLUTION, actor_player="P0", state=snapshot,
                payload=ResolutionPayload(),
            )

    def test_subkind_belongs_to_the_playability_kind(self, make_record):
        with pytest.raises(ValueError, match="subkind is the playability kind"):
            make_record(
                kind=RecordKind.COMBAT,
                subkind=PlayabilitySubkind.DECISION,
                payload=CombatPayload(),
            )

    def test_the_payload_type_must_match_the_kind(self, make_record):
        with pytest.raises(ValueError, match="takes a ResolutionPayload"):
            make_record(moment=Moment.RESOLUTION, payload=CombatPayload())

    def test_the_payload_type_must_match_the_moment(self, make_record):
        with pytest.raises(ValueError, match="takes an? ActivationPayload"):
            make_record(moment=Moment.ACTIVATION, payload=ResolutionPayload())


class TestCollectionMetadataStaysOutOfTheModel:
    def test_the_four_metadata_fields_are_named(self):
        assert COLLECTION_METADATA_FIELDS == {
            "mode", "interventional", "fork", "synthetic",
        }

    def test_no_metadata_field_reaches_the_model_input_projection(
        self, make_record,
    ):
        record = make_record(interventional=True, fork=True)
        assert not COLLECTION_METADATA_FIELDS & set(record.model_input_fields())

    def test_the_projection_keeps_the_game_situation(self, make_record):
        fields = make_record().model_input_fields()
        assert set(fields) == {
            "kind", "moment", "subkind", "actor_player", "ability", "state",
            "payload",
        }

    def test_identity_fields_are_not_model_inputs_either(self, make_record):
        """ids join records to a split; they say nothing about the game."""
        fields = make_record().model_input_fields()
        assert not {"record_id", "run_id", "game_id", "timestamp"} & set(fields)


class TestIdentity:
    def test_record_id_carries_the_worker_index(self, make_record):
        assert make_record(record_id="run-uuid.4.10237").worker == "4"

    def test_a_record_is_immutable_once_written(self, make_record):
        record = make_record()
        with pytest.raises(AttributeError):
            record.link_id = "pair-1"
