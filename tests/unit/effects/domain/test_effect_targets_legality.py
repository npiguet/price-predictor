"""A playability record's legality fields: ``target_legal``,
``attacker_legal``, ``blocker_legal``, ``min_blockers``,
``attack_target_legal`` and ``combat_requirement``, trained without the gate
(root spec section 11.4)."""

from __future__ import annotations

import functools

from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    Candidate,
    ForbiddenEntity,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
)
from tests.unit.effects.domain.effect_targets_support import (
    _entity,
    _on_board,
    _record,
    _resolution,
)

_RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"


@functools.cache
def _fixture() -> dict:
    """The gen-3 fixture's records, parsed, by record id."""
    from effects.infrastructure.shard_io import read_shard
    from tests.unit.effects.builders import GEN3_RECORDS

    return {record.record_id: record for record in read_shard(GEN3_RECORDS)}


class TestLegalityBits:
    def test_a_legal_target_sets_its_bit(self):
        record = _record(
            PlayabilityDecisionPayload(candidates=(
                Candidate(ability=(), can_play=True, affordable=True,
                          has_legal_target=True, legal_targets=("E1",)),
            )),
            kind=RecordKind.PLAYABILITY, moment=None,
            subkind=PlayabilitySubkind.DECISION,
        )
        assert derive_targets(record)["E1"].fields["target_legal"] == 1

    def test_legal_and_forbidden_attackers_both_get_a_bit(self):
        record = _record(
            PlayabilityAttackersPayload(
                legal_attackers=("E1",),
                forbidden=(ForbiddenEntity(entity="E2"),),
            ),
            kind=RecordKind.PLAYABILITY, moment=None,
            subkind=PlayabilitySubkind.ATTACKERS,
        )
        targets = derive_targets(record)
        assert targets["E1"].fields["attacker_legal"] == 1
        assert targets["E2"].fields["attacker_legal"] == 0

    def test_min_blockers_is_read_at_the_anchored_attacker(self):
        """Not at the blockers — the restriction belongs to the attacker."""
        record = _record(
            PlayabilityBlockersPayload(
                anchor_attacker="E9", legal_blockers=("E1",), min_blockers=2,
            ),
            kind=RecordKind.PLAYABILITY, moment=None,
            subkind=PlayabilitySubkind.BLOCKERS,
        )
        targets = derive_targets(record)
        assert targets["E9"].fields["min_blockers"] == 2
        assert "min_blockers" not in targets["E1"].fields

    def test_a_forbidden_blocker_gets_a_zero_bit(self):
        record = _record(
            PlayabilityBlockersPayload(
                anchor_attacker="E9", legal_blockers=("E1",),
                forbidden=(ForbiddenEntity(entity="E2"),),
            ),
            kind=RecordKind.PLAYABILITY, moment=None,
            subkind=PlayabilitySubkind.BLOCKERS,
        )
        targets = derive_targets(record)
        assert targets["E1"].fields["blocker_legal"] == 1
        assert targets["E2"].fields["blocker_legal"] == 0


class TestTargetLegalityOnADecisionRecord:
    """Story 5 acceptance 3: ``target_legal`` on every card and player of the
    record, gate-free (root spec section 11.4)."""

    SHREDDERS_TECHNIQUE = f"{_RUN}.5-mwqw5tpydi06.4292"

    def test_it_is_one_inside_the_legal_targets_and_zero_on_every_other_entity(self):
        record = _fixture()[self.SHREDDERS_TECHNIQUE]
        legal = set(record.payload.candidates[0].legal_targets)
        assert legal == {"E40", "E41", "E42", "E55", "E68", "E69", "E72"}
        targets = derive_targets(record)
        refs = [e.id for e in record.state.entities] + [p.id for p in record.state.players]
        assert {"P0", "P1"} <= set(refs)
        for ref in refs:
            assert targets[ref].fields["target_legal"] == (1 if ref in legal else 0), ref

    def test_the_record_trains_no_gate(self):
        targets = derive_targets(_fixture()[self.SHREDDERS_TECHNIQUE])
        assert targets.gated is False
        assert not any(entry.affected for entry in targets.values())

    def test_a_candidate_with_no_targeting_clause_leaves_it_unsupervised(self):
        land_play = next(
            record for record in _fixture().values()
            if record.subkind == PlayabilitySubkind.DECISION
            and record.payload.candidates[0].land_play
        )
        targets = derive_targets(land_play)
        assert not any("target_legal" in entry.fields for entry in targets.values())

    def test_a_clause_with_no_legal_target_is_zero_everywhere(self):
        record = _on_board(
            PlayabilityDecisionPayload(candidates=(
                Candidate(ability=(), can_play=False, affordable=True, has_legal_target=False),
            )),
            _entity("E1"), kind=RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.DECISION,
        )
        targets = derive_targets(record)
        assert {ref: entry.fields["target_legal"] for ref, entry in targets.items()} == {
            "E1": 0, "P0": 0, "P1": 0,
        }

    def test_an_effect_record_still_trains_the_gate(self):
        assert derive_targets(_resolution(
            Event(type=EventType.DESTROYED, subjects=("E1",)),
        )).gated is True


class TestCombatLegality:
    def test_an_anchored_attackers_record_labels_every_defender(self):
        record = _on_board(
            PlayabilityAttackersPayload(
                anchor_attacker="E1", legal_defenders=("P1", "E3"), must_attack=True,
                legal_attackers=("E1",),
            ),
            _entity("E1", types=("creature",)),
            _entity("E3", types=("planeswalker",), controller="P1"),
            _entity("E4", types=("battle",), controller="P1"),
            _entity("E5", types=("creature",), controller="P1"),
            kind=RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.ATTACKERS,
        )
        targets = derive_targets(record)
        legal = {ref: e.fields["attack_target_legal"] for ref, e in targets.items()
                 if "attack_target_legal" in e.fields}
        assert legal == {"P0": 0, "P1": 1, "E3": 1, "E4": 0}
        assert targets["E1"].fields["combat_requirement"] == 1
        assert targets.gated is False

    def test_a_declared_anchor_carries_the_tap_its_attack_made(self):
        """FR-037: tapped where ``tapped_by_attack`` is true, unchanged where false."""
        for tapped_by_attack, expected in ((True, 1), (False, 0)):
            record = _on_board(
                PlayabilityAttackersPayload(
                    anchor_attacker="E1", legal_defenders=("P1",),
                    tapped_by_attack=tapped_by_attack,
                ),
                _entity("E1", types=("creature",)),
                kind=RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.ATTACKERS,
            )
            entry = derive_targets(record)["E1"]
            assert entry.fields["tap_state"] == expected
            assert entry.affected

    def test_a_fixture_attack_that_tapped_its_anchor_reads_tapped(self):
        record = _fixture()[f"{_RUN}.0-mwqvqdyrb9c0.14"]
        assert record.payload.tapped_by_attack is True
        assert derive_targets(record)["E74"].fields["tap_state"] == 1

    def test_a_blockers_record_labels_the_block_requirement_on_every_listed_creature(self):
        record = _fixture()[f"{_RUN}.2-mwqvqdxrusnv.2464"]
        payload = record.payload
        targets = derive_targets(record)
        must = set(payload.must_block)
        for ref in (*payload.legal_blockers, *(f.entity for f in payload.forbidden)):
            assert targets[ref].fields["combat_requirement"] == (1 if ref in must else 0)
        assert targets.gated is False
