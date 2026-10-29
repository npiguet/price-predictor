"""Gate 2's direction scorer and the perturbation it reads (FR-119 – FR-122).

Three things have to hold for the gate's number to mean anything, and each is
tested here on its own.

The **perturbation** must actually remove the keyword. It reaches the model
through two channels — an ability token where the keyword is printed or
attachment-granted, an overlay bit where it is granted until end of turn — and a
strip that cleared only one of them would be measuring a board that still has
the keyword.

The **decoding** must turn each head's own layout into one comparable number. A
count head predicts a log rate, a signed delta predicts a direction and a
magnitude separately, and a categorical predicts logits over classes; reading
any of them as a raw scalar would compare two numbers that are not on the same
scale.

And the **counter and the scorer must describe one population**, because the
under-sampled verdict and the agreement verdict are read off the same row.
"""

from __future__ import annotations

import math

import pytest
import torch

from effects.application.gate_two import (
    KeywordResolver,
    KeywordScore,
    qualifying_observations,
    read_field,
    score_keywords,
    score_observation,
    subject_id,
)
from effects.application.surface_batching import IdentityTable, SurfaceBatcher
from effects.domain.baselines import BaselineMasks
from effects.domain.damage_step_keywords import (
    KEYWORDS_BY_NAME,
    Subject,
    combat_participant,
)
from effects.domain.effect_head_input import (
    OVERLAY_KEYWORDS_GEN2,
    THERMOMETER_STEPS,
    SlotKind,
)
from effects.domain.effect_head_layout import (
    FIELD_SLICES,
    FIELDS_BY_NAME,
    PER_ENTITY_FIELDS,
    PER_ENTITY_WIDTH,
    ZONE_OUTCOMES,
)
from effects.domain.provenance import ProvenanceKey, SidecarLine
from effects.domain.records import CombatPayload, EffectRecord, RecordKind
from effects.domain.state_snapshot import (
    CombatStatus,
    EntityState,
    GlobalState,
    GrantedTemporary,
    PlayerState,
    PowerToughness,
    StateSnapshot,
)
from effects.infrastructure.model_runner import Runnable
from tests.unit.effects.builders import make_batcher, slots_of_kind

# ── a board: one first-striking attacker, one blocker ────────────────────

_FIRST_STRIKE = ProvenanceKey("cardsfolder/w/white_knight.txt", 0, "static", 0)
_PUMP = ProvenanceKey("cardsfolder/w/white_knight.txt", 0, "activated", 0)
_OTHER = ProvenanceKey("cardsfolder/g/grizzly_bears.txt", 0, "static", 0)

_LINES = {
    _FIRST_STRIKE: SidecarLine(
        line_index=0, line_kind="static", provenance=(_FIRST_STRIKE,),
        script_text="First Strike",
    ),
    _PUMP: SidecarLine(
        line_index=1, line_kind="activated", provenance=(_PUMP,),
        script_text="W: this creature gets +1/+0",
    ),
    _OTHER: SidecarLine(
        line_index=0, line_kind="static", provenance=(_OTHER,),
        script_text="Trample",
    ),
}


class _Sidecars:
    def line_for(self, key):
        return _LINES.get(key)

    def line_or_none(self, key):
        return _LINES.get(key)

    def prose_for(self, key):
        return None

    def get(self, script_file):
        raise KeyError(f"{script_file}: no sidecar, so no charm options")


def _creature(
    entity_id: str,
    power: int,
    toughness: int,
    *,
    controller: str = "P0",
    keywords: tuple[str, ...] = (),
    printed: tuple[ProvenanceKey, ...] = (),
    combat: CombatStatus | None = None,
) -> EntityState:
    return EntityState(
        id=entity_id, name=entity_id, zone="battlefield", controller=controller,
        types=("creature",), pt=PowerToughness(base=(power, toughness)),
        combat=combat, printed=printed,
        granted_temporary=GrantedTemporary(keywords=keywords),
    )


#: Each damage step's phase, keyed by its substep, as a real record spells them.
_STEPS = {
    "first_strike": "combat_first_strike_damage",
    "regular": "combat_damage",
}


def _combat_record(
    *entities: EntityState, game: str = "g1", substep: str = "regular",
) -> EffectRecord:
    return EffectRecord(
        record_id=f"{game}.1", run_id="run", timestamp="t", game_id=game,
        kind=RecordKind.COMBAT, actor_player="P0",
        payload=CombatPayload(attackers=("A",)),
        state=StateSnapshot(
            global_=GlobalState(
                turn=4, phase=_STEPS[substep], active="P0", priority="P0",
                stack_size=0, combat_substep=substep,
            ),
            players=(
                PlayerState(id="P0", life=20, hand=3, library=30, graveyard=1),
                PlayerState(id="P1", life=17, hand=4, library=28, graveyard=2),
            ),
            entities=entities,
        ),
    )


def _printed_board(
    game: str = "g1", substep: str = "first_strike",
) -> EffectRecord:
    """The attacker prints first strike; the blocker prints nothing relevant.

    The first-strike step's record by default, the one record the first-strike
    row reads: the attacker's 2 damage lands on the 2/2 blocker and kills it.
    """
    attacker = _creature(
        "A", 2, 2, printed=(_FIRST_STRIKE, _PUMP),
        combat=CombatStatus(attacking="P1", blocked_by=("B0",), became_blocked=True),
    )
    blocker = _creature(
        "B0", 2, 2, controller="P1", printed=(_OTHER,),
        combat=CombatStatus(blocking=("A",)),
    )
    return _combat_record(attacker, blocker, game=game, substep=substep)


def _overlay_board(game: str = "g1") -> EffectRecord:
    """The attacker was granted first strike until end of turn."""
    attacker = _creature(
        "A", 2, 2, keywords=("first_strike",),
        combat=CombatStatus(attacking="P1", blocked_by=("B0",), became_blocked=True),
    )
    blocker = _creature(
        "B0", 2, 2, controller="P1", printed=(_OTHER,),
        combat=CombatStatus(blocking=("A",)),
    )
    return _combat_record(attacker, blocker, game=game, substep="first_strike")


def _batcher() -> SurfaceBatcher:
    """A real batcher whose ``e`` comes from a free table rather than a model.

    The identity baseline's own path, borrowed here because it needs no
    tokenizer and no encoder: what these tests are about is the surface, and an
    encoder would only be a second thing that could fail.
    """
    return make_batcher(
        tokenizer=None, sidecars=_Sidecars(),
        masks=BaselineMasks(identity_embedding=True),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
        identity_table=IdentityTable(
            [line.script_text for line in _LINES.values()], 4,
        ),
    )


def _rows(batcher: SurfaceBatcher, records) -> dict[str, int]:
    texts = batcher.batch_texts(records)
    return {text: row for row, text in enumerate(sorted(texts))}


# ── the perturbation ─────────────────────────────────────────────────────


class TestStrippingTheAbilityChannel:
    def test_the_keywords_own_ability_slot_is_dropped(self):
        record = _printed_board()
        batcher = _batcher()
        rows = _rows(batcher, [record])
        base = batcher.surface_for(record, rows)
        stripped = batcher.surface_for(
            record, rows, strip_keywords={"A": frozenset({"first_strike"})},
        )
        assert _ability_count(base, "A") == 2
        assert _ability_count(stripped, "A") == 1

    def test_the_entitys_other_lines_survive(self):
        """Only the keyword line goes; the activated ability is untouched."""
        record = _printed_board()
        batcher = _batcher()
        rows = _rows(batcher, [record])
        stripped = batcher.surface_for(
            record, rows, strip_keywords={"A": frozenset({"first_strike"})},
        )
        kept = [
            slot.e for slot in slots_of_kind(stripped, SlotKind.ABILITY)
            if slot.entity_id == "A"
        ]
        assert kept == [rows["W: this creature gets +1/+0"]]

    def test_another_entitys_lines_are_intact(self):
        record = _printed_board()
        batcher = _batcher()
        rows = _rows(batcher, [record])
        stripped = batcher.surface_for(
            record, rows, strip_keywords={"A": frozenset({"first_strike"})},
        )
        assert _ability_count(stripped, "B0") == 1

    def test_stripping_a_keyword_the_entity_does_not_carry_changes_nothing(self):
        record = _printed_board()
        batcher = _batcher()
        rows = _rows(batcher, [record])
        base = batcher.surface_for(record, rows)
        stripped = batcher.surface_for(
            record, rows, strip_keywords={"A": frozenset({"lifelink"})},
        )
        assert base.slots == stripped.slots

    def test_the_slot_is_dropped_rather_than_zeroed(self):
        """A zero token would still say "this creature has an ability here"."""
        record = _printed_board()
        batcher = _batcher()
        rows = _rows(batcher, [record])
        stripped = batcher.surface_for(
            record, rows, strip_keywords={"A": frozenset({"first_strike"})},
        )
        assert all(
            slot.e != (0.0,) * 4
            for slot in slots_of_kind(stripped, SlotKind.ABILITY)
        )


class TestStrippingTheOverlayChannel:
    def _overlay_bit(self, surface, entity_id: str) -> float:
        slot = next(
            s for s in slots_of_kind(surface, SlotKind.CARD) if s.entity_id == entity_id
        )
        # The multi-hot sits at a fixed offset from the end of the card block;
        # find it by the one value the base board sets.
        return _overlay_column(slot.features, "first_strike")

    def test_the_granted_keyword_bit_is_cleared(self):
        record = _overlay_board()
        batcher = _batcher()
        rows = _rows(batcher, [record])
        base = batcher.surface_for(record, rows)
        stripped = batcher.surface_for(
            record, rows, strip_keywords={"A": frozenset({"first_strike"})},
        )
        assert self._overlay_bit(base, "A") == 1.0
        assert self._overlay_bit(stripped, "A") == 0.0


class TestBatchThreading:
    def test_the_unperturbed_record_in_the_same_batch_is_unchanged(self):
        record = _printed_board()
        batcher = _batcher()
        _batch, surfaces, _encoded = batcher.build(
            [record, record], _NoEncoder(),
            strip_per_record=[None, {"A": frozenset({"first_strike"})}],
        )
        assert _ability_count(surfaces[0], "A") == 2
        assert _ability_count(surfaces[1], "A") == 1

    def test_no_strip_list_builds_exactly_as_before(self):
        record = _printed_board()
        batcher = _batcher()
        _batch, surfaces, _encoded = batcher.build([record], _NoEncoder())
        assert _ability_count(surfaces[0], "A") == 2


def _ability_count(surface, entity_id: str) -> int:
    return sum(
        1 for slot in slots_of_kind(surface, SlotKind.ABILITY)
        if slot.entity_id == entity_id
    )


def _overlay_column(features: tuple[float, ...], keyword: str) -> float:
    """Read one overlay-keyword bit out of a ``[CARD]`` slot's features.

    The multi-hot is followed by a fixed tail (granted-ability count, three
    relation flags, the stack block, the two per-kind overlays and the gen-2
    layout's three thermometer codes), so the block's position is found by
    counting back from the end rather than by re-deriving every field ahead of
    it.
    """
    tail = 2 + 3 + 6 + 3 + 2 + 3 * THERMOMETER_STEPS
    end = len(features) - tail
    start = end - (len(OVERLAY_KEYWORDS_GEN2) + 1)
    return features[start + OVERLAY_KEYWORDS_GEN2.index(keyword)]


class _NoEncoder:
    """The encoder the identity batcher never calls.

    It still has to be an object with ``eval()``, because the scorer puts both
    halves of the model into evaluation mode before it reads anything.
    """

    def __call__(self, **batch):
        raise AssertionError("the identity batcher must not encode")

    def eval(self):
        return self


# ── decoding one head into one number ────────────────────────────────────


def _vector(**values: float) -> torch.Tensor:
    out = torch.zeros(PER_ENTITY_WIDTH)
    for name, value in values.items():
        start, _end = FIELD_SLICES[name]
        out[start] = value
    return out


def _affected(vector: torch.Tensor) -> torch.Tensor:
    """``vector`` with the gate certain the entity was affected."""
    from effects.domain.effect_head_layout import GATE_INDEX

    vector[GATE_INDEX] = 20.0
    return vector


class TestReadField:
    def test_a_count_field_reads_as_its_rate(self):
        vector = _affected(_vector(damage_taken=math.log(3.0)))
        value = read_field(vector, FIELDS_BY_NAME["damage_taken"], None)
        assert value == pytest.approx(3.0)

    def test_a_categorical_field_reads_as_the_classs_probability(self):
        vector = _affected(torch.zeros(PER_ENTITY_WIDTH))
        start, end = FIELD_SLICES["zone_outcome"]
        vector[start:end] = -20.0
        vector[start + ZONE_OUTCOMES.index("died")] = 20.0
        value = read_field(vector, FIELDS_BY_NAME["zone_outcome"], "died")
        assert value == pytest.approx(1.0, abs=1e-6)

    def test_a_signed_delta_reads_as_its_expected_signed_value(self):
        """Direction and magnitude are separate heads; one number needs both."""
        vector = _affected(torch.zeros(PER_ENTITY_WIDTH))
        start, _end = FIELD_SLICES["life_delta"]
        vector[start] = -20.0       # negative
        vector[start + 1] = -20.0   # zero
        vector[start + 2] = 20.0    # positive
        vector[start + 3] = math.log(4.0)
        value = read_field(vector, FIELDS_BY_NAME["life_delta"], None)
        assert value == pytest.approx(4.0, abs=1e-4)

    def test_a_negative_signed_delta_reads_negative(self):
        vector = _affected(torch.zeros(PER_ENTITY_WIDTH))
        start, _end = FIELD_SLICES["life_delta"]
        vector[start] = 20.0
        vector[start + 1] = -20.0
        vector[start + 2] = -20.0
        vector[start + 3] = math.log(4.0)
        value = read_field(vector, FIELDS_BY_NAME["life_delta"], None)
        assert value == pytest.approx(-4.0, abs=1e-4)

    def test_every_reading_is_scaled_by_the_affected_gate(self):
        """The field heads train only where an entity was affected."""
        vector = _vector(damage_taken=math.log(3.0))  # gate logit 0: P = 0.5
        value = read_field(vector, FIELDS_BY_NAME["damage_taken"], None)
        assert value == pytest.approx(1.5)

    def test_a_field_type_the_table_should_never_name_raises(self):
        spec = next(s for s in PER_ENTITY_FIELDS if s.name == "tap_state")
        with pytest.raises(ValueError, match="tap_state"):
            read_field(torch.zeros(PER_ENTITY_WIDTH), spec, None)


# ── agreement ────────────────────────────────────────────────────────────


_FIELDS = tuple(PER_ENTITY_FIELDS)
_ACTIVE = frozenset(spec.name for spec in _FIELDS)


def _participant(record, entity_id: str):
    return combat_participant(record.state, record.state.entity(entity_id))


class TestScoreObservation:
    """Which effects a combat can show, and whether each moved the right way."""

    def _first_strike(self, opponent_damage_base, opponent_damage_stripped):
        record = _printed_board()
        participant = _participant(record, "A")
        base = {"B0": _vector(damage_taken=math.log(opponent_damage_base))}
        stripped = {"B0": _vector(damage_taken=math.log(opponent_damage_stripped))}
        # Only the damage effect, so the zone effect cannot decide the verdict.
        row = _only(KEYWORDS_BY_NAME["first_strike"], "damage_taken")
        return score_observation(
            row, participant, base, stripped,
            active=_ACTIVE, keywords_of=_overlay_keywords,
        )

    def test_a_change_in_the_rules_direction_agrees(self):
        """With first strike the carrier's hit lands in this step, so the
        blocker takes more with it than without: present > absent."""
        agreed, per_effect = self._first_strike(3.0, 1.0)
        assert agreed
        assert per_effect == {"opponent.damage_taken": True}

    def test_a_change_the_other_way_disagrees(self):
        agreed, per_effect = self._first_strike(1.0, 3.0)
        assert not agreed
        assert per_effect == {"opponent.damage_taken": False}

    def test_no_change_at_all_disagrees(self):
        """A model that ignored the keyword is not a model that got it right."""
        agreed, _per_effect = self._first_strike(2.0, 2.0)
        assert not agreed

    def test_a_move_in_the_affected_gate_alone_agrees(self):
        """Stripped of first strike, the blocker takes nothing in this step.

        The head says so through the gate, while the damage head — supervised
        only where an entity was affected — still reads how much a hit would
        deal. A reading of the field head alone sees no change and calls the
        keyword ignored.
        """
        from effects.domain.effect_head_layout import GATE_INDEX

        record = _printed_board()
        participant = _participant(record, "A")
        hit = _vector(damage_taken=math.log(3.0))
        hit[GATE_INDEX] = 4.0
        untouched = _vector(damage_taken=math.log(3.0))
        untouched[GATE_INDEX] = -4.0
        agreed, per_effect = score_observation(
            _only(KEYWORDS_BY_NAME["first_strike"], "damage_taken"), participant,
            {"B0": hit}, {"B0": untouched},
            active=_ACTIVE, keywords_of=_overlay_keywords,
        )
        assert agreed
        assert per_effect == {"opponent.damage_taken": True}

    def test_every_effect_must_agree(self):
        record = _printed_board()
        participant = _participant(record, "A")
        base = {"B0": _zone_and_damage(damage=3.0, died_logit=0.0)}
        stripped = {"B0": _zone_and_damage(damage=1.0, died_logit=1.0)}
        agreed, per_effect = score_observation(
            KEYWORDS_BY_NAME["first_strike"], participant, base, stripped,
            active=_ACTIVE, keywords_of=_overlay_keywords,
        )
        # Damage moved the right way; the death probability moved the wrong way.
        assert per_effect == {
            "opponent.damage_taken": True, "opponent.zone_outcome/died": False,
        }
        assert not agreed

    def test_an_effect_this_combat_cannot_show_is_skipped_not_failed(self):
        """Infect's two halves are exclusive: an unblocked attacker poisons a
        player and damages no creature, so the creature fields have nothing to
        say and must not be counted as disagreements."""
        attacker = _creature("A", 2, 2, combat=CombatStatus(attacking="P1"))
        record = _combat_record(attacker)
        participant = _participant(record, "A")
        base = {"A": torch.zeros(PER_ENTITY_WIDTH), "P1": _player(poison=2.0, life=0.0)}
        stripped = {"A": torch.zeros(PER_ENTITY_WIDTH), "P1": _player(poison=0.0, life=-2.0)}
        agreed, per_effect = score_observation(
            KEYWORDS_BY_NAME["infect"], participant, base, stripped,
            active=_ACTIVE, keywords_of=_overlay_keywords,
        )
        assert set(per_effect) == {
            "defending_player.poison_delta", "defending_player.life_delta",
        }
        assert agreed

    def test_a_blocked_infecter_is_scored_on_the_creature_half_alone(self):
        record = _combat_record(*_blocked_pair(keywords=("infect",)))
        participant = _participant(record, "A")
        base = {"B0": _wither_side(counters=2.0, damage=0.0)}
        stripped = {"B0": _wither_side(counters=0.0, damage=2.0)}
        agreed, per_effect = score_observation(
            KEYWORDS_BY_NAME["infect"], participant, base, stripped,
            active=_ACTIVE, keywords_of=_overlay_keywords,
        )
        assert set(per_effect) == {
            "opponent.counters_delta_m1m1", "opponent.damage_taken",
        }
        assert agreed

    def test_a_row_with_no_applicable_effect_scores_nothing(self):
        """A 0-power carrier deals no damage, so lifelink moves nothing."""
        attacker = _creature("A", 0, 4, combat=CombatStatus(attacking="P1"))
        record = _combat_record(attacker)
        participant = _participant(record, "A")
        vectors = {"P0": torch.zeros(PER_ENTITY_WIDTH)}
        agreed, per_effect = score_observation(
            KEYWORDS_BY_NAME["lifelink"], participant, vectors, vectors,
            active=_ACTIVE, keywords_of=_overlay_keywords,
        )
        assert per_effect == {}
        assert not agreed

    def test_a_field_the_run_never_trained_is_not_scored(self):
        record = _printed_board()
        participant = _participant(record, "A")
        base = {"B0": _zone_and_damage(damage=3.0, died_logit=1.0)}
        stripped = {"B0": _zone_and_damage(damage=1.0, died_logit=0.0)}
        _agreed, per_effect = score_observation(
            KEYWORDS_BY_NAME["first_strike"], participant, base, stripped,
            active=_ACTIVE - {"damage_taken"}, keywords_of=_overlay_keywords,
        )
        assert set(per_effect) == {"opponent.zone_outcome/died"}


def _overlay_keywords(entity) -> set[str]:
    return set(entity.granted_temporary.keywords)


def _blocked_pair(*, keywords: tuple[str, ...] = ()):
    attacker = _creature(
        "A", 2, 2, keywords=keywords,
        combat=CombatStatus(attacking="P1", blocked_by=("B0",),
                            became_blocked=True),
    )
    blocker = _creature(
        "B0", 2, 5, controller="P1", combat=CombatStatus(blocking=("A",)),
    )
    return attacker, blocker


def _signed(vector: torch.Tensor, name: str, value: float) -> None:
    """Write one signed-delta field as an unambiguous direction and magnitude."""
    start, _end = FIELD_SLICES[name]
    sign = 2 if value > 0 else 0 if value < 0 else 1
    for offset in range(3):
        vector[start + offset] = 20.0 if offset == sign else -20.0
    vector[start + 3] = math.log(abs(value)) if value else -20.0


def _player(*, poison: float, life: float) -> torch.Tensor:
    vector = torch.zeros(PER_ENTITY_WIDTH)
    _signed(vector, "poison_delta", poison)
    _signed(vector, "life_delta", life)
    return vector


def _wither_side(*, counters: float, damage: float) -> torch.Tensor:
    vector = torch.zeros(PER_ENTITY_WIDTH)
    _signed(vector, "counters_delta_m1m1", counters)
    start, _end = FIELD_SLICES["damage_taken"]
    vector[start] = math.log(damage) if damage else -20.0
    return vector

def _only(row, field_name: str):
    import dataclasses

    return dataclasses.replace(
        row, effects=tuple(e for e in row.effects if e.field == field_name),
    )


def _zone_and_damage(*, damage: float, died_logit: float) -> torch.Tensor:
    vector = _vector(damage_taken=math.log(damage))
    start, _end = FIELD_SLICES["zone_outcome"]
    vector[start + ZONE_OUTCOMES.index("died")] = died_logit
    return vector


# ── qualification: the counter and the scorer read one population ────────


class TestSubjectResolution:
    def test_the_carrier_is_itself(self):
        record = _printed_board()
        participant = _participant(record, "A")
        effect = KEYWORDS_BY_NAME["indestructible"].effects[0]
        assert effect.subject is Subject.CARRIER
        assert subject_id(effect, participant) == "A"

    def test_the_opponent_is_the_first_creature_it_fights(self):
        record = _printed_board()
        participant = _participant(record, "A")
        effect = next(
            e for e in KEYWORDS_BY_NAME["double_strike"].effects
            if e.subject is Subject.OPPONENT
        )
        assert subject_id(effect, participant) == "B0"

    def test_the_controller_is_a_player(self):
        record = _printed_board()
        participant = _participant(record, "A")
        effect = KEYWORDS_BY_NAME["lifelink"].effects[0]
        assert subject_id(effect, participant) == "P0"

    def test_the_defending_player_is_the_one_being_attacked(self):
        record = _printed_board()
        participant = _participant(record, "A")
        effect = KEYWORDS_BY_NAME["trample"].effects[0]
        assert subject_id(effect, participant) == "P1"


class TestQualifyingObservations:
    def test_a_printed_keyword_is_found_through_the_sidecar(self):
        resolver = KeywordResolver(_Sidecars())
        found = qualifying_observations(
            _printed_board(), (KEYWORDS_BY_NAME["first_strike"],), resolver,
        )
        assert [(row.keyword, p.carrier.id) for row, p in found] == [
            ("first_strike", "A"),
        ]

    def test_an_overlay_keyword_is_found_without_a_sidecar(self):
        resolver = KeywordResolver(None)
        found = qualifying_observations(
            _overlay_board(), (KEYWORDS_BY_NAME["first_strike"],), resolver,
        )
        assert len(found) == 1

    @pytest.mark.parametrize("keyword", ["first_strike", "double_strike"])
    def test_a_regular_step_record_is_never_a_strike_first_observation(
        self, keyword,
    ):
        """The first striker already struck; that step's own record is the one
        the keyword changes, and the regular step's board starts after it."""
        attacker = _creature(
            "A", 2, 2, keywords=(keyword,),
            combat=CombatStatus(attacking="P1", blocked_by=("B0",),
                                became_blocked=True),
        )
        blocker = _creature(
            "B0", 2, 2, controller="P1", combat=CombatStatus(blocking=("A",)),
        )
        unblocked = _creature(
            "A", 2, 2, keywords=(keyword,), combat=CombatStatus(attacking="P1"),
        )
        rows = (KEYWORDS_BY_NAME[keyword],)
        for record in (
            _combat_record(attacker, blocker),
            _combat_record(unblocked),
            _printed_board(substep="regular"),
        ):
            assert record.state.global_.combat_substep == "regular"
            assert qualifying_observations(
                record, rows, KeywordResolver(_Sidecars()),
            ) == []

    def test_a_carrier_the_predicate_rejects_is_not_an_observation(self):
        """Trample on a 2/2 blocked by a 2/2 runs nothing over."""
        attacker = _creature(
            "A", 2, 2, keywords=("trample",),
            combat=CombatStatus(attacking="P1", blocked_by=("B0",),
                                became_blocked=True),
        )
        blocker = _creature(
            "B0", 2, 2, controller="P1", combat=CombatStatus(blocking=("A",)),
        )
        record = _combat_record(attacker, blocker)
        found = qualifying_observations(
            record, (KEYWORDS_BY_NAME["trample"],), KeywordResolver(None),
        )
        assert found == []

    def test_an_unblocked_double_striker_is_still_an_observation(self):
        """Its first hit lands on the player, which the row reads as life."""
        attacker = _creature(
            "A", 2, 2, keywords=("double_strike",),
            combat=CombatStatus(attacking="P1"),
        )
        record = _combat_record(attacker, substep="first_strike")
        found = qualifying_observations(
            record, (KEYWORDS_BY_NAME["double_strike"],), KeywordResolver(None),
        )
        assert len(found) == 1

    def test_an_observation_with_nothing_readable_is_not_counted(self):
        """A 0-power double striker deals no damage in either step."""
        attacker = _creature(
            "A", 0, 4, keywords=("double_strike",),
            combat=CombatStatus(attacking="P1"),
        )
        record = _combat_record(attacker, substep="first_strike")
        found = qualifying_observations(
            record, (KEYWORDS_BY_NAME["double_strike"],), KeywordResolver(None),
        )
        assert found == []

    def test_two_carriers_in_one_record_are_two_observations(self):
        first = _creature(
            "A", 2, 2, keywords=("lifelink",),
            combat=CombatStatus(attacking="P1", blocked_by=("B0",),
                                became_blocked=True),
        )
        second = _creature(
            "B0", 2, 2, controller="P1", keywords=("lifelink",),
            combat=CombatStatus(blocking=("A",)),
        )
        record = _combat_record(first, second)
        found = qualifying_observations(
            record, (KEYWORDS_BY_NAME["lifelink"],), KeywordResolver(None),
        )
        assert sorted(p.carrier.id for _row, p in found) == ["A", "B0"]


# ── the whole scorer, against a model that reacts to the strip ───────────


class _AbilityCountingModel:
    """A stand-in whose predicted damage and death rise with the record's
    ability count.

    Not a model of anything — it exists so the scorer's two builds differ in a
    way that is a function of the perturbation and nothing else, which is what
    makes an agreement number here a test of the wiring rather than of a
    randomly initialised head. The build that keeps the keyword keeps one more
    ability, so it predicts more damage and a likelier death, which is the way
    first strike moves its opponent.
    """

    def eval(self):
        return self

    def __call__(self, **batch):
        return batch

    def per_entity(self, batch):
        slot_kinds = batch["slot_kinds"]
        abilities = (slot_kinds == int(SlotKind.ABILITY)).sum(dim=1).float()
        row = torch.zeros(slot_kinds.shape[0], PER_ENTITY_WIDTH)
        row[:, FIELD_SLICES["damage_taken"][0]] = 0.5 * abilities
        died = FIELD_SLICES["zone_outcome"][0] + ZONE_OUTCOMES.index("died")
        row[:, died] = 0.5 * abilities
        return row[:, None, :].expand(-1, slot_kinds.shape[1], -1)


def _runnable() -> Runnable:
    return Runnable(_NoEncoder(), _AbilityCountingModel(), _batcher(), _FIELDS)


class TestScoreKeywords:
    def test_it_scores_the_qualifying_records(self):
        records = [_printed_board(game=f"g{i}") for i in range(3)]
        scores = score_keywords(
            records, (KEYWORDS_BY_NAME["first_strike"],), _runnable(),
        )
        score = scores["first_strike"]
        assert score.qualifying == 3
        assert score.agreeing == 3

    def test_it_reports_agreement_per_field(self):
        records = [_printed_board()]
        scores = score_keywords(
            records, (KEYWORDS_BY_NAME["first_strike"],), _runnable(),
        )
        assert set(scores["first_strike"].per_field) == {
            "opponent.damage_taken", "opponent.zone_outcome/died",
        }

    def test_a_keyword_no_record_qualifies_for_scores_nothing(self):
        scores = score_keywords(
            [_printed_board()], (KEYWORDS_BY_NAME["wither"],), _runnable(),
        )
        assert scores["wither"] == KeywordScore()

    def test_an_empty_corpus_scores_every_row_at_zero(self):
        scores = score_keywords(
            [], (KEYWORDS_BY_NAME["lifelink"],), _runnable(),
        )
        assert scores["lifelink"].qualifying == 0


# ── the evaluator's own wiring ───────────────────────────────────────────


class TestEvaluatorWiring:
    """The gate-2 block must produce a computed agreement, not a zero.

    It reported ``agreeing_records=0`` for every keyword until the scorer
    existed, which routed all eight to a probe whatever the model did — a canary
    that always sings the same note.
    """

    @staticmethod
    def _load(_records):
        return _runnable()

    def test_it_scores_the_game_disjoint_combat_records(self):
        from effects.application.gate_two import run_gate_two

        verdicts = run_gate_two([_printed_board(game="gd-1")], self._load)

        first_strike = next(v for v in verdicts if v.keyword == "first_strike")
        assert first_strike.qualifying_records == 1
        assert first_strike.direction_agreement == 1.0

    def test_the_reason_lists_agreement_per_field(self):
        from effects.application.gate_two import run_gate_two

        verdicts = run_gate_two([_printed_board(game="gd-1")], self._load)

        first_strike = next(v for v in verdicts if v.keyword == "first_strike")
        assert "damage_taken 100%" in first_strike.reason
        assert "zone_outcome/died 100%" in first_strike.reason

    def test_no_game_disjoint_combat_record_leaves_every_keyword_at_zero(self):
        from effects.application.gate_two import run_gate_two

        def _never(_records):
            raise AssertionError("no records: the model must not be loaded")

        verdicts = run_gate_two([], _never)

        assert len(verdicts) == 8
        assert all(v.qualifying_records == 0 for v in verdicts)
        assert all("under-sampled" in v.reason for v in verdicts)


# ── who is counted (FR-119, FR-120) ──────────────────────────────────────


def _two_first_strikers() -> EffectRecord:
    """Two first-striking attackers in one combat, each blocked by its own 2/2."""
    creatures = []
    for index in range(2):
        creatures.append(_creature(
            f"A{index}", 2, 2, printed=(_FIRST_STRIKE,),
            combat=CombatStatus(
                attacking="P1", blocked_by=(f"B{index}",), became_blocked=True,
            ),
        ))
        creatures.append(_creature(
            f"B{index}", 2, 2, controller="P1", printed=(_OTHER,),
            combat=CombatStatus(blocking=(f"A{index}",)),
        ))
    return _combat_record(*creatures, substep="first_strike")


def _scores(records):
    return score_keywords(
        records, (KEYWORDS_BY_NAME["first_strike"],), _runnable(),
    )["first_strike"]


class TestPopulation:
    def test_a_fork_record_never_qualifies(self):
        """A damage-step probe re-runs a real combat: a near-duplicate board."""
        import dataclasses

        probe = dataclasses.replace(
            _printed_board(game="gd-1"), record_id="gd-1.2", fork=True,
            mirror_of="gd-1.1",
        )
        assert _scores([probe]).qualifying == 0
        assert _scores([_printed_board(game="gd-1"), probe]).qualifying == 1

    def test_two_carriers_in_one_record_count_one_qualifying_record(self):
        """FR-119's threshold counts records; agreement averages the carriers."""
        score = _scores([_two_first_strikers()])
        assert score.qualifying == 1
        assert score.observations == 2
        assert score.agreeing == 2

    def test_the_verdict_reads_agreement_over_observations(self):
        from effects.application.gate_two import evaluate_keyword

        verdict = evaluate_keyword(
            KEYWORDS_BY_NAME["first_strike"],
            KeywordScore(qualifying=300, agreeing=300, observations=400),
        )
        assert verdict.qualifying_records == 300
        assert verdict.observations == 400
        assert verdict.direction_agreement == pytest.approx(0.75)


class TestGateTwoRecords:
    def _write(self, directory, *records):
        from effects.infrastructure.record_io import write_shard

        write_shard(directory / "shard-00001.jsonl.gz", records)

    def test_it_reads_the_recorded_games_of_the_curated_game_disjoint_stratum(
        self, tmp_path,
    ):
        from effects.application.gate_two import gate_two_records
        from effects.infrastructure.corpus_store import CorpusStore
        from effects.infrastructure.effect_model_store import SplitProvenance

        corpus = tmp_path / "corpus"
        recorded = _printed_board(game="gd-1")
        other = _printed_board(game="gd-2")
        self._write(CorpusStore(corpus).game_disjoint_dir, recorded, other)
        provenance = SplitProvenance(game_disjoint_games=("gd-1",))

        read = list(gate_two_records(provenance, corpus_path=corpus))
        assert [r.record_id for r in read] == [recorded.record_id]
