"""Gate 2's decoding and agreement on one observation (FR-119 – FR-122).

The decoding must turn each head's own layout into one comparable number. A
count head predicts a log rate, a signed delta predicts a direction and a
magnitude separately, and a categorical predicts logits over classes; reading
any of them as a raw scalar would compare two numbers that are not on the same
scale.
"""

from __future__ import annotations

import math

import pytest
import torch

from effects.application.gate_two import (
    PairedVectors,
    read_field,
    score_observation,
)
from effects.domain.damage_step_keywords import (
    KEYWORDS_BY_NAME,
    combat_participant,
)
from effects.domain.effect_head_layout import (
    FIELD_SLICES,
    FIELDS_BY_NAME,
    PER_ENTITY_FIELDS,
    PER_ENTITY_WIDTH,
    ZONE_OUTCOMES,
)
from effects.domain.state_snapshot import (
    CombatStatus,
)
from tests.unit.effects.application.gate_two_support import (
    _combat_record,
    _creature,
    _printed_board,
)
from tests.unit.effects.damage_step_boards import CreatureTraits

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
    return combat_participant(record, record.state.entity(entity_id), _overlay_keywords)


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
            row, participant, PairedVectors(base, stripped),
            active=_ACTIVE,
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
            PairedVectors({"B0": hit}, {"B0": untouched}),
            active=_ACTIVE,
        )
        assert agreed
        assert per_effect == {"opponent.damage_taken": True}

    def test_every_effect_must_agree(self):
        record = _printed_board()
        participant = _participant(record, "A")
        base = {"B0": _zone_and_damage(damage=3.0, died_logit=0.0)}
        stripped = {"B0": _zone_and_damage(damage=1.0, died_logit=1.0)}
        agreed, per_effect = score_observation(
            KEYWORDS_BY_NAME["first_strike"], participant, PairedVectors(base, stripped),
            active=_ACTIVE,
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
        attacker = _creature("A", 2, 2, CreatureTraits(combat=CombatStatus(attacking="P1")))
        record = _combat_record(attacker)
        participant = _participant(record, "A")
        base = {"A": torch.zeros(PER_ENTITY_WIDTH), "P1": _player(poison=2.0, life=0.0)}
        stripped = {"A": torch.zeros(PER_ENTITY_WIDTH), "P1": _player(poison=0.0, life=-2.0)}
        agreed, per_effect = score_observation(
            KEYWORDS_BY_NAME["infect"], participant, PairedVectors(base, stripped),
            active=_ACTIVE,
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
            KEYWORDS_BY_NAME["infect"], participant, PairedVectors(base, stripped),
            active=_ACTIVE,
        )
        assert set(per_effect) == {
            "opponent.counters_delta_m1m1", "opponent.damage_taken",
        }
        assert agreed

    def test_a_row_with_no_applicable_effect_scores_nothing(self):
        """A 0-power carrier deals no damage, so lifelink moves nothing."""
        attacker = _creature("A", 0, 4, CreatureTraits(combat=CombatStatus(attacking="P1")))
        record = _combat_record(attacker)
        participant = _participant(record, "A")
        vectors = {"P0": torch.zeros(PER_ENTITY_WIDTH)}
        agreed, per_effect = score_observation(
            KEYWORDS_BY_NAME["lifelink"], participant, PairedVectors(vectors, vectors),
            active=_ACTIVE,
        )
        assert per_effect == {}
        assert not agreed

    def test_a_field_the_run_never_trained_is_not_scored(self):
        record = _printed_board()
        participant = _participant(record, "A")
        base = {"B0": _zone_and_damage(damage=3.0, died_logit=1.0)}
        stripped = {"B0": _zone_and_damage(damage=1.0, died_logit=0.0)}
        _agreed, per_effect = score_observation(
            KEYWORDS_BY_NAME["first_strike"], participant, PairedVectors(base, stripped),
            active=_ACTIVE - {"damage_taken"},
        )
        assert set(per_effect) == {"opponent.zone_outcome/died"}


def _overlay_keywords(entity) -> set[str]:
    return set(entity.granted_temporary.keywords)


def _blocked_pair(*, keywords: tuple[str, ...] = ()):
    attacker = _creature(
        "A", 2, 2, CreatureTraits(
            keywords=keywords,
            combat=CombatStatus(attacking="P1", blocked_by=("B0",),
                                became_blocked=True),
        ),
    )
    blocker = _creature(
        "B0", 2, 5, CreatureTraits(controller="P1", combat=CombatStatus(blocking=("A",))),
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
