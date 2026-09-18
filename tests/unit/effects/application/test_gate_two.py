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
    count_qualifying,
    count_qualifying_all,
    qualifying_observations,
    read_field,
    score_keywords,
    score_observation,
    subject_id,
)
from effects.application.surface_batching import (
    IDENTITY_TABLE_SIZE,
    SurfaceBatcher,
)
from effects.application.train_effect_model import VariantMasks
from effects.domain.damage_step_keywords import (
    KEYWORDS_BY_NAME,
    Subject,
    combat_participant,
)
from effects.domain.effect_head_input import OVERLAY_KEYWORDS, SlotKind
from effects.domain.effect_model import (
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

    def prose_for(self, key):
        return None


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


def _combat_record(*entities: EntityState, game: str = "g1") -> EffectRecord:
    return EffectRecord(
        record_id=f"{game}.1", run_id="run", timestamp="t", game_id=game,
        kind=RecordKind.COMBAT, actor_player="P0",
        payload=CombatPayload(attackers=("A",)),
        state=StateSnapshot(
            global_=GlobalState(
                turn=4, phase="combat_damage", active="P0", priority="P0",
                stack_size=0, combat_substep="regular",
            ),
            players=(
                PlayerState(id="P0", life=20, hand=3, library=30, graveyard=1),
                PlayerState(id="P1", life=17, hand=4, library=28, graveyard=2),
            ),
            entities=entities,
        ),
    )


def _printed_board(game: str = "g1") -> EffectRecord:
    """The attacker prints first strike; the blocker prints nothing relevant."""
    attacker = _creature(
        "A", 2, 2, printed=(_FIRST_STRIKE, _PUMP),
        combat=CombatStatus(attacking="P1", blocked_by=("B0",), became_blocked=True),
    )
    blocker = _creature(
        "B0", 2, 2, controller="P1", printed=(_OTHER,),
        combat=CombatStatus(blocking=("A",)),
    )
    return _combat_record(attacker, blocker, game=game)


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
    return _combat_record(attacker, blocker, game=game)


def _batcher() -> SurfaceBatcher:
    """A real batcher whose ``e`` comes from a free table rather than a model.

    The identity baseline's own path, borrowed here because it needs no
    tokenizer and no encoder: what these tests are about is the surface, and an
    encoder would only be a second thing that could fail.
    """
    return SurfaceBatcher(
        tokenizer=None, sidecars=_Sidecars(),
        masks=VariantMasks(identity_embedding=True),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
        identity_table=torch.nn.Embedding(IDENTITY_TABLE_SIZE, 4),
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
            slot.e for slot in stripped.of_kind(SlotKind.ABILITY)
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
            for slot in stripped.of_kind(SlotKind.ABILITY)
        )


class TestStrippingTheOverlayChannel:
    def _overlay_bit(self, surface, entity_id: str) -> float:
        slot = next(
            s for s in surface.of_kind(SlotKind.CARD) if s.entity_id == entity_id
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
        _batch, surfaces = batcher.build(
            [record, record], _NoEncoder(),
            strip_per_record=[None, {"A": frozenset({"first_strike"})}],
        )
        assert _ability_count(surfaces[0], "A") == 2
        assert _ability_count(surfaces[1], "A") == 1

    def test_no_strip_list_builds_exactly_as_before(self):
        record = _printed_board()
        batcher = _batcher()
        _batch, surfaces = batcher.build([record], _NoEncoder())
        assert _ability_count(surfaces[0], "A") == 2


def _ability_count(surface, entity_id: str) -> int:
    return sum(
        1 for slot in surface.of_kind(SlotKind.ABILITY)
        if slot.entity_id == entity_id
    )


def _overlay_column(features: tuple[float, ...], keyword: str) -> float:
    """Read one overlay-keyword bit out of a ``[CARD]`` slot's features.

    The multi-hot is followed by a fixed tail (granted-ability count, three
    relation flags, the stack block and the two per-kind overlays), so the
    block's position is found by counting back from the end rather than by
    re-deriving every field ahead of it.
    """
    tail = 2 + 3 + 6 + 3 + 2
    end = len(features) - tail
    start = end - (len(OVERLAY_KEYWORDS) + 1)
    return features[start + OVERLAY_KEYWORDS.index(keyword)]


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


class TestReadField:
    def test_a_count_field_reads_as_its_rate(self):
        vector = _vector(damage_taken=math.log(3.0))
        value = read_field(vector, FIELDS_BY_NAME["damage_taken"], None)
        assert value == pytest.approx(3.0)

    def test_a_categorical_field_reads_as_the_classs_probability(self):
        vector = torch.zeros(PER_ENTITY_WIDTH)
        start, end = FIELD_SLICES["zone_outcome"]
        vector[start:end] = -20.0
        vector[start + ZONE_OUTCOMES.index("died")] = 20.0
        value = read_field(vector, FIELDS_BY_NAME["zone_outcome"], "died")
        assert value == pytest.approx(1.0, abs=1e-6)

    def test_a_signed_delta_reads_as_its_expected_signed_value(self):
        """Direction and magnitude are separate heads; one number needs both."""
        vector = torch.zeros(PER_ENTITY_WIDTH)
        start, _end = FIELD_SLICES["life_delta"]
        vector[start] = -20.0       # negative
        vector[start + 1] = -20.0   # zero
        vector[start + 2] = 20.0    # positive
        vector[start + 3] = math.log(4.0)
        value = read_field(vector, FIELDS_BY_NAME["life_delta"], None)
        assert value == pytest.approx(4.0, abs=1e-4)

    def test_a_negative_signed_delta_reads_negative(self):
        vector = torch.zeros(PER_ENTITY_WIDTH)
        start, _end = FIELD_SLICES["life_delta"]
        vector[start] = 20.0
        vector[start + 1] = -20.0
        vector[start + 2] = -20.0
        vector[start + 3] = math.log(4.0)
        value = read_field(vector, FIELDS_BY_NAME["life_delta"], None)
        assert value == pytest.approx(-4.0, abs=1e-4)

    def test_a_field_type_the_table_should_never_name_raises(self):
        spec = next(s for s in PER_ENTITY_FIELDS if s.name == "tap_state")
        with pytest.raises(ValueError, match="tap_state"):
            read_field(torch.zeros(PER_ENTITY_WIDTH), spec, None)


# ── agreement ────────────────────────────────────────────────────────────


_FIELDS = tuple(PER_ENTITY_FIELDS)


def _participant(record, entity_id: str):
    return combat_participant(record.state, record.state.entity(entity_id))


class TestScoreObservation:
    """Which effects a combat can show, and whether each moved the right way."""

    def _first_strike(self, carrier_damage_base, carrier_damage_stripped):
        record = _printed_board()
        participant = _participant(record, "A")
        base = {"A": _vector(damage_taken=math.log(carrier_damage_base))}
        stripped = {"A": _vector(damage_taken=math.log(carrier_damage_stripped))}
        # Only the damage effect, so the zone effect cannot decide the verdict.
        row = _only(KEYWORDS_BY_NAME["first_strike"], "damage_taken")
        return score_observation(
            row, participant, base, stripped,
            fields=_FIELDS, keywords_of=_overlay_keywords,
        )

    def test_a_change_in_the_rules_direction_agrees(self):
        """First strike means less damage comes back, so present < absent."""
        agreed, per_effect = self._first_strike(1.0, 3.0)
        assert agreed
        assert per_effect == {"carrier.damage_taken": True}

    def test_a_change_the_other_way_disagrees(self):
        agreed, per_effect = self._first_strike(3.0, 1.0)
        assert not agreed
        assert per_effect == {"carrier.damage_taken": False}

    def test_no_change_at_all_disagrees(self):
        """A model that ignored the keyword is not a model that got it right."""
        agreed, _per_effect = self._first_strike(2.0, 2.0)
        assert not agreed

    def test_every_effect_must_agree(self):
        record = _printed_board()
        participant = _participant(record, "A")
        base = {"A": _zone_and_damage(damage=1.0, died_logit=1.0)}
        stripped = {"A": _zone_and_damage(damage=3.0, died_logit=0.0)}
        agreed, per_effect = score_observation(
            KEYWORDS_BY_NAME["first_strike"], participant, base, stripped,
            fields=_FIELDS, keywords_of=_overlay_keywords,
        )
        # Damage moved the right way; the death probability moved the wrong way.
        assert per_effect == {
            "carrier.damage_taken": True, "carrier.zone_outcome/died": False,
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
            fields=_FIELDS, keywords_of=_overlay_keywords,
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
            fields=_FIELDS, keywords_of=_overlay_keywords,
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
            fields=_FIELDS, keywords_of=_overlay_keywords,
        )
        assert per_effect == {}
        assert not agreed

    def test_a_field_the_run_never_trained_is_not_scored(self):
        record = _printed_board()
        participant = _participant(record, "A")
        without_damage = tuple(
            spec for spec in PER_ENTITY_FIELDS if spec.name != "damage_taken"
        )
        base = {"A": _zone_and_damage(damage=3.0, died_logit=0.0)}
        stripped = {"A": _zone_and_damage(damage=1.0, died_logit=1.0)}
        _agreed, per_effect = score_observation(
            KEYWORDS_BY_NAME["first_strike"], participant, base, stripped,
            fields=without_damage, keywords_of=_overlay_keywords,
        )
        assert set(per_effect) == {"carrier.zone_outcome/died"}


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
        effect = KEYWORDS_BY_NAME["first_strike"].effects[0]
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
        """It hits the player twice, which the row carries as a life effect."""
        attacker = _creature(
            "A", 2, 2, keywords=("double_strike",),
            combat=CombatStatus(attacking="P1"),
        )
        record = _combat_record(attacker)
        found = qualifying_observations(
            record, (KEYWORDS_BY_NAME["double_strike"],), KeywordResolver(None),
        )
        assert len(found) == 1

    def test_an_observation_with_nothing_readable_is_not_counted(self):
        """A 0-power double striker deals no damage twice."""
        attacker = _creature(
            "A", 0, 4, keywords=("double_strike",),
            combat=CombatStatus(attacking="P1"),
        )
        record = _combat_record(attacker)
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


class TestCounting:
    def test_it_counts_the_population_the_scorer_scores(self):
        records = [_printed_board(), _overlay_board()]
        counts = count_qualifying_all(
            records, (KEYWORDS_BY_NAME["first_strike"],), _Sidecars(),
        )
        assert counts["first_strike"] == 2

    def test_a_non_combat_record_is_never_counted(self):
        from effects.domain.records import Moment, ResolutionPayload

        record = _printed_board()
        resolution = EffectRecord(
            record_id="x", run_id="run", timestamp="t", game_id="g",
            kind=RecordKind.RESOLUTION, moment=Moment.RESOLUTION,
            actor_player="P0", payload=ResolutionPayload(),
            state=record.state, ability=(_PUMP,),
        )
        counts = count_qualifying_all(
            [resolution], (KEYWORDS_BY_NAME["first_strike"],), _Sidecars(),
        )
        assert counts["first_strike"] == 0

    def test_without_sidecars_a_printed_keyword_is_invisible(self):
        assert count_qualifying(
            [_printed_board()], KEYWORDS_BY_NAME["first_strike"], None,
        ) == 0

    def test_one_row_at_a_time_matches_the_one_pass_count(self):
        records = [_printed_board(), _overlay_board()]
        row = KEYWORDS_BY_NAME["first_strike"]
        assert count_qualifying(records, row, _Sidecars()) == (
            count_qualifying_all(records, (row,), _Sidecars())[row.keyword]
        )


# ── the whole scorer, against a model that reacts to the strip ───────────


class _AbilityCountingModel:
    """A stand-in whose every output falls as the entity keeps more abilities.

    Not a model of anything — it exists so the scorer's two builds differ in a
    way that is a function of the perturbation and nothing else, which is what
    makes an agreement number here a test of the wiring rather than of a
    randomly initialised head.
    """

    def eval(self):
        return self

    def __call__(self, **batch):
        return batch

    def per_entity(self, batch):
        slot_kinds = batch["slot_kinds"]
        abilities = (slot_kinds == int(SlotKind.ABILITY)).sum(dim=1).float()
        ramp = torch.arange(1, PER_ENTITY_WIDTH + 1, dtype=torch.float32)
        row = -0.5 * abilities[:, None, None] * ramp[None, None, :]
        return row.expand(-1, slot_kinds.shape[1], -1)


class TestScoreKeywords:
    def test_it_scores_the_qualifying_records(self):
        records = [_printed_board(game=f"g{i}") for i in range(3)]
        scores = score_keywords(
            records, (KEYWORDS_BY_NAME["first_strike"],),
            _NoEncoder(), _AbilityCountingModel(), _batcher(), _Sidecars(),
            fields=_FIELDS,
        )
        score = scores["first_strike"]
        assert score.qualifying == 3
        assert score.agreeing == 3

    def test_it_reports_agreement_per_field(self):
        records = [_printed_board()]
        scores = score_keywords(
            records, (KEYWORDS_BY_NAME["first_strike"],),
            _NoEncoder(), _AbilityCountingModel(), _batcher(), _Sidecars(),
            fields=_FIELDS,
        )
        assert set(scores["first_strike"].per_field) == {
            "carrier.damage_taken", "carrier.zone_outcome/died",
        }

    def test_a_keyword_no_record_qualifies_for_scores_nothing(self):
        scores = score_keywords(
            [_printed_board()], (KEYWORDS_BY_NAME["wither"],),
            _NoEncoder(), _AbilityCountingModel(), _batcher(), _Sidecars(),
            fields=_FIELDS,
        )
        assert scores["wither"] == KeywordScore()

    def test_an_empty_corpus_scores_every_row_at_zero(self):
        scores = score_keywords(
            [], (KEYWORDS_BY_NAME["lifelink"],),
            _NoEncoder(), _AbilityCountingModel(), _batcher(), _Sidecars(),
            fields=_FIELDS,
        )
        assert scores["lifelink"].qualifying == 0


# ── the evaluator's own wiring ───────────────────────────────────────────


class TestEvaluatorWiring:
    """The gate-2 block must produce a computed agreement, not a zero.

    It reported ``agreeing_records=0`` for every keyword until the scorer
    existed, which routed all eight to a probe whatever the model did — a canary
    that always sings the same note.
    """

    def _provenance(self, game_disjoint):
        from effects.infrastructure.effect_model_store import SplitProvenance

        return SplitProvenance(
            card_disjoint_games=("card-1",),
            game_disjoint_games=tuple(game_disjoint),
        )

    def _config(self, tmp_path):
        from effects.application.evaluate_effect_model import (
            EvaluateEffectModelConfig,
        )

        return EvaluateEffectModelConfig(
            records_dir=tmp_path, cards_folders=(tmp_path / "cardsfolder",),
        )

    def test_it_scores_the_game_disjoint_combat_records(self, tmp_path, monkeypatch):
        from effects.application import evaluate_effect_model as evaluator

        records = [_printed_board(game="gd-1"), _printed_board(game="train-1")]
        monkeypatch.setattr(
            evaluator, "_gate_two_records", lambda config, provenance: records[:1],
        )
        monkeypatch.setattr(
            evaluator, "_gate_two_runnable",
            lambda *a, **k: (
                _NoEncoder(), _AbilityCountingModel(), _batcher(), _FIELDS,
            ),
        )

        verdicts = evaluator.run_gate_two(
            self._config(tmp_path), _Checkpoint(self._provenance(("gd-1",))),
            vocab_path=tmp_path / "vocab.txt",
            keyword_path=tmp_path / "keywords.json",
        )

        first_strike = next(v for v in verdicts if v.keyword == "first_strike")
        assert first_strike.qualifying_records == 1
        assert first_strike.direction_agreement == 1.0

    def test_the_reason_lists_agreement_per_field(self, tmp_path, monkeypatch):
        from effects.application import evaluate_effect_model as evaluator

        monkeypatch.setattr(
            evaluator, "_gate_two_records",
            lambda config, provenance: [_printed_board(game="gd-1")],
        )
        monkeypatch.setattr(
            evaluator, "_gate_two_runnable",
            lambda *a, **k: (
                _NoEncoder(), _AbilityCountingModel(), _batcher(), _FIELDS,
            ),
        )

        verdicts = evaluator.run_gate_two(
            self._config(tmp_path), _Checkpoint(self._provenance(("gd-1",))),
            vocab_path=tmp_path / "vocab.txt",
            keyword_path=tmp_path / "keywords.json",
        )

        first_strike = next(v for v in verdicts if v.keyword == "first_strike")
        assert "damage_taken 100%" in first_strike.reason
        assert "zone_outcome/died 100%" in first_strike.reason

    def test_no_game_disjoint_combat_record_leaves_every_keyword_at_zero(
        self, tmp_path, monkeypatch,
    ):
        from effects.application import evaluate_effect_model as evaluator

        monkeypatch.setattr(
            evaluator, "_gate_two_records", lambda config, provenance: [],
        )

        def _never(*_args, **_kwargs):
            raise AssertionError("no records: the model must not be loaded")

        monkeypatch.setattr(evaluator, "_gate_two_runnable", _never)

        verdicts = evaluator.run_gate_two(
            self._config(tmp_path), _Checkpoint(self._provenance(())),
            vocab_path=tmp_path / "vocab.txt",
            keyword_path=tmp_path / "keywords.json",
        )

        assert len(verdicts) == 8
        assert all(v.qualifying_records == 0 for v in verdicts)
        assert all("under-sampled" in v.reason for v in verdicts)


class _Checkpoint:
    def __init__(self, provenance) -> None:
        self.provenance = provenance
