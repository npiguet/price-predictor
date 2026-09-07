"""Every Forge effect API maps to an event type or an explicit exclusion (T025).

Runs against the checked-in class list in
:mod:`effects.domain.forge_effect_apis`, so it needs no JVM and no ``../forge``
checkout. A Forge upgrade that adds an effect API fails here, which is the
point: a new API is a new thing an ability can do, and a corpus that silently
stops describing it cannot be repaired without recollecting.
"""

from __future__ import annotations

import pytest

from effects.domain.event_schema import (
    EFFECT_API_EVENTS,
    EVENT_PARAMS,
    EXCLUDED_EFFECT_APIS,
    Event,
    EventType,
    event_types_for,
)
from effects.domain.forge_effect_apis import (
    EFFECT_APIS,
    GAME_EVENTS,
    REPLACEMENT_TYPES,
    TRIGGER_TYPES,
)


class TestCoverage:
    def test_every_effect_api_is_mapped_or_excluded(self):
        covered = set(EFFECT_API_EVENTS) | set(EXCLUDED_EFFECT_APIS)
        missing = sorted(set(EFFECT_APIS) - covered)
        assert not missing, (
            f"{len(missing)} Forge effect APIs are neither mapped in "
            f"EFFECT_API_EVENTS nor excluded with a reason: {missing}"
        )

    def test_nothing_is_mapped_that_forge_does_not_have(self):
        covered = set(EFFECT_API_EVENTS) | set(EXCLUDED_EFFECT_APIS)
        stale = sorted(covered - set(EFFECT_APIS))
        assert not stale, (
            "the vocabulary maps effect APIs that no longer exist in the "
            f"checked-in Forge class list; regenerate or remove: {stale}"
        )

    def test_no_api_is_both_mapped_and_excluded(self):
        both = sorted(set(EFFECT_API_EVENTS) & set(EXCLUDED_EFFECT_APIS))
        assert not both, f"ambiguous coverage for: {both}"

    def test_every_exclusion_states_a_reason(self):
        blank = sorted(k for k, v in EXCLUDED_EFFECT_APIS.items() if len(v) < 15)
        assert not blank, f"exclusions without a usable reason: {blank}"

    def test_every_mapped_api_names_at_least_one_event_type(self):
        empty = sorted(k for k, v in EFFECT_API_EVENTS.items() if not v)
        assert not empty, (
            f"mapped to no event type; these belong in EXCLUDED_EFFECT_APIS "
            f"with a reason: {empty}"
        )

    @pytest.mark.parametrize("api", sorted(EFFECT_API_EVENTS))
    def test_a_mapped_api_resolves_to_known_event_types(self, api: str):
        assert all(isinstance(t, EventType) for t in event_types_for(api))

    @pytest.mark.parametrize("api", sorted(EXCLUDED_EFFECT_APIS))
    def test_an_excluded_api_resolves_to_no_event_type(self, api: str):
        assert event_types_for(api) == ()

    def test_an_unknown_api_fails_loudly(self):
        with pytest.raises(KeyError, match="neither mapped"):
            event_types_for("SomeEffectForgeAddedLastWeek")


class TestVocabulary:
    def test_the_checked_in_forge_lists_are_populated(self):
        """A regeneration that silently produced nothing would pass everything else."""
        assert len(EFFECT_APIS) > 150
        assert len(TRIGGER_TYPES) > 100
        assert len(GAME_EVENTS) > 40
        assert len(REPLACEMENT_TYPES) > 20

    def test_event_type_values_are_unique_and_snake_case(self):
        values = [t.value for t in EventType]
        assert len(values) == len(set(values))
        assert all(v == v.lower() and " " not in v for v in values)

    def test_every_normalized_type_is_a_real_event_type(self):
        assert set(EVENT_PARAMS) <= set(EventType)

    def test_the_vocabulary_is_smaller_than_forges_api_list(self):
        """Types name outcomes, not the script API that produced them."""
        assert len(EventType) < len(EFFECT_APIS)

    def test_the_common_outcome_families_are_all_present(self):
        for name in (
            "ZONE_CHANGE", "DAMAGE_DEALT", "LIFE_CHANGE", "COUNTER_CHANGE",
            "PT_CHANGE", "KEYWORD_CHANGE", "TOKEN_CREATED", "CARD_DRAWN",
        ):
            assert hasattr(EventType, name)


class TestEventNormalization:
    def test_an_event_accepts_the_params_its_type_normalizes(self):
        event = Event(
            type=EventType.DAMAGE_DEALT,
            subjects=("E12",),
            params={"amount": 3, "combat": True},
        )
        assert event.params["amount"] == 3

    def test_an_event_rejects_a_param_the_schema_does_not_normalize(self):
        with pytest.raises(ValueError, match="does not normalize"):
            Event(type=EventType.DAMAGE_DEALT, params={"amonut": 3})

    def test_a_type_with_no_params_takes_none(self):
        assert Event(type=EventType.LIBRARY_SHUFFLED).params == {}
        with pytest.raises(ValueError, match="does not normalize"):
            Event(type=EventType.LIBRARY_SHUFFLED, params={"count": 1})

    def test_an_event_round_trips_through_its_dict_form(self):
        event = Event(
            type=EventType.ZONE_CHANGE,
            subjects=("E1", "E2"),
            params={"from_zone": "battlefield", "to_zone": "graveyard"},
            duration=None,
            attributed_to="0",
        )
        assert Event.from_dict(event.as_dict()) == event

    def test_a_continuous_outcome_carries_its_duration(self):
        event = Event(
            type=EventType.PT_CHANGE,
            params={"power_delta": 3, "toughness_delta": 3},
            duration="end_of_turn",
        )
        assert event.duration == "end_of_turn"
