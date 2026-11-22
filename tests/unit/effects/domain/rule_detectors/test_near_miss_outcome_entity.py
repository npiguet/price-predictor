"""A tag whose outcome is an absence names where the outcome would have fallen
(finding 7.8).

A rule test reads a rule's per-entity outputs at the tag's outcome entities,
so a ``does_not_fire`` tag naming none gives its rule no reading on that
polarity. ``outcome_event``'s absence tags (a scoped near miss without the
event, an ``absent`` declaration's ``fires`` side) name the acting clause's
targets as the effect half's refs block holds them, or its controller where
it targets nothing. ``resolution_outcome`` names the stack object on both
sides already, and a ``trigger`` record supervises no per-entity output, so
``trigger_mode`` names none on either side. Each case reads a real gen-3
fixture record.
"""

from __future__ import annotations

from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.outcome_entities import outcome_entities
from effects.domain.rule_detectors.outcome_event import FAMILY as OUTCOME
from effects.domain.rule_detectors.outcome_event import acting_api
from effects.domain.rule_detectors.resolution_outcome import FAMILY as RESOLUTION
from effects.domain.rule_detectors.trigger_mode import FAMILY as TRIGGER
from effects.domain.rule_detectors.trigger_mode import acting_root_params

FIRES, DOES_NOT_FIRE = Polarity.FIRES, Polarity.DOES_NOT_FIRE


def _effect_half(by_kind, sidecars, *, targeted: bool):
    """A patched effect half acting through a line of known API, holding no
    ``token_created`` event, with or without targets."""
    return next(r for r in by_kind("resolution")
                if r.moment.value == "resolution" and r.mode.value == "patched"
                and acting_api(r, sidecars) is not None
                and bool(r.state.refs.targets) == targeted
                and not any(e.type.value == "token_created" for e in r.payload.events))


def _is_card(record, entity_id) -> bool:
    entity = record.state.entity(entity_id)
    return entity is not None and entity.token_script_id is None


def _no_token(record, sidecars, **params):
    return {"event": "token_created", "api": acting_api(record, sidecars), **params}


def test_a_scoped_near_miss_names_the_acting_clause_s_targets(by_kind, gen3_sidecars):
    record = _effect_half(by_kind, gen3_sidecars, targeted=True)
    found = OUTCOME.match(record, gen3_sidecars, _no_token(record, gen3_sidecars))
    assert found.polarity is DOES_NOT_FIRE
    assert found.outcome == record.state.refs.targets


def test_a_scoped_near_miss_targeting_nothing_names_its_controller(by_kind, gen3_sidecars):
    record = _effect_half(by_kind, gen3_sidecars, targeted=False)
    found = OUTCOME.match(record, gen3_sidecars, _no_token(record, gen3_sidecars))
    assert found.polarity is DOES_NOT_FIRE
    assert found.outcome == (record.actor_player,)


def test_an_absent_declaration_s_fires_side_names_the_acting_clause_s_targets(
        by_kind, gen3_sidecars):
    record = _effect_half(by_kind, gen3_sidecars, targeted=True)
    found = OUTCOME.match(record, gen3_sidecars, _no_token(record, gen3_sidecars, absent=True))
    assert found.polarity is FIRES
    assert found.outcome == record.state.refs.targets


def test_a_qualified_near_miss_names_where_the_events_that_missed_fell(
        by_kind, gen3_sidecars):
    """A card's ``zone_change`` read for ``subject_is_token``: the event happened and
    fails the qualifier, so the near miss names where it fell (task FR1: a qualified
    near miss needs an event of the declared type; without one it is no near miss)."""
    record = next(r for r in by_kind("resolution")
                  if r.moment.value == "resolution" and r.mode.value == "patched"
                  and acting_api(r, gen3_sidecars) is not None
                  and any(e.type.value == "zone_change" and outcome_entities(r, [e])
                          and all(_is_card(r, s) for s in e.subjects) for e in r.payload.events))
    params = {"event": "zone_change", "api": acting_api(record, gen3_sidecars),
              "subject_is_token": True}
    found = OUTCOME.match(record, gen3_sidecars, params)
    assert found.polarity is DOES_NOT_FIRE
    missed = [e for e in record.payload.events if e.type.value == "zone_change"
              and e.subjects and all(_is_card(record, s) for s in e.subjects)]
    assert found.outcome == outcome_entities(record, missed)


def test_a_qualified_declaration_finds_no_near_miss_on_a_record_without_its_event(
        by_kind, gen3_sidecars):
    record = _effect_half(by_kind, gen3_sidecars, targeted=False)
    params = _no_token(record, gen3_sidecars, subject_is_token=True)
    assert OUTCOME.match(record, gen3_sidecars, params) is None


def test_a_resolution_outcome_near_miss_names_the_stack_object(by_kind):
    record = next(r for r in by_kind("resolution")
                  if r.moment.value == "resolution" and r.payload.outcome is not None
                  and r.payload.outcome.value == "resolved" and r.state.refs.as_cast
                  and r.state.refs.source)
    found = RESOLUTION.match(record, None, {"outcome": "fizzled", "targeted": True})
    assert found.polarity is DOES_NOT_FIRE
    assert found.outcome == (record.state.refs.source,)


def test_a_trigger_near_miss_names_no_outcome_entity(by_kind, gen3_sidecars):
    """The trigger's effect arrives as its resolution's record: no per-entity
    output is supervised on a trigger record, so neither side names one."""
    record = next(r for r in by_kind("trigger") if not r.payload.fired
                  and (acting_root_params(r, gen3_sidecars) or {}).get("Mode"))
    mode = acting_root_params(record, gen3_sidecars)["Mode"]
    found = TRIGGER.match(record, gen3_sidecars, {"mode": mode})
    assert found.polarity is DOES_NOT_FIRE
    assert found.outcome == ()
