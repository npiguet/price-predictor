"""``outcome_event``'s ``source_dies`` qualifier on real gen-3 combat records."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.outcome_event import FAMILY


def _is_creature(record, entity_id) -> bool:
    entity = record.state.entity(entity_id)
    return entity is not None and "creature" in {t.lower() for t in entity.types}


class TestSourceDies:
    """``source_dies`` reads whether a damage event's source dies in the same
    record (510.2: a creature dying in a combat damage step still dealt its
    damage there), on a real combat record, which names the source as the
    event's ``source`` and ``cause`` both. Its near miss is a first striker's
    victim: a creature with power above 0 dealt damage in the first-strike
    step, dying there having dealt none. A step in which no such creature
    dies, a 0-power creature dying, and a creature dying in the regular step
    having dealt none there are neither side."""

    DECLARED = {"event": "damage_dealt", "combat": True, "source_dies": True}

    @staticmethod
    def _combat(by_kind):
        """A real combat record holding combat damage to a creature, and its
        damage events, the first one dealt to that creature."""
        for record in by_kind("combat"):
            dealt = [e for e in record.payload.events if e.type is EventType.DAMAGE_DEALT
                     and e.params.get("combat") and e.params.get("cause")]
            dealt.sort(key=lambda e: not _is_creature(record, e.subjects[0]))
            if dealt and _is_creature(record, dealt[0].subjects[0]):
                return record, dealt
        pytest.fail("no combat record in the fixture deals combat damage to a creature")

    @staticmethod
    def _events(record, events):
        return replace(record, payload=replace(record.payload, events=tuple(events)))

    @staticmethod
    def _dies(subject):
        return Event(type=EventType.ZONE_CHANGE, subjects=(subject,),
                     params={"from_zone": "battlefield", "to_zone": "graveyard"})

    @staticmethod
    def _in_step(record, step, victim=None, power=None):
        """``record`` as taken in damage step ``step``, ``victim``'s base power
        set to ``power`` when given."""
        state = replace(record.state, global_=replace(record.state.global_, combat_substep=step))
        if victim is not None:
            entities = tuple(
                replace(e, pt=replace(e.pt, base=(power - e.pt.boosts[0] - e.pt.counters[0],
                                                  e.pt.base[1])))
                if e.id == victim else e for e in state.entities)
            state = replace(state, entities=entities)
        return replace(record, state=state)

    def _struck(self, by_kind, step, power=2):
        """A real combat record's striker dealing its damage and living, the
        creature it kills, of power ``power``, having dealt none, in ``step``."""
        record, dealt = self._combat(by_kind)
        striker = dealt[0]
        victim = striker.subjects[0]
        assert victim != striker.params["cause"]
        struck = self._events(record, (striker, self._dies(victim)))
        return self._in_step(struck, step, victim, power), victim

    def test_a_source_that_dies_in_the_step_fires(self, by_kind):
        record, dealt = self._combat(by_kind)
        source = dealt[0].params["cause"]
        traded = self._events(record, (*dealt, self._dies(source)))
        found = FAMILY.match(traded, None, self.DECLARED)
        assert found.polarity is Polarity.FIRES and found.observed

    def test_a_first_strikers_victim_is_the_near_miss(self, by_kind):
        """The first-strike step: the striker deals its damage and lives, the
        creature it kills had damage to deal and dealt none."""
        struck_first, victim = self._struck(by_kind, "first_strike")
        found = FAMILY.match(struck_first, None, self.DECLARED)
        assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
        assert victim in found.outcome

    def test_a_victim_dying_in_the_regular_step_is_neither_side(self, by_kind):
        """A first striker killed in the regular step dealt its damage in the
        first-strike step's record, so this one shows it dealing none; a
        creature without first strike dying there is no first striker's victim."""
        struck, _victim = self._struck(by_kind, "regular")
        assert FAMILY.match(struck, None, self.DECLARED) is None

    def test_a_zero_power_victim_is_neither_side(self, by_kind):
        """A 0-power creature dealt no damage because it had none to deal."""
        struck_first, _victim = self._struck(by_kind, "first_strike", power=0)
        assert FAMILY.match(struck_first, None, self.DECLARED) is None

    def test_a_record_naming_no_step_shows_no_victim(self, by_kind):
        struck, _victim = self._struck(by_kind, None)
        assert FAMILY.match(struck, None, self.DECLARED) is None

    def test_a_damage_step_in_which_no_creature_dies_is_neither_side(self, by_kind):
        """An ordinary unblocked attack, or a block both creatures survive:
        no creature dies, so nothing shows whether the damage was dealt at once."""
        record, dealt = self._combat(by_kind)
        assert FAMILY.match(self._events(record, dealt), None, self.DECLARED) is None

    def test_a_creature_dying_that_was_dealt_no_damage_there_is_neither_side(self, by_kind):
        record, dealt = self._combat(by_kind)
        bystander = next(e.id for e in record.state.entities
                         if "creature" in {t.lower() for t in e.types}
                         and all(e.id not in d.subjects and e.id != d.params["cause"]
                                 for d in dealt))
        died = self._events(record, (*dealt, self._dies(bystander)))
        assert FAMILY.match(died, None, self.DECLARED) is None

    def test_false_flips_the_sides(self, by_kind):
        record, dealt = self._combat(by_kind)
        striker = dealt[0]
        declared = {**self.DECLARED, "source_dies": False}
        died = self._events(record, (striker, self._dies(striker.params["cause"])))
        # A combat has no acting line to scope the near miss by (finding 7.1).
        assert FAMILY.match(died, None, declared) is None
        assert FAMILY.match(self._events(record, (striker,)), None,
                            declared).polarity is Polarity.FIRES

    def test_the_source_is_read_before_the_cause(self, by_kind):
        record, dealt = self._combat(by_kind)
        named = [replace(e, params={**e.params, "source": e.params["cause"], "cause": "E0"})
                 for e in dealt]
        traded = self._events(record, (*named, self._dies(dealt[0].params["cause"])))
        assert FAMILY.match(traded, None, self.DECLARED).polarity is Polarity.FIRES

    def test_it_reads_an_event_that_names_a_source_only(self):
        FAMILY.validate({"event": "damage_prevented", "source_dies": True})
        with pytest.raises(ValueError, match="source_dies"):
            FAMILY.validate({"event": "zone_change", "source_dies": True})
        with pytest.raises(ValueError, match="qualifier"):
            FAMILY.validate({"event": "damage_dealt", "api": "DealDamage", "absent": True,
                             "source_dies": True})
