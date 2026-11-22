"""A damage-prevented copy of a combat record never keeps a block its
protection forbids (702.16f, 702.16j): FR7, finding 12.3 (CPC1 C-1)."""

from __future__ import annotations

from tests.unit.effects.builders import make_record
from tests.unit.effects.domain.test_protection_copies import (
    CreatureLook,
    _creature,
    _damage,
    _state,
)
from effects.domain.protection_model import ProtectionOutcome
from effects.domain.protection_opportunities import qualify
from effects.domain.records import CombatPayload, RecordKind
from effects.domain.rule_outcomes import ProtectionQuality, QualityKind

_GOBLIN = ProtectionQuality(QualityKind.CREATURE_TYPE, "goblin")
_CREATURE = ProtectionQuality(QualityKind.CARD_TYPE, "creature")


def _combat(attacker_look=CreatureLook(), blocker_look=CreatureLook(), *, extra=()):
    attacker = _creature("A1", 6, look=attacker_look)
    blocker = _creature("B1", 6, look=blocker_look)
    return make_record(
        RecordKind.COMBAT, generation=3, state=_state((attacker, blocker, *extra)),
        payload=CombatPayload(
            attackers=("A1",), blocks={"A1": ("B1",)},
            events=(_damage("A1", 2, "B1", combat=True), _damage("B1", 2, "A1", combat=True)),
        ),
    )


def _by_protected(record):
    return {o.protected: o for o in qualify(record) if o.outcome is ProtectionOutcome.DAMAGE_PREVENTED}


def test_an_attacker_is_not_protected_from_its_own_blocker():
    """Heavy Ballista blocked by Heavy Ballista: no protection from Humans
    (702.16f would forbid the block), none from everything (702.16j)."""
    found = _by_protected(_combat())
    assert "A1" not in found


def test_a_blocker_keeps_its_opportunity():
    blocker = _by_protected(_combat())["B1"]
    assert blocker.covered == ("A1",) and blocker.everything
    assert _GOBLIN in blocker.qualities and _CREATURE in blocker.qualities


def test_a_quality_no_blocker_has_stays_for_the_attacker():
    """The blocker deals no damage; a source that is no blocker does (a
    trigger's damage). Its elf quality is no blocker's and stays; creature,
    which the goblin blocker has too, goes, and so does protection from
    everything."""
    from dataclasses import replace
    elf = _creature("X1", 6, look=CreatureLook(subtypes=("elf",)))
    record = _combat(extra=(elf,))
    record = replace(record, payload=replace(
        record.payload, events=(_damage("A1", 1, "X1", combat=True),)))
    found = _by_protected(record)["A1"]
    assert found.covered == ("X1",) and not found.everything
    assert ProtectionQuality(QualityKind.CREATURE_TYPE, "elf") in found.qualities
    assert _CREATURE not in found.qualities
