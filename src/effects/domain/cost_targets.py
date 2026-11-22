"""What a resolution's cost half trains: what its payment did (task G11, D7).

Whether an entity was affected is the effect half's question, so a cost half
trains no affected gate (:func:`effect_targets.derive_targets`). Its payment
still moved permanents and cards, and those moves train without the gate at
the entities ``payload.costs`` names:

- a sacrificed permanent and a discarded card go to the graveyard, an exiled
  card to exile, a returned permanent to its owner's hand (``zone_outcome``);
- a tapped permanent's ``tap_state`` is 1, an untapped one's 0;
- the counters paid or put are the source's counter deltas, where the source
  is on the battlefield: ``costs.counters`` names a count per type and no
  permanent, and a source anywhere else holds no counter;
- the payer, the source's controller, loses ``costs.life``.

Nothing else trains: the cast card's own move to the stack is the cast, not an
outcome, and a cost half's silence says nothing changed
(:attr:`effect_head_layout.FieldSpec.silent_by`). ``revealed``, ``exerted``,
``convoked`` and ``delved`` name no field here; ``mana_by_color`` is the
verdict head's ``mana_paid``.
"""

from __future__ import annotations

from collections.abc import Callable

from effects.domain.effect_head_layout import (
    ZONE_OUTCOMES,
    HeadField,
    ZoneOutcome,
    counter_delta_field,
)
from effects.domain.records import ActivationPayload, Costs, EffectRecord
from effects.domain.state_snapshot import EntityState, Zone

#: ``costs`` channel -> the zone outcome of each card it names.
_MOVES: tuple[tuple[str, ZoneOutcome], ...] = (
    ("sacrificed", ZoneOutcome.DIED),
    ("discarded", ZoneOutcome.DIED),
    ("exiled", ZoneOutcome.EXILED),
    ("returned", ZoneOutcome.TO_HAND),
)
#: ``costs`` channel -> the tap state of each permanent it names.
_TAPS: tuple[tuple[str, int], ...] = (("tapped", 1), ("untapped", 0))


def is_cost_half(record: EffectRecord) -> bool:
    """Whether ``record`` is a resolution's cost half."""
    return isinstance(record.payload, ActivationPayload)


def apply_paid_costs(record: EffectRecord, slot: Callable) -> None:
    """Write what a cost half's payment did into the entries ``slot`` gives
    each paying entity, marking each affected; nothing on any other record."""
    if not is_cost_half(record):
        return
    costs: Costs = record.payload.costs
    for channel, outcome in _MOVES:
        _set_each(slot, getattr(costs, channel), HeadField.ZONE_OUTCOME,
                  ZONE_OUTCOMES.index(outcome))
    for channel, state in _TAPS:
        _set_each(slot, getattr(costs, channel), HeadField.TAP_STATE, state)
    source = _source(record)
    if source is None:
        return
    if costs.counters and source.zone == Zone.BATTLEFIELD:
        _paid_counters(slot(source.id), costs.counters)
    if costs.life:
        _paid(slot(source.controller)).add(HeadField.LIFE_DELTA, -costs.life)


def _source(record: EffectRecord) -> EntityState | None:
    source = record.state.refs.source
    return record.state.entity(source) if source is not None else None


def _set_each(slot: Callable, refs: tuple[str, ...], name: str, value: int) -> None:
    for ref in refs:
        _paid(slot(ref)).set(name, value)


def _paid_counters(entry, counters: dict[str, int]) -> None:
    _paid(entry)
    for counter, delta in counters.items():
        entry.add(counter_delta_field(counter), delta)


def _paid(entry):
    """``entry``, affected: the field targets of a gated field train only at an
    affected entity, and a cost half trains no gate to be wrong about it."""
    entry.affected = True
    return entry
