"""What a resolution's cost half trains: what its payment did.

Whether an entity was affected is the effect half's question, so a cost half
trains no affected gate (:func:`effect_targets.derive_targets`). Its payment
still moved permanents and cards, and those moves train without the gate at
the entities ``payload.costs`` names:

- a sacrificed permanent and a discarded card go to the graveyard, an exiled
  card to exile, a returned permanent to its owner's hand (``zone_outcome``);
- a tapped permanent's ``tap_state`` is 1, an untapped one's 0;
- the counters paid or put are the source's counter deltas where the record
  shows the source bore them: ``costs.counters`` names a net count per type
  and no permanent (the Java ledger sums every counter change of the payment,
  whatever it changed), so a removal books at the source only when the
  source is on the battlefield and its snapshot, read before the payment,
  holds at least that many counters of the type (Iron Spider, Stark
  Upgrade's ``RemoveAnyCounter`` of +1/+1 it does not hold; The Filigree
  Sylex's oil); an addition books there only for loyalty (a loyalty cost
  always counts on its own planeswalker, 606.4), since the record carries no
  cost script and a ``blight`` or ``PutCounter`` cost may count on another
  permanent; every other change leaves the source's field of that type
  unknown, never silent;
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
    ("sacrificed", ZoneOutcome.TO_GRAVEYARD),
    ("discarded", ZoneOutcome.TO_GRAVEYARD),
    ("exiled", ZoneOutcome.EXILED),
    ("returned", ZoneOutcome.TO_HAND),
)
#: The one counter type whose cost always counts on the acting permanent.
_LOYALTY = "LOYALTY"
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
    if costs.counters:
        _paid_counters(slot(source.id), source, costs.counters)
    if costs.life:
        _paid(slot(source.controller)).add(HeadField.LIFE_DELTA, -costs.life)


def paid_entities(record: EffectRecord) -> tuple[str, ...]:
    """The entities :func:`apply_paid_costs` trains at, in the order it
    reaches them: a cost half's tag's outcome entities, where a
    rule's outcome fell when the payment is the outcome; none on any other
    record."""
    reached: dict[str, _Reached] = {}
    apply_paid_costs(record, lambda ref: reached.setdefault(ref, _Reached()))
    return tuple(reached)


class _Reached:
    """A slot that keeps nothing: :func:`paid_entities` asks only who paid."""

    affected = False

    def set(self, name: str, value: float) -> None:
        """What the payment wrote, not kept."""

    def add(self, name: str, value: float) -> None:
        """What the payment wrote, not kept."""

    def forget(self, name: str) -> None:
        """What the payment left unknown, not kept."""


def _source(record: EffectRecord) -> EntityState | None:
    source = record.state.refs.source
    return record.state.entity(source) if source is not None else None


def _set_each(slot: Callable, refs: tuple[str, ...], name: str, value: int) -> None:
    for ref in refs:
        _paid(slot(ref)).set(name, value)


def _paid_counters(entry, source: EntityState, counters: dict[str, int]) -> None:
    _paid(entry)
    for counter, delta in counters.items():
        if _source_bore(source, counter, delta):
            entry.add(counter_delta_field(counter), delta)
        else:
            entry.forget(counter_delta_field(counter))


def _source_bore(source: EntityState, counter: str, delta: int) -> bool:
    """Whether the record shows ``source`` itself changing ``delta`` counters
    of the type: it holds them (a removal), or it is a loyalty cost."""
    if source.zone != Zone.BATTLEFIELD:
        return False
    if delta < 0:
        return source.counters.get(counter, 0) >= -delta
    return counter == _LOYALTY


def _paid(entry):
    """``entry``, affected: the field targets of a gated field train only at an
    affected entity, and a cost half trains no gate to be wrong about it."""
    entry.affected = True
    return entry
