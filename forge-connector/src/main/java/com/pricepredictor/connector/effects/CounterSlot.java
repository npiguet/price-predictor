package com.pricepredictor.connector.effects;

import java.util.Map;

/**
 * Which counter moved and by how much, signed: positive for counters put on,
 * negative for counters removed.
 *
 * <p>Forge says it three ways. {@code CounterAdded} and {@code
 * CounterRemoved} fire once per counter, their {@code CounterAmount} and
 * {@code NewCounterAmount} the count after that one counter moved, so each
 * is one counter. {@code CounterAddedOnce} and {@code CounterRemovedOnce}
 * carry the number moved as {@code CounterAmount}. The {@code AddCounter}
 * replacement hands over {@code CounterMap}, a per-player multiset that it
 * rewrites <b>in place</b> — which is also why the engine-side copy of the
 * parameter map has to reach inside it, or both halves of the record read
 * the same however carefully this reads them. The {@code RemoveCounter}
 * replacement carries the count the removal would leave ({@code Result}),
 * so what it removes is that count less the card's count now
 * ({@link RunParamSlots#removal}).
 *
 * <p>A map naming several counter types at once, read whole, has one delta
 * and no single type: the total is the honest answer and a joined type name
 * would be a value no reader could match. Read for one type, it is that
 * type's count alone, summed over the map's players (0 where the map holds
 * none of it).
 *
 * @param amount how the mode's amount is read, before the map's {@code CounterMap}
 * @param sign   the direction the mode moves a count
 */
record CounterSlot(RunParamAmount amount, ModeRow.Sign sign) implements ModeRow.Slot {

    @Override
    public void read(EffectEvent event, ModeRow.RunParams params) {
        if (params.counterType() != null) {
            oneType(event, params);
            return;
        }
        Object type = params.get("CounterType");
        if (type != null) {
            event.param(EventParam.COUNTER_TYPE, CardSpelling.counterName(type));
        }
        Integer read = amount.read(params.byName());
        if (read != null) {
            event.param(EventParam.DELTA, sign.apply(read));
            return;
        }
        wholeMap(event, params, type == null);
    }

    private void oneType(EffectEvent event, ModeRow.RunParams params) {
        String only = params.counterType();
        event.param(EventParam.COUNTER_TYPE, only);
        Map<String, Integer> total = RunParamValues.counterMapTotals(params.get("CounterMap"));
        event.param(EventParam.DELTA, sign.apply(total.getOrDefault(only, 0)));
    }

    /** The {@code CounterMap} summed, its type named when it holds one and no other names it. */
    private void wholeMap(EffectEvent event, ModeRow.RunParams params, boolean untyped) {
        Map<String, Integer> total = RunParamValues.counterMapTotals(params.get("CounterMap"));
        if (total.isEmpty()) {
            return;
        }
        if (total.size() == 1 && untyped) {
            event.param(EventParam.COUNTER_TYPE, total.keySet().iterator().next());
        }
        event.param(EventParam.DELTA, sign.apply(RunParamValues.sumOf(total.values())));
    }
}
