package com.pricepredictor.connector.effects;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * How one trigger or replacement mode's run-parameter map is read: the event
 * type it is, what it is about, and the slots it fills, in the order they
 * are written.
 *
 * <p>A row says everything its mode reads. Forge spells the same quantity
 * differently per mode, and fills one key with an amount in one mode and an
 * ordinal or a running total in another, so nothing is read for a mode
 * because of its event type alone.
 *
 * @param type     the event type the mode's event is
 * @param subjects what the event is about, read off the map
 * @param slots    the parameters the event carries, each read off the map
 */
record ModeRow(EventType type, Subjects subjects, List<Slot> slots) {

    static ModeRow of(EventType type, Subjects subjects, Slot... slots) {
        return new ModeRow(type, subjects, List.of(slots));
    }

    /** What an event is about: the entities the map names under the row's key. */
    @FunctionalInterface
    interface Subjects {
        Set<String> of(Map<String, Object> byName);
    }

    /** One parameter, or a few written together, read off the map into the event. */
    @FunctionalInterface
    interface Slot {
        void read(EffectEvent event, RunParams params);
    }

    /**
     * A run-parameter map keyed by {@code AbilityKey} name, and the one counter
     * type it is read for (null to read it whole): an {@code AddCounter}
     * replacement's record over several types is written once per type.
     */
    record RunParams(Map<String, Object> byName, String counterType) {

        Object get(String key) {
            return byName.get(key);
        }
    }

    /**
     * The direction a mode moves a life total or a counter count: a gain is
     * written positive, a loss negative, and an amount read already signed
     * as it was read.
     */
    enum Sign {
        GAIN, LOSS, AS_READ;

        Integer apply(Integer amount) {
            if (amount == null || this == AS_READ) {
                return amount;
            }
            int size = Math.abs(amount);
            return this == LOSS ? -size : size;
        }
    }
}
