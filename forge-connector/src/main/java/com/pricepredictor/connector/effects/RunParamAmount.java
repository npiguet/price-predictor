package com.pricepredictor.connector.effects;

import java.util.Collection;
import java.util.Map;

import static com.pricepredictor.connector.effects.RunParamValues.firstInt;

/** How a mode's amount is read off its map; null where the map holds none. */
@FunctionalInterface
interface RunParamAmount {
    Integer read(Map<String, Object> byName);

    /** Nothing read: the counter channel's own reading of the map applies. */
    RunParamAmount NONE = byName -> null;
    /** One: a mode Forge fires once per card or per counter. */
    RunParamAmount ONE = byName -> 1;

    /** The first of these keys holding a number. */
    static RunParamAmount key(String... keys) {
        return byName -> firstInt(byName, keys);
    }

    /** How many objects the collection under {@code key} holds. */
    static RunParamAmount sizeOf(String key) {
        return byName -> byName.get(key) instanceof Collection<?> many ? many.size() : null;
    }
}
