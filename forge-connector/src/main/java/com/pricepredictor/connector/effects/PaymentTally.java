package com.pricepredictor.connector.effects;

import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

/**
 * What a payment did to counters and life, as the bus told it while the
 * payment ran.
 *
 * <p>Neither survives on the {@link forge.game.spellability.SpellAbility}: a
 * removed counter is in no paid list, and the engine's own life figure is the
 * last {@code payLife} call, not their sum.
 *
 * @param counters the net count per counter type (upper case), in type order:
 *                 negative for counters removed, positive for counters put
 * @param life     the life paid, the sum of every payment
 */
record PaymentTally(Map<String, Integer> counters, int life) {

    /** Nothing moved. */
    static final PaymentTally NONE = new PaymentTally(Map.of(), 0);

    PaymentTally {
        counters = Collections.unmodifiableMap(new TreeMap<>(counters));
    }
}
