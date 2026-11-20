package com.pricepredictor.connector.effects;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Forge's produced-mana text ({@code "G"}, {@code "G G"}, {@code "1"}), read
 * the one way both the mana-activation channel and a mana event's run
 * parameters read it, so a tapped land and a replaced mana production are
 * comparable rows rather than two spellings of one fact.
 */
final class ProducedMana {

    private ProducedMana() {
    }

    /**
     * Forge's produced-mana string as counts per colour letter.
     *
     * <p>Returned as a map rather than rendered here, because
     * {@link EffectEvent#param} renders a map as a JSON object and a string as
     * a JSON string — and the reader expects {@code {"B":1}}, not {@code "{...}"}.
     */
    static Map<String, Integer> byColor(String produced) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (String atom : produced.split(" ")) {
            if (atom.isEmpty()) {
                continue;
            }
            // A bare number is that many generic; anything else is one mana of
            // the colour it names, "C" included.
            if (atom.chars().allMatch(Character::isDigit)) {
                counts.merge("C", Integer.parseInt(atom), Integer::sum);
            } else {
                counts.merge(atom.toUpperCase(Locale.ROOT), 1, Integer::sum);
            }
        }
        return counts;
    }
}
