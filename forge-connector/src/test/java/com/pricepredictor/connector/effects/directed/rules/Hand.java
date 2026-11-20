package com.pricepredictor.connector.effects.directed.rules;

import java.util.List;

/** The cards a scenario board puts in a hand, by name, in the order they are put there. */
record Hand(List<String> names) {

    /** A hand of {@code names}, in the order they are put there. */
    static Hand of(String... names) {
        return new Hand(List.of(names));
    }
}
