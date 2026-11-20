package com.pricepredictor.connector.effects.directed.rules;

import java.util.List;

/** The cards a scenario board puts in a library, by name, the top card first. */
record Library(List<String> names) {

    /** A library of {@code names}, the top card first; none for an empty library. */
    static Library of(String... names) {
        return new Library(List.of(names));
    }
}
