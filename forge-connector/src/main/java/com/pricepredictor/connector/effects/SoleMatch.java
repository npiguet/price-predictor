package com.pricepredictor.connector.effects;

import java.util.function.Predicate;

/**
 * The only item of a list that matches, refusing when several do.
 *
 * <p>Every fallback of {@link ProvenanceResolver} that recognises a line by
 * resemblance refuses on two candidates: a record with no key is still
 * trainable through its state and payload, while a record with the wrong key
 * silently trains the wrong line.
 */
final class SoleMatch {

    private SoleMatch() {
    }

    /** The position of the only item that matches, or -1 when none or several do. */
    static <T> int indexOf(Iterable<? extends T> items, Predicate<? super T> matches) {
        int index = 0;
        int hit = -1;
        for (T item : items) {
            if (matches.test(item)) {
                if (hit >= 0) {
                    return -1;
                }
                hit = index;
            }
            index++;
        }
        return hit;
    }

    /** The only item that matches, or null when none or several do. */
    static <T> T of(Iterable<? extends T> items, Predicate<? super T> matches) {
        T hit = null;
        boolean found = false;
        for (T item : items) {
            if (matches.test(item)) {
                if (found) {
                    return null;
                }
                hit = item;
                found = true;
            }
        }
        return hit;
    }
}
