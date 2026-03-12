package com.pricepredictor.connector;

/**
 * Mutable counter for assigning sequential action numbers to abilities.
 */
public final class ActionCounter {
    private int value;

    public ActionCounter(int start) {
        this.value = start;
    }

    public int next() {
        return ++value;
    }
}
