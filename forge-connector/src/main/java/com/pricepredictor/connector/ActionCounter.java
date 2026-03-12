package com.pricepredictor.connector;

/**
 * Mutable counter for assigning sequential action numbers to abilities.
 */
final class ActionCounter {
    private int value;

    ActionCounter(int start) {
        this.value = start;
    }

    int next() {
        return ++value;
    }

    Integer nextIfActionable(AbilityType type) {
        return type.isActionable() ? next() : null;
    }
}
