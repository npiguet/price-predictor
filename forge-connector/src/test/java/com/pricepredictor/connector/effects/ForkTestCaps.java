package com.pricepredictor.connector.effects;

/** The caps the fork record tests start from. */
final class ForkTestCaps {

    private ForkTestCaps() {
    }

    /** The caps every fork test starts from: room for every mana record. */
    static TestCaps forkCaps() {
        return TestCaps.caps().manaCap(2000);
    }
}
