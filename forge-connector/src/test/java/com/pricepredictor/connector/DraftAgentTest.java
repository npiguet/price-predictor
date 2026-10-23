package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Tests for {@link DraftAgent}; pure, so it runs in the default build. */
class DraftAgentTest {

    @Test
    void randomOverrideFractionByAgent() {
        assertEquals(0.0, DraftAgent.randomOverrideFraction("forge-full"));
        assertEquals(0.30, DraftAgent.randomOverrideFraction("forge-r30"));
        assertEquals(1.0, DraftAgent.randomOverrideFraction("forge-r100"));
        assertEquals(0.0, DraftAgent.randomOverrideFraction("unknown"));
    }
}
