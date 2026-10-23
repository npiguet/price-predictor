package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import static com.pricepredictor.connector.MatchFixtures.HEADER;
import static com.pricepredictor.connector.MatchFixtures.RUN_ID;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** The five leading fields both sealed corpus lines share. */
class MatchHeaderTest {

    @Test
    void theHeaderAloneIsFiveFieldsWithAnIsoTimestamp() {
        assertEquals("2026-04-22T14:30:05Z;" + RUN_ID + ";RVR;forge-best;gen-2", HEADER.line());
    }

    @Test
    void theRestFollowsTheHeaderFieldByField() {
        assertEquals("2026-04-22T14:30:05Z;" + RUN_ID + ";RVR;forge-best;gen-2;x;;y",
                HEADER.line("x", "", "y"));
    }
}
