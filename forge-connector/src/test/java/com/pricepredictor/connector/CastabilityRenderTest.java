package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The {@code {name: verdict}} JSON object the Python side reads
 * ({@code effects.infrastructure.castability_connector}). Forge-free.
 */
class CastabilityRenderTest {

    @Test
    void rendersOneVerdictPerNameAsAJsonObjectInOrder() {
        Map<String, String> verdicts = new LinkedHashMap<>();
        verdicts.put("Lightning Bolt", CastabilityMain.CASTABLE);
        verdicts.put("Forest", CastabilityMain.UNCASTABLE);

        assertEquals("{\"Lightning Bolt\":\"castable\",\"Forest\":\"uncastable\"}",
                CastabilityMain.render(verdicts));
    }

    @Test
    void rendersNoVerdictsAsAnEmptyObject() {
        assertEquals("{}", CastabilityMain.render(Map.of()));
    }

    @Test
    void escapesQuotesInsideACardName() {
        assertEquals("{\"Kongming, \\\"Sleeping Dragon\\\"\":\"castable\"}",
                CastabilityMain.render(
                        Map.of("Kongming, \"Sleeping Dragon\"", CastabilityMain.CASTABLE)));
    }
}
