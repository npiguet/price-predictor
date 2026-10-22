package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The caps a collecting worker runs under, and how they reach it from the
 * supervisor.
 */
class CollectionCapsTest {

    @Test
    void theDefaultCapsAreTheContracts() {
        CollectionCaps defaults = CollectionCaps.defaults();
        // One record per mana ability per game: this collector is built per
        // game, and a Mountain's dozen taps observe a board that barely moved.
        assertEquals(1, defaults.manaCap());
        assertEquals(0.1, defaults.playabilityRate());
        assertEquals(2, defaults.interventionsPerGame());
        assertEquals(2, defaults.probesPerGame());
        // A worker started without the property probes nothing; the eight
        // keywords a run probes by default are the supervisor's to send.
        assertTrue(defaults.probeKeywords().isEmpty());
    }

    /**
     * The caps arrive as system properties the supervisor sets.
     *
     * <p>Every one is a per-worker-process quantity the Python side cannot
     * observe, so the flags are inert unless this reads them — which is exactly
     * the state they were in before: parsed by the CLI and never sent.
     */
    @Test
    void capsAreReadFromSystemProperties() {
        Map<String, String> properties = Map.of(
                "effect.mana.cap", "7",
                "effect.playability.rate", "0.5",
                "effect.interventions.per.game", "3",
                "effect.probes.per.game", "4",
                "effect.probe.keywords", "wither, infect");
        properties.forEach(System::setProperty);
        try {
            CollectionCaps caps = CollectionCaps.fromSystemProperties();
            assertEquals(7, caps.manaCap());
            assertEquals(0.5, caps.playabilityRate());
            assertEquals(3, caps.interventionsPerGame());
            assertEquals(4, caps.probesPerGame());
            assertEquals(List.of("wither", "infect"), caps.probeKeywords());
        } finally {
            properties.keySet().forEach(System::clearProperty);
        }
    }

    @Test
    void anUnsetPropertyKeepsItsDefault() {
        // A worker started without instrumentation sets none of them, and one
        // started by an older supervisor may set only some.
        CollectionCaps caps = CollectionCaps.fromSystemProperties();
        assertEquals(CollectionCaps.defaults(), caps);
    }

    @Test
    void aMalformedPropertyKeepsItsDefaultRatherThanFailing() {
        System.setProperty(CollectionCaps.MANA_CAP, "not-a-number");
        PrintStream realErr = System.err;
        ByteArrayOutputStream warnings = new ByteArrayOutputStream();
        System.setErr(new PrintStream(warnings, true, StandardCharsets.UTF_8));
        try {
            assertEquals(
                    CollectionCaps.defaults().manaCap(),
                    CollectionCaps.fromSystemProperties().manaCap());
        } finally {
            System.setErr(realErr);
            System.clearProperty(CollectionCaps.MANA_CAP);
        }
        String warning = warnings.toString(StandardCharsets.UTF_8).strip();
        assertEquals(1, warning.lines().count(), warning);
        assertTrue(warning.contains(CollectionCaps.MANA_CAP) && warning.contains("not-a-number"),
                warning);
    }

    @Test
    void probesAreOffUntilKeywordsAreNamed() {
        assertFalse(CollectionCaps.defaults().probesEnabled());
        assertTrue(new CollectionCaps(
                2000, 0.1, 2, 2, List.of("wither"), 0.0, 0.0)
                .probesEnabled());
    }

    /** The two knobs of the random seat travel like every other cap. */
    @Test
    void theRandomSeatKnobsAreReadFromSystemProperties() {
        System.setProperty("effect.random.seat.share", "0.125");
        System.setProperty("effect.random.seat.probability", "0.3");
        try {
            CollectionCaps caps = CollectionCaps.fromSystemProperties();
            assertEquals(0.125, caps.randomSeatShare());
            assertEquals(0.3, caps.randomSeatProbability());
            assertTrue(caps.randomSeatEnabled());
        } finally {
            System.clearProperty("effect.random.seat.share");
            System.clearProperty("effect.random.seat.probability");
        }
        assertEquals(0.0, CollectionCaps.defaults().randomSeatShare());
        assertFalse(CollectionCaps.defaults().randomSeatEnabled());
    }
}
