package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.StdStreams;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

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
    void aMalformedPropertyKeepsItsDefaultRatherThanFailing() throws Throwable {
        System.setProperty(CollectionCaps.MANA_CAP, "not-a-number");
        String warning;
        try {
            warning = StdStreams.captureStderr(() -> assertEquals(
                    CollectionCaps.defaults().manaCap(),
                    CollectionCaps.fromSystemProperties().manaCap())).strip();
        } finally {
            System.clearProperty(CollectionCaps.MANA_CAP);
        }
        assertEquals(1, warning.lines().count(), warning);
        assertTrue(warning.contains(CollectionCaps.MANA_CAP) && warning.contains("not-a-number"),
                warning);
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

    /**
     * {@code effect.aimed.at} is read by its presence (FR-009a): unset, the run
     * aims at nothing; set blank, at no rule in particular, which a coverage
     * run with no {@code --directed-fork-rules} is; set, at its rules.
     */
    @Test
    void theAimedAtRulesAreReadByThePropertysPresence() {
        assertEquals(Optional.empty(), CollectionCaps.fromSystemProperties().aimedAt());
        try {
            System.setProperty(CollectionCaps.AIMED_AT, "");
            assertEquals(Optional.of(List.of()), CollectionCaps.fromSystemProperties().aimedAt());
            System.setProperty(CollectionCaps.AIMED_AT, "704.5f, 702.19b");
            assertEquals(Optional.of(List.of("704.5f", "702.19b")),
                    CollectionCaps.fromSystemProperties().aimedAt());
        } finally {
            System.clearProperty(CollectionCaps.AIMED_AT);
        }
        assertEquals(Optional.empty(), CollectionCaps.defaults().aimedAt());
    }

    /**
     * {@code effect.count.dropped.events} (FR-041): off unless the supervisor
     * passes {@code --count-dropped-events}, which sends it {@code true}.
     */
    @Test
    void theDroppedEventCountIsOffUnlessThePropertySaysTrue() {
        assertFalse(CollectionCaps.defaults().countDroppedEvents());
        assertFalse(CollectionCaps.fromSystemProperties().countDroppedEvents());
        try {
            System.setProperty(CollectionCaps.COUNT_DROPPED_EVENTS, "true");
            assertTrue(CollectionCaps.fromSystemProperties().countDroppedEvents());
            System.setProperty(CollectionCaps.COUNT_DROPPED_EVENTS, "false");
            assertFalse(CollectionCaps.fromSystemProperties().countDroppedEvents());
        } finally {
            System.clearProperty(CollectionCaps.COUNT_DROPPED_EVENTS);
        }
    }

    /**
     * A game's counts are appended as one {@code event_type, phase, count}
     * line each, so a lifetime's file is the concatenation of its games and
     * the supervisor sums the lines.
     */
    @Test
    void aGamesDroppedEventsAppendOneTabSeparatedLinePerTypeAndPhase(@TempDir Path dir)
            throws Exception {
        Path file = dir.resolve("run.0-l1" + DroppedEvents.SUFFIX);
        DroppedEvents first = new DroppedEvents();
        first.count(EventType.ZONE_CHANGE, "upkeep");
        first.count(EventType.ZONE_CHANGE, "upkeep");
        first.count(EventType.MANA_PRODUCED, "main1");
        first.appendTo(file);
        DroppedEvents second = new DroppedEvents();
        second.count(EventType.ZONE_CHANGE, "upkeep");
        second.appendTo(file);
        // A game that dropped nothing appends nothing.
        new DroppedEvents().appendTo(file);

        assertEquals(List.of(
                        "zone_change\tupkeep\t2", "mana_produced\tmain1\t1",
                        "zone_change\tupkeep\t1"),
                Files.readAllLines(file));
    }

    /** Before a game has a phase, an event is counted under {@code none}. */
    @Test
    void anEventBeforeAnyPhaseIsCountedUnderNone() {
        assertEquals(DroppedEvents.NO_PHASE, DroppedEvents.phaseOf(null));
        assertEquals("none", DroppedEvents.NO_PHASE);
    }
}
