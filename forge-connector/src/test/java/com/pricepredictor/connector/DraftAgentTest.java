package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/** Tests for {@link DraftAgent}; pure, so it runs in the default build. */
class DraftAgentTest {

    @Test
    void randomOverrideFractionByAgent() {
        assertEquals(0.0, DraftAgent.randomOverrideFraction("forge-full"));
        assertEquals(0.30, DraftAgent.randomOverrideFraction("forge-r30"));
        assertEquals(1.0, DraftAgent.randomOverrideFraction("forge-r100"));
    }

    @Test
    void anUnknownLabelIsRefusedNotDraftedAsForgeAi() {
        IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class,
                () -> DraftAgent.randomOverrideFraction("forge-r3O"));
        assertTrue(thrown.getMessage().contains("forge-r3O"), thrown.getMessage());
    }

    @Test
    void aMixLabelNeitherDraftedNorExternalIsRefusedAtParseTimeNamingIt() {
        AgentMix mix = AgentMix.parse("forge-full:5,forge-r30:1,gen-3:2,forge-r3O:1");

        IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class,
                () -> DraftAgent.requireKnown(mix.labels(), Set.of("gen-3")));
        assertTrue(thrown.getMessage().contains("forge-r3O"), thrown.getMessage());
        assertFalse(thrown.getMessage().contains("[gen-3"), thrown.getMessage());
    }

    @Test
    void aMixOfKnownAndExternalLabelsIsAccepted() {
        AgentMix mix = AgentMix.parse("forge-full:5,forge-r100:1,gen-3:2");
        assertDoesNotThrow(() -> DraftAgent.requireKnown(mix.labels(), Set.of("gen-3")));
    }
}
