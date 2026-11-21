package com.pricepredictor.connector.effects;

import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementHandler;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerHandler;

/** Triggers and replacements of a chosen mode, parsed the way a test needs them, not taken off a card. */
final class TestParsed {

    private TestParsed() {
    }

    /**
     * A trigger of a chosen mode, on a host card the mode does not care about.
     *
     * <p>Parsed rather than taken off a real card, because what is under test
     * is the reading of one mode's run parameters and finding a printed card
     * per mode would tie the test to the set list.
     */
    static Trigger triggerOfMode(String mode) {
        return TriggerHandler.parseTrigger(
                "Mode$ " + mode + " | TriggerZones$ Battlefield"
                        + " | TriggerDescription$ under test",
                TestCards.build("Paralyze"), true);
    }

    /** A replacement of a chosen mode, parsed for the reason {@link #triggerOfMode} is. */
    static ReplacementEffect replacementOfMode(String mode) {
        return ReplacementHandler.parseReplacement(
                "Event$ " + mode + " | Description$ under test",
                TestCards.build("Paralyze"), true);
    }
}
