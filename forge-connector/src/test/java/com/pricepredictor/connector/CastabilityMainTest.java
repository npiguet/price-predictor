package com.pricepredictor.connector;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.CastabilityMain.CASTABLE;
import static com.pricepredictor.connector.CastabilityMain.UNCASTABLE;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The castability consult's verdicts, against a live Forge.
 *
 * <p>Integration-tagged because the verdict is the engine's own answer to
 * "does this card have a spell ability", which needs the card database and
 * the card factory. The card names are deliberately the most ordinary
 * representative of each kind, so a wrong verdict here is a wrong verdict
 * for the whole kind.
 */
@Tag("integration")
@ExtendWith(ForgeExtension.class)
class CastabilityMainTest {

    private static String verdict(String name) {
        Map<String, String> verdicts = CastabilityMain.consult(List.of(name));
        assertTrue(verdicts.containsKey(name), name + " should get a verdict");
        return verdicts.get(name);
    }

    @Test
    void anInstantIsCastable() {
        assertEquals(CASTABLE, verdict("Lightning Bolt"));
    }

    @Test
    void aSorceryIsCastable() {
        assertEquals(CASTABLE, verdict("Wrath of God"));
    }

    /**
     * Benediction of Moons has no {@code A:SP$} line at all: its spell is
     * synthesised by the Haunt keyword. Only the engine knows that, which is
     * why the consult instantiates the card instead of reading its script.
     */
    @Test
    void aSpellSynthesisedFromAKeywordIsCastable() {
        assertEquals(CASTABLE, verdict("Benediction of Moons"));
    }

    @Test
    void aCreatureIsCastable() {
        assertEquals(CASTABLE, verdict("Grizzly Bears"));
    }

    @Test
    void aNoncreaturePermanentIsCastable() {
        assertEquals(CASTABLE, verdict("Sol Ring"));
    }

    @Test
    void aLandIsUncastable() {
        assertEquals(UNCASTABLE, verdict("Forest"));
    }

    @Test
    void aNonbasicLandIsUncastable() {
        assertEquals(UNCASTABLE, verdict("Wasteland"));
    }

    @Test
    void aNameForgeDoesNotKnowIsOmittedNotJudged() {
        assertEquals(Map.of(), CastabilityMain.consult(List.of("No Such Card Was Ever Printed")));
    }

    /** The consult ranks; it never drops a card Forge recognises. */
    @Test
    void everyRecognisedNameGetsExactlyOneVerdictInRequestOrder() {
        List<String> names = List.of(
                "Forest", "Lightning Bolt", "No Such Card Was Ever Printed", "Grizzly Bears");

        Map<String, String> verdicts = CastabilityMain.consult(names);

        assertEquals(List.of("Forest", "Lightning Bolt", "Grizzly Bears"),
                List.copyOf(verdicts.keySet()));
    }
}
