package com.pricepredictor.connector;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;
import java.util.Map;

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

    private static CastabilityMain.Verdict verdict(String name) {
        Map<String, CastabilityMain.Verdict> verdicts = CastabilityMain.consult(List.of(name));
        assertTrue(verdicts.containsKey(name), name + " should get a verdict");
        return verdicts.get(name);
    }

    @ParameterizedTest(name = "{0}")
    @CsvSource(delimiter = ';', value = {
            "an instant is castable;                                     Lightning Bolt;       CASTABLE",
            "a sorcery is castable;                                      Wrath of God;         CASTABLE",
            // Benediction of Moons has no A:SP$ line at all: its spell is synthesised by the
            // Haunt keyword. Only the engine knows that, which is why the consult instantiates
            // the card instead of reading its script.
            "a spell synthesised from a keyword (Benediction of Moons) is castable; Benediction of Moons; CASTABLE",
            "a creature is castable;                                     Grizzly Bears;        CASTABLE",
            "a noncreature permanent is castable;                        Sol Ring;             CASTABLE",
            "a basic land is uncastable;                                 Forest;               UNCASTABLE",
            "a nonbasic land is uncastable;                              Wasteland;            UNCASTABLE",
    })
    void theEngineJudgesEachKindOfCard(String ignoredCase, String name, CastabilityMain.Verdict expected) {
        assertEquals(expected, verdict(name));
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

        Map<String, CastabilityMain.Verdict> verdicts = CastabilityMain.consult(names);

        assertEquals(List.of("Forest", "Lightning Bolt", "Grizzly Bears"),
                List.copyOf(verdicts.keySet()));
    }
}
