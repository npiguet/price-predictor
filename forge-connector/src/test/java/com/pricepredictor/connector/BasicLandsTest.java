package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** {@link BasicLands} against Forge's real card database and type rule. */
@ExtendWith(ForgeExtension.class)
class BasicLandsTest {

    private static final List<String> BASICS = List.of(
            "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
            "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
            "Snow-Covered Mountain", "Snow-Covered Forest");

    @Test
    void everyBasicLandIsRecognisedByName() {
        for (String basic : BASICS) {
            assertTrue(BasicLands.isBasic(basic), basic);
        }
        assertFalse(BasicLands.isBasic("Lightning Bolt"));
    }

    @Test
    void aNameForgeDoesNotKnowIsNotABasicLand() {
        assertFalse(BasicLands.isBasic("Not A Card Name"));
    }

    @Test
    void theCardAndItsNameGetTheSameAnswer() {
        for (String name : List.of("Snow-Covered Island", "Wastes", "Lightning Bolt", "Mutavault")) {
            assertEquals(BasicLands.isBasic(name),
                    BasicLands.isBasic(DeckMaterializer.lookUp(name).orElseThrow()), name);
        }
    }
}
