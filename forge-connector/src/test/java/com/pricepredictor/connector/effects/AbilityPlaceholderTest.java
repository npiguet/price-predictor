package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A trigger's {@code ABILITY} placeholder filled with a die roll's sentence
 * does not repeat the word the description already ends with.
 */
@ExtendWith(ForgeExtension.class)
class AbilityPlaceholderTest {

    private static final Path CARDS = ForgeEnvironmentInitializer.findCardsFolder();

    @Test
    void aDieRollFillingThePlaceholderDropsTheRepeatedThen() {
        // Druid of the Emerald Grove: "When Druid of the Emerald Grove enters, search your
        // library for up to two basic land cards and reveal them, then roll a d20."
        List<String> lines = ConvertedScript.fromFile(new RulesParser(), CARDS,
                "d/druid_of_the_emerald_grove.txt").texts();
        assertTrue(lines.contains("triggered: when CARDNAME enters, search your library for up to"
                + " two basic land cards and reveal them, then roll a d20."), lines.toString());
    }
}
