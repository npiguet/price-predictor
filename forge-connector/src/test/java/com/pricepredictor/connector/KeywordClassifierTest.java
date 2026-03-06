package com.pricepredictor.connector;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

class KeywordClassifierTest {

    @ParameterizedTest
    @ValueSource(strings = {
            "flying", "trample", "deathtouch", "vigilance", "first strike",
            "double strike", "haste", "lifelink", "reach", "flash",
            "indestructible", "menace", "hexproof", "defender"
    })
    void passiveKeywordsAreNotActivatable(String keyword) {
        assertFalse(KeywordClassifier.isActivatable(keyword));
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "kicker", "cycling", "equip", "ninjutsu", "evoke",
            "flashback", "morph", "bestow", "dash", "unearth",
            "madness", "foretell", "escape", "crew"
    })
    void activatableKeywordsAreActivatable(String keyword) {
        assertTrue(KeywordClassifier.isActivatable(keyword));
    }

    @ParameterizedTest
    @ValueSource(strings = {"unknown_keyword", "made_up", "notreal"})
    void unknownKeywordsDefaultToPassive(String keyword) {
        assertFalse(KeywordClassifier.isActivatable(keyword));
    }

    @ParameterizedTest
    @ValueSource(strings = {"Flying", "KICKER", "Cycling"})
    void classificationIsCaseInsensitive(String keyword) {
        // Flying is passive, Kicker/Cycling are activatable
        if (keyword.equalsIgnoreCase("flying")) {
            assertFalse(KeywordClassifier.isActivatable(keyword));
        } else {
            assertTrue(KeywordClassifier.isActivatable(keyword));
        }
    }
}
