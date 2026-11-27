package com.pricepredictor.connector;

import static com.pricepredictor.connector.FaceAbilities.abilitiesOfType;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Assertions on the text the parser converts a card to. */
final class ConvertedTextAssertions {

    private ConvertedTextAssertions() {
    }

    /** The face has an ability of {@code type}, and each of {@code containsTexts} is in one of them. */
    static void assertHasAbility(CardFace card, AbilityType type, String... containsTexts) {
        var abilities = abilitiesOfType(card, type);
        assertFalse(abilities.isEmpty(), "Expected at least one " + type + " but got: " + card.abilities());
        for (String text : containsTexts) {
            assertTrue(abilities.stream().anyMatch(a -> a.descriptionText().contains(text)),
                    "Expected '" + text + "' in " + type + " abilities but got: " + abilities);
        }
    }

    /** How many times {@code needle} appears in {@code text}, overlaps counted. */
    static int occurrences(String text, String needle) {
        int found = 0;
        for (int at = text.indexOf(needle); at >= 0; at = text.indexOf(needle, at + 1)) {
            found++;
        }
        return found;
    }

    /** {@code needle} appears in {@code text}, and only once: a duplicated clause would repeat it. */
    static void assertOccursOnce(String text, String needle) {
        assertEquals(1, occurrences(text, needle), "'" + needle + "' must appear exactly once: " + text);
    }

    /** No static or activated line of the face prints the raw ETBReplacement keyword. */
    static void assertNoRawEtbReplacementKeyword(CardFace card) {
        long raw = card.abilities().stream()
                .filter(ConvertedTextAssertions::isKeywordLine)
                .filter(a -> a.descriptionText().contains("etbreplacement"))
                .count();
        assertEquals(0, raw, "ETBReplacement should not appear as raw keyword");
    }

    private static boolean isKeywordLine(Ability ability) {
        return ability.type() == AbilityType.STATIC || ability.type() == AbilityType.ACTIVATED;
    }
}
