package com.pricepredictor.connector;

import java.util.List;

/**
 * One line of a validation matches file: both decks fully specified,
 * {@code deckA_card1|...|card40;deckB_card1|...|card40}.
 *
 * @param deckA deck A's card names, in order, duplicates kept
 * @param deckB deck B's card names
 */
record ValidationMatch(List<String> deckA, List<String> deckB) {

    /**
     * The match a line names.
     *
     * @throws IllegalArgumentException when the line has no deck B field
     */
    static ValidationMatch parse(String line) {
        String[] decks = line.split(CardNameList.FIELD_SEPARATOR, 2);
        if (decks.length < 2) {
            throw new IllegalArgumentException(
                    "Malformed validation match (need 'deckA;deckB'): " + line);
        }
        return new ValidationMatch(CardNameList.parse(decks[0]), CardNameList.parse(decks[1]));
    }
}
