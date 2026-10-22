package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardEdition;
import forge.item.SealedTemplate;

import java.util.Optional;

/** Which booster a set's packs are opened from, for the pool, match and draft workers. */
final class BoosterTemplates {

    private BoosterTemplates() {
    }

    /**
     * The set's draft booster template, or empty.
     *
     * <p>Read off the edition rather than {@code StaticData.getBoosters()}: that
     * map holds the same Draft templates (keyed by bare code, other booster
     * types under an affixed key), but only under each set's own code, while
     * the editions lookup also resolves a set's alias and two-letter code.
     *
     * <p>Empty rather than an exception for a set Forge does not know: the set
     * code can come from a decks file or a {@code -D} property, and an unknown
     * edition is a configuration error the caller names, not a
     * {@code NullPointerException} from inside the lookup.
     */
    static Optional<SealedTemplate> forSet(String setCode) {
        return edition(setCode).map(edition -> edition.getBoosterTemplate("Draft"));
    }

    /**
     * The code the set's edition goes by, which an alias resolves to, or
     * {@code setCode} unchanged when Forge knows no such set.
     *
     * <p>What a corpus line records: every downstream join compares set codes
     * as strings, so a pool opened under {@code CON} and one opened under
     * {@code CFX} must not read as two sets.
     */
    static String canonicalCode(String setCode) {
        return edition(setCode).map(CardEdition::getCode).orElse(setCode);
    }

    private static Optional<CardEdition> edition(String setCode) {
        return Optional.ofNullable(StaticData.instance().getEditions().get(setCode));
    }
}
