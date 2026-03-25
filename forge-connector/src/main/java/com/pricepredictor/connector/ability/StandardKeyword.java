package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.cost.Cost;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordWithTypeInterface;

/**
 * Standard keyword ability — handles defined keywords and the undefined keyword fallback.
 * Covers KICKER dual cost, AFFINITY type description, and internal keyword reminder text.
 */
public record StandardKeyword(AbilityType type, String descriptionText) implements Ability {

    public static StandardKeyword of(KeywordInterface ki, Keyword kw) {
        boolean activatable = !ki.getAbilities().isEmpty();
        String title = ki.getTitle();

        if (kw == Keyword.UNDEFINED) {
            if (title == null || title.isEmpty()) {
                title = ki.getOriginal();
            }
        } else {
            if (title == null || title.isEmpty() || !title.trim().equals(title)) {
                title = ki.getOriginal();
            }
            if (AbilityType.isInternalKeyword(kw)) {
                String reminder = ki.getReminderText();
                if (reminder != null && !reminder.isEmpty()) {
                    title = reminder;
                }
            }
            if (kw == Keyword.KICKER) {
                String[] kickerParts = ki.getOriginal().split(":", 3);
                if (kickerParts.length >= 3) {
                    Cost cost2 = new Cost(kickerParts[2], false);
                    title = title + " and/or " + cost2.toSimpleString();
                }
            }
            if (kw == Keyword.AFFINITY && ki instanceof KeywordWithTypeInterface kti) {
                title = "Affinity for " + kti.getTypeDescription();
            }
            if (kw == Keyword.PROTECTION) {
                // getOriginal() returns the raw script keyword, e.g.:
                //   "Protection:Creature"         (simple type)
                //   "Protection:Card.MultiColor:multicolored"  (filter with description)
                // Build "Protection from <target>" from it.
                // Color-based protection ("K:Protection from white") already has the
                // correct format in getOriginal(), so no special-casing needed there.
                String original = ki.getOriginal();
                int firstColon = original.indexOf(':');
                if (firstColon >= 0) {
                    String details = original.substring(firstColon + 1);
                    // If the details contain a second colon, the text after it is the
                    // human-readable target (e.g. "Card.MultiColor:multicolored" → "multicolored").
                    int innerColon = details.indexOf(':');
                    if (innerColon >= 0) {
                        details = details.substring(innerColon + 1);
                    }
                    title = "Protection from " + details.toLowerCase();
                }
                // else: getOriginal() had no colon (shouldn't happen), leave title as-is
            }
        }

        AbilityType kwType = AbilityType.classifyKeyword(kw, activatable, !ki.getTriggers().isEmpty());
        return new StandardKeyword(kwType, AbilityDescription.applyCasing(title));
    }
}
