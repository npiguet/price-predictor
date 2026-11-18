package com.pricepredictor.connector.ability;

import com.google.common.base.Strings;
import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordWithTypeInterface;

import java.util.EnumSet;
import java.util.Set;

/**
 * Standard keyword ability — handles defined keywords and the undefined keyword fallback.
 * Covers KICKER dual cost, AFFINITY type description, and internal keyword reminder text.
 */
public record StandardKeyword(AbilityType type, String descriptionText) implements Ability {

    /** Keywords Forge uses internally, which no card prints by name. */
    private static final Set<Keyword> INTERNAL_KEYWORDS =
            EnumSet.of(Keyword.MAYFLASHCOST, Keyword.MAYFLASHSAC);

    public static StandardKeyword of(KeywordInterface ki, Keyword kw) {
        AbilityType kwType = KeywordTypes.classify(
                kw, !ki.getAbilities().isEmpty(), !ki.getTriggers().isEmpty());
        return new StandardKeyword(kwType, AbilityDescription.applyCasing(fullTitle(ki, kw)));
    }

    /** The keyword's own title, then its cost-reduction sentence when it has one. */
    private static String fullTitle(KeywordInterface ki, Keyword kw) {
        String title = titleFor(ki, kw);
        String reduceCost = new ReduceCost(ki.getOriginal()).sentence(kw);
        return reduceCost == null ? title : title + ". " + reduceCost;
    }

    /** The keyword's base title, overridden where the keyword words its own. */
    private static String titleFor(KeywordInterface ki, Keyword kw) {
        String title = baseTitle(ki, kw);
        return switch (kw) {
            case KICKER -> Kicker.of(ki).titled(title);
            case AFFINITY -> ki instanceof KeywordWithTypeInterface kti
                    ? "Affinity for " + kti.getTypeDescription() : title;
            case CRAFT -> Craft.titleOf(ki);
            case PROTECTION -> new Protection(ki.getOriginal()).title();
            default -> title;
        };
    }

    /**
     * The title a keyword starts from, before {@link #of}'s per-keyword
     * overrides: Forge's own title, or the script's original text where that
     * title is empty.
     *
     * <p>A defined keyword also falls back to the original when its title
     * carries surrounding whitespace, and an internal keyword reads as its
     * reminder text whenever it has one. An undefined keyword has neither
     * rule, and keeps any non-empty title.
     */
    private static String baseTitle(KeywordInterface ki, Keyword kw) {
        String title = ki.getTitle();
        if (kw == Keyword.UNDEFINED) {
            return Strings.isNullOrEmpty(title) ? ki.getOriginal() : title;
        }
        if (INTERNAL_KEYWORDS.contains(kw) && !Strings.isNullOrEmpty(ki.getReminderText())) {
            return ki.getReminderText();
        }
        return Strings.isNullOrEmpty(title) || !title.trim().equals(title)
                ? ki.getOriginal() : title;
    }
}
