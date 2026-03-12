package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.keyword.Companion;
import forge.game.keyword.KeywordInterface;

/**
 * Companion keyword — includes the companion restriction description.
 */
public record CompanionKeyword(String descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.STATIC;
    }

    public static CompanionKeyword of(KeywordInterface ki, Companion comp) {
        String compDesc = comp.getDescription();
        if (compDesc != null && !compDesc.isEmpty()) {
            compDesc = AbilityDescription.stripReminderText(compDesc);
        }
        String compTitle = (compDesc != null && !compDesc.isEmpty())
                ? ki.getTitle() + " \u2014 " + compDesc : ki.getTitle();
        return new CompanionKeyword(AbilityDescription.applyCasing(compTitle));
    }
}
