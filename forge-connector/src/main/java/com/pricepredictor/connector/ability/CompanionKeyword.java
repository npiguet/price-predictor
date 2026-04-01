package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
import forge.game.keyword.Companion;
import forge.game.keyword.KeywordInterface;

import java.util.Optional;

/**
 * Companion keyword — includes the companion restriction description.
 */
public record CompanionKeyword(NonBlankString descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.STATIC;
    }

    public static CompanionKeyword of(KeywordInterface ki, Companion comp) {
        String compTitle = Optional.ofNullable(comp.getDescription())
                .filter(d -> !d.isEmpty())
                .map(AbilityDescription::stripReminderText)
                .filter(d -> !d.isEmpty())
                .map(d -> ki.getTitle() + " \u2014 " + d)
                .orElse(ki.getTitle());
        return new CompanionKeyword(NonBlankString.require(AbilityDescription.applyCasing(compTitle)));
    }
}
