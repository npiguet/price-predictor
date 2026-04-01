package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.staticability.StaticAbility;
import forge.game.staticability.StaticAbilityMode;

/**
 * Ability wrapping a Forge StaticAbility. Dynamically returns STATIC or ADDITIONAL_COST
 * for self-targeting RaiseCost/OptionalCost statics.
 */
public record StaticAbilityEntry(AbilityType type, String descriptionText) implements Ability {

    public static StaticAbilityEntry of(StaticAbility sa) {
        if (sa.getKeyword() != null) return null;
        String normalized = AbilityDescription.normalize(sa.getParam("Description"));
        if (normalized == null) return null;

        AbilityType effectiveType = AbilityType.STATIC;
        if (sa.checkMode(StaticAbilityMode.RaiseCost) || sa.checkMode(StaticAbilityMode.OptionalCost)) {
            String validCard = sa.getParam("ValidCard");
            String affected = sa.getParam("Affected");
            boolean selfCost = "Card.Self".equals(validCard) || "Card.Self".equals(affected);
            if (selfCost) {
                effectiveType = AbilityType.ADDITIONAL_COST;
                normalized = ForgeParams.stripAdditionalCostPrefix(normalized);
            }
        }

        return new StaticAbilityEntry(effectiveType, normalized);
    }
}
