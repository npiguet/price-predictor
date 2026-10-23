package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.staticability.StaticAbility;
import forge.game.staticability.StaticAbilityMode;

/**
 * The line of a Forge StaticAbility: its normalized {@code Description$}. The
 * type is fixed when the line is built — ADDITIONAL_COST for a RaiseCost or
 * OptionalCost static that applies to the card itself, STATIC otherwise. A
 * keyword's static renders as the keyword line instead, so it has none.
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
                normalized = AbilityDescription.stripAdditionalCostPrefix(normalized);
            }
        }

        return new StaticAbilityEntry(effectiveType, normalized);
    }
}
