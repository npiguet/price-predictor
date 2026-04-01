package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
import forge.game.staticability.StaticAbility;
import forge.game.staticability.StaticAbilityMode;

import java.util.Optional;

/**
 * Ability wrapping a Forge StaticAbility. Dynamically returns STATIC or ADDITIONAL_COST
 * for self-targeting RaiseCost/OptionalCost statics.
 */
public record StaticAbilityEntry(AbilityType type, NonBlankString descriptionText) implements Ability {

    public static Optional<StaticAbilityEntry> of(StaticAbility sa) {
        if (sa.getKeyword() != null) return Optional.empty();
        String desc = sa.getParam("Description");
        return NonBlankString.of(desc).flatMap(d -> AbilityDescription.normalize(d.value())).map(normalized -> {
            AbilityType effectiveType = AbilityType.STATIC;
            if (sa.checkMode(StaticAbilityMode.RaiseCost) || sa.checkMode(StaticAbilityMode.OptionalCost)) {
                String validCard = sa.getParam("ValidCard");
                String affected = sa.getParam("Affected");
                boolean selfCost = "Card.Self".equals(validCard) || "Card.Self".equals(affected);
                if (selfCost) {
                    return ForgeParams.stripAdditionalCostPrefix(normalized)
                            .map(stripped -> new StaticAbilityEntry(AbilityType.ADDITIONAL_COST, stripped))
                            .orElse(new StaticAbilityEntry(effectiveType, normalized));
                }
            }
            return new StaticAbilityEntry(effectiveType, normalized);
        });
    }
}
