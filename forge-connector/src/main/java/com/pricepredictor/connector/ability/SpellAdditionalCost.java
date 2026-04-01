package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.Optional;

/**
 * Additional cost extracted from a spell ability's cost description.
 */
public record SpellAdditionalCost(String descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.ADDITIONAL_COST;
    }

    public static Optional<SpellAdditionalCost> of(SpellAbility sa) {
        String costDesc = sa.getCostDescription();
        if (costDesc == null || costDesc.isBlank()) return Optional.empty();
        return AbilityDescription.normalize(costDesc.trim())
                .map(ForgeParams::stripAdditionalCostPrefix)
                .filter(n -> !n.isEmpty())
                .map(SpellAdditionalCost::new);
    }
}
