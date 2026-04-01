package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

/**
 * Additional cost extracted from a spell ability's cost description.
 */
public record SpellAdditionalCost(String descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.ADDITIONAL_COST;
    }

    public static SpellAdditionalCost of(SpellAbility sa) {
        String costDesc = sa.getCostDescription();
        if (costDesc == null) return null;
        String normalized = AbilityDescription.normalize(costDesc.trim());
        if (normalized == null) return null;
        normalized = ForgeParams.stripAdditionalCostPrefix(normalized);
        if (normalized.isEmpty()) return null;
        return new SpellAdditionalCost(normalized);
    }
}
