package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

/**
 * Additional cost extracted from a spell ability's cost description.
 */
public record SpellAdditionalCost(String descriptionText) implements Ability {

    private static final String ADDITIONAL_COST_PREFIX = "as an additional cost to cast this spell, ";

    @Override
    public AbilityType type() {
        return AbilityType.ADDITIONAL_COST;
    }

    public static SpellAdditionalCost of(SpellAbility sa) {
        String costDesc = sa.getCostDescription();
        if (costDesc == null) return null;
        String normalized = AbilityDescription.normalize(costDesc.trim());
        if (normalized == null) return null;
        if (normalized.startsWith(ADDITIONAL_COST_PREFIX)) {
            normalized = normalized.substring(ADDITIONAL_COST_PREFIX.length());
        }
        if (normalized.isEmpty()) return null;
        return new SpellAdditionalCost(normalized);
    }
}
