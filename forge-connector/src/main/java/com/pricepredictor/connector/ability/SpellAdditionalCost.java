package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

/**
 * A spell's additional cost on a line of its own, for a spell whose chain
 * states nothing else (a permanent's). A spell that states an effect opens
 * its {@link SpellLine} with the cost's sentence instead.
 */
public record SpellAdditionalCost(String descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.ADDITIONAL_COST;
    }

    public static SpellAdditionalCost of(SpellAbility sa) {
        String sentence = sentenceOf(sa);
        return sentence == null ? null
                : new SpellAdditionalCost(AbilityDescription.stripAdditionalCostPrefix(sentence));
    }

    /**
     * The additional cost as the card states it ("as an additional cost to
     * cast this spell, sacrifice a land."), or null if {@code sa} has none.
     */
    public static String sentenceOf(SpellAbility sa) {
        String costDesc = sa.getCostDescription();
        if (costDesc == null) return null;
        String normalized = AbilityDescription.normalize(costDesc.trim());
        if (normalized == null) return null;
        return AbilityDescription.stripAdditionalCostPrefix(normalized).isEmpty() ? null : normalized;
    }
}
