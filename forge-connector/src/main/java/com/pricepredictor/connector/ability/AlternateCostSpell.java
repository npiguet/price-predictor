package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

/**
 * Alternate cost spell ability (e.g., Cleave). Built from NonBasicSpell spell abilities
 * that have PrecostDesc and CostDesc parameters.
 */
public record AlternateCostSpell(String descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.ALTERNATE_COST;
    }

    public static AlternateCostSpell of(SpellAbility sa) {
        String precost = sa.getParam("PrecostDesc");
        String costDesc = sa.getParam("CostDesc");
        if (precost == null || costDesc == null) return null;
        return new AlternateCostSpell(AbilityDescription.applyCasing(precost + " " + costDesc));
    }
}
