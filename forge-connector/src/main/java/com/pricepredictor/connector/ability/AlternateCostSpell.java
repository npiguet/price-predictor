package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
import forge.game.spellability.SpellAbility;

import java.util.Optional;

import static com.pricepredictor.connector.AbilityDescription.applyCasing;

/**
 * Alternate cost spell ability (e.g., Cleave). Built from NonBasicSpell spell abilities
 * that have PrecostDesc and CostDesc parameters.
 */
public record AlternateCostSpell(NonBlankString descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.ALTERNATE_COST;
    }

    public static Optional<AlternateCostSpell> of(SpellAbility sa) {
        return NonBlankString.of(sa.getParam("PrecostDesc")).flatMap(precost ->
                NonBlankString.of(sa.getParam("CostDesc")).map(costDesc ->
                        new AlternateCostSpell(applyCasing(precost + " " + costDesc))
                )
        );
    }
}
