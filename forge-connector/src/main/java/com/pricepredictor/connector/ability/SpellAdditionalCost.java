package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
import forge.game.spellability.SpellAbility;

import java.util.Optional;

/**
 * Additional cost extracted from a spell ability's cost description.
 */
public record SpellAdditionalCost(NonBlankString descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.ADDITIONAL_COST;
    }

    public static Optional<SpellAdditionalCost> of(SpellAbility sa) {
        return NonBlankString.of(sa.getCostDescription())
                .flatMap(AbilityDescription::normalize)
                .flatMap(ForgeParams::stripAdditionalCostPrefix)
                .map(SpellAdditionalCost::new);
    }
}
