package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.Optional;

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

    public static Optional<SpellAdditionalCost> of(SpellAbility sa) {
        return Optional.ofNullable(sentenceOf(sa)).map(sentence ->
                new SpellAdditionalCost(AbilityDescription.stripAdditionalCostPrefix(sentence)));
    }

    /**
     * The additional cost as the card states it ("as an additional cost to
     * cast this spell, sacrifice a land."), or null if {@code sa} has none.
     * An ability word alone ({@code PrecostDesc$ Parley —}, which Forge
     * prints before the cost) is no cost: the spell's description opens with
     * it already.
     */
    public static String sentenceOf(SpellAbility sa) {
        String normalized = normalized(sa.getCostDescription());
        if (normalized == null || normalized.equals(normalized(sa.getParam("PrecostDesc")))) {
            return null;
        }
        String cost = withoutAbilityWord(normalized, normalized(sa.getParam("PrecostDesc")));
        return AbilityDescription.stripAdditionalCostPrefix(cost).isEmpty() ? null : cost;
    }

    /** {@code cost} without the ability word the line opens with ({@code precost}), if any. */
    private static String withoutAbilityWord(String cost, String precost) {
        return precost != null && cost.startsWith(precost)
                ? cost.substring(precost.length()).trim() : cost;
    }

    private static String normalized(String description) {
        return description == null ? null : AbilityDescription.normalize(description.trim());
    }
}
