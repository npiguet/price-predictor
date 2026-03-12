package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.replacement.ReplacementEffect;

/**
 * Ability wrapping a Forge ReplacementEffect.
 */
public record ReplacementAbilityEntry(String descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.REPLACEMENT;
    }

    public static ReplacementAbilityEntry of(ReplacementEffect re) {
        if (re.getKeyword() != null) return null;
        String normalized = AbilityDescription.normalize(re.getParam("Description"));
        if (normalized == null) return null;
        return new ReplacementAbilityEntry(normalized);
    }
}
