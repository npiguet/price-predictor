package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
import forge.game.replacement.ReplacementEffect;

import java.util.Optional;

/**
 * Ability wrapping a Forge ReplacementEffect.
 */
public record ReplacementAbilityEntry(NonBlankString descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.REPLACEMENT;
    }

    public static Optional<ReplacementAbilityEntry> of(ReplacementEffect re) {
        if (re.getKeyword() != null) return Optional.empty();
        String desc = re.getParam("Description");
        if (desc == null || desc.isEmpty()) return Optional.empty();
        return AbilityDescription.normalize(desc).map(ReplacementAbilityEntry::new);
    }
}
