package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityType;

import java.util.List;
import java.util.Objects;

/**
 * Catch-all ability implementation for pre-computed text.
 * Used for CARDNAME keywords, AlternateAdditionalCost, land mana,
 * reclassified levels, and in tests.
 */
public record TextAbility(
        AbilityType type,
        String descriptionText,
        int ordinal,
        List<Ability> subAbilities
) implements Ability {

    /** Convenience constructor: no ordinal, no sub-abilities. */
    public TextAbility(AbilityType type, String descriptionText) {
        this(type, descriptionText, 0, List.of());
    }

    /** Convenience constructor: fixed ordinal, no sub-abilities. */
    public TextAbility(AbilityType type, String descriptionText, int ordinal) {
        this(type, descriptionText, ordinal, List.of());
    }

    public TextAbility {
        Objects.requireNonNull(type, "type must not be null");
        Objects.requireNonNull(descriptionText, "descriptionText must not be null");
        if (descriptionText.isEmpty()) {
            throw new IllegalArgumentException("descriptionText must not be empty");
        }
        Objects.requireNonNull(subAbilities, "subAbilities must not be null");
        subAbilities = List.copyOf(subAbilities);
    }
}
