package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;

import java.util.List;
import java.util.Objects;

/**
 * Catch-all ability implementation for pre-computed text.
 * Used for CARDNAME keywords, AlternateAdditionalCost, land mana,
 * reclassified levels, and in tests.
 */
public record TextAbility(
        AbilityType type,
        NonBlankString descriptionText,
        int ordinal,
        List<Ability> subAbilities
) implements Ability {

    /** Convenience constructor: no ordinal, no sub-abilities. */
    public TextAbility(AbilityType type, NonBlankString descriptionText) {
        this(type, descriptionText, 0, List.of());
    }

    /** Convenience constructor: fixed ordinal, no sub-abilities. */
    public TextAbility(AbilityType type, NonBlankString descriptionText, int ordinal) {
        this(type, descriptionText, ordinal, List.of());
    }

    // Bridge constructors for callers that pass raw String literals.
    public TextAbility(AbilityType type, String descriptionText) {
        this(type, NonBlankString.require(descriptionText), 0, List.of());
    }

    public TextAbility(AbilityType type, String descriptionText, int ordinal) {
        this(type, NonBlankString.require(descriptionText), ordinal, List.of());
    }

    public TextAbility(AbilityType type, String descriptionText, int ordinal, List<Ability> subAbilities) {
        this(type, NonBlankString.require(descriptionText), ordinal, subAbilities);
    }

    public TextAbility {
        Objects.requireNonNull(type, "type must not be null");
        Objects.requireNonNull(descriptionText, "descriptionText must not be null");
        Objects.requireNonNull(subAbilities, "subAbilities must not be null");
        subAbilities = List.copyOf(subAbilities);
    }
}
