package com.pricepredictor.connector;

import java.util.List;
import java.util.Objects;

/**
 * Represents a single card face after conversion from Forge script format.
 */
public record ConvertedCard(
        String name,
        String manaCost,
        String types,
        String powerToughness,
        String loyalty,
        String defense,
        String colors,
        String text,
        List<AbilityLine> abilities
) {

    public ConvertedCard {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("name must not be null or empty");
        }
        if (types == null || types.isEmpty()) {
            throw new IllegalArgumentException("types must not be null or empty");
        }
        Objects.requireNonNull(abilities, "abilities must not be null");
        abilities = List.copyOf(abilities);
    }
}
