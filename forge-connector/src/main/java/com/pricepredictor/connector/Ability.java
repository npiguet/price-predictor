package com.pricepredictor.connector;

import java.util.List;
import java.util.Objects;

/**
 * A single ability on a card face. An ability knows its type, description, and can format itself.
 * Abilities can contain sub-abilities (e.g., charm choices).
 */
public record Ability(
        AbilityType type,
        AbilityDescription description,
        Integer actionNumber,
        List<Ability> subAbilities
) {

    /** Convenience constructor for the common case (no sub-abilities). */
    public Ability(AbilityType type, AbilityDescription description, Integer actionNumber) {
        this(type, description, actionNumber, List.of());
    }

    public Ability {
        Objects.requireNonNull(type, "type must not be null");
        Objects.requireNonNull(description, "description must not be null");
        if (actionNumber != null && actionNumber < 1) {
            throw new IllegalArgumentException("actionNumber must be >= 1 when present");
        }
        if (actionNumber != null && !type.isActionable()) {
            throw new IllegalArgumentException("actionNumber must be null for non-actionable type " + type);
        }
        Objects.requireNonNull(subAbilities, "subAbilities must not be null");
        subAbilities = List.copyOf(subAbilities);
    }

    /**
     * Formats this ability line for output.
     * Actionable types include the action number in brackets: {@code activated[1]: {T}: add {G}.}
     * Non-actionable types omit it: {@code static: flying}
     */
    public String formatLine() {
        if (type.isActionable() && actionNumber != null) {
            return type.getOutputPrefix() + "[" + actionNumber + "]: " + description.text();
        }
        return type.getOutputPrefix() + ": " + description.text();
    }

    /**
     * Formats this ability and all sub-abilities as a text block.
     */
    public String formatBlock() {
        StringBuilder sb = new StringBuilder(formatLine());
        for (Ability sub : subAbilities) {
            sb.append('\n').append(sub.formatLine());
        }
        return sb.toString();
    }
}
