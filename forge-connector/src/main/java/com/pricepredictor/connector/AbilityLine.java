package com.pricepredictor.connector;

import java.util.Objects;

/**
 * A single ability line in converted card output.
 */
public record AbilityLine(AbilityType type, String description, Integer actionNumber) {

    public AbilityLine {
        Objects.requireNonNull(type, "type must not be null");
        if (description == null || description.isEmpty()) {
            throw new IllegalArgumentException("description must not be null or empty");
        }
        if (actionNumber != null && actionNumber < 1) {
            throw new IllegalArgumentException("actionNumber must be >= 1 when present");
        }
        if (actionNumber != null && !type.isActionable()) {
            throw new IllegalArgumentException("actionNumber must be null for non-actionable type " + type);
        }
    }

    /**
     * Formats this ability line for output.
     * Actionable types include the action number in brackets: {@code activated[1]: {T}: add {G}.}
     * Non-actionable types omit it: {@code keyword: flying}
     */
    public String formatLine() {
        if (type.isActionable() && actionNumber != null) {
            return type.getOutputPrefix() + "[" + actionNumber + "]: " + description;
        }
        return type.getOutputPrefix() + ": " + description;
    }
}
