package com.pricepredictor.connector.ability;

/**
 * The marker strings Forge's card scripts put in a description in place of
 * text, and the questions the converter asks of them.
 */
public final class DescriptionMarkers {

    private DescriptionMarkers() {}

    /** Where a trigger's description stands for the text of the ability it executes. */
    public static final String ABILITY_PLACEHOLDER = "ABILITY";

    /** The {@code TriggerDescription$} of a trigger that prints no line of its own. */
    public static final String BLANK_TRIGGER = "Blank";

    /** Whether {@code description} holds the {@link #ABILITY_PLACEHOLDER}. */
    public static boolean hasAbilityPlaceholder(String description) {
        return description.contains(ABILITY_PLACEHOLDER);
    }

    /**
     * Whether a trigger with this {@code TriggerDescription$} prints no line:
     * it has none, or it is {@link #BLANK_TRIGGER}.
     */
    public static boolean printsNoLine(String triggerDescription) {
        return triggerDescription == null || BLANK_TRIGGER.equals(triggerDescription);
    }
}
