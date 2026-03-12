package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A single ability on a card face. An ability knows its type, description, and can format itself.
 * Abilities can contain sub-abilities (e.g., charm choices).
 */
public record Ability(
        AbilityType type,
        String description,
        Integer actionNumber,
        List<Ability> subAbilities
) {

    /** Convenience constructor for the common case (no sub-abilities). */
    public Ability(AbilityType type, String description, Integer actionNumber) {
        this(type, description, actionNumber, List.of());
    }

    public Ability {
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
            return type.getOutputPrefix() + "[" + actionNumber + "]: " + description;
        }
        return type.getOutputPrefix() + ": " + description;
    }

    // --- Text normalization ---

    private static final Pattern BRACE_SYMBOL = Pattern.compile("\\{[^}]+\\}");
    private static final Pattern REMINDER_TEXT = Pattern.compile("\\s*\\([^)]*\\)");
    private static final Pattern PLACEHOLDER_WORD = Pattern.compile("\\b(cardname|nickname|alternate)\\b");
    private static final Pattern VARIABLE_X = Pattern.compile("(?<![a-z])x(?![a-z])");

    /**
     * Normalize a description: strip reminder text and apply casing rules.
     */
    public static String normalizeDescription(String text) {
        return applyCasing(stripReminderText(text));
    }

    static String stripReminderText(String text) {
        if (text == null) {
            return null;
        }
        return REMINDER_TEXT.matcher(text).replaceAll("").trim();
    }

    /**
     * Apply text casing rules: lowercase all text, then restore CARDNAME/NICKNAME/ALTERNATE
     * and brace symbols to uppercase.
     */
    static String applyCasing(String text) {
        if (text == null) {
            return null;
        }

        // Protect brace symbols by extracting them
        List<String> braceSymbols = new ArrayList<>();
        Matcher m = BRACE_SYMBOL.matcher(text);
        StringBuilder sb = new StringBuilder();
        int lastEnd = 0;
        while (m.find()) {
            braceSymbols.add(m.group().toUpperCase());
            sb.append(text, lastEnd, m.start());
            sb.append("\0BRACE").append(braceSymbols.size() - 1).append('\0');
            lastEnd = m.end();
        }
        sb.append(text, lastEnd, text.length());

        // Lowercase everything
        String lowered = sb.toString().toLowerCase();

        // Restore brace symbols (uppercase)
        for (int i = 0; i < braceSymbols.size(); i++) {
            lowered = lowered.replace("\0brace" + i + "\0", braceSymbols.get(i));
        }

        // Restore placeholders to uppercase (word-boundary-aware to avoid corrupting substrings)
        lowered = PLACEHOLDER_WORD.matcher(lowered).replaceAll(mr -> mr.group().toUpperCase());

        // Restore variable X to uppercase (standalone X not inside words like "exile")
        lowered = VARIABLE_X.matcher(lowered).replaceAll("X");

        return lowered;
    }
}
