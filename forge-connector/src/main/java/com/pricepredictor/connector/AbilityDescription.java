package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Static utility methods for normalizing ability description text.
 */
public final class AbilityDescription {

    private AbilityDescription() {}

    // --- Normalization patterns ---

    private static final Pattern BRACE_SYMBOL = Pattern.compile("\\{[^}]+\\}");
    private static final Pattern REMINDER_TEXT = Pattern.compile("\\s*\\([^)]*\\)");
    private static final Pattern PLACEHOLDER_WORD = Pattern.compile("\\b(cardname|nickname|alternate)\\b");
    private static final Pattern VARIABLE_X = Pattern.compile("(?<![a-z])x(?![a-z])");

    /**
     * Full normalization returning null for empty/null input.
     * Pipeline: strip reminder text, then apply casing.
     */
    public static String normalize(String raw) {
        if (raw == null || raw.isEmpty()) return null;
        String result = applyCasing(stripReminderText(raw));
        return (result == null || result.isEmpty()) ? null : result;
    }

    /** Strip parenthesized reminder text. */
    public static String stripReminderText(String text) {
        if (text == null) {
            return null;
        }
        return REMINDER_TEXT.matcher(text).replaceAll("").trim();
    }

    /** Lowercase text, preserving brace symbols, placeholders, and standalone X. */
    public static String applyCasing(String text) {
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
