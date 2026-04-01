package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.StringJoiner;
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
     * Full normalization returning empty Optional for empty input or when stripping reduces to nothing.
     * Pipeline: strip reminder text, then apply casing.
     */
    public static @lombok.NonNull Optional<NonBlankString> normalize(@lombok.NonNull String raw) {
        if (raw.isEmpty()) return Optional.empty();
        String stripped = stripReminderText(raw);
        if (stripped.isEmpty()) return Optional.empty();
        return Optional.of(applyCasing(stripped));
    }

    /** Strip parenthesized reminder text. */
    public static @lombok.NonNull String stripReminderText(@lombok.NonNull String text) {
        return REMINDER_TEXT.matcher(text).replaceAll("").trim();
    }

    /**
     * Strip parenthesized reminder text from a NonBlankString.
     * Returns empty if stripping removes all content.
     */
    public static @lombok.NonNull Optional<NonBlankString> stripReminderText(@lombok.NonNull NonBlankString text) {
        return NonBlankString.of(stripReminderText(text.value()));
    }

    /**
     * Join items with commas and a final "or" disjunction.
     * ["A"] → "A", ["A","B"] → "A or B", ["A","B","C"] → "A, B, or C"
     */
    public static @lombok.NonNull String joinDisjunction(List<String> items) {
        return switch (items.size()) {
            case 0 -> "";
            case 1 -> items.get(0);
            case 2 -> items.get(0) + " or " + items.get(1);
            default -> {
                StringJoiner sj = new StringJoiner(", ");
                for (int i = 0; i < items.size() - 1; i++) sj.add(items.get(i));
                yield sj + ", or " + items.get(items.size() - 1);
            }
        };
    }

    /**
     * Replace Forge-internal VERT placeholder with " | " so dice-outcome descriptions
     * render correctly (e.g. "1—9 VERT Tap all…" → "1—9 | Tap all…").
     */
    public static @lombok.NonNull String replaceVert(@lombok.NonNull String text) {
        return text.replace(" VERT ", " | ").replace("VERT", "");
    }

    /** Lowercase text, preserving brace symbols, placeholders, and standalone X. */
    public static @lombok.NonNull NonBlankString applyCasing(@lombok.NonNull NonBlankString text) {
        return applyCasing(text.value());
    }

    public static @lombok.NonNull NonBlankString applyCasing(@lombok.NonNull String text) {
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

        String lowered = sb.toString().toLowerCase();

        // Restore brace symbols — {W}, {U}, {B}, {R}, {G}, {T}, etc. are MTG mana/tap symbols
        // whose uppercase form is the canonical notation expected by downstream consumers.
        for (int i = 0; i < braceSymbols.size(); i++) {
            lowered = lowered.replace("\0brace" + i + "\0", braceSymbols.get(i));
        }

        // Restore placeholders to uppercase (word-boundary-aware to avoid corrupting substrings)
        lowered = PLACEHOLDER_WORD.matcher(lowered).replaceAll(mr -> mr.group().toUpperCase());

        // Normalize NICKNAME to CARDNAME: both refer to the card's own name; use one placeholder.
        lowered = lowered.replace("NICKNAME", "CARDNAME");

        // Restore variable X to uppercase (standalone X not inside words like "exile")
        lowered = VARIABLE_X.matcher(lowered).replaceAll("X");

        return NonBlankString.require(lowered);
    }
}
