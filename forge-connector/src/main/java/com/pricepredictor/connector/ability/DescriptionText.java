package com.pricepredictor.connector.ability;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Text-level repairs to the description an entry prints. */
final class DescriptionText {

    private DescriptionText() {}

    private static final Pattern LEADING_WORD = Pattern.compile("(\\p{L}+)\\s+");
    private static final Pattern LAST_WORD = Pattern.compile("(\\p{L}+)\\s*$");
    private static final Pattern RAW_SELECTOR = Pattern.compile("\\b[a-z]+\\.[a-z]+\\+");

    /** True when {@code text} holds a Forge card selector ({@code card.other+youown}), not prose. */
    static boolean hasRawSelector(String text) {
        return text != null && RAW_SELECTOR.matcher(text).find();
    }

    /**
     * {@code text} with its {@code placeholder} written as {@code filler},
     * less a leading word of the filler the text before the placeholder
     * already ends with: Druid of the Emerald Grove's "reveal them, then
     * ABILITY" filled with "then roll a d20." reads "then roll a d20.".
     */
    static String fill(String text, String placeholder, String filler) {
        int at = text.indexOf(placeholder);
        if (at < 0) return text;
        String before = text.substring(0, at);
        Matcher lead = LEADING_WORD.matcher(filler);
        String rest = lead.lookingAt() && endsWithWord(before, lead.group(1))
                ? filler.substring(lead.end()) : filler;
        return before + rest + text.substring(at + placeholder.length());
    }

    /** Whether the last word of {@code text} is {@code word}, in any case. */
    private static boolean endsWithWord(String text, String word) {
        Matcher last = LAST_WORD.matcher(text);
        return last.find() && last.group(1).equalsIgnoreCase(word);
    }
}
