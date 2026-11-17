package com.pricepredictor.connector.ability;

import java.util.regex.Pattern;

/** Text-level repairs to the description an entry prints. */
final class DescriptionText {

    private DescriptionText() {}

    private static final Pattern RAW_SELECTOR = Pattern.compile("\\b[a-z]+\\.[a-z]+\\+");

    /** True when {@code text} holds a Forge card selector ({@code card.other+youown}), not prose. */
    static boolean hasRawSelector(String text) {
        return text != null && RAW_SELECTOR.matcher(text).find();
    }

    /**
     * {@code text} followed by the sentences of {@code extra} it does not
     * already hold: an effect the description states is not printed twice.
     */
    static String appendMissing(String text, String extra) {
        StringBuilder result = new StringBuilder(text);
        for (String sentence : extra.split("(?<=[.!?])\\s+")) {
            if (!sentence.isEmpty() && !result.toString().contains(sentence)) {
                if (result.length() > 0) result.append(' ');
                result.append(sentence);
            }
        }
        return result.toString();
    }
}
