package com.pricepredictor.connector;

import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Whether one line of a face restates another: every word of the first,
 * leaving out the face's own name and words that carry no content, has a word
 * of the second with the same five opening letters. The five letters make
 * "control" and "controlled" one word, the way Coffin Queen's triggers
 * restate its activated line.
 */
final class RepeatedLines {

    private static final Pattern WORD = Pattern.compile("[a-z0-9'+/{}-]+");
    private static final int STEM = 5;
    private static final Set<String> NO_CONTENT = Set.of(
            "a", "an", "the", "by", "of", "that", "it", "its", "to", "and", "or", "cardname");

    private RepeatedLines() {}

    /** Whether {@code text} says nothing {@code other} does not say. */
    static boolean repeats(String text, String other) {
        Set<String> words = stems(text);
        return !words.isEmpty() && stems(other).containsAll(words);
    }

    private static Set<String> stems(String text) {
        Set<String> stems = new HashSet<>();
        Matcher m = WORD.matcher(text.toLowerCase(Locale.ROOT));
        while (m.find()) {
            String word = m.group();
            if (!NO_CONTENT.contains(word)) {
                stems.add(word.substring(0, Math.min(STEM, word.length())));
            }
        }
        return stems;
    }
}
