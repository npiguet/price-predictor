package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A card's own name inside its ability lines, written as the {@code CARDNAME}
 * placeholder every other mention of the card already is.
 *
 * <p>The names are a face's own and, for a split card, the whole card's
 * ({@code fire // ice}); the longer is tried first so the whole name is one
 * placeholder. A name matches as whole words and in any case, since the
 * converted text is lower-cased but for its placeholders. Three things are
 * not the card's name and stay:
 * <ul>
 *   <li>the short name of a legendary card ("chandra"): a common word or
 *       another card's name too often ("eleven", "storm", a token's name);</li>
 *   <li>a face name that is a creature type (Aurochs, Assembly-Worker):
 *       Forge scripts write a self-reference as CARDNAME, so the literal is
 *       the type;</li>
 *   <li>a match inside a longer card's name ("keeper of kookus" on Kookus).</li>
 * </ul>
 */
final class OwnName {

    private static final String PLACEHOLDER = "CARDNAME";

    private final String faceName;
    private final String wholeName;
    private final Pattern pattern;
    private List<String> longerNames;

    private OwnName(String faceName, String wholeName, Pattern pattern) {
        this.faceName = faceName.toLowerCase(Locale.ROOT);
        this.wholeName = wholeName.toLowerCase(Locale.ROOT);
        this.pattern = pattern;
    }

    /** The substitution for a face named {@code faceName} of a card with {@code wholeName}. */
    static OwnName of(String faceName, String wholeName) {
        List<String> names = new ArrayList<>();
        if (!ForgeCardNames.isCreatureType(faceName)) names.add(faceName);
        if (!wholeName.equals(faceName)) names.add(wholeName);
        if (names.isEmpty()) return new OwnName(faceName, wholeName, null);
        names.sort(Comparator.comparingInt(String::length).reversed());
        StringBuilder alternatives = new StringBuilder();
        for (String name : names) {
            if (!alternatives.isEmpty()) alternatives.append('|');
            alternatives.append(Pattern.quote(name));
        }
        return new OwnName(faceName, wholeName, Pattern.compile(
                "(?<![\\p{L}\\p{N}_])(?:" + alternatives + ")(?![\\p{L}\\p{N}_])",
                Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE));
    }

    /** {@code prose} with the name written as the placeholder. */
    String apply(String prose) {
        if (pattern == null) return prose;
        Matcher matcher = pattern.matcher(prose);
        StringBuilder out = new StringBuilder();
        while (matcher.find()) {
            String replacement = isPartOfLongerName(prose, matcher.start(), matcher.end())
                    ? matcher.group() : PLACEHOLDER;
            matcher.appendReplacement(out, Matcher.quoteReplacement(replacement));
        }
        return matcher.appendTail(out).toString();
    }

    /** Whether the match is the face's name within another card's longer name. */
    private boolean isPartOfLongerName(String prose, int start, int end) {
        if (!prose.regionMatches(true, start, faceName, 0, end - start)
                || end - start != faceName.length()) {
            return false; // the whole card's name, not the face's
        }
        for (String longer : longerNames()) {
            if (!longer.equals(wholeName) && covers(prose, start, longer)) return true;
        }
        return false;
    }

    /** Whether {@code longer} stands in the prose around the face name at {@code start}. */
    private boolean covers(String prose, int start, String longer) {
        for (int at = longer.indexOf(faceName); at >= 0; at = longer.indexOf(faceName, at + 1)) {
            int from = start - at;
            if (from >= 0 && prose.regionMatches(true, from, longer, 0, longer.length())) {
                return true;
            }
        }
        return false;
    }

    /** Looked up once, and only for a face whose name stands in a line. */
    private List<String> longerNames() {
        if (longerNames == null) longerNames = ForgeCardNames.longerNamesHolding(faceName);
        return longerNames;
    }
}
