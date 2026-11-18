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
 * <p>The name is matched as Forge writes a self-reference it does not write
 * as {@code CARDNAME}: the literal name, in its own case, as whole words, in
 * the raw description before it is lower-cased ({@link AbilityDescription#applyCasing}).
 * The same word in another case is prose ("until end of turn" on Turn, "roll
 * a six-sided die" on Six-Sided Die). The names are a face's own and, for a
 * split card, the whole card's ({@code Fire // Ice}); the longer is tried
 * first so the whole name is one placeholder. These are not the card's name
 * and stay:
 * <ul>
 *   <li>the short name of a legendary card ("Chandra"): a common word or
 *       another card's name too often ("Eleven", "Storm", a token's name);</li>
 *   <li>a face name Forge writes as something else in title case: a creature
 *       type (Aurochs, Assembly-Worker), a keyword (Fear, Flashback), a
 *       keyword action (Exile, Regenerate) or a set's name (War of the Spark);</li>
 *   <li>a match inside a longer card's or token's name ("Keeper of Kookus" on
 *       Kookus, "Stangg Twin" on Stangg).</li>
 * </ul>
 *
 * <p>The substitution holds while a face is parsed ({@link OwnNameScope}):
 * every description of the face is cased then, and its name header, cased
 * with {@link AbilityDescription#lowerCase}, keeps the name.
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
        if (!ForgeCardNames.namesSomethingElse(faceName)) names.add(faceName);
        if (!wholeName.equals(faceName)) names.add(wholeName);
        if (names.isEmpty()) return new OwnName(faceName, wholeName, null);
        names.sort(Comparator.comparingInt(String::length).reversed());
        StringBuilder alternatives = new StringBuilder();
        for (String name : names) {
            if (!alternatives.isEmpty()) alternatives.append('|');
            alternatives.append(Pattern.quote(name));
        }
        return new OwnName(faceName, wholeName, Pattern.compile(
                "(?<![\\p{L}\\p{N}_])(?:" + alternatives + ")(?![\\p{L}\\p{N}_])"));
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

    /** Whether the match is the face's name within another card's or a token's longer name. */
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
