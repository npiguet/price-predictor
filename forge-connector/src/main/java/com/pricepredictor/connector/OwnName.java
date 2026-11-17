package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Pattern;

/**
 * A card's own name inside its ability lines, written as the {@code CARDNAME}
 * placeholder every other mention of the card already is.
 *
 * <p>The names are a face's own and, for a split card, the whole card's
 * ({@code fire // ice}); the longer is tried first so the whole name is one
 * placeholder. A name matches as whole words and in any case, since the
 * converted text is lower-cased but for its placeholders. The short name of a
 * legendary card ("chandra") is not one of them: it is a common word or
 * another card's name too often ("eleven", "storm", a token's name).
 */
final class OwnName {

    private static final String PLACEHOLDER = "CARDNAME";

    private final Pattern pattern;

    private OwnName(Pattern pattern) {
        this.pattern = pattern;
    }

    /** The substitution for a face named {@code faceName} of a card with {@code wholeName}. */
    static OwnName of(String faceName, String wholeName) {
        List<String> names = new ArrayList<>(List.of(faceName));
        if (!wholeName.equals(faceName)) names.add(wholeName);
        names.sort(Comparator.comparingInt(String::length).reversed());
        StringBuilder alternatives = new StringBuilder();
        for (String name : names) {
            if (!alternatives.isEmpty()) alternatives.append('|');
            alternatives.append(Pattern.quote(name));
        }
        return new OwnName(Pattern.compile(
                "(?<![\\p{L}\\p{N}_])(?:" + alternatives + ")(?![\\p{L}\\p{N}_])",
                Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE));
    }

    /** {@code prose} with the name written as the placeholder. */
    String apply(String prose) {
        return pattern.matcher(prose).replaceAll(PLACEHOLDER);
    }
}
