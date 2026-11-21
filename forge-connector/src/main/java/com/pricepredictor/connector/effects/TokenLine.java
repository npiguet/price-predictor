package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;

/**
 * One of a contribution's token lists, in the order its tokens are first
 * written.
 *
 * <p>The keyword, type and colour lists are sets rather than lists: a static
 * that writes to two type layers can name the same type twice, and the second
 * says nothing the first did not. Insertion order keeps a {@code =} marker
 * ahead of what it marks. An empty token says nothing there and is dropped,
 * and a keyword is spelled lower-case. The ability and hidden-keyword lists
 * keep every token written, as written.
 */
final class TokenLine {

    private static final Predicate<String> SAYS_SOMETHING = token -> token != null && !token.isEmpty();

    private final Collection<String> tokens;
    private final Predicate<String> kept;
    private final UnaryOperator<String> spelling;

    private TokenLine(Collection<String> tokens, Predicate<String> kept, UnaryOperator<String> spelling) {
        this.tokens = tokens;
        this.kept = kept;
        this.spelling = spelling;
    }

    /** The keyword list: each keyword once, lower-case. */
    static TokenLine keywords() {
        return new TokenLine(new LinkedHashSet<>(), SAYS_SOMETHING, keyword -> keyword.toLowerCase(Locale.ROOT));
    }

    /** A type or colour list: each token once, as spelled. */
    static TokenLine distinct() {
        return new TokenLine(new LinkedHashSet<>(), SAYS_SOMETHING, UnaryOperator.identity());
    }

    /** An ability or hidden-keyword list: every token written, as spelled. */
    static TokenLine everyToken() {
        return new TokenLine(new ArrayList<>(), token -> true, UnaryOperator.identity());
    }

    void add(LayerToken token) {
        String spelled = token.spelled();
        if (kept.test(spelled)) {
            tokens.add(spelling.apply(spelled));
        }
    }

    /** The list as a JSON array of strings. */
    String json() {
        return Json.stringArray(tokens);
    }
}
