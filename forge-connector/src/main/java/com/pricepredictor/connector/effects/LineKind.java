package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * What a card's line is, as a line's {@code kind} spells it.
 *
 * <p>{@code LineKind} in {@code state_snapshot.py} is the other half of this
 * vocabulary, in the same order. It is {@link TraitKind} with {@code
 * activated} told apart from {@code spell}: a spell ability on the stack or in
 * hand is a {@link #SPELL}, any other ability a player activates is {@link
 * #ACTIVATED}.
 */
enum LineKind implements WireNamed {
    SPELL("spell"),
    ACTIVATED("activated"),
    TRIGGER("trigger"),
    STATIC("static"),
    REPLACEMENT("replacement"),
    KEYWORD("keyword");

    private final String wireName;

    LineKind(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code kind} a line carries. */
    @Override
    public String wireName() {
        return wireName;
    }
}
