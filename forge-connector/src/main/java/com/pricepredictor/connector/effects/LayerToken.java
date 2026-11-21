package com.pricepredictor.connector.effects;

/**
 * One token of a contribution's keyword, type, colour, ability or hidden
 * keyword list: what a static does to one entry of that list.
 *
 * <p>A token the static adds is spelled as its bare text, one it removes as
 * {@code -} and the text, and the lone {@code =} marks the tokens after it as
 * a list the static sets outright rather than adds to.
 */
record LayerToken(Sign sign, String text) {

    /** What the static does with a token's text. */
    enum Sign {
        ADDS,
        REMOVES,
        SETS
    }

    /** The {@code =} marker: the tokens after it are the list the static sets. */
    static final LayerToken SETS_THE_LINE = new LayerToken(Sign.SETS, "");

    static LayerToken added(String text) {
        return new LayerToken(Sign.ADDS, text);
    }

    static LayerToken removed(String text) {
        return new LayerToken(Sign.REMOVES, text);
    }

    /** The token as the record spells it. */
    String spelled() {
        return switch (sign) {
            case ADDS -> text;
            case REMOVES -> "-" + text;
            case SETS -> "=";
        };
    }
}
