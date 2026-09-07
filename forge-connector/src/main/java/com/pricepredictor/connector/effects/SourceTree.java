package com.pricepredictor.connector.effects;

/**
 * The three converted source trees, which do not share a directory layout.
 *
 * <p>{@code cardsfolder} is letter-keyed and the other two are flat, so one
 * filename names different files in different trees: Ajani's Pridemate is
 * {@code cardsfolder/a/ajanis_pridemate.txt} on one side and
 * {@code tokenscripts/ajanis_pridemate.txt} on the other. Token scripts stay
 * out of the card tree because their converted filenames collide with card ones
 * and the sealed pipeline reads that tree as its card corpus.
 *
 * <p>Mirrors {@code SOURCE_TREE_LAYOUTS} in the Python
 * {@code sealed.infrastructure.converted_card_locator}.
 */
public final class SourceTree {

    private SourceTree() {
    }

    public static final String CARDSFOLDER = "cardsfolder";
    public static final String TOKENSCRIPTS = "tokenscripts";
    public static final String VARIANT_SCRIPTS = "variant-scripts";

    public static boolean isFlat(String tree) {
        return TOKENSCRIPTS.equals(tree) || VARIANT_SCRIPTS.equals(tree);
    }

    public static boolean isKnown(String tree) {
        return CARDSFOLDER.equals(tree) || isFlat(tree);
    }
}
