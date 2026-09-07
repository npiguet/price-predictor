package com.pricepredictor.connector.effects;

import java.text.Normalizer;
import java.util.Locale;

/**
 * Card name to on-disk filename, mirroring the Python
 * {@code price_predictor.infrastructure.card_filenames.sanitize_card_name}.
 *
 * <p>The two implementations must agree exactly. A provenance key's
 * {@code script_file} is written by this side and resolved by the Python side,
 * and the join fails loudly on a mismatch — which is the designed behaviour,
 * but a needless failure if the sanitizers merely disagree about punctuation.
 * {@code CardFilenamesTest} pins the agreement against the same cases the
 * Python suite uses.
 *
 * <p>Forge itself cannot supply the path: {@code CardRules.getNormalizedName()}
 * is only set by the two-argument {@code readCard(script, filename)}, which
 * nothing in the engine calls, so it is null for every card in a live game.
 */
public final class CardFilenames {

    private CardFilenames() {
    }

    /** Alchemy/Arena rebalanced cards live in this directory rather than a letter one. */
    public static final String REBALANCED_DIR = "rebalanced";

    /**
     * The filename stem for a card name, without extension or directory.
     *
     * <p>NFKD-decomposes accents, lowercases, strips punctuation, and collapses
     * runs of {@code _} into one — so "Anchovy &amp; Banana Pizza" becomes
     * {@code anchovy_banana_pizza} rather than a double underscore, matching
     * Forge's own filename convention.
     */
    public static String sanitize(String name) {
        String ascii = Normalizer.normalize(name, Normalizer.Form.NFKD)
                .replaceAll("\\p{M}+", "");
        String sanitized = ascii.toLowerCase(Locale.ROOT)
                .replace(" // ", "_")
                .replace(" ", "_")
                .replace("'", "")
                .replace(",", "")
                .replace(":", "")
                .replace("!", "")
                .replace("\"", "")
                .replace("&", "")
                .replace("+", "")
                .replace(".", "")
                .replace("-", "_")
                .replace("/", "");
        sanitized = sanitized.replaceAll("_+", "_");
        return trimUnderscores(sanitized);
    }

    private static String trimUnderscores(String value) {
        int start = 0;
        int end = value.length();
        while (start < end && value.charAt(start) == '_') start++;
        while (end > start && value.charAt(end - 1) == '_') end--;
        return value.substring(start, end);
    }

    /**
     * The tree-prefixed script path a provenance key names, e.g.
     * {@code cardsfolder/a/ajanis_pridemate.txt}.
     *
     * <p>Written with forward slashes on every platform, because the string
     * itself is the key both sides compare.
     */
    public static String scriptFile(String tree, String cardName) {
        String stem;
        String directory;
        if (cardName.length() > 2 && cardName.regionMatches(true, 0, "A-", 0, 2)) {
            // "A-Akki Ronin" keeps its literal "a-" prefix (the general
            // sanitizer would turn the hyphen into an underscore) and lives
            // under cardsfolder/rebalanced/.
            stem = "a-" + sanitize(cardName.substring(2));
            directory = REBALANCED_DIR;
        } else {
            stem = sanitize(cardName);
            directory = stem.isEmpty() ? "_" : stem.substring(0, 1);
        }
        if (SourceTree.isFlat(tree)) {
            return tree + "/" + stem + ".txt";
        }
        return tree + "/" + directory + "/" + stem + ".txt";
    }
}
