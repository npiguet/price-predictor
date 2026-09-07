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
 * <p>This is the <b>fallback</b>, not the primary route. Forge's folder loader
 * calls the two-argument {@code readCard(script, filename)}, so a card loaded
 * from {@code cardsfolder} carries its own file stem in
 * {@code CardRules.getNormalizedName()}, and {@link ProvenanceKey} asks for that
 * first — Forge's filenames disagree with its card names often enough that a
 * sanitizer alone loses cards. The sanitizer still names variants, and still
 * covers a card with no rules attached.
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
    /**
     * The tree-prefixed path for a stem that is already a filename.
     *
     * <p>Token scripts are filed by what the token <em>is</em> — colours, power,
     * toughness, type, abilities — not by its name:
     * {@code c_1_1_eldrazi_scion_sac.txt} is the Eldrazi Scion. Sanitizing the
     * name would name a file that does not exist, and the two 1/1 Eldrazi Scions
     * that differ only in their sacrifice ability would collide besides.
     */
    public static String scriptFileForStem(String tree, String stem) {
        if (SourceTree.isFlat(tree)) {
            return tree + "/" + stem + ".txt";
        }
        String directory;
        if (stem.isEmpty()) {
            directory = "_";
        } else if (stem.startsWith("a-")) {
            // Alchemy rebalances keep their literal "a-" and live in their own
            // directory rather than under "a".
            directory = REBALANCED_DIR;
        } else {
            directory = stem.substring(0, 1);
        }
        return tree + "/" + directory + "/" + stem + ".txt";
    }

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
