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
 * records the file it read each card from, so {@link ProvenanceKey} asks
 * {@code CardRules.getPath()} first and then its file stem in
 * {@code CardRules.getNormalizedName()} — Forge's filenames disagree with its
 * card names often enough that a sanitizer alone loses cards. The sanitizer
 * still names variants, and still covers a card with no rules attached.
 *
 * <p>The <em>directory</em> half of a path is not sanitized at all: it is
 * looked up in {@link SourceTree}, because no rule derives it correctly for
 * every card and a derived-but-wrong path mis-joins in silence.
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
     * {@code cardsfolder/a/ajanis_pridemate.txt}, for a stem that is already a
     * filename.
     *
     * <p>Written with forward slashes on every platform, because the string
     * itself is the key both sides compare.
     *
     * <p>Token scripts are filed by what the token <em>is</em> — colours, power,
     * toughness, type, abilities — not by its name:
     * {@code c_1_1_eldrazi_scion_sac.txt} is the Eldrazi Scion. Sanitizing the
     * name would name a file that does not exist, and the two 1/1 Eldrazi Scions
     * that differ only in their sacrifice ability would collide besides.
     *
     * <p>{@link SourceTree} is asked first for a letter-keyed tree, because the
     * tree is the authority on where it files a card and the initial is only a
     * convention it keeps 99.7% of the time. Derivation answers for the rest —
     * a stem the tree does not hold at all still has to name something.
     */
    public static String scriptFileForStem(String tree, String stem) {
        if (SourceTree.isFlat(tree)) {
            return tree + "/" + stem + ".txt";
        }
        String directory = SourceTree.directoryOf(tree, stem);
        if (directory == null) {
            directory = derivedDirectory(stem);
        }
        return tree + "/" + directory + "/" + stem + ".txt";
    }

    /**
     * Where the letter-keyed convention would put a stem.
     *
     * <p>Shared with {@link SourceTree}, which indexes the tree by asking this
     * of every file it holds and keeping only the answers the tree disagrees
     * with. One function, both sides: an index of exceptions built against a
     * different rule than the one it corrects would correct the wrong files.
     */
    static String derivedDirectory(String stem) {
        if (stem.isEmpty()) {
            return "_";
        }
        if (stem.startsWith("a-")) {
            // Alchemy rebalances keep their literal "a-" and live in their own
            // directory rather than under "a".
            return REBALANCED_DIR;
        }
        return stem.substring(0, 1);
    }

    /** The stem Forge's own filenames use for a card name. */
    static String stemOf(String cardName) {
        if (cardName.length() > 2 && cardName.regionMatches(true, 0, "A-", 0, 2)) {
            // "A-Akki Ronin" keeps its literal "a-" prefix; the general
            // sanitizer would turn the hyphen into an underscore.
            return "a-" + sanitize(cardName.substring(2));
        }
        return sanitize(cardName);
    }

    public static String scriptFile(String tree, String cardName) {
        return scriptFileForStem(tree, stemOf(cardName));
    }
}
