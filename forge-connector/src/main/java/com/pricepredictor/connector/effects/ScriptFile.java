package com.pricepredictor.connector.effects;

/**
 * One source script as a provenance key names it: the {@link SourceTree} it
 * lies in and its path within that tree.
 *
 * <p>The tree is half the name, not a prefix to be glued on wherever a path is
 * needed: one filename occurs in more than one tree, and the Python side looks
 * for the sidecar under exactly the tree-prefixed path, so the converter that
 * writes a key and the collector that names one have to assemble it the same
 * way. This record is that one way.
 *
 * @param tree         the tree the script lies in
 * @param relativePath the path within the tree, held with forward slashes
 *                     whatever the platform's separator
 */
public record ScriptFile(SourceTree tree, String relativePath) {

    private static final String EXTENSION = ".txt";

    public ScriptFile {
        relativePath = relativePath.replace('\\', '/');
    }

    /**
     * The tree-prefixed path, e.g. {@code cardsfolder/a/ajanis_pridemate.txt}.
     *
     * <p>Written with forward slashes on every platform, because the string
     * itself is the key both sides compare.
     */
    public String key() {
        return tree.wireName() + "/" + relativePath;
    }

    /** The file's name without its directory or {@code .txt} extension. */
    public String stem() {
        String name = relativePath.substring(relativePath.lastIndexOf('/') + 1);
        return name.endsWith(EXTENSION)
                ? name.substring(0, name.length() - EXTENSION.length()) : name;
    }

    /** The directory within the tree, or the empty string for a file at its root. */
    public String directory() {
        int slash = relativePath.lastIndexOf('/');
        return slash < 0 ? "" : relativePath.substring(0, slash);
    }
}
