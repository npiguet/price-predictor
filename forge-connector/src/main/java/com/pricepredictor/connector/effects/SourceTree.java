package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;

import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.HashMap;
import java.util.Map;

/**
 * The three converted source trees, which do not share a directory layout, and
 * where each of them actually files a card.
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
 *
 * <p>"Letter-keyed" describes the tree; it does not generate it. Forge files
 * {@code +2 Mace} under {@code p/}, {@code 1996 World Champion} under
 * {@code n/}, {@code P. Joven and Chandler} under {@code j/}, and every
 * unreleased card under {@code upcoming/} whatever its initial — 86 files at
 * the time of writing. The converter walks the tree and keys each sidecar by
 * the path it read the file from, so a collector that <em>derives</em> a path
 * instead disagrees with the sidecar for exactly those cards, and the mis-join
 * is silent: only a {@code keyword} key is checked against its sidecar, so a
 * fabricated {@code cardsfolder/+/+2_mace.txt} joins to nothing and nothing
 * says so. This class is therefore where a caller asks the tree where a card
 * lives; {@link CardFilenames} derives only what the tree does not answer.
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

    // -- where a loader says it read a file ------------------------------

    /**
     * The tree-relative path a loader-reported file path names, or null when
     * the path does not lie in that tree.
     *
     * <p>This is the authoritative route, because it is the tree's own record:
     * {@code CardStorageReader} stores the path of the file it read a card from
     * in {@code CardRules.getPath()}, and the converter keys each sidecar by
     * the same path relativised against the same root. Anchoring on the tree
     * segment rather than on whichever root the reader was handed makes the two
     * agree whether Forge was pointed at an absolute or a relative
     * {@code cardsfolder}.
     *
     * <p>A bare {@code <directory>/<file>.txt} is accepted too: that is what a
     * path looks like when Forge reads {@code cardsfolder.zip} rather than the
     * loose files, where the entry names carry no tree segment at all.
     */
    public static String relativePathIn(String tree, String loaderPath) {
        if (tree == null || loaderPath == null || loaderPath.isEmpty()) return null;
        String path = loaderPath.replace('\\', '/');
        String anchor = "/" + tree + "/";
        int at = path.lastIndexOf(anchor);
        String relative;
        if (at >= 0) {
            relative = path.substring(at + anchor.length());
        } else if (path.startsWith(tree + "/")) {
            relative = path.substring(tree.length() + 1);
        } else if (isZipEntryShaped(path)) {
            relative = path;
        } else {
            return null;
        }
        return relative.endsWith(".txt") ? relative : null;
    }

    /** Whether a path could only be a {@code <directory>/<file>.txt} zip entry. */
    private static boolean isZipEntryShaped(String path) {
        int slash = path.indexOf('/');
        return path.indexOf(':') < 0 && slash > 0 && slash == path.lastIndexOf('/');
    }

    // -- where the tree files a stem -------------------------------------

    /**
     * The directory the tree files a stem under, or null when the tree has
     * nothing to say — either it files the stem exactly where
     * {@link CardFilenames#derivedDirectory} would, or it does not hold that
     * stem at all and the caller has nothing better than the derived answer.
     *
     * <p>Only the disagreements are kept. The tree is ~33,700 files and 86 of
     * them are filed somewhere other than under their initial, so an index of
     * the exceptions is three orders of magnitude smaller than an index of the
     * tree while answering the same question — and it is built from the tree
     * itself, so it stays right as the tree changes rather than freezing
     * today's list of oddities into code. The walk happens at most once per JVM
     * and only when something asks: the collector's first route is
     * {@link #relativePathIn}, which costs nothing, so a run in which every
     * card came from the tree never builds this at all.
     */
    public static String directoryOf(String tree, String stem) {
        if (isFlat(tree) || stem == null || stem.isEmpty()) return null;
        return exceptions(tree).get(stem);
    }

    /**
     * Point a tree at its on-disk root, discarding any index already built.
     *
     * <p>A caller that knows where the tree is says so; everything else falls
     * back to Forge's own {@code cardsfolder}. Tests index a temporary tree
     * this way rather than the installation's.
     */
    public static synchronized void useTree(String tree, Path root) {
        ROOTS.put(tree, root);
        INDEXES.remove(tree);
    }

    /** Forget every installed root and index, so one test cannot bias another. */
    static synchronized void forgetTrees() {
        ROOTS.clear();
        INDEXES.clear();
    }

    private static final Map<String, Path> ROOTS = new HashMap<>();
    private static final Map<String, Map<String, String>> INDEXES = new HashMap<>();

    private static synchronized Map<String, String> exceptions(String tree) {
        Map<String, String> index = INDEXES.get(tree);
        if (index == null) {
            index = buildExceptions(rootOf(tree));
            INDEXES.put(tree, index);
        }
        return index;
    }

    private static Path rootOf(String tree) {
        Path installed = ROOTS.get(tree);
        if (installed != null) return installed;
        if (!CARDSFOLDER.equals(tree)) return null;
        try {
            return ForgeEnvironmentInitializer.findCardsFolder();
        } catch (RuntimeException e) {
            // Nothing to consult: derivation is all the caller has, and a tree
            // that cannot be located must not take the collector down with it.
            return null;
        }
    }

    private static Map<String, String> buildExceptions(Path root) {
        Map<String, String> found = new HashMap<>();
        if (root == null || !Files.isDirectory(root)) return found;
        try {
            Files.walkFileTree(root, new SimpleFileVisitor<Path>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                    String name = file.getFileName().toString();
                    if (!name.endsWith(".txt")) return FileVisitResult.CONTINUE;
                    String stem = name.substring(0, name.length() - ".txt".length());
                    Path parent = root.relativize(file).getParent();
                    String directory =
                            parent == null ? "" : parent.toString().replace('\\', '/');
                    if (!directory.equals(CardFilenames.derivedDirectory(stem))) {
                        found.put(stem, directory);
                    }
                    return FileVisitResult.CONTINUE;
                }
            });
        } catch (IOException e) {
            // A half-walked tree still answers for what it saw, and the rest
            // falls back to derivation — which is what happened before this
            // existed, so a partial answer is never worse than no index.
        }
        return found;
    }
}
