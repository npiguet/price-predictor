package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.WireNamed;

import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

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
 * unreleased card under {@code upcoming/} whatever its initial. The converter
 * walks the tree and keys each sidecar by
 * the path it read the file from, so a collector that <em>derives</em> a path
 * instead disagrees with the sidecar for exactly those cards, and the mis-join
 * is silent: only a {@code keyword} key is checked against its sidecar, so a
 * fabricated {@code cardsfolder/+/+2_mace.txt} joins to nothing and nothing
 * says so. This class is therefore where a caller asks the tree where a card
 * lives; {@link CardFilenames} derives only what the tree does not answer.
 */
public enum SourceTree implements WireNamed {

    CARDSFOLDER("cardsfolder", false),
    TOKENSCRIPTS("tokenscripts", true),
    VARIANT_SCRIPTS("variant-scripts", true);

    private final String wireName;
    private final boolean flat;

    SourceTree(String wireName, boolean flat) {
        this.wireName = wireName;
        this.flat = flat;
    }

    /**
     * The tree's directory name, which is also the first segment of every
     * provenance key's {@code script_file} the Python side splits on.
     */
    @Override
    public String wireName() {
        return wireName;
    }

    /** Whether the tree files every script at its root rather than under a directory. */
    public boolean isFlat() {
        return flat;
    }

    // -- where a loader says it read a file ------------------------------

    /**
     * The tree-relative path a loader-reported file path names, or empty when
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
    public Optional<String> relativePathIn(String loaderPath) {
        if (loaderPath == null || loaderPath.isEmpty()) return Optional.empty();
        String path = loaderPath.replace('\\', '/');
        String anchor = "/" + wireName + "/";
        int at = path.lastIndexOf(anchor);
        String relative;
        if (at >= 0) {
            relative = path.substring(at + anchor.length());
        } else if (path.startsWith(wireName + "/")) {
            relative = path.substring(wireName.length() + 1);
        } else if (isZipEntryShaped(path)) {
            relative = path;
        } else {
            return Optional.empty();
        }
        return Optional.of(relative).filter(r -> r.endsWith(".txt"));
    }

    /** Whether a path could only be a {@code <directory>/<file>.txt} zip entry. */
    private static boolean isZipEntryShaped(String path) {
        int slash = path.indexOf('/');
        return path.indexOf(':') < 0 && slash > 0 && slash == path.lastIndexOf('/');
    }

    // -- where the tree files a stem -------------------------------------

    /**
     * The directory the tree files a stem under, or empty when the tree has
     * nothing to say — either it files the stem exactly where
     * {@link CardFilenames#derivedDirectory} would, or it does not hold that
     * stem at all and the caller has nothing better than the derived answer.
     *
     * <p>Only the disagreements are kept. A few dozen of the tree's tens of
     * thousands of files are filed somewhere other than under their initial, so
     * an index of the exceptions is orders of magnitude smaller than an index
     * of the tree while answering the same question — and it is built from the tree
     * itself, so it stays right as the tree changes rather than freezing
     * today's list of oddities into code. The walk happens at most once per JVM
     * and only when something asks: the collector's first route is
     * {@link #relativePathIn}, which costs nothing, so a run in which every
     * card came from the tree never builds this at all.
     */
    public Optional<String> directoryOf(String stem) {
        if (flat || stem == null || stem.isEmpty()) return Optional.empty();
        return Optional.ofNullable(exceptions(this).get(stem));
    }

    /** Guarded by the class lock, which also keeps a tree from being walked twice. */
    private static final Map<SourceTree, Map<String, String>> INDEXES =
            new EnumMap<>(SourceTree.class);

    private static synchronized Map<String, String> exceptions(SourceTree tree) {
        return INDEXES.computeIfAbsent(tree, t -> buildExceptions(t, t.root()));
    }

    /** Forge's own {@code cardsfolder}; no other tree is letter-keyed. */
    private Path root() {
        if (this != CARDSFOLDER) return null;
        try {
            return ForgeEnvironmentInitializer.findCardsFolder();
        } catch (RuntimeException e) {
            // Nothing to consult: derivation is all the caller has, and a tree
            // that cannot be located must not take the collector down with it.
            return null;
        }
    }

    /**
     * The stems {@code root} files somewhere other than the convention says.
     *
     * <p>Package-private so a test can hand it a root that does not exist: a
     * tree that cannot be found yields no exceptions, never an error.
     */
    static Map<String, String> buildExceptions(SourceTree tree, Path root) {
        Map<String, String> found = new HashMap<>();
        if (root == null || !Files.isDirectory(root)) return found;
        try {
            Files.walkFileTree(root, new SimpleFileVisitor<Path>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                    if (!file.getFileName().toString().endsWith(".txt")) {
                        return FileVisitResult.CONTINUE;
                    }
                    ScriptFile script =
                            new ScriptFile(tree, root.relativize(file).toString());
                    if (!script.directory().equals(
                            CardFilenames.derivedDirectory(script.stem()))) {
                        found.put(script.stem(), script.directory());
                    }
                    return FileVisitResult.CONTINUE;
                }
            });
        } catch (IOException e) {
            // A half-walked tree still answers for what it saw, and the rest
            // falls back to derivation, which is the answer with no index at
            // all, so a partial answer is never worse than none.
        }
        return found;
    }
}
