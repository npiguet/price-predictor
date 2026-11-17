package com.pricepredictor.connector;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

/**
 * The output trees {@link ConvertMain} writes, checked before anything is
 * written.
 *
 * <p>A conversion into a tree that already holds converted files would mix
 * two conversions: a script that fails, or that Forge has since deleted,
 * keeps its previous pair, which every reader still loads. So such a tree is
 * refused, unless the caller asks for it to be replaced, in which case its
 * converted files are swept out first. The sweep takes only what a
 * conversion writes ({@link #isConverted}) and the directories it leaves
 * empty, never the tree itself, a file of another kind (the {@code run.json}
 * a caller keeps there) or anything outside the tree.
 *
 * <p>Whatever is asked, a tree that is a filesystem root, holds the working
 * directory, is a repository's root, or overlaps a source tree or the other
 * output tree is refused: writing 34,000 converted files there, or sweeping
 * it, is never what was meant.
 */
final class ConvertedTrees {

    private static final String PROVENANCE_SUFFIX = ".provenance.json";

    private ConvertedTrees() {
    }

    /**
     * Refuse each output tree that is unsafe to write or sweep.
     *
     * @throws ToolMain.Exit with {@link ExitCode#BAD_CONFIG}, the tree named
     */
    static void refuseUnsafe(List<Path> outputs, List<Path> sources) throws IOException {
        Path workingDirectory = canonical(Path.of(""));
        for (Path output : outputs) {
            Path tree = canonical(output);
            String problem = problemWith(tree, workingDirectory);
            if (problem == null) {
                problem = overlap(tree, sources, "the source tree");
            }
            if (problem == null) {
                problem = overlap(tree, others(outputs, output), "the other output tree");
            }
            if (problem != null) {
                throw ToolMain.badConfig("refusing to convert into " + tree + ": " + problem);
            }
        }
    }

    /**
     * Refuse an output tree that already holds a converted file.
     *
     * @throws ToolMain.Exit with {@link ExitCode#BAD_CONFIG}, the tree named
     */
    static void refuseOccupied(List<Path> outputs) throws IOException {
        for (Path output : outputs) {
            if (holdsConvertedFiles(output)) {
                throw ToolMain.badConfig(output + " already holds converted files; pass --replace"
                        + " to sweep them out first, or convert into an empty directory");
            }
        }
    }

    /**
     * Remove every converted file under {@code output}, then every directory
     * that leaves empty below it.
     *
     * @return how many files were removed
     * @throws IOException naming the file or directory that could not be removed
     */
    static int sweep(Path output) throws IOException {
        if (!Files.isDirectory(output)) {
            return 0;
        }
        List<Path> deepestFirst;
        try (Stream<Path> paths = Files.walk(output)) {
            deepestFirst = paths.filter(path -> !path.equals(output))
                    .sorted(Comparator.reverseOrder())
                    .toList();
        }
        int removed = 0;
        for (Path path : deepestFirst) {
            if (Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS)) {
                removeIfEmpty(path);
            } else if (isConverted(path)) {
                remove(path);
                removed++;
            }
        }
        return removed;
    }

    /**
     * Whether a conversion writes files by this name: a converted text, a
     * provenance sidecar, or either one's temporary name ({@link AtomicFiles})
     * left by a killed run.
     */
    static boolean isConverted(Path file) {
        String name = file.getFileName().toString();
        if (name.endsWith(AtomicFiles.TEMPORARY_SUFFIX)) {
            name = name.substring(0, name.length() - AtomicFiles.TEMPORARY_SUFFIX.length());
        }
        return name.endsWith(".txt") || name.endsWith(PROVENANCE_SUFFIX);
    }

    private static boolean holdsConvertedFiles(Path output) throws IOException {
        if (!Files.isDirectory(output)) {
            return false;
        }
        try (Stream<Path> paths = Files.walk(output)) {
            return paths.anyMatch(path -> Files.isRegularFile(path) && isConverted(path));
        }
    }

    /** Why {@code tree} is never an output tree, or null when it may be. */
    private static String problemWith(Path tree, Path workingDirectory) {
        if (tree.getParent() == null) {
            return "it is a filesystem root";
        }
        if (workingDirectory.startsWith(tree)) {
            return "it holds the working directory " + workingDirectory;
        }
        if (Files.exists(tree.resolve(".git"))) {
            return "it is a repository's root";
        }
        return null;
    }

    /** Which of {@code trees} {@code tree} lies in or holds, named, or null. */
    private static String overlap(Path tree, List<Path> trees, String kind) throws IOException {
        for (Path other : trees) {
            Path otherTree = canonical(other);
            if (tree.startsWith(otherTree) || otherTree.startsWith(tree)) {
                return "it overlaps " + kind + " " + otherTree;
            }
        }
        return null;
    }

    private static List<Path> others(List<Path> outputs, Path output) {
        return outputs.stream().filter(other -> other != output).toList();
    }

    /** The absolute path, links resolved as far as the path exists. */
    private static Path canonical(Path path) throws IOException {
        Path absolute = path.toAbsolutePath().normalize();
        Path existing = absolute;
        while (existing != null && !Files.exists(existing)) {
            existing = existing.getParent();
        }
        return existing == null ? absolute : existing.toRealPath().resolve(existing.relativize(absolute));
    }

    private static void removeIfEmpty(Path directory) throws IOException {
        boolean empty;
        try (DirectoryStream<Path> entries = Files.newDirectoryStream(directory)) {
            empty = !entries.iterator().hasNext();
        }
        if (empty) {
            remove(directory);
        }
    }

    private static void remove(Path path) throws IOException {
        try {
            Files.delete(path);
        } catch (IOException e) {
            throw new IOException("cannot sweep " + path + ": " + BatchConverter.describe(e), e);
        }
    }
}
