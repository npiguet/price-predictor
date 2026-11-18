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
 * a caller keeps there, {@code check-convert}'s {@code check_report.txt}) or
 * anything outside the tree.
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
        return sweep(output, Files::delete);
    }

    /** Removes one path; a test substitutes one that fails. */
    @FunctionalInterface
    interface Remover {
        void remove(Path path) throws IOException;
    }

    /**
     * {@link #sweep(Path)} through {@code remover}. The converted files go
     * pair by pair, each sidecar before its text, so a file that cannot be
     * removed leaves only whole pairs behind (and the failure names it).
     */
    static int sweep(Path output, Remover remover) throws IOException {
        if (!Files.isDirectory(output)) {
            return 0;
        }
        List<Path> files;
        List<Path> directoriesDeepestFirst;
        try (Stream<Path> paths = Files.walk(output)) {
            List<Path> all = paths.filter(path -> !path.equals(output)).toList();
            // Decided before anything is removed: a text is converted only
            // while its sidecar is there.
            files = all.stream()
                    .filter(path -> !Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS))
                    .filter(ConvertedTrees::isConverted)
                    .sorted(Comparator.comparing((Path path) -> path.getParent().toString())
                            .thenComparing(ConvertedTrees::stem)
                            .thenComparing(path -> path.getFileName().toString().endsWith(".txt")))
                    .toList();
            directoriesDeepestFirst = all.stream()
                    .filter(path -> Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS))
                    .sorted(Comparator.reverseOrder())
                    .toList();
        }
        for (Path file : files) {
            remove(file, remover);
        }
        for (Path directory : directoriesDeepestFirst) {
            removeIfEmpty(directory, remover);
        }
        return files.size();
    }

    /**
     * Whether a conversion writes this file: a provenance sidecar, a
     * converted text (a {@code .txt} with its sidecar beside it; any other
     * text, such as {@code check-convert}'s report, is the caller's), or
     * either one's temporary name ({@link AtomicFiles}) left by a killed run.
     */
    static boolean isConverted(Path file) {
        String name = file.getFileName().toString();
        if (name.endsWith(AtomicFiles.TEMPORARY_SUFFIX)) {
            String written = name.substring(0, name.length() - AtomicFiles.TEMPORARY_SUFFIX.length());
            return written.endsWith(".txt") || written.endsWith(PROVENANCE_SUFFIX);
        }
        if (name.endsWith(PROVENANCE_SUFFIX)) {
            return true;
        }
        return name.endsWith(".txt")
                && Files.isRegularFile(file.resolveSibling(stem(file) + PROVENANCE_SUFFIX));
    }

    /** The file name without its {@code .txt} or sidecar suffix. */
    private static String stem(Path file) {
        String name = file.getFileName().toString();
        if (name.endsWith(PROVENANCE_SUFFIX)) {
            return name.substring(0, name.length() - PROVENANCE_SUFFIX.length());
        }
        return name.endsWith(".txt") ? name.substring(0, name.length() - ".txt".length()) : name;
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

    private static void removeIfEmpty(Path directory, Remover remover) throws IOException {
        boolean empty;
        try (DirectoryStream<Path> entries = Files.newDirectoryStream(directory)) {
            empty = !entries.iterator().hasNext();
        }
        if (empty) {
            remove(directory, remover);
        }
    }

    private static void remove(Path path, Remover remover) throws IOException {
        try {
            remover.remove(path);
        } catch (IOException e) {
            throw new IOException("cannot sweep " + path + ": " + BatchConverter.describe(e), e);
        }
    }
}
