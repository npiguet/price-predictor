package com.pricepredictor.connector;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

/**
 * Files written together or not at all: a converted card's text and its
 * provenance sidecar, which every reader pairs by name.
 *
 * <p>Each file is written in full under a temporary name beside its target,
 * then renamed over the target, so no reader sees half a file. A rename can
 * fail where a write would not: on Windows a file another process holds open
 * cannot be replaced. When one does, the targets already renamed get their
 * previous bytes back, or are removed when they had none, so what is on disk
 * is the old set whole or the new set whole, and the failure names the
 * target it could not replace.
 */
final class AtomicFiles {

    /** Appended to a target's name for the file written before the rename. */
    static final String TEMPORARY_SUFFIX = ".tmp";

    /** One file of the set: where it goes, and its whole text. */
    record Entry(Path target, String text) {
    }

    private AtomicFiles() {
    }

    /**
     * Write every entry, in order, all or none.
     *
     * @throws IOException naming the target that could not be written or
     *         replaced; every target then holds what it held before
     */
    static void write(List<Entry> entries) throws IOException {
        List<Staged> staged = new ArrayList<>(entries.size());
        try {
            for (Entry entry : entries) {
                staged.add(Staged.of(entry));
            }
            commit(staged);
        } finally {
            for (Staged each : staged) {
                discard(each.temporary());
            }
        }
    }

    /** The temporary name {@code target} is written under before its rename. */
    static Path temporaryFor(Path target) {
        return target.resolveSibling(target.getFileName() + TEMPORARY_SUFFIX);
    }

    private static void commit(List<Staged> staged) throws IOException {
        int committed = 0;
        try {
            for (Staged each : staged) {
                each.commit();
                committed++;
            }
        } catch (IOException failure) {
            for (int i = committed - 1; i >= 0; i--) {
                staged.get(i).restore(failure);
            }
            throw failure;
        }
    }

    /** Rename {@code from} over {@code to} in one step. */
    private static void replace(Path from, Path to) throws IOException {
        try {
            Files.move(from, to, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new IOException("cannot replace " + to + ": " + BatchConverter.describe(e), e);
        }
    }

    /**
     * Remove a temporary file that was not renamed. A failure is not
     * reported: the target is already whole either way, and a converted
     * tree's {@code --replace} sweep removes a leftover temporary file.
     */
    private static void discard(Path temporary) {
        try {
            Files.deleteIfExists(temporary);
        } catch (IOException ignored) {
            // Left for the next sweep; see above.
        }
    }

    /**
     * An entry written under its temporary name, with what its target held
     * before ({@code null} when it held nothing), for {@link #restore}.
     */
    private record Staged(Path target, Path temporary, byte[] previous) {

        static Staged of(Entry entry) throws IOException {
            Path target = entry.target();
            byte[] previous = Files.isRegularFile(target) ? Files.readAllBytes(target) : null;
            Path temporary = temporaryFor(target);
            Files.writeString(temporary, entry.text(), StandardCharsets.UTF_8);
            return new Staged(target, temporary, previous);
        }

        void commit() throws IOException {
            replace(temporary, target);
        }

        /** Put back what the target held before {@link #commit}. */
        void restore(IOException failure) {
            try {
                if (previous == null) {
                    Files.deleteIfExists(target);
                } else {
                    Files.write(temporary, previous);
                    replace(temporary, target);
                }
            } catch (IOException e) {
                failure.addSuppressed(e);
            }
        }
    }
}
