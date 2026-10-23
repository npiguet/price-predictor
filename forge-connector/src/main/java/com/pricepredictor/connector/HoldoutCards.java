package com.pricepredictor.connector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/**
 * The card names a depleted run keeps out of every booster.
 *
 * <p>With these absent from every pool, no collected game holds a held-out
 * card and none has to be discarded from the effect model's training corpus.
 * An empty set is an ordinary full-strength run, which is what supplies the
 * card-disjoint validation stratum.
 *
 * <p>The list is the newline-delimited file {@code effects holdout-cards}
 * writes, read the same way by {@link PoolMain} ({@code --exclude-cards}) and
 * the match worker ({@code -Dsealed.exclude.cards}).
 */
final class HoldoutCards {

    /** The system property naming the list file. */
    static final String EXCLUDE_CARDS = "sealed.exclude.cards";

    /** A full-strength run: nothing held out. */
    static final HoldoutCards NONE = new HoldoutCards(Set.of());

    private final Set<String> folded;

    private HoldoutCards(Set<String> folded) {
        this.folded = Set.copyOf(folded);
    }

    /** These names held out, in any spelling {@link #fold} agrees on. */
    static HoldoutCards of(Collection<String> names) {
        Set<String> folded = new HashSet<>(names.size());
        for (String name : names) {
            folded.add(fold(name));
        }
        return new HoldoutCards(folded);
    }

    /** The names in {@code file}, one per line; blank lines are skipped. */
    static HoldoutCards load(Path file) throws IOException {
        return of(Files.readAllLines(file).stream()
                .map(String::trim)
                .filter(line -> !line.isEmpty())
                .toList());
    }

    /**
     * The list named by {@code -Dsealed.exclude.cards}, or {@link #NONE} when
     * the property is unset.
     *
     * @throws IllegalStateException when the property names a file that
     *         cannot be read: a depleted run that silently fell back to
     *         full-strength pools would put held-out cards in training games
     */
    static HoldoutCards fromSystemProperty() {
        String path = System.getProperty(EXCLUDE_CARDS);
        if (path == null || path.isBlank()) {
            return NONE;
        }
        try {
            return load(Path.of(path.trim()));
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Cannot read -D" + EXCLUDE_CARDS + "=" + path
                            + ". A depleted run that silently fell back to full-strength "
                            + "pools would put held-out cards in training games.", e);
        }
    }

    /** Whether {@code cardName} is held out. */
    boolean contains(String cardName) {
        return folded.contains(fold(cardName));
    }

    int size() {
        return folded.size();
    }

    /**
     * A card name in the one spelling both sides of the boundary agree on.
     *
     * <p>The list is written from names read out of the converted card tree,
     * and converted text is lowercased: it says {@code soul echo} where Forge
     * says {@code Soul Echo}. Comparing the two directly matches nothing, and
     * a run told to deplete its pools depletes none of them without saying so.
     *
     * <p>{@code Locale.ROOT} to agree with the Python side's {@code lower()}.
     * A Turkish default locale folds {@code I} to a dotless {@code i}, and the
     * two languages would then disagree about the holdout.
     */
    static String fold(String name) {
        return name.toLowerCase(Locale.ROOT);
    }
}
