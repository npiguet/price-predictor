package com.pricepredictor.connector;

import forge.game.CardTraitBase;

import java.util.ArrayList;
import java.util.List;

/**
 * What one keyword rendered: the lines it printed, in order, and, for a
 * keyword printing a line per trait it built (a saga's chapters, haunt's
 * halves), which of those lines each such trait printed.
 *
 * <p>Built in the one pass that renders the keyword, so the line a built
 * trait's key is declared on is read off this value rather than reported on
 * the side. A keyword printing its lines as a whole names no trait line.
 * Without the trait lines, every trait a several-line keyword built would be
 * claimed by every one of its lines, and the join, which reads the first line
 * holding a key, would give a saga's chapter II record chapter I's text.
 *
 * @param lines      the lines, in the order the face lists them
 * @param traitLines the line each built trait printed, where the keyword
 *                   prints a line per trait; a trait printing none is absent
 */
public record KeywordRendering(List<Ability> lines, List<TraitLine> traitLines) {

    /** The line {@code trait}, one of the keyword's built traits, printed. */
    public record TraitLine(CardTraitBase trait, Ability line) {}

    public KeywordRendering {
        lines = List.copyOf(lines);
        traitLines = List.copyOf(traitLines);
    }

    /** A keyword printing {@code lines} as a whole. */
    public static KeywordRendering of(List<Ability> lines) {
        return new KeywordRendering(lines, List.of());
    }

    /** A keyword printing {@code line} alone, or nothing when it is null. */
    public static KeywordRendering of(Ability line) {
        return of(line == null ? List.of() : List.of(line));
    }

    /**
     * The line {@code trait} printed, or null when the keyword names none for
     * it. By identity, as Forge's traits are told apart.
     */
    public Ability lineOf(CardTraitBase trait) {
        for (TraitLine traitLine : traitLines) {
            if (traitLine.trait() == trait) return traitLine.line();
        }
        return null;
    }

    /**
     * A keyword printing {@code line} alone, as the text of {@code trait} (a
     * Visit's trigger, Backup's trigger), or nothing when the line is null;
     * a null trait names none, and the line is the keyword's own text.
     */
    public static KeywordRendering ofTrait(Ability line, CardTraitBase trait) {
        if (line == null) return of(List.of());
        Builder rendering = new Builder();
        rendering.add(line);
        if (trait != null) rendering.printed(trait, line);
        return rendering.build();
    }

    /**
     * The traits that printed {@code line}, in the order the keyword named
     * them, or none when the line is the keyword's own text rather than a
     * trait's (a Class level's line is its level-up ability's and its
     * static's). By identity.
     */
    public List<CardTraitBase> traitsOf(Ability line) {
        return traitLines.stream()
                .filter(traitLine -> traitLine.line() == line)
                .map(TraitLine::trait)
                .toList();
    }

    /** Collects a keyword's lines and its traits' lines while it renders. */
    public static final class Builder {
        private final List<Ability> lines = new ArrayList<>();
        private final List<TraitLine> traitLines = new ArrayList<>();

        /** Append a line the keyword prints. */
        public void add(Ability line) {
            lines.add(line);
        }

        /** Note that {@code trait} printed {@code line}; a null line notes nothing. */
        public void printed(CardTraitBase trait, Ability line) {
            if (line != null) traitLines.add(new TraitLine(trait, line));
        }

        public KeywordRendering build() {
            return new KeywordRendering(lines, traitLines);
        }
    }
}
