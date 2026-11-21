package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.Map;
import java.util.NavigableMap;
import java.util.TreeMap;
import java.util.WeakHashMap;

/**
 * The clause notes of one game, which join a timestamped floating effect to
 * the line that drew it ({@link FloatingEffects}): at each live clause's
 * {@code ON_CLAUSE_RESOLVING}, the game's timestamp, the line resolving and
 * its {@code Duration$}, closed at {@code ON_CLAUSE_RESOLVED}. A row's
 * timestamp falls inside the innermost note open when Forge drew it.
 *
 * <p>The notes are kept per game in a weak registry, because the clause hook
 * and the several snapshot builders over one game are built apart.
 */
final class ClauseNotes {

    private static final Map<Game, ClauseNotes> BY_GAME =
            Collections.synchronizedMap(new WeakHashMap<>());

    /** The notes of {@code game}, made on first use. */
    static ClauseNotes of(Game game) {
        return BY_GAME.computeIfAbsent(game, g -> new ClauseNotes(null, 0L));
    }

    /**
     * The notes of a fork of {@code live}, made on first use: its own
     * clauses' notes, and the live game's for every timestamp at or before
     * the copy point (B4). The copy keeps the live game's timestamps, so a
     * row drawn before the fork is the live game's to name; a fork's own
     * notes start empty.
     */
    static ClauseNotes ofFork(Game fork, Game live) {
        return BY_GAME.computeIfAbsent(fork, g -> new ClauseNotes(of(live), fork.getTimestamp()));
    }

    /**
     * Forget {@code game}'s notes: its recording has closed, or its fork has
     * ended. A note still open (a fork whose body threw mid-clause) holds
     * its clause and so its game, which the weak key alone never lets go.
     */
    static void release(Game game) {
        if (game != null) {
            BY_GAME.remove(game);
        }
    }

    /** How many games' notes are held, for a test. */
    static int gamesHeld() {
        return BY_GAME.size();
    }

    /** Whether notes of {@code game} are held, for a test. */
    static boolean holds(Game game) {
        return BY_GAME.containsKey(game);
    }

    /** One live clause's resolution: the timestamps Forge drew while it ran. */
    static final class Note {
        private final long start;
        private final ProvenanceKey line;
        /** The ref of the card whose line it is, or null. */
        private final String host;
        private final String duration;
        /**
         * The clause while it runs, to close the right note; dropped once
         * closed, because a clause holds its game and the registry holds
         * games only weakly.
         */
        private SpellAbility clause;
        /** The game's timestamp when the clause finished; open until then. */
        private long end = Long.MAX_VALUE;

        private Note(long start, ProvenanceKey line, String host, String duration, SpellAbility clause) {
            this.start = start;
            this.line = line;
            this.host = host;
            this.duration = duration;
            this.clause = clause;
        }

        /** The key of the line that drew the clause's rows, or null. */
        ProvenanceKey line() {
            return line;
        }

        /** The ref of the card whose line it is, or null. */
        String host() {
            return host;
        }

        /** The clause's {@code Duration$}, unbucketed. */
        String duration() {
            return duration;
        }

        private boolean covers(long timestamp) {
            return timestamp > start && timestamp <= end;
        }
    }

    /** By (start, order noted): clauses nest, and nested ones can share a start. */
    private final NavigableMap<long[], Note> notes = new TreeMap<>((a, b) ->
            a[0] != b[0] ? Long.compare(a[0], b[0]) : Long.compare(a[1], b[1]));
    private final Deque<Note> open = new ArrayDeque<>();
    private long noted;
    /** The live game's notes, for a fork's; null for a live game's own. */
    private final ClauseNotes copiedFrom;
    /** The game's timestamp when a fork was copied; the live game's notes cover up to it. */
    private final long copyPoint;

    private ClauseNotes(ClauseNotes copiedFrom, long copyPoint) {
        this.copiedFrom = copiedFrom;
        this.copyPoint = copyPoint;
    }

    /** A live clause begins resolving: note the timestamp, its line and its duration. */
    synchronized void clauseResolving(Game game, SpellAbility clause) {
        Note note = new Note(game.getTimestamp(), ProvenanceResolver.of(clause),
                Refs.ofNullable(clause.getHostCard()), noteDuration(clause), clause);
        notes.put(new long[] {note.start, noted++}, note);
        open.push(note);
    }

    /** A live clause finished, or threw: close its note. */
    synchronized void clauseResolved(Game game, SpellAbility clause) {
        for (var it = open.iterator(); it.hasNext(); ) {
            Note note = it.next();
            if (note.clause == clause) {
                note.end = game.getTimestamp();
                note.clause = null;
                it.remove();
                return;
            }
        }
    }

    /**
     * The innermost note that covers a timestamp, or null: the live game's,
     * for a fork's timestamp at or before its copy point.
     */
    Note noteAt(long timestamp) {
        if (copiedFrom != null && timestamp <= copyPoint) {
            return copiedFrom.noteAt(timestamp);
        }
        return ownNoteAt(timestamp);
    }

    private synchronized Note ownNoteAt(long timestamp) {
        for (Note note : notes.headMap(new long[] {timestamp, Long.MAX_VALUE}, false)
                .descendingMap().values()) {
            if (note.covers(timestamp)) {
                return note;
            }
        }
        return null;
    }

    /**
     * A noted clause's duration, as a {@code Duration$} value: its declared
     * one, else, for a {@code GainControl} clause, the one its {@code
     * LoseControl$} says: {@code EOT} until end of turn, an object's change
     * ({@code LeavesPlay}, {@code LoseControl}, {@code Untap}, ...) until it
     * leaves, and none at all for good.
     */
    static String noteDuration(SpellAbility clause) {
        String declared = FloatingEffects.declaredDuration(clause);
        if (declared != null || !gainsControl(clause)) {
            return declared;
        }
        return loseControlDuration(clause.getParam("LoseControl"));
    }

    private static boolean gainsControl(SpellAbility clause) {
        return clause != null && clause.getApi() == ApiType.GainControl;
    }

    /** A {@code GainControl} clause's {@code LoseControl$}, as a {@code Duration$} value. */
    private static String loseControlDuration(String lose) {
        if (lose == null || lose.isEmpty()) {
            return "Permanent";
        }
        if (lose.equals("EOT")) {
            return null;
        }
        return lose.startsWith("UntilTheEndOf") ? lose : "UntilHostLeavesPlay";
    }
}
