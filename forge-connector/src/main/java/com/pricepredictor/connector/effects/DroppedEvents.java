package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * One game's events that reached no record, counted by event type and engine
 * phase (FR-029, FR-041).
 *
 * <p>Three ways an event reaches none: it arrives during a player's action at
 * priority that is no cast (a special action, a mana ability outside any
 * payment); it is kept with the orphans while no bracket is open and then
 * discarded rather than rescued at the game's end; or it is a state-based
 * death with no held resolution and no open damage step to take it.
 *
 * <p>Counted whatever the run, since a count costs one map entry; only a run
 * under {@link CollectionCaps#countDroppedEvents()} writes them, appending the
 * game's counts to its worker lifetime's file at the end of each game ({@link
 * #appendTo}). The supervisor sums the files of its run when the run ends.
 */
final class DroppedEvents {

    /**
     * The file beside a shard that its worker lifetime's counts go to:
     * {@code {run_id}.{worker}-{lifetime}.dropped-events.tsv}. Not a shard
     * suffix, so no shard reader takes it for one.
     */
    static final String SUFFIX = ".dropped-events.tsv";

    /** The phase of an event published before the game has one. */
    static final String NO_PHASE = "none";

    /** What one count is of. */
    record Key(String eventType, String phase) {
    }

    private final Map<Key, Long> counts = new LinkedHashMap<>();

    /** One event of {@code type} reached no record, in {@code phase}. */
    void count(EventType type, String phase) {
        counts.merge(new Key(type.wireName(), phase), 1L, Long::sum);
    }

    /** The engine phase {@code game} is in now, as it is counted under. */
    static String phaseOf(Game game) {
        PhaseHandler handler = game == null ? null : game.getPhaseHandler();
        PhaseType phase = handler == null ? null : handler.getPhase();
        return phase == null ? NO_PHASE : phase.name().toLowerCase(Locale.ROOT);
    }

    /** The counts so far. */
    Map<Key, Long> counts() {
        return Collections.unmodifiableMap(counts);
    }

    /**
     * Append the counts as {@code event_type<TAB>phase<TAB>count} lines.
     *
     * <p>Appended once per game rather than written when the worker exits:
     * the supervisor stops a worker by killing its process tree, which runs
     * no shutdown hook, so a count held until exit would never be written.
     * A worker killed mid-game loses that game's counts alone, and a line cut
     * off by the kill is one the reader skips.
     */
    void appendTo(Path file) throws IOException {
        if (counts.isEmpty()) {
            return;
        }
        StringBuilder lines = new StringBuilder();
        counts.forEach((key, count) -> lines.append(key.eventType()).append('\t')
                .append(key.phase()).append('\t').append(count).append('\n'));
        Files.writeString(file, lines, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }
}
