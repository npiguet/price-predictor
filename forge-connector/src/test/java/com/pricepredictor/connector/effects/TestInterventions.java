package com.pricepredictor.connector.effects;

import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

/** {@link InterventionScheduler}'s watch on the lines it forces, for a test outside this package. */
public final class TestInterventions {

    private TestInterventions() {
    }

    /** Tell {@code watch} of each line forced from now on, or stop telling with null. */
    public static void watch(BiConsumer<SpellAbility, Player> watch) {
        InterventionScheduler.forcedWatch = watch;
    }

    /** The scheduler's spread over a typical game, in game turns. */
    public static int typicalGameTurns() {
        return InterventionScheduler.TYPICAL_GAME_TURNS;
    }

    /**
     * The forced resolutions' records in a closed shard, each parsed:
     * interventional resolution halves with no partner, which leaves out a
     * protection fork's; none when nothing was written to it.
     */
    public static List<Map<String, Object>> forcedRecords(Path shard) throws IOException {
        if (!Files.exists(shard)) {
            return List.of();
        }
        return ShardRun.readShard(shard).stream()
                .map(RecordLines::parse)
                .filter(TestInterventions::forced)
                .toList();
    }

    private static boolean forced(Map<String, Object> record) {
        return Boolean.TRUE.equals(record.get("interventional"))
                && "resolution".equals(record.get("kind"))
                && record.get("link_id") == null;
    }
}
