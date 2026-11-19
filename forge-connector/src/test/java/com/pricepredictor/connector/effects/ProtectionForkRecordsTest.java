package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static com.pricepredictor.connector.effects.RecordLines.eventsNaming;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What a protection fork's grant itself sets off is in the fork's records:
 * the state-based check the grant makes true runs as the fork's premise,
 * under a record of its own, with the fork's collectors listening.
 */
@ExtendWith(ForgeExtension.class)
class ProtectionForkRecordsTest {

    @TempDir
    Path tempDir;

    /**
     * Protection from enchantments granted to a creature wearing Pacifism:
     * the Aura falls off in the check after the grant, and one fork record
     * holds both its {@code unattached} and its move to the graveyard.
     */
    @Test
    void anAuraTheGrantMakesFallOffIsInAForkRecord() throws Throwable {
        Game game = ForkPlayForwardTest.board();
        Player opponent = game.getPlayers().get(1);
        Card bears = game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        Card pacifism = TestBoard.inPlay(game, opponent, "Pacifism");
        pacifism.attachToEntity(bears, null);

        List<String> forked = protectionFork(game, new ProtectionFork.Grant(bears, "Protection:Enchantment"));

        String ref = Refs.of(pacifism);
        List<String> fell = forked.stream().filter(record -> fellOff(record, ref)).toList();
        assertEquals(1, fell.size(),
                () -> "no one fork record holds Pacifism's unattached and its move to the graveyard: " + forked);
        Map<String, Object> premise = parse(fell.get(0));
        assertEquals("turn_action", premise.get("kind"));
        assertEquals("fork_edit", premise.get("subkind"));
        assertEquals(Boolean.TRUE, premise.get("fork"));
        assertEquals(Boolean.TRUE, premise.get("interventional"));
        assertEquals(forked.get(0), fell.get(0), "the premise is the fork's first record");
        for (String record : forked) {
            assertEquals("protection", parse(record).get("fork_kind"), record);
        }
    }

    /**
     * Protection from enchantments granted to a creature wearing Holy Strength:
     * the Aura falls off in the premise's state-based check, and the stats the
     * Aura gave go with it as a {@code pt_change} on the creature, in the
     * premise record as the live priority check publishes it.
     */
    @Test
    void theStatsAnAuraTheGrantMakesFallOffGaveAreInTheForkRecord() throws Throwable {
        Game game = ForkPlayForwardTest.board();
        Player active = game.getPlayers().get(0);
        Card bears = game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        TestBoard.inPlay(game, active, "Holy Strength").attachToEntity(bears, null);
        game.getAction().checkStateEffects(true);

        List<String> forked = protectionFork(game, new ProtectionFork.Grant(bears, "Protection:Enchantment"));

        List<Map<String, Object>> changes = eventsNaming(forked.get(0), Refs.of(bears)).stream()
                .filter(event -> "pt_change".equals(event.get("type"))).toList();
        assertEquals(1, changes.size(),
                () -> "the premise record holds no pt_change on the Bears: " + forked.get(0));
        assertEquals(-1, ((Number) params(changes.get(0)).get("power_delta")).intValue());
        assertEquals(-2, ((Number) params(changes.get(0)).get("toughness_delta")).intValue());
    }

    /**
     * What the fork dropped (an attack's taps, its turn played on) is in the
     * live game's counts, under the fork's kind: the live game, which plays
     * nothing here, counts nothing of its own.
     */
    @Test
    void aKeptForksDroppedEventsAreCountedUnderItsKind() throws Throwable {
        Game game = ForkPlayForwardTest.board();
        Card bears = game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        Map<DroppedEvents.Key, Long> counts = new HashMap<>();

        protectionFork(game, new ProtectionFork.Grant(bears, "Protection from everything"),
                recording -> counts.putAll(recording.droppedEvents().counts()));

        assertFalse(counts.isEmpty(), "the fork dropped something over its turn");
        assertTrue(counts.keySet().stream().allMatch(key -> key.phase().startsWith("protection:")),
                () -> "a count not under the fork's kind: " + counts);
    }

    /** Take {@code grant}'s protection fork over {@code game} and read what it wrote. */
    private List<String> protectionFork(Game game, ProtectionFork.Grant grant) throws Exception {
        return protectionFork(game, grant, recording -> { });
    }

    /** The same, {@code after} handed the live recording once the fork is taken. */
    private List<String> protectionFork(
            Game game, ProtectionFork.Grant grant, Consumer<GameRecording> after) throws Exception {
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(0).probes(0).build();
        RecordShardWriter shard = ShardRun.writer(tempDir);
        try (GameRecording recording = new GameRecording(game, shard, ShardRun.GAME_ID, caps, 7L, null)) {
            recording.open();
            assertTrue(new ProtectionFork(game, new ForkBudget(caps), recording.playForward(), 7L).take(grant),
                    "the fork was kept");
            after.accept(recording);
        } finally {
            shard.close();
        }
        return ShardRun.readShard(shard.path());
    }

    /** Whether {@code record} holds the Aura {@code ref}'s unattached and its move to the graveyard. */
    private static boolean fellOff(String record, String ref) {
        List<Map<String, Object>> events = eventsNaming(record, ref);
        return events.stream().anyMatch(event -> "unattached".equals(event.get("type")))
                && events.stream().anyMatch(ProtectionForkRecordsTest::movedToTheGraveyard);
    }

    private static boolean movedToTheGraveyard(Map<String, Object> event) {
        return "zone_change".equals(event.get("type")) && "graveyard".equals(params(event).get("to_zone"));
    }
}
