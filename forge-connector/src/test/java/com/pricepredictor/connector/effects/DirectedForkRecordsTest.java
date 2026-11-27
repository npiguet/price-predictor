package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.DirectedRules;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.PtOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static com.pricepredictor.connector.effects.ForkTestCaps.forkCaps;
import static com.pricepredictor.connector.effects.RecordEvents.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordPicks.records;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The records of the directed forks: the rule, the variant and the operation
 * each names.
 */
@ExtendWith(ForgeExtension.class)
class DirectedForkRecordsTest {

    @TempDir
    Path tempDir;

    // ── directed forks ───────────────────────────────────────

    /**
     * 704.5f's rule, for these tests only: as the end step begins with
     * Grizzly Bears on the battlefield, its toughness becomes 5.
     */
    private static DirectedRule tougherBears() {
        return DirectedRule.of("704.5f", "704.5f-1", live -> {
            if (live.getPhaseHandler().getPhase() != PhaseType.END_OF_TURN) {
                return Optional.empty();
            }
            return live.getCardsIn(ZoneType.Battlefield).stream()
                    .filter(card -> card.getName().equals("Grizzly Bears")).findFirst()
                    .map(bears -> Binding.empty().with("creature", bears));
        }, new DirectedRule.Variant("tougher", new Application(ForkOperation.PT,
                binding -> new PtOperation(binding.card("creature"), null, 5))));
    }

    /** The live turn of the Sneak Attack board played out with the rule enabled; the shard's lines. */
    private List<String> directedTurn(CollectionCaps caps) throws Throwable {
        Game game = ForkPlayForwardTest.board();
        ForkPlayForwardTest.sneakAttack(game);
        RecordShardWriter shard = ShardRun.writer(tempDir);
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(game, shard, ShardRun.GAME_ID, caps)
                        .seeded(7L)
                        .withForkSeats(ForkSeats.NONE)
                        .withDirected(DirectedRules.of(tougherBears()), new DirectedForkTally()))) {
            recording.open();
            ForkPlayForwardTest.playTheTurnOut(game);
        } finally {
            shard.close();
        }
        return ShardRun.readShard(shard.path());
    }

    private static List<String> split(List<String> all, boolean fork) {
        List<String> kept = new ArrayList<>();
        for (String record : all) {
            if (Boolean.valueOf(fork).equals(parse(record).get("fork"))) {
                kept.add(record);
            }
        }
        return kept;
    }

    /**
     * Every record of a directed fork names its one rule, is a
     * non-interventional fork and says which variant and operation it
     * applied, and mirrors no live record. The live game's records carry
     * none of it on a match-outcomes run.
     */
    @Test
    void aDirectedForksRecordsNameTheRuleTheVariantAndTheOperation() throws Throwable {
        List<String> all = directedTurn(
                forkCaps().interventions(0).probes(0).directed(1, 1.0, "704.5f").build());
        List<String> forked = split(all, true);

        assertFalse(forked.isEmpty(), "the fork at the end step was kept and recorded");
        for (String record : forked) {
            Map<String, Object> parsed = parse(record);
            assertEquals(List.of("704.5f"), parsed.get("aimed_at"), record);
            assertEquals(Boolean.FALSE, parsed.get("interventional"), record);
            assertEquals(Map.of("variant", "tougher", "operation", "pt"), parsed.get("directed"), record);
            assertNull(parsed.get("mirror_of"), record);
            assertEquals(ShardRun.GAME_ID, parsed.get("game_id"), record);
        }
        for (String record : split(all, false)) {
            assertFalse(parse(record).containsKey("aimed_at"), record);
            assertFalse(parse(record).containsKey("directed"), record);
        }
    }

    /**
     * The edit publishes a stats change on the fork's bus, but the fork's
     * collectors start listening after it: no fork record holds the
     * toughness change as an event the turn caused.
     */
    @Test
    void theEditItselfIsNoEventOfTheFork() throws Throwable {
        List<String> all = directedTurn(
                forkCaps().interventions(0).probes(0).directed(1, 1.0, "704.5f").build());

        for (String record : split(all, true)) {
            assertFalse(eventTypes(record).contains("pt_change"), record);
        }
    }

    /**
     * On a run aimed at several rules (collect-variants), the live records
     * carry the run's list and a directed fork's records the one rule it was
     * taken for.
     */
    @Test
    void aDirectedForkOnAnAimedRunNamesItsOwnRule() throws Throwable {
        List<String> all = directedTurn(forkCaps().interventions(0).probes(0)
                .aimedAt("704.5f", "702.19b").directed(1, 1.0, "704.5f").build());

        assertFalse(split(all, true).isEmpty());
        for (String record : split(all, true)) {
            assertEquals(List.of("704.5f"), parse(record).get("aimed_at"), record);
        }
        for (String record : split(all, false)) {
            assertEquals(List.of("704.5f", "702.19b"), parse(record).get("aimed_at"), record);
        }
    }

    /**
     * Feature 025 story 3 acceptance 1 on the registered 704.5f: the active player
     * holds Disfigure and an untapped Swamp and the opponent's Drudge
     * Skeletons is a 1/1, so as the main phase begins the collector forks,
     * raises the Skeletons' toughness through {@code pt}, plays the fork to
     * the end of the turn (its end step sacrifices Sneak Attack's Hill Giant)
     * and writes the fork's records, each showing the 1/3 the fork made.
     */
    @Test
    void theRegisteredToughnessZeroRuleForksATougherCreatureToTheEndOfTheTurn() throws Throwable {
        Game game = ForkPlayForwardTest.board();
        ForkPlayForwardTest.sneakAttack(game);
        Player active = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.inPlay(game, active, "Swamp");
        TestBoard.inHand(game, active, "Disfigure");
        Card skeletons = TestBoard.inPlay(game, opponent, "Drudge Skeletons");
        CollectionCaps caps = forkCaps().interventions(0).probes(0).directed(1, 1.0, "704.5f").build();
        RecordShardWriter shard = ShardRun.writer(tempDir);
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(game, shard, ShardRun.GAME_ID, caps)
                        .seeded(7L))) {
            recording.open();
            // The main phase's boundary, before either seat has had priority
            // in it: given priority earlier, the AI casts Disfigure at once.
            TestBoard.enterStep(game, PhaseType.MAIN1);
            ForkPlayForwardTest.playTheTurnOut(game);
        } finally {
            shard.close();
        }
        List<String> forked = split(ShardRun.readShard(shard.path()), true);

        assertFalse(forked.isEmpty(), "the fork was kept and recorded");
        boolean sawSkeletons = false;
        for (String record : forked) {
            Map<String, Object> parsed = parse(record);
            assertEquals(List.of("704.5f"), parsed.get("aimed_at"), record);
            assertEquals(Map.of("variant", "tougher", "operation", "pt"), parsed.get("directed"), record);
            for (Object entity : (List<?>) RecordLines.asMap(parsed.get("state")).get("entities")) {
                Map<String, Object> fields = RecordLines.asMap(entity);
                if (("E" + skeletons.getId()).equals(fields.get("id"))) {
                    assertEquals(List.of(1L, 3L), fields.get("base_pt"), record);
                    sawSkeletons = true;
                }
            }
        }
        assertTrue(sawSkeletons, "a fork record shows the Skeletons");
    }
}
