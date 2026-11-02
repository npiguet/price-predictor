package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A card's first change in power and toughness is recorded, and a change is
 * recorded only by the records whose snapshots precede it.
 *
 * <p>The bus says only that a card's stats changed, and Forge says it after the
 * change is applied, so a differ with no earlier reading of the card has
 * nothing to compare against: the first pump on a creature would leave no
 * {@code pt_change}, and its wearing off would read as a lone -3/-3. Said
 * late enough, a change has already reached the snapshot of the record it
 * is published into. These
 * drive the real engine, because what is under test is when the engine
 * publishes relative to when it changes the card.
 */
@ExtendWith(ForgeExtension.class)
class FirstStatChangeTest {

    @TempDir
    Path tempDir;

    /** The rendered start of a +3/+3 on a card. */
    private static String pumpOf(Card card) {
        return "{\"type\":\"pt_change\",\"subjects\":[\"E" + card.getId()
                + "\"],\"params\":{\"power_delta\":3,\"toughness_delta\":3";
    }

    /** A creature that has just entered, then pumped, writes the pump. */
    @Test
    void aCreatureThatJustEnteredRecordsItsFirstPump() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        // Entered under the bracket's eye, so the pump is the first change it sees.
        AtomicReference<Card> bears = new AtomicReference<>();
        List<String> records = ShardRun.observed(tempDir, game, bracket -> {
            bears.set(TestBoard.onBattlefield(game, caster, "Grizzly Bears"));
            SpellAbility growth = TestBoard.spellOf(TestBoard.inHand(game, caster, "Giant Growth"), caster);
            growth.resetTargets();
            growth.getTargets().add(bears.get());
            game.getStack().add(growth);
            game.getStack().resolveStack();
            TestBoard.nextStep(game);
        });

        String shard = String.join("\n", records);
        assertTrue(shard.contains(pumpOf(bears.get())), shard);
    }

    /**
     * A fork reports the first change of a card it never saw change before:
     * its sink lives for one resolution, so every card on the copied board
     * is new to it.
     */
    @Test
    void aForkRecordsTheFirstPumpOfACreature() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.onBattlefield(game, caster, "Grizzly Bears");
        SpellAbility growth = TestBoard.spellOf(TestBoard.inHand(game, caster, "Giant Growth"), caster);
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(1).probes(0).build();

        String shard = String.join("\n",
                ShardRun.forked(tempDir, game, caps, 1L, growth).records());
        assertTrue(shard.contains("\"interventional\":true"), "no fork was written: " + shard);
        assertTrue(shard.contains(pumpOf(bears)), shard);
    }

    private static final String DAXOS = "cardsfolder/d/daxos_blessed_by_the_sun.txt";

    /** A creature's net toughness as a record's state shows it. */
    private static long toughnessIn(String record, Card card) {
        Object entities = RecordLines.asMap(RecordLines.parse(record).get("state")).get("entities");
        for (Object each : (List<?>) entities) {
            Map<String, Object> entity = RecordLines.asMap(each);
            if (("E" + card.getId()).equals(entity.get("id"))) {
                Map<String, Object> boosts = RecordLines.asMap(entity.get("pt_boosts"));
                return (Long) ((List<?>) entity.get("base_pt")).get(1)
                        + (Long) ((List<?>) boosts.get("temporary")).get(1)
                        + (Long) ((List<?>) boosts.get("counters")).get(1);
            }
        }
        throw new AssertionError("no entity " + card + " in " + record);
    }

    /**
     * A change the record's own snapshot already shows is not its event.
     * Daxos's toughness is its controller's devotion to white; a Lightning
     * Bolt kills Savannah Lions, and the state-based loop applies the
     * statics — Daxos shrinks from 3 to 2 — before it puts Daxos's dies
     * trigger on the stack, whose bracket's board shows the 2. Forge
     * publishes that shrink only at the loop's end, inside the trigger's
     * bracket: measured from the differ's last reading, the trigger's half
     * reported a -1 its snapshot had already taken.
     */
    @Test
    void aChangeTheSnapshotAlreadyShowsIsNotTheRecordsEvent() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card lions = TestBoard.onBattlefield(game, opponent, "Savannah Lions");
        Card daxos = TestBoard.onBattlefield(game, opponent, "Daxos, Blessed by the Sun");
        SpellAbility bolt = TestBoard.spellOf(TestBoard.inHand(game, caster, "Lightning Bolt"), caster);
        bolt.resetTargets();
        bolt.getTargets().add(lions);
        // A card moved onto the battlefield by hand is not in the last-state
        // battlefield a dies trigger looks back at.
        game.getTriggerHandler().resetActiveTriggers(false, null);
        game.copyLastState();
        int lifeBefore = opponent.getLife();

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, game, "for the state-based loop's signals", () -> {
                    // Read under the collector's eye, so the shrink has a
                    // reading to differ from.
                    game.getAction().checkStaticAbilities();
                    game.getStack().add(bolt);
                    game.getStack().resolveStack();
                    // The loop kills the Lions and stacks the trigger; once
                    // both players pass, the trigger resolves.
                    for (int step = 0; step < 20 && opponent.getLife() == lifeBefore; step++) {
                        game.getPhaseHandler().mainLoopStep();
                    }
                    assertEquals(lifeBefore + 1, opponent.getLife(), "the dies trigger resolved");
                    TestBoard.nextStep(game);
                });

        String boltHalf = RecordLines.only(records, "resolution", "cardsfolder/l/lightning_bolt.txt");
        String triggerHalf = RecordLines.only(records, "resolution", DAXOS, "trigger");
        // The shape: the toughness changed between the two records, and the
        // second one's snapshot already shows it.
        assertEquals(3L, toughnessIn(boltHalf, daxos), boltHalf);
        assertEquals(2L, toughnessIn(triggerHalf, daxos), triggerHalf);
        assertEquals(List.of("life_change"), RecordLines.eventTypes(triggerHalf), triggerHalf);
    }
}
