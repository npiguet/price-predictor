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
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A card's first change in power and toughness is recorded.
 *
 * <p>The bus says only that a card's stats changed, and Forge says it after the
 * change is applied, so a differ with no earlier reading of the card has
 * nothing to compare against: the first pump on a creature would leave no
 * {@code pt_change}, and its wearing off would read as a lone -3/-3. These
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
}
