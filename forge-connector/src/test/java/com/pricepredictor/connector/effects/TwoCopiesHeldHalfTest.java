package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static com.pricepredictor.connector.effects.RecordAbilities.linkOf;
import static com.pricepredictor.connector.effects.RecordPicks.records;
import static com.pricepredictor.connector.effects.TestBoard.castAndResolve;
import static com.pricepredictor.connector.effects.TestBoard.nextStep;
import static com.pricepredictor.connector.effects.TestBoard.spellOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A resolution that makes two copies of itself as it finishes keeps its
 * deaths in its half.
 *
 * <p>The first copy pushed while the resolution finishes is not a player's
 * next action, and nor is the second: the state-based check still to come is
 * the resolution's, whose half takes the deaths. Real engine, real cards:
 * Chain of Acid's copy, which Twinning Staff makes two.
 */
@ExtendWith(ForgeExtension.class)
class TwoCopiesHeldHalfTest {

    @TempDir
    Path tempDir;

    private static String deathOf(Card card) {
        return "{\"type\":\"zone_change\",\"subjects\":[\"E" + card.getId()
                + "\"],\"params\":{\"from_zone\":\"battlefield\",\"to_zone\":\"graveyard\"";
    }

    /** Chain of Acid's effect half: the resolution record whose ability is its script. */
    private static List<String> chainOfAcidHalves(List<String> records) {
        return records(records, "resolution").stream()
                .filter(record -> linkOf(record) != null && record.contains("chain_of_acid"))
                .toList();
    }

    /**
     * Chain of Acid destroys the Glorious Anthem keeping a damaged Grizzly
     * Bears alive, and the Anthem's controller, who has Twinning Staff,
     * copies the spell twice. Both copies are pushed as Forge finishes the
     * resolution, before the state-based check that kills the Bears.
     */
    @Test
    void aSecondCopyPushedAsTheResolutionFinishesKeepsTheDeathInTheHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.inPlay(game, caster, "Mountain");
        for (int i = 0; i < 4; i++) {
            TestBoard.inPlay(game, opponent, "Forest");
        }
        TestBoard.inPlay(game, opponent, "Twinning Staff");
        Card anthem = TestBoard.inPlay(game, opponent, "Glorious Anthem");
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        game.getAction().checkStaticAbilities();
        bears.setDamage(2);
        SpellAbility chain = spellOf(TestBoard.inHand(game, caster, "Chain of Acid"), caster);
        chain.resetTargets();
        chain.getTargets().add(anthem);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, game, "to resolve a spell its target's controller copies twice", () -> {
                    castAndResolve(game, chain);
                    assertEquals(2, game.getStack().size(), "the Anthem's controller copied it twice");
                    assertTrue(game.getCardsIn(ZoneType.Graveyard).stream()
                            .anyMatch(c -> c.getId() == bears.getId()), "the Bears died");
                    nextStep(game);
                });

        List<String> halves = chainOfAcidHalves(records);
        assertEquals(1, halves.size(), records.toString());
        assertTrue(halves.get(0).contains(deathOf(bears)),
                "the death joins the record: " + halves.get(0));
    }
}
