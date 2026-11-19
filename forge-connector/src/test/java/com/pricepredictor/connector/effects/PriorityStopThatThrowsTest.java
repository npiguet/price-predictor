package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.AbilityStatic;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static com.pricepredictor.connector.effects.TestCaps.caps;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** A priority stop whose enumeration throws leaves no stop for the seat to take (finding 8.6). */
@ExtendWith(ForgeExtension.class)
class PriorityStopThatThrowsTest {

    @TempDir
    Path tempDir;

    /** An ability whose enumeration throws, as a script the engine cannot evaluate does. */
    private static final class Broken extends AbilityStatic {
        private boolean throwing;

        Broken(Card card) {
            super(card, new Cost("0", true), null);
        }

        @Override
        public void resolve() {
        }

        @Override
        public boolean isManaAbility() {
            if (throwing) {
                throw new IllegalStateException("cannot be evaluated");
            }
            return super.isManaAbility();
        }
    }

    @Test
    void aStopThatThrowsIsNotServedFromTheStopBefore() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        TestBoard.inPlay(game, a, "Forest");
        TestBoard.inPlay(game, a, "Forest");
        TestBoard.inHand(game, a, "Grizzly Bears");
        Card forest = TestBoard.inPlay(game, a, "Forest");
        Broken broken = new Broken(forest);

        ShardRun.shardOf(tempDir, game, caps().manaCap(0).build(), (bracket, collectors) -> {
            PriorityCandidates candidates = collectors.candidates();
            candidates.settle(a);
            forest.addSpellAbility(broken);
            broken.throwing = true;
            assertThrows(IllegalStateException.class, () -> candidates.settle(a));
            broken.throwing = false;

            List<PriorityCandidates.Candidate> taken = candidates.takeFor(a);

            assertTrue(taken.stream().anyMatch(c -> c.ability() == broken),
                    "the stop before was served again, not a fresh enumeration");
        });
    }
}
