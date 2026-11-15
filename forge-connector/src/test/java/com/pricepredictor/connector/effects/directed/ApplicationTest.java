package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** A variant's {@link Application}: what its fork keeps (FR-010c). */
@ExtendWith(ForgeExtension.class)
class ApplicationTest {

    /** Each kept set adds to those named before it; none replaces another. */
    @Test
    void keptRecordsCombine() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card spell = TestBoard.inHand(game, you, "Lightning Bolt");
        Card watcher = TestBoard.inPlay(game, you, "Grizzly Bears");
        Card other = TestBoard.inPlay(game, you, "Hill Giant");
        Binding binding = Binding.empty().with("spell", spell).with("watcher", watcher).with("other", other);

        KeptRecords kept = new Application(ForkOperation.LIFE, fork -> new LifeOperation(you, 19))
                .keepingNegativesOf("watcher")
                .keepingCandidatesOf("spell")
                .keepingCandidatesOf("other")
                .kept(binding);

        assertEquals(Set.of(spell.getId(), other.getId()), kept.candidateHosts());
        assertEquals(Set.of(watcher.getId()), kept.triggerHosts());
    }
}
