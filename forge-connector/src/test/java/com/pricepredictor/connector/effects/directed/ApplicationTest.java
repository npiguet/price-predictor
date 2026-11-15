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
import static org.junit.jupiter.api.Assertions.assertSame;

/** A variant's {@link Application}: what its fork keeps (FR-010c) and the roles its edit makes. */
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

    /**
     * A card the edit makes is bound once the edit is applied, for a forced
     * play to name; the roles bound before are kept, and every later
     * {@code forcing} or {@code keeping} keeps the new role too.
     */
    @Test
    void aRoleTheEditMakesIsBoundAfterTheEdit() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card watcher = TestBoard.inPlay(game, you, "Grizzly Bears");
        Card made = TestBoard.inPlay(game, you, "Hill Giant");
        Binding binding = Binding.empty().with("watcher", watcher).with("player", you);

        Application application = new Application(ForkOperation.LIFE, fork -> new LifeOperation(you, 19))
                .making("made", (fork, copy) -> made)
                .keepingNegativesOf("watcher");

        Binding after = application.afterEdit(binding, game);
        assertSame(made, after.card("made"));
        assertSame(watcher, after.card("watcher"));
        assertSame(you, after.player("player"));
        Application plain = new Application(ForkOperation.LIFE, fork -> new LifeOperation(you, 19));
        assertEquals(binding, plain.afterEdit(binding, game));
    }
}
