package com.pricepredictor.connector;

import forge.LobbyPlayer;
import forge.ai.LobbyPlayerAi;
import forge.deck.Deck;
import forge.game.player.RegisteredPlayer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The one thing that keeps effect-record collection honest against Forge's
 * full-simulation AI mode.
 *
 * <p>{@code GameSimulator} runs the real resolution pipeline against a
 * {@code GameCopier} clone of the live game purely to evaluate a candidate
 * move — and every effect-record listener in this branch (the clause hook,
 * the outcome hook, the rewrite listener) is a plain JVM-wide static with no
 * {@code Game} reference of its own. If that mode were ever entered during a
 * collected game, a hypothetical resolution would fire the same static
 * listeners as a real one and enter the corpus indistinguishable from it.
 *
 * <p>It is off today because {@link GamePlayer#registeredPlayers} builds both
 * seats with a null {@code Set<AIOption>}: {@code LobbyPlayerAi} only sets an
 * option when it is given a non-empty set, and {@code AiController} only
 * enters hybrid or full simulation when an option is set. This test reads
 * that off the actual {@link LobbyPlayerAi} instances {@link GamePlayer}
 * builds, not off a re-declaration of the constant, so a future edit that
 * starts passing an {@code AIOption} fails here rather than silently
 * poisoning every collection channel at once.
 */
@ExtendWith(ForgeExtension.class)
class GamePlayerTest {

    @Test
    void theRegisteredPlayersCarryNoAiOption() throws ReflectiveOperationException {
        List<RegisteredPlayer> players =
                GamePlayer.registeredPlayers(
                        new Deck(), new Deck(), new GamePlayer.SeatPlan(Optional.empty(), 0.0, null));

        assertTrue(players.size() >= 2, "expected both seats: " + players);
        for (RegisteredPlayer registered : players) {
            LobbyPlayer lobbyPlayer = registered.getPlayer();
            assertTrue(lobbyPlayer instanceof LobbyPlayerAi,
                    "expected an AI-controlled seat, got " + lobbyPlayer);
            assertNull(aiOptionOf((LobbyPlayerAi) lobbyPlayer),
                    lobbyPlayer.getName() + " carries an AIOption, which would let "
                            + "AiController enter full/hybrid simulation");
        }
    }

    /**
     * {@code LobbyPlayerAi} exposes no getter for the option it was built
     * with -- reasonably, since nothing but {@code AiController} is meant to
     * read it -- so this reads the private field directly rather than
     * re-declaring what "no option" means.
     */
    private static Object aiOptionOf(LobbyPlayerAi player) throws ReflectiveOperationException {
        Field option = LobbyPlayerAi.class.getDeclaredField("option");
        option.setAccessible(true);
        return option.get(player);
    }

    /**
     * A recording whose close throws an {@code Error} after its game threw
     * (an {@code OutOfMemoryError} in the shard writer) leaves the game's
     * failure the one reported, the close's attached to it as suppressed.
     * A plain {@code Error} stands in for it: JUnit rethrows an
     * {@code OutOfMemoryError} as unrecoverable, ending the test JVM.
     */
    @Test
    void aCloseThatThrowsAnErrorAfterTheGameThrewIsKeptWithTheGamesFailure() {
        IllegalStateException game = new IllegalStateException("the game threw");
        Error close = new Error("the shard writer ran out of memory");

        GamePlayer.closeRecording(() -> {
            throw close;
        }, game);

        assertArrayEquals(new Throwable[] {close}, game.getSuppressed());
    }

    /** With no game failure to keep, the close's own failure is the one thrown. */
    @Test
    void aCloseThatThrowsAfterTheGameEndedThrows() {
        Error close = new Error("the shard writer ran out of memory");

        assertSame(close, assertThrows(Error.class, () -> GamePlayer.closeRecording(() -> {
            throw close;
        }, null)));
    }
}
