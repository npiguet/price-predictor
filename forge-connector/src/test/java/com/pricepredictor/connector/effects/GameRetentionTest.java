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

import java.lang.ref.WeakReference;
import java.nio.file.Path;
import java.util.List;

import static com.pricepredictor.connector.effects.TestBoard.castAndResolve;
import static com.pricepredictor.connector.effects.TestBoard.spellOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A recorded game, and a fork of it, are let go once their recording or
 * fork is over: nothing the collectors keep across games holds them.
 *
 * <p>The library card a record's event names is rendered from a card built
 * fresh from its printing and cached per game in a weak-keyed map; the built
 * card holds its game, so the entry kept its own key alive and every game
 * and fork a worker recorded stayed in its heap. Some thirty games into a
 * JVM's life its games slowed tenfold under the collector, and two workers
 * of a four-worker run died of {@code OutOfMemoryError}.
 */
@ExtendWith(ForgeExtension.class)
class GameRetentionTest {

    private static final String WHY = "to record a draw as a worker records it";

    @TempDir
    Path tempDir;

    private static void library(Game game, Player owner, String name, int count) {
        for (int i = 0; i < count; i++) {
            game.getAction().moveToLibrary(TestBoard.card(game, owner, name), null, null);
        }
    }

    private static boolean rendersALibraryCard(List<String> records) {
        return records.stream().anyMatch(record -> record.contains("\"zone\":\"library\""));
    }

    /** Divination resolved and recorded on a fresh board; the game held only weakly. */
    private WeakReference<Game> recordADraw() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        library(game, caster, "Hill Giant", 8);
        Card divination = TestBoard.inHand(game, caster, "Divination");
        int held = LibraryCapture.gamesHeld();

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            Card onStack = game.getAction().moveToStack(divination, null);
            castAndResolve(game, spellOf(onStack, caster));
        });

        assertTrue(rendersALibraryCard(records), "the draws render the cards as they were in the library");
        assertEquals(held, LibraryCapture.gamesHeld(), "the closed recording's game holds no built card");
        return new WeakReference<>(game);
    }

    @Test
    void aRecordedGameIsLetGoOnceItsRecordingCloses() throws Throwable {
        WeakReference<Game> game = recordADraw();

        for (int i = 0; i < 50 && game.get() != null; i++) {
            System.gc();
            Thread.sleep(20);
        }

        assertNull(game.get(), "the recorded game is still reachable after its recording closed");
    }

    @Test
    void aForkIsLetGoOnceItEnds() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        library(game, caster, "Hill Giant", 8);
        Card divination = TestBoard.inHand(game, caster, "Divination");
        SpellAbility spell = divination.getFirstSpellAbility();
        spell.setActivatingPlayer(caster);
        assertFalse(LibraryCapture.holds(game), "a fresh game holds no built card");
        int held = LibraryCapture.gamesHeld();

        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                tempDir, game, TestCaps.caps().interventions(1).probes(0).build(), 1L, spell, WHY);

        assertTrue(fork.wrote(), "the fork forced Divination");
        assertTrue(rendersALibraryCard(fork.records()), "the fork's draws render library cards");
        int live = LibraryCapture.holds(game) ? 1 : 0;
        assertEquals(held, LibraryCapture.gamesHeld() - live, "the ended fork holds no built card");
    }
}
