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
import java.util.Optional;

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

    /**
     * A fork thrown away at its score check (here for want of a perspective)
     * was already snapshotted by the copy's fidelity audit, which builds its
     * library cards; it never reaches the body, whose end released the rest.
     * At most fewer games are held afterwards (an earlier test's collected),
     * never the discarded copy.
     */
    @Test
    void aForkDiscardedAtItsScoreCheckIsLetGo() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        library(game, game.getPlayers().get(0), "Hill Giant", 8);
        int built = LibraryCapture.gamesHeld();
        int noted = ClauseNotes.gamesHeld();
        boolean[] ran = {false};

        Optional<?> written = new Forks.ForkRun("probe", game, () -> { }, () -> null).take(
                (verified, snapshots) -> {
                    ran[0] = true;
                    return Optional.of(verified);
                },
                (verified, produced) -> produced);

        assertTrue(written.isEmpty(), "a fork with no perspective writes nothing");
        assertFalse(ran[0], "a fork with no perspective never runs its body");
        int liveBuilt = LibraryCapture.holds(game) ? 1 : 0;
        int liveNoted = ClauseNotes.holds(game) ? 1 : 0;
        assertTrue(LibraryCapture.gamesHeld() - liveBuilt <= built,
                "the discarded fork's built cards are held: " + LibraryCapture.gamesHeld() + " > " + built);
        assertTrue(ClauseNotes.gamesHeld() - liveNoted <= noted,
                "the discarded fork's notes are held: " + ClauseNotes.gamesHeld() + " > " + noted);
    }

    /**
     * A recorded game whose clause note was left open (its clause threw
     * before the resolved hook) is let go with its recording: the open note
     * holds the clause and so the game, which the weak key alone never frees.
     */
    @Test
    void aRecordedGameWithANoteLeftOpenIsLetGo() throws Throwable {
        WeakReference<Game> game = recordANoteLeftOpen();

        for (int i = 0; i < 50 && game.get() != null; i++) {
            System.gc();
            Thread.sleep(20);
        }

        assertNull(game.get(), "the recorded game is still reachable through its open note");
    }

    private WeakReference<Game> recordANoteLeftOpen() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility spell = TestBoard.inHand(game, caster, "Divination").getFirstSpellAbility();
        spell.setActivatingPlayer(caster);

        ShardRun.recordedWithClauseHook(tempDir, game, TestCaps.caps().interventions(0).probes(0).build(), WHY,
                () -> ClauseNotes.of(game).clauseResolving(game, spell));

        assertFalse(ClauseNotes.holds(game), "the closed recording's game holds no note");
        return new WeakReference<>(game);
    }
}
