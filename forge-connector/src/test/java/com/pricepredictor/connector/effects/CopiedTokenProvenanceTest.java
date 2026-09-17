package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.StaticData;
import forge.ai.LobbyPlayerAi;
import forge.ai.simulation.GameCopier;
import forge.deck.Deck;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The whole path a forked combat record takes, run once against the real
 * copier.
 *
 * <p>{@code ProvenanceKeyTest} reproduces {@code GameCopier}'s shape by hand --
 * a bare {@code Card} carrying the original's name and image key and no paper
 * card -- which is the right unit to pin, and the wrong one to trust alone: it
 * asserts what this project believes the copier does. This test asks the copier
 * instead. A Food token is put on a real two-player board, the board is forked
 * with {@link GameCopier}, and the fork's copy is rendered by the collectors'
 * own {@link SnapshotBuilder}.
 *
 * <p>The assertion is on the copy's {@code printed} keys (FR-150): every
 * ability-bearing token in a forked record arrives without a paper card, and
 * before the fix its lines keyed into {@code cardsfolder/f/food_token.txt} --
 * a file that does not exist, so the sidecar join failed loudly and the record
 * carried no ability text at all.
 *
 * <p>Its own class rather than {@code ForkCollectorTest}, whose class doc says
 * outright that forking a live game is not its job and whose fixture is a
 * player-less {@code TestCards.game()} the copier cannot clone; and not
 * {@code SnapshotBuilderTest}, which is deliberately game-free and does not
 * even initialise the card database.
 */
@ExtendWith(ForgeExtension.class)
class CopiedTokenProvenanceTest {

    private static final int[] STAGE_TWO = {1, 2, 3};

    /** The script the Food token is filed under, and the file the bug named. */
    private static final String TOKEN_SCRIPT =
            "\"script_file\":\"tokenscripts/c_a_food_sac.txt\"";
    private static final String CARD_SCRIPT = "cardsfolder/f/food_token.txt";

    private static final String PRINTED_KEY = "\"printed\":[";

    /**
     * A two-player game in a main phase, the shape {@code GameCopier} expects:
     * it clones the match's registered players and replays the phase handler's
     * turn onto the fork, so a game that has taken no turn at all forks with a
     * null active player and the copy's own event bus logs an NPE.
     */
    private static Game twoPlayerGameInAMainPhase() {
        Deck deck = new Deck();
        List<RegisteredPlayer> players = List.of(
                new RegisteredPlayer(deck).setPlayer(new LobbyPlayerAi("a", null)),
                new RegisteredPlayer(deck).setPlayer(new LobbyPlayerAi("b", null)));
        GameRules rules = new GameRules(GameType.Constructed);
        Game game = new Game(
                players, rules, new Match(rules, players, "CopiedTokenProvenanceTest"));
        game.getPhaseHandler().devModeSet(PhaseType.MAIN1, game.getPlayers().get(0));
        return game;
    }

    /**
     * A Food token on a forked board keys its printed line into the token tree.
     *
     * <p>The end-to-end reading of the unit test: the fork's token really does
     * arrive with no paper card, and its image key really is what names the
     * script it came from.
     */
    @Test
    void aTokenOnAForkedBoardKeysIntoTheTokenTree() {
        Game game = twoPlayerGameInAMainPhase();
        Player owner = game.getPlayers().get(0);
        Card food = CardFactory.getCard(
                StaticData.instance().getAllTokens().getToken("c_a_food_sac"),
                owner, game);
        game.getAction().moveToPlay(food, owner, null, null);

        Game fork = new GameCopier(game).makeCopy();
        Card copied = fork.getCardsIn(ZoneType.Battlefield).stream()
                .filter(c -> food.getName().equals(c.getName()))
                .findFirst().orElseThrow();

        assertNotNull(food.getPaperCard(), "the mainline token has its PaperToken");
        assertNull(copied.getPaperCard(),
                "GameCopier rebuilds tokens through TokenInfo, without a paper card");

        String forked = new SnapshotBuilder(fork, STAGE_TWO).toJson(null, List.of());

        assertTrue(forked.contains(TOKEN_SCRIPT), forked);
        assertFalse(forked.contains(CARD_SCRIPT), forked);

        // And the whole block, not just the file: a forked record is meant to
        // read exactly like the record the same board would have written
        // un-forked, which is the property the join actually needs.
        String mainline = new SnapshotBuilder(game, STAGE_TWO).toJson(null, List.of());
        assertEquals(printedBlock(mainline), printedBlock(forked),
                "the fork's printed keys differ from the board it was forked from");
    }

    /** The one entity's {@code printed} array, as rendered. */
    private static String printedBlock(String snapshot) {
        int open = snapshot.indexOf(PRINTED_KEY);
        assertTrue(open >= 0, snapshot);
        int close = snapshot.indexOf(']', open);
        return snapshot.substring(open, close + 1);
    }
}
