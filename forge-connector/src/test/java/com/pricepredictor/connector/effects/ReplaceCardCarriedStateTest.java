package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.ai.simulation.GameCopier;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What {@code replace_card} refuses because it cannot carry it, and what it
 * carries, on permanents the engine itself made (finding 6.7): a merged
 * (mutated) permanent, a melded one and a token are refused, since their
 * other cards would be left orphaned or the token would become a real card;
 * a hasty creature that entered this turn stays summoning sick as the card
 * in its place; the prediction the preconditions read leaves a
 * characteristic-defining P/T out, as the edit does. Beside them, the
 * {@code life} edit sets the life the turn started with too.
 */
@ExtendWith(ForgeExtension.class)
class ReplaceCardCarriedStateTest {

    private final Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
    private final Player you = game.getPlayers().get(0);

    private Card permanent(String name) {
        Card card = TestBoard.inPlay(game, you, name);
        card.setGameTimestamp(game.getNextTimestamp());
        return card;
    }

    private static void assertRefused(Game game, Card old, String reason) {
        ForkRefusal refusal = assertThrows(ForkRefusal.class,
                () -> new ReplaceCardOperation(old, "Hill Giant").apply(game));
        assertEquals(reason, refusal.reason());
        assertTrue(old.isInZone(ZoneType.Battlefield), "the refusal changed the board");
    }

    /** Pouncing Shoreshark cast for its mutate cost onto Grizzly Bears: one permanent of two cards. */
    @Test
    void aMergedPermanentIsRefused() {
        Card bears = permanent("Grizzly Bears");
        Card shark = game.getAction().moveToStack(TestBoard.card(game, you, "Pouncing Shoreshark"), null);
        SpellAbility mutate = shark.getSpells().stream().filter(SpellAbility::isMutate).findFirst().orElseThrow();
        mutate.setActivatingPlayer(you);
        mutate.getTargets().add(bears);
        TestBoard.castAndResolve(game, mutate);
        Card merged = you.getCardsIn(ZoneType.Battlefield).stream()
                .filter(Card::hasMergedCard).findFirst().orElseThrow();
        assertRefused(game, merged, ReplaceCardOperation.MERGED);
    }

    /** Hanweir Battlements' meld with Hanweir Garrison: Hanweir, the Writhing Township is two cards. */
    @Test
    void aMeldedPermanentIsRefused() {
        Card battlements = permanent("Hanweir Battlements");
        permanent("Hanweir Garrison");
        SpellAbility meld = TestBoard.ability(battlements, ApiType.Meld);
        meld.setActivatingPlayer(you);
        TestBoard.castAndResolve(game, meld);
        Card township = you.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Hanweir, the Writhing Township")).findFirst().orElseThrow();
        assertNotNull(township.getMeldedWith());
        assertRefused(game, township, ReplaceCardOperation.MELDED);
    }

    /** Raise the Alarm's Soldier: a real card in its place would go to a graveyard where the token ceases to exist. */
    @Test
    void aTokenIsRefused() {
        TestBoard.castUntargeted(game, you, "Raise the Alarm");
        game.getStack().resolveStack();
        Card soldier = you.getCardsIn(ZoneType.Battlefield).stream().filter(Card::isToken).findFirst().orElseThrow();
        assertRefused(game, soldier, ReplaceCardOperation.TOKEN);
    }

    /**
     * Raging Goblin (haste) entered this turn: in a fork's copy, the card put
     * in its place without haste is summoning sick, as the Goblin would be
     * without its haste.
     */
    @Test
    void aHastyCreatureThatEnteredThisTurnIsReplacedBySickness() {
        Card goblin = TestBoard.onBattlefield(game, you, "Raging Goblin");
        goblin.setGameTimestamp(game.getNextTimestamp());
        assertFalse(goblin.hasSickness(), "haste hides the sickness");
        GameCopier copier = new GameCopier(game);
        Game fork = copier.makeCopy();
        Card forked = (Card) copier.find(goblin);

        new ReplaceCardOperation(forked, "Grizzly Bears").apply(fork);

        Card bears = fork.getPlayers().get(0).getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        assertTrue(bears.hasSickness(), "the Bears entered this turn and has no haste");
    }

    /**
     * Tarmogoyf over empty graveyards is a 0/1 by its characteristic-defining
     * ability, 1 damage marked: Grizzly Bears in its place is a 2/2 with that
     * damage and lives, since the edit carries no characteristic-defining row.
     */
    @Test
    void theReplacementPredictionLeavesACharacteristicDefiningPtOut() {
        Card goyf = permanent("Tarmogoyf");
        game.getAction().checkStaticAbilities();
        assertEquals(1, goyf.getNetToughness(), "Tarmogoyf's own toughness here");
        goyf.setDamage(1);
        assertFalse(ReplaceCardOperation.replacementDies(goyf, "Grizzly Bears"));
        assertEquals(2, ReplaceCardOperation.replacementPower(goyf, "Grizzly Bears"));
    }

    @Test
    void theLifeEditSetsTheLifeTheTurnStartedWith() {
        new LifeOperation(you, 7).apply(game);
        assertEquals(7, you.getLife());
        assertEquals(7, you.getLifeStartedThisTurnWith());
    }
}
