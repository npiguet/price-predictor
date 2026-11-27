package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.ai.simulation.GameCopier;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CounterEnumType;
import forge.game.keyword.Keyword;
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
 * carries, on permanents the engine itself made: a merged
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

    private Card forkReplace(Card old, String name, Game[] forkOut) {
        GameCopier copier = new GameCopier(game);
        Game fork = copier.makeCopy();
        forkOut[0] = fork;
        new ReplaceCardOperation((Card) copier.find(old), name).apply(fork);
        return fork.getPlayers().get(0).getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(name)).findFirst().orElseThrow();
    }

    /** 701.60 (suspect): a suspected permanent has menace and cannot block, whatever card it is. */
    @Test
    void aSuspectedPermanentStaysSuspected() {
        Card bears = permanent("Grizzly Bears");
        bears.setSuspected(true);
        game.getAction().checkStaticAbilities();
        Card mystic = forkReplace(bears, "Elvish Mystic", new Game[1]);
        assertTrue(mystic.isSuspected());
        assertTrue(mystic.hasKeyword(Keyword.MENACE));
    }

    /** 701.54 (the Ring tempts you): the player's ring-bearer is the card in the old one's place. */
    @Test
    void aRingBearerStaysTheRingBearer() {
        Card giant = permanent("Hill Giant");
        you.setRingBearer(giant);
        Game[] fork = new Game[1];
        Card courser = forkReplace(giant, "Centaur Courser", fork);
        assertTrue(courser.isRingBearer());
        assertEquals(courser, fork[0].getPlayers().get(0).getRingBearer());
        assertTrue(courser.isInZone(ZoneType.Battlefield));
    }

    /** 702.171 (saddle): a Mount saddled this turn stays saddled. */
    @Test
    void aSaddledPermanentStaysSaddled() {
        Card bears = permanent("Grizzly Bears");
        bears.setSaddled(true);
        bears.setSaddledByThisTurn(new CardCollection(permanent("Hill Giant")));
        Card mystic = forkReplace(bears, "Elvish Mystic", new Game[1]);
        assertTrue(mystic.isSaddled());
        assertEquals(1, mystic.getTimesSaddledThisTurn());
        assertEquals(1, mystic.getSaddledByThisTurn().size());
    }

    /** 701.35 (detain): Forge keeps no way to read who detained it, so the edit refuses. */
    @Test
    void aDetainedPermanentIsRefused() {
        Card wurm = permanent("Craw Wurm");
        wurm.detain(game.getPlayers().get(1));
        assertFalse(ReplaceCardOperation.carries(wurm));
        assertRefused(game, wurm, ReplaceCardOperation.DETAINED);
    }

    /**
     * 702.99 (cipher): the encoded spell is tied to its creature. A hand-built link
     * (no cipher spell is cast in this board): the engine's own link is this one.
     */
    @Test
    void aCreatureHoldingAnEncodedSpellIsRefused() {
        Card bears = permanent("Grizzly Bears");
        Card spell = TestBoard.card(game, you, "Hill Giant");
        bears.addEncodedCard(spell);
        spell.setEncodingCard(bears);
        assertFalse(ReplaceCardOperation.carries(bears));
        assertRefused(game, bears, ReplaceCardOperation.ENCODED);
    }

    /** 702.87 (level up) and 711.2: Caravan Escort at level 5 is a 5/5; Youthful Knight in its place is 2/1. */
    @Test
    void aLevelerPredictsWithoutItsLevelRows() {
        Card escort = permanent("Caravan Escort");
        escort.setCounters(CounterEnumType.LEVEL, 5);
        game.getAction().checkStaticAbilities();
        escort.setDamage(2);
        assertEquals(2, ReplaceCardOperation.replacementPower(escort, "Youthful Knight"));
        assertTrue(ReplaceCardOperation.replacementDies(escort, "Youthful Knight"));
        Game[] fork = new Game[1];
        ForkRefusal refusal = assertThrows(ForkRefusal.class, () -> forkReplace(escort, "Youthful Knight", fork));
        assertEquals(ReplaceCardOperation.DIES, refusal.reason());
    }

    /** 611.3 (a static's boost): Kird Ape is a 2/3 with a Forest; Elvish Mystic in its place is 1/1 and the prediction says so. */
    @Test
    void aSelfPumpPredictsWithoutItsBoostRows() {
        permanent("Forest");
        Card ape = permanent("Kird Ape");
        game.getAction().checkStaticAbilities();
        assertEquals(2, ape.getNetPower());
        assertEquals(1, ReplaceCardOperation.replacementPower(ape, "Elvish Mystic"));
        Game[] fork = new Game[1];
        assertEquals(1, forkReplace(ape, "Elvish Mystic", fork).getNetPower());
    }

    @Test
    void theLifeEditSetsTheLifeTheTurnStartedWith() {
        new LifeOperation(you, 7).apply(game);
        assertEquals(7, you.getLife());
        assertEquals(7, you.getLifeStartedThisTurnWith());
    }
}
