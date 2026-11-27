package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.Binding;
import forge.game.card.Card;
import forge.game.card.CardCopyService;
import forge.game.card.token.TokenInfo;
import forge.game.phase.PhaseType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.Set;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static com.pricepredictor.connector.effects.directed.rules.RuleTwinChecks.assertOnlyRemoved;
import static com.pricepredictor.connector.effects.directed.rules.RuleTwinChecks.assertOnlyReplaced;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The registered directed rules of a creature changed in place: toughness
 * 0 (704.5f), champion (607.2k), meld (701.42a, 712.4a, 712.14c),
 * several permanents exploring (701.44d) and keywords shared by "the
 * same is true for" (702.1c). Each rule's scenario base binds, a board
 * missing it binds nothing, and the fork differs from its twin by the
 * variant's one fact, as {@link RegisteredRulesTest} sets out.
 */
@ExtendWith(ForgeExtension.class)
class RegisteredRulesCreaturesTest {

    // ── 704.5f, toughness 0 ────────────────────────

    /** 704.5f-1's base: A holds Disfigure and an untapped Swamp; B's Drudge Skeletons is a 1/1. */
    private static RuleBoard disfigureBoard(String creature) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.card(board.you, "Swamp");
        board.subject = board.card(board.opponent, creature);
        board.other = TestBoard.inHand(board.game, board.you, "Disfigure");
        return board;
    }

    @Test
    void toughnessZeroBindsTheCreatureTheSpellWouldKill() {
        RuleBoard board = disfigureBoard("Drudge Skeletons");
        Binding binding = ToughnessZero.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("creature"));
        assertEquals(board.other, binding.card("spell"));
    }

    @Test
    void toughnessZeroBindsNothingWhenTheSpellLeavesTheCreatureAlive() {
        assertTrue(ToughnessZero.bind(disfigureBoard("Hill Giant").game).isEmpty());
    }

    @Test
    void toughnessZeroBindsNothingWithoutTheManaToCastTheSpell() {
        RuleBoard board = disfigureBoard("Drudge Skeletons");
        board.you.getLandsInPlay().forEach(land -> land.setTapped(true));
        assertTrue(ToughnessZero.bind(board.game).isEmpty());
    }

    @Test
    void toughnessZeroForksATougherCreature() {
        RuleBoard board = disfigureBoard("Drudge Skeletons");
        RuleTwin twin = board.twin(ToughnessZero.rule());
        assertEquals(Set.of(id(board.subject) + ".base_pt"), twin.changes());
        Card skeletons = twin.fork().card("creature");
        assertEquals(3, skeletons.getNetToughness(), "Disfigure now leaves it at toughness 1");
        assertEquals(1, skeletons.getNetPower());
    }

    // ── 607.2k, champion: remove_permanent ──────────────────────────────

    private static RuleBoard championBoard() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        for (int i = 0; i < 5; i++) {
            board.card(board.you, "Plains");
        }
        board.subject = board.card(board.you, "Grizzly Bears");
        board.other = TestBoard.inHand(board.game, board.you, "Changeling Hero");
        return board;
    }

    @Test
    void championBindsTheOnlyCreatureToChampionAndTheCard() {
        RuleBoard board = championBoard();
        Binding binding = Champion.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("creature"));
        assertEquals(board.other, binding.card("card"));
    }

    @Test
    void championBindsNothingWithTwoCreaturesToChampion() {
        RuleBoard board = championBoard();
        board.card(board.you, "Hill Giant");
        assertTrue(Champion.bind(board.game).isEmpty());
    }

    @Test
    void championForksTheBoardWithoutTheCreature() {
        RuleBoard board = championBoard();
        assertOnlyRemoved(board.twin(Champion.rule()), board.subject, "creature");
    }

    // ── 701.42a, 712.4a and 712.14c, meld: replace_card ─────────────────

    /**
     * A's {@code host} beside A's {@code partner}, a card or a token copy of
     * one (built as a game copy builds a token: the card's copiable values,
     * its front face alone); the partner is the subject. The copy calls the
     * deprecated {@code copyCopiableCharacteristics} because the game copy does,
     * and Forge offers nothing in its place.
     */
    @SuppressWarnings("deprecation")
    private static RuleBoard meldBoard(String host, String partner, boolean token, int mountains) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        for (int i = 0; i < mountains; i++) {
            board.card(board.you, "Mountain");
        }
        board.other = board.card(board.you, host);
        if (token) {
            Card card = TestBoard.card(board.game, board.you, partner);
            Card copy = new TokenInfo(card).makeOneToken(board.you, board.game.nextCardId());
            new CardCopyService(card).copyCopiableCharacteristics(copy, null, null);
            board.subject = board.game.getAction().moveToPlay(copy, board.you, null, null);
            board.subject.setSickness(false);
            board.subject.setGameTimestamp(board.game.getNextTimestamp());
        } else {
            board.subject = board.card(board.you, partner);
        }
        return board;
    }

    @Test
    void meldBindsTheHostAndACreatureThePartnerCanReplace() {
        RuleBoard board = meldBoard("Graf Rats", "Hill Giant", false, 0);
        Binding binding = Meld.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("host"));
        assertEquals(board.subject, binding.card("stand-in"));
    }

    @Test
    void meldBindsNothingBesideThePartnerOrATokenAlone() {
        assertTrue(Meld.bind(meldBoard("Graf Rats", "Midnight Scavengers", false, 0).game).isEmpty());
        assertTrue(Meld.bind(meldBoard("Graf Rats", "Hill Giant", true, 0).game).isEmpty());
    }

    @Test
    void meldBindsNothingInTheSecondMainPhase() {
        RuleBoard board = meldBoard("Graf Rats", "Hill Giant", false, 0);
        board.game.getPhaseHandler().devModeSet(PhaseType.MAIN2, board.you);
        assertTrue(Meld.bind(board.game).isEmpty());
    }

    /** An activated meld is a seat's choice: only a trigger melds with no choice made. */
    @Test
    void meldBindsNothingForAnActivatedMeld() {
        assertTrue(Meld.bind(meldBoard("Hanweir Battlements", "Grizzly Bears", false, 5).game).isEmpty());
    }

    @Test
    void meldForksThePartnerInTheStandInsPlace() {
        RuleBoard board = meldBoard("Graf Rats", "Hill Giant", false, 0);
        assertOnlyReplaced(board.twin(Meld.rule()), board.subject, "Midnight Scavengers");
    }

    // ── 701.44d, several permanents exploring: replace_card ─────────────

    /** Hakbal of the Surging Soul beside A's {@code other}, over a library of Forest then Shock. */
    private static RuleBoard hakbalBoard(String other) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.other = board.card(board.you, "Hakbal of the Surging Soul");
        board.subject = board.card(board.you, other);
        board.library(board.you, "Forest", "Shock");
        return board;
    }

    @Test
    void severalExploreBindsTheCreatureBesideTheOnlyMerfolk() {
        RuleBoard board = hakbalBoard("Grizzly Bears");
        assertEquals(board.subject, SeveralExplore.bind(board.game).orElseThrow().card("creature"));
    }

    /** Two Merfolk already explore two at a time, the side the counts hold, whatever else is there. */
    @Test
    void severalExploreBindsNothingWithTwoMerfolk() {
        RuleBoard board = hakbalBoard(SeveralExplore.BRANCHWALKER);
        assertTrue(SeveralExplore.bind(board.game).isEmpty());
        board.card(board.you, "Grizzly Bears");
        assertTrue(SeveralExplore.bind(board.game).isEmpty(), "beside a creature the Branchwalker could replace");
    }

    @Test
    void severalExploreBindsNothingWithoutACreatureToReplace() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.card(board.you, "Hakbal of the Surging Soul");
        board.library(board.you, "Forest", "Shock");
        assertTrue(SeveralExplore.bind(board.game).isEmpty());
    }

    @Test
    void severalExploreBindsNothingWithOneCardInTheLibrary() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.card(board.you, "Hakbal of the Surging Soul");
        board.card(board.you, "Grizzly Bears");
        board.library(board.you, "Forest");
        assertTrue(SeveralExplore.bind(board.game).isEmpty());
    }

    /** Two damage marked on the Bears would kill a 2/1 Branchwalker in its place. */
    @Test
    void severalExploreBindsNothingWhenTheBranchwalkerWouldDie() {
        RuleBoard board = hakbalBoard("Grizzly Bears");
        board.subject.setDamage(1);
        assertTrue(SeveralExplore.bind(board.game).isEmpty());
    }

    @Test
    void severalExploreBindsNothingAtAnotherBoundary() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN2);
        board.card(board.you, "Hakbal of the Surging Soul");
        board.card(board.you, "Grizzly Bears");
        board.library(board.you, "Forest", "Shock");
        assertTrue(SeveralExplore.bind(board.game).isEmpty());
    }

    @Test
    void severalExploreForksAMerfolkBranchwalkerInTheBearsPlace() {
        RuleBoard board = hakbalBoard("Grizzly Bears");
        assertOnlyReplaced(board.twin(SeveralExplore.rule()), board.subject, SeveralExplore.BRANCHWALKER);
    }

    // ── 702.1c, keywords shared by "the same is true for": replace_card ─

    /** A's Concerted Effort beside A's {@code creatures} as the upkeep begins; the first is the subject. */
    static RuleBoard concertedBoard(String... creatures) {
        RuleBoard board = new RuleBoard(PhaseType.UPKEEP);
        board.card(board.you, "Concerted Effort");
        for (String name : creatures) {
            Card creature = board.card(board.you, name);
            if (board.subject == null) {
                board.subject = creature;
            }
        }
        return board;
    }

    @Test
    void sharedKeywordsBindsTheOneCreatureWithAListedKeyword() {
        RuleBoard board = concertedBoard("White Knight", "Grizzly Bears");
        assertEquals(board.subject, SharedKeywords.bind(board.game).orElseThrow().card("creature"));
    }

    @Test
    void sharedKeywordsBindsNothingWithTwoHoldersOrNoOtherCreature() {
        assertTrue(SharedKeywords.bind(concertedBoard("White Knight", SimultaneousDamage.KNIGHT).game).isEmpty());
        assertTrue(SharedKeywords.bind(concertedBoard("White Knight").game).isEmpty());
    }

    @Test
    void sharedKeywordsForksHillGiantInTheKnightsPlace() {
        RuleBoard board = concertedBoard("White Knight", "Grizzly Bears");
        assertOnlyReplaced(board.twin(SharedKeywords.rule()), board.subject, Cards.GIANT);
    }
}
