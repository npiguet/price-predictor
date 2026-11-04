package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.BoardStates;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.DirectedRules;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.KeywordEdit;
import forge.ai.simulation.GameCopier;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.combat.Combat;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static com.pricepredictor.connector.effects.directed.BoardChanges.renamed;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The registered directed rules (T094, T095): on a {@code TestBoard} set up
 * as each rule's scenario sets up its base variant, the precondition binds the
 * scenario's objects, and on a board missing the base it binds nothing; and
 * the fork, copied as a directed fork copies the live game and edited by the
 * rule's variant, differs from its real twin, the live board, by that
 * variant's one fact (story 3, acceptance 3). One rule per operation, so the
 * ten together cover every operation.
 *
 * <p>The twin comparison reads the board-only snapshot a record would carry,
 * member by member ({@code BoardChanges}), and checks first that the unedited
 * copy reads as the live board does, so what differs afterwards is the edit's.
 */
@ExtendWith(ForgeExtension.class)
class RegisteredRulesTest {

    // ── the registry ────────────────────────────────────────────────────

    @Test
    void theRegisteredRulesCoverEveryOperation() {
        Set<ForkOperation> covered = EnumSet.noneOf(ForkOperation.class);
        for (DirectedRule rule : DirectedRules.registered().rules()) {
            rule.variants().values().forEach(application -> covered.add(application.operation()));
        }
        assertEquals(EnumSet.allOf(ForkOperation.class), covered);
    }

    // ── 704.5f, toughness 0 (T094, acceptance 1) ────────────────────────

    /** 704.5f-1's base: A holds Disfigure and an untapped Swamp; B's Drudge Skeletons is a 1/1. */
    private static Board disfigureBoard(String creature) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Swamp");
        board.subject = board.card(board.opponent, creature);
        board.other = TestBoard.inHand(board.game, board.you, "Disfigure");
        return board;
    }

    @Test
    void toughnessZeroBindsTheCreatureTheSpellWouldKill() {
        Board board = disfigureBoard("Drudge Skeletons");
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
        Board board = disfigureBoard("Drudge Skeletons");
        board.you.getLandsInPlay().forEach(land -> land.setTapped(true));
        assertTrue(ToughnessZero.bind(board.game).isEmpty());
    }

    @Test
    void toughnessZeroForksATougherCreature() {
        Board board = disfigureBoard("Drudge Skeletons");
        Twin twin = board.twin(ToughnessZero.rule());
        assertEquals(Set.of(id(board.subject) + ".base_pt"), twin.changes());
        Card skeletons = twin.fork().card("creature");
        assertEquals(3, skeletons.getNetToughness(), "Disfigure now leaves it at toughness 1");
        assertEquals(1, skeletons.getNetPower());
    }

    // ── 122.1d, stun counter: counters ──────────────────────────────────

    private static Board stunBoard(boolean tapped) {
        Board board = new Board(PhaseType.UNTAP);
        board.subject = board.card(board.you, "Grizzly Bears");
        board.subject.setTapped(tapped);
        board.subject.setCounters(CounterEnumType.STUN, 1);
        return board;
    }

    @Test
    void stunCounterBindsATappedPermanentWithAStunCounterAsTheUntapStepBegins() {
        Board board = stunBoard(true);
        assertEquals(board.subject, StunCounter.bind(board.game).orElseThrow().card("permanent"));
    }

    @Test
    void stunCounterBindsNothingOnAnUntappedPermanent() {
        assertTrue(StunCounter.bind(stunBoard(false).game).isEmpty());
    }

    @Test
    void stunCounterForksThePermanentWithoutItsStunCounter() {
        Board board = stunBoard(true);
        Twin twin = board.twin(StunCounter.rule());
        // The stun counter carries its untap replacement as a line of its own.
        assertEquals(Set.of(id(board.subject) + ".counters", id(board.subject) + ".lines"), twin.changes());
        assertEquals(0, twin.fork().card("permanent").getCounters(CounterEnumType.STUN));
    }

    // ── 400.6, a replaced zone change: remove_permanent ─────────────────

    private static Board restInPeaceBoard(String permanent) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Mountain");
        board.subject = board.card(board.you, permanent);
        board.card(board.opponent, "Grizzly Bears");
        return board;
    }

    @Test
    void graveyardReplacementBindsRestInPeace() {
        Board board = restInPeaceBoard("Rest in Peace");
        assertEquals(board.subject, GraveyardReplacement.bind(board.game).orElseThrow().card("permanent"));
    }

    /** Darksteel Colossus replaces only its own move to a graveyard. */
    @Test
    void graveyardReplacementBindsNothingForAReplacementOfItsOwnCard() {
        assertTrue(GraveyardReplacement.bind(restInPeaceBoard("Darksteel Colossus").game).isEmpty());
    }

    @Test
    void graveyardReplacementForksTheBoardWithoutRestInPeace() {
        Board board = restInPeaceBoard("Rest in Peace");
        Twin twin = board.twin(GraveyardReplacement.rule());
        String rip = id(board.subject);
        assertTrue(twin.changes().contains("-" + rip), twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(change.equals("-" + rip) || change.endsWith(".order_rank")
                            || change.startsWith("-link ") && change.contains("=" + rip),
                    "removing changed " + change + "; all: " + twin.changes());
        }
        assertNull(twin.fork().card("permanent").getZone());
    }

    // ── 701.12g, a life exchange: add_permanent ─────────────────────────

    private static Board treeBoard(int opponentLife) {
        Board board = new Board(PhaseType.MAIN1);
        board.subject = board.card(board.you, "Tree of Perdition");
        board.opponent.setLife(opponentLife, null);
        return board;
    }

    @Test
    void lifeExchangeBindsTheTreeAndAnOpponentItWouldRaise() {
        Board board = treeBoard(7);
        Binding binding = LifeExchange.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("permanent"));
        assertEquals(board.opponent, binding.player("opponent"));
    }

    @Test
    void lifeExchangeBindsNothingWhenTheExchangeWouldLowerTheLife() {
        assertTrue(LifeExchange.bind(treeBoard(20).game).isEmpty());
    }

    @Test
    void lifeExchangeForksEverlastingTormentUnderTheOpponent() {
        Board board = treeBoard(7);
        Twin twin = board.twin(LifeExchange.rule());
        Player opponent = twin.fork().player("opponent");
        Card torment = opponent.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(LifeExchange.TORMENT)).findFirst().orElseThrow();
        String added = id(torment);
        assertTrue(twin.changes().contains("+" + added), twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(change.equals("+" + added) || change.startsWith("+link ") && change.contains("=" + added),
                    "adding changed " + change + "; all: " + twin.changes());
        }
        assertFalse(opponent.canGainLife());
    }

    // ── 702.13b, intimidate: colour_type ────────────────────────────────

    /** Bladetusk Boar (red, intimidate) attacks B, who controls {@code blocker}; no blocker declared yet. */
    private static Board intimidateBoard(String blocker) {
        Board board = new Board(PhaseType.COMBAT_DECLARE_BLOCKERS);
        board.other = board.card(board.you, "Bladetusk Boar");
        board.subject = board.card(board.opponent, blocker);
        Combat combat = new Combat(board.you);
        combat.addAttacker(board.other, board.opponent);
        board.game.getPhaseHandler().setCombat(combat);
        return board;
    }

    @Test
    void intimidateBindsTheAttackerAndAGreenCreature() {
        Board board = intimidateBoard("Grizzly Bears");
        Binding binding = Intimidate.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("attacker"));
        assertEquals(board.subject, binding.card("blocker"));
    }

    @Test
    void intimidateBindsNothingWhenEveryCreatureMayBlock() {
        assertTrue(Intimidate.bind(intimidateBoard("Hill Giant").game).isEmpty());
        assertTrue(Intimidate.bind(intimidateBoard("Phyrexian Walker").game).isEmpty());
    }

    @Test
    void intimidateForksARedBlocker() {
        Board board = intimidateBoard("Grizzly Bears");
        Twin twin = board.twin(Intimidate.rule());
        assertEquals(Set.of(id(board.subject) + ".colors"), twin.changes());
        Card bears = twin.fork().card("blocker");
        assertTrue(bears.getColor().hasRed());
        assertFalse(bears.getColor().hasGreen());
    }

    // ── 702.63c, vanishing instances: keyword ───────────────────────────

    private static Board djinnBoard(int instances) {
        Board board = new Board(PhaseType.UPKEEP);
        board.subject = board.card(board.you, "Maelstrom Djinn");
        for (int i = 0; i < instances; i++) {
            KeywordEdit.add(board.game, board.subject, "Vanishing");
        }
        board.subject.setCounters(CounterEnumType.TIME, 2);
        assertEquals(instances, board.subject.getAmountOfKeyword(Keyword.VANISHING));
        return board;
    }

    @Test
    void vanishingInstancesBindsAPermanentWithTwo() {
        Board board = djinnBoard(2);
        assertEquals(board.subject, VanishingInstances.bind(board.game).orElseThrow().card("permanent"));
    }

    @Test
    void vanishingInstancesBindsNothingOnOne() {
        assertTrue(VanishingInstances.bind(djinnBoard(1).game).isEmpty());
    }

    @Test
    void vanishingInstancesForksOneInstance() {
        Board board = djinnBoard(2);
        Twin twin = board.twin(VanishingInstances.rule());
        String djinn = id(board.subject);
        assertFalse(twin.changes().isEmpty());
        assertTrue(Set.of(djinn + ".lines", djinn + ".granted_temporary").containsAll(twin.changes()),
                twin.changes().toString());
        Card forked = twin.fork().card("permanent");
        assertEquals(1, forked.getAmountOfKeyword(Keyword.VANISHING));
        assertTrue(forked.hasKeyword(Keyword.FLYING), "the djinn's other keywords stay");
    }

    // ── 702.105a, dethrone: life ────────────────────────────────────────

    private static Board dethroneBoard(int opponentLife) {
        Board board = new Board(PhaseType.COMBAT_DECLARE_ATTACKERS);
        board.subject = board.card(board.you, "Enraged Revolutionary");
        board.you.setLife(15, null);
        board.opponent.setLife(opponentLife, null);
        return board;
    }

    @Test
    void dethroneBindsTheCreatureAndTheOpponentWithTheMostLife() {
        for (int life : List.of(20, 15)) {
            Board board = dethroneBoard(life);
            Binding binding = Dethrone.bind(board.game).orElseThrow();
            assertEquals(board.subject, binding.card("attacker"));
            assertEquals(board.opponent, binding.player("opponent"));
        }
    }

    @Test
    void dethroneBindsNothingAgainstAnOpponentWithLessLife() {
        assertTrue(Dethrone.bind(dethroneBoard(14).game).isEmpty());
    }

    @Test
    void dethroneForksTheOpponentOneLifeBelow() {
        Board board = dethroneBoard(20);
        Twin twin = board.twin(Dethrone.rule());
        assertEquals(Set.of("P1.life"), twin.changes());
        assertEquals(14, twin.fork().player("opponent").getLife());
    }

    // ── 702.127a, aftermath: move_card ──────────────────────────────────

    @Test
    void aftermathBindsTheSplitCardInTheGraveyard() {
        Board board = new Board(PhaseType.MAIN1);
        board.subject = TestBoard.inGraveyard(board.game, board.you, "Cut // Ribbons");
        assertEquals(board.subject, Aftermath.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void aftermathBindsNothingInHand() {
        Board board = new Board(PhaseType.MAIN1);
        TestBoard.inHand(board.game, board.you, "Cut // Ribbons");
        assertTrue(Aftermath.bind(board.game).isEmpty());
    }

    @Test
    void aftermathForksTheCardInHand() {
        Board board = new Board(PhaseType.MAIN1);
        board.subject = TestBoard.inGraveyard(board.game, board.you, "Cut // Ribbons");
        Twin twin = board.twin(Aftermath.rule());
        String cut = id(board.subject);
        assertTrue(twin.changes().containsAll(Set.of("P0.hand", "P0.graveyard")), twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(Set.of("P0.hand", "P0.graveyard").contains(change) || change.startsWith(cut + "."),
                    "moving changed " + change + "; all: " + twin.changes());
        }
        assertTrue(twin.fork().card("card").getZone().is(ZoneType.Hand));
    }

    // ── 702.154a, enlist: tap ───────────────────────────────────────────

    private static Board enlistBoard() {
        Board board = new Board(PhaseType.COMBAT_DECLARE_ATTACKERS);
        board.other = board.card(board.you, "Barkweave Crusher");
        board.subject = board.card(board.you, "Hill Giant");
        return board;
    }

    @Test
    void enlistBindsTheAttackerAndTheOneCreatureItCouldEnlist() {
        Board board = enlistBoard();
        Binding binding = Enlist.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("attacker"));
        assertEquals(board.subject, binding.card("enlisted"));
    }

    @Test
    void enlistBindsNothingWhenTheOtherCreatureCameThisTurn() {
        Board board = enlistBoard();
        board.subject.setSickness(true);
        assertTrue(Enlist.bind(board.game).isEmpty());
    }

    @Test
    void enlistBindsNothingWhenTwoCreaturesCouldBeEnlisted() {
        Board board = enlistBoard();
        board.card(board.you, "Grizzly Bears");
        assertTrue(Enlist.bind(board.game).isEmpty());
    }

    @Test
    void enlistForksTheOtherCreatureTapped() {
        Board board = enlistBoard();
        Twin twin = board.twin(Enlist.rule());
        assertEquals(Set.of(id(board.subject) + ".tapped"), twin.changes());
        assertTrue(twin.fork().card("enlisted").isTapped());
    }

    // ── 703.4m, simultaneous combat damage: replace_card ────────────────

    /** A's Grizzly Bears attacks, B's {@code blocker} blocks it; the first-strike damage step begins. */
    private static Board blockedBoard(String blocker) {
        Board board = new Board(PhaseType.COMBAT_FIRST_STRIKE_DAMAGE);
        board.other = board.card(board.you, "Grizzly Bears");
        board.subject = board.card(board.opponent, blocker);
        TestBoard.blockedCombat(board.game, board.you, board.opponent, board.other, board.subject);
        return board;
    }

    @Test
    void simultaneousDamageBindsTheBlockedAttackerAndItsBlocker() {
        Board board = blockedBoard("Grizzly Bears");
        Binding binding = SimultaneousDamage.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("attacker"));
        assertEquals(board.subject, binding.card("blocker"));
    }

    @Test
    void simultaneousDamageBindsNothingWhenTheBlockerStrikesFirst() {
        assertTrue(SimultaneousDamage.bind(blockedBoard("Youthful Knight").game).isEmpty());
    }

    @Test
    void simultaneousDamageForksAFirstStrikeBlockerInTheBlockersPlace() {
        Board board = blockedBoard("Grizzly Bears");
        Twin twin = board.twin(SimultaneousDamage.rule());
        Game fork = twin.forkGame();
        Card knight = fork.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(SimultaneousDamage.KNIGHT)).findFirst().orElseThrow();
        // The live board with the knight's id wherever the blocker's stood:
        // what is left differs only in what makes the card that card, its
        // first strike included (granted_temporary lists every keyword).
        Set<String> left = changes(renamed(twin.real(), id(board.subject), id(knight)), twin.forked());
        Set<String> identity = Set.of("name", "colors", "subtypes", "mana_value", "mana_cost",
                "base_pt", "lines", "granted_temporary");
        for (String change : left) {
            assertTrue(change.startsWith(id(knight) + ".")
                            && identity.contains(change.substring(change.indexOf('.') + 1)),
                    "replacing changed " + change + "; all: " + left);
        }
        assertTrue(fork.getCombat().isBlocking(knight, twin.fork().card("attacker")));
        assertTrue(knight.hasFirstStrike());
    }

    // ── the board and its fork ──────────────────────────────────────────

    /**
     * What a directed fork of the live game looks like once the variant is
     * applied: the changes from the live board, both snapshots, and the
     * fork's binding and game.
     */
    private record Twin(Set<String> changes, Map<String, Object> real, Map<String, Object> forked,
                        Binding fork, Game forkGame) {
    }

    /**
     * A started two-seat game in {@code phase} of your turn: you (P0, the
     * scenarios' A) and the opponent (P1, B).
     */
    private static final class Board {
        final Game game;
        final Player you;
        final Player opponent;
        /** The object the variant changes. */
        Card subject;
        /** The other object a precondition binds, where it binds two. */
        Card other;

        Board(PhaseType phase) {
            game = TestBoard.startedTwoPlayerGame(phase);
            you = game.getPlayers().get(0);
            opponent = game.getPlayers().get(1);
        }

        /** A permanent since the turn began, with a timestamp of its own. */
        Card card(Player owner, String name) {
            Card card = TestBoard.inPlay(game, owner, name);
            card.setGameTimestamp(game.getNextTimestamp());
            return card;
        }

        /**
         * Copy the game as a directed fork does, map the precondition's
         * binding into the copy and apply the rule's one variant, then compare
         * the copy with the live board. The live board is left as it was.
         */
        Twin twin(DirectedRule rule) {
            assertEquals(1, rule.variants().size(), rule.rule() + " registers one variant");
            Binding binding = rule.precondition(game).orElseThrow(
                    () -> new AssertionError(rule.rule() + ": the precondition does not hold"));
            Map<String, Object> real = BoardStates.of(game);
            GameCopier copier = new GameCopier(game);
            Game fork = copier.makeCopy();
            assertEquals(Set.of(), changes(real, BoardStates.of(fork)), "the unedited copy differs");

            Binding forked = binding.inFork(copier);
            rule.variants().values().iterator().next().edit(forked).apply(fork);

            Map<String, Object> edited = BoardStates.of(fork);
            assertEquals(real, BoardStates.of(game), "the edit reached the live game");
            return new Twin(changes(real, edited), real, edited, forked, fork);
        }
    }
}
