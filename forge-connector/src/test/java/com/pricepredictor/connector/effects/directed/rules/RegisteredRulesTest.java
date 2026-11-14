package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.BoardStates;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.TestForks;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.DirectedRules;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.KeywordEdit;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import com.pricepredictor.connector.effects.directed.operations.CountersOperation;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import forge.StaticData;
import forge.ai.simulation.GameCopier;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCopyService;
import forge.game.card.CardFactory;
import forge.game.card.CounterEnumType;
import forge.game.card.token.TokenInfo;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Function;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static com.pricepredictor.connector.effects.directed.BoardChanges.renamed;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
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

    /**
     * You at 8 life beside Enraged Revolutionary (2 power, 1 more from its
     * dethrone counter) and a 2-power double striker: counted once the
     * reach is 5, below the 7 the cut opponent keeps, but the double striker
     * deals its 2 twice, and 7 is lethal.
     */
    @Test
    void dethroneCountsADoubleStrikersDamageTwice() {
        Board board = dethroneBoard(20);
        board.you.setLife(8, null);
        board.card(board.you, "Mirran Crusader");
        assertEquals(7, Dethrone.attackReach(board.you));
        assertTrue(Dethrone.bind(board.game).isEmpty(), "the cut opponent is in the attack's reach");
    }

    @Test
    void dethroneBindsNothingWhenSomethingCanGrowTheAttack() {
        Board exalted = dethroneBoard(20);
        exalted.card(exalted.you, "Akrasan Squire");
        assertTrue(Dethrone.bind(exalted.game).isEmpty(), "exalted");
        Board pump = dethroneBoard(20);
        pump.card(pump.you, "Forest");
        TestBoard.inHand(pump.game, pump.you, "Giant Growth");
        assertTrue(Dethrone.bind(pump.game).isEmpty(), "an instant pump it has the land for");
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

    // ── 510.2 (703.4m), simultaneous combat damage: replace_card ────────

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

    // ── 104.2b, a win at a life total: life ─────────────────────────────

    private static Board sovereignBoard(int life) {
        Board board = new Board(PhaseType.UPKEEP);
        board.subject = board.card(board.you, "Felidar Sovereign");
        board.you.setLife(life, null);
        return board;
    }

    @Test
    void lifeTotalWinBindsTheSovereignAtFortyLife() {
        Board board = sovereignBoard(40);
        Binding binding = LifeTotalWin.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("permanent"));
        assertEquals(board.you, binding.player("player"));
    }

    @Test
    void lifeTotalWinBindsNothingOneLifeShort() {
        assertTrue(LifeTotalWin.bind(sovereignBoard(39).game).isEmpty());
    }

    @Test
    void lifeTotalWinForksThePlayerOneLifeShort() {
        Twin twin = sovereignBoard(45).twin(LifeTotalWin.rule());
        assertEquals(Set.of("P0.life"), twin.changes());
        assertEquals(39, twin.fork().player("player").getLife());
    }

    // ── 104.3d, ten poison counters: counters ───────────────────────────

    private static Board poisonBoard(int poison) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Swamp");
        board.opponent.setCounters(CounterEnumType.POISON, poison, null, false);
        return board;
    }

    @Test
    void poisonLossBindsThePlayerOneCounterShort() {
        Board board = poisonBoard(9);
        assertEquals(board.opponent, PoisonLoss.bind(board.game).orElseThrow().player("player"));
    }

    @Test
    void poisonLossBindsNothingTwoCountersShort() {
        assertTrue(PoisonLoss.bind(poisonBoard(8).game).isEmpty());
    }

    @Test
    void poisonLossForksThePlayerWithEightCounters() {
        Twin twin = poisonBoard(9).twin(PoisonLoss.rule());
        assertEquals(Set.of("P1.poison"), twin.changes());
        assertEquals(8, twin.fork().player("player").getPoisonCounters());
    }

    // ── 104.6, restarting the game: counters ────────────────────────────

    private static Board karnBoard(int loyalty) {
        Board board = new Board(PhaseType.MAIN1);
        board.subject = board.card(board.you, "Karn Liberated");
        board.subject.setCounters(CounterEnumType.LOYALTY, loyalty);
        return board;
    }

    @Test
    void restartGameBindsKarnWithFourteenLoyalty() {
        Board board = karnBoard(14);
        assertEquals(board.subject, RestartGame.bind(board.game).orElseThrow().card("planeswalker"));
    }

    @Test
    void restartGameBindsNothingOneLoyaltyShort() {
        assertTrue(RestartGame.bind(karnBoard(13).game).isEmpty());
    }

    @Test
    void restartGameForksKarnOneLoyaltyShort() {
        Board board = karnBoard(16);
        Twin twin = board.twin(RestartGame.rule());
        assertEquals(Set.of(id(board.subject) + ".counters"), twin.changes());
        assertEquals(13, twin.fork().card("planeswalker").getCounters(CounterEnumType.LOYALTY));
    }

    // ── 111.10n, a Sorcerer Role: remove_permanent ──────────────────────

    private static Board vendorBoard(int plains) {
        Board board = new Board(PhaseType.MAIN1);
        board.other = board.card(board.you, "Spellbook Vendor");
        for (int i = 0; i < plains; i++) {
            board.subject = board.card(board.you, "Plains");
        }
        return board;
    }

    @Test
    void sorcererRoleBindsTheLastManaSource() {
        Board board = vendorBoard(1);
        assertEquals(board.subject, SorcererRole.bind(board.game).orElseThrow().card("mana source"));
    }

    @Test
    void sorcererRoleBindsNothingWithManaToSpare() {
        assertTrue(SorcererRole.bind(vendorBoard(2).game).isEmpty());
    }

    @Test
    void sorcererRoleForksTheBoardWithoutTheManaSource() {
        Board board = vendorBoard(1);
        assertOnlyRemoved(board.twin(SorcererRole.rule()), board.subject, "mana source");
    }

    // ── 111.10p, a Virtuous Role: remove_permanent ──────────────────────

    private static Board elliverBoard() {
        Board board = new Board(PhaseType.MAIN1);
        for (String land : List.of("Forest", "Forest", "Forest", "Plains")) {
            board.card(board.you, land);
        }
        board.subject = board.card(board.you, "Grizzly Bears");
        board.other = TestBoard.inHand(board.game, board.you, "Ellivere of the Wild Court");
        return board;
    }

    @Test
    void virtuousRoleBindsTheOnlyOtherCreatureAndEllivere() {
        Board board = elliverBoard();
        Binding binding = VirtuousRole.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("creature"));
        assertEquals(board.other, binding.card("card"));
    }

    @Test
    void virtuousRoleBindsNothingWithTwoTargets() {
        Board board = elliverBoard();
        board.card(board.you, "Hill Giant");
        assertTrue(VirtuousRole.bind(board.game).isEmpty());
    }

    @Test
    void virtuousRoleForksTheBoardWithoutTheCreature() {
        Board board = elliverBoard();
        assertOnlyRemoved(board.twin(VirtuousRole.rule()), board.subject, "creature");
    }

    // ── 111.10v, a Mutagen token: remove_permanent ──────────────────────

    private static Board mutagenBoard(boolean islandTapped) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Crustacean Commando");
        Card token = CardFactory.getCard(
                StaticData.instance().getAllTokens().getToken("c_a_mutagen_sac"), board.you, board.game);
        board.other = board.game.getAction().moveToPlay(token, board.you, null, null);
        board.other.setGameTimestamp(board.game.getNextTimestamp());
        board.subject = board.card(board.you, "Island");
        board.subject.setTapped(islandTapped);
        return board;
    }

    @Test
    void mutagenTokenBindsTheTokenAndTheLastManaSource() {
        Board board = mutagenBoard(false);
        Binding binding = MutagenToken.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("token"));
        assertEquals(board.subject, binding.card("mana source"));
    }

    @Test
    void mutagenTokenBindsNothingWithNoManaLeft() {
        assertTrue(MutagenToken.bind(mutagenBoard(true).game).isEmpty());
    }

    @Test
    void mutagenTokenForksTheBoardWithoutTheManaSource() {
        Board board = mutagenBoard(false);
        assertOnlyRemoved(board.twin(MutagenToken.rule()), board.subject, "mana source");
    }

    // ── 118.11, a replaced sacrifice cost: remove_permanent ─────────────

    private static Board bombardmentBoard(boolean restInPeace) {
        Board board = new Board(PhaseType.MAIN1);
        board.other = board.card(board.you, "Goblin Bombardment");
        board.subject = board.card(board.you, "Grizzly Bears");
        if (restInPeace) {
            board.card(board.you, "Rest in Peace");
        }
        return board;
    }

    @Test
    void replacedSacrificeCostBindsTheOnlyCreatureToSacrifice() {
        Board board = bombardmentBoard(true);
        Binding binding = ReplacedSacrificeCost.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("creature"));
        assertEquals(board.other, binding.card("permanent"));
    }

    @Test
    void replacedSacrificeCostBindsNothingWithoutAReplacement() {
        assertTrue(ReplacedSacrificeCost.bind(bombardmentBoard(false).game).isEmpty());
    }

    @Test
    void replacedSacrificeCostBindsNothingWithTwoCreaturesToSacrifice() {
        Board board = bombardmentBoard(true);
        board.card(board.you, "Hill Giant");
        assertTrue(ReplacedSacrificeCost.bind(board.game).isEmpty());
    }

    @Test
    void replacedSacrificeCostForksTheBoardWithoutTheCreature() {
        Board board = bombardmentBoard(true);
        assertOnlyRemoved(board.twin(ReplacedSacrificeCost.rule()), board.subject, "creature");
    }

    // ── 118.12b, an optional search: move_card ──────────────────────────

    /** Elfhame Sanctuary as A's upkeep begins, over A's library listed top card first. */
    private static Board sanctuaryBoard(String... library) {
        Board board = new Board(PhaseType.UPKEEP);
        board.card(board.you, "Elfhame Sanctuary");
        for (String name : library) {
            Card card = board.game.getAction().moveToLibrary(
                    TestBoard.card(board.game, board.you, name), -1, null, null);
            if (name.equals("Forest")) {
                board.subject = card;
            }
        }
        return board;
    }

    @Test
    void optionalSearchBindsTheOnlyBasicLandInTheLibrary() {
        Board board = sanctuaryBoard("Shock", "Forest");
        assertEquals(board.subject, OptionalSearch.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void optionalSearchBindsNothingWithTwoBasicLands() {
        assertTrue(OptionalSearch.bind(sanctuaryBoard("Forest", "Shock", "Forest").game).isEmpty());
    }

    @Test
    void optionalSearchForksALightningBoltInTheLandsPlace() {
        Board board = sanctuaryBoard("Shock", "Forest");
        Twin twin = board.twin(OptionalSearch.rule());
        List<Card> library = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library));
        assertEquals(List.of("Shock", OptionalSearch.BOLT), library.stream().map(Card::getName).toList());
        String forest = id(board.subject);
        String bolt = id(library.get(1));
        for (String change : twin.changes()) {
            assertTrue(change.equals("-" + forest) || change.equals("+" + bolt)
                            || change.startsWith("-link ") && change.contains("=" + forest)
                            || change.startsWith("+link ") && change.contains("=" + bolt),
                    "replacing changed " + change + "; all: " + twin.changes());
        }
    }

    // ── 119.7, an exchange a can't-gain-life effect stops: life ─────────

    private static Board redemptionBoard(boolean leyline) {
        Board board = new Board(PhaseType.MAIN1);
        board.subject = board.card(board.you, "Tree of Redemption");
        if (leyline) {
            board.card(board.you, "Leyline of Punishment");
        }
        board.you.setLife(5, null);
        return board;
    }

    @Test
    void cantGainLifeExchangeBindsTheTreeAndItsController() {
        Board board = redemptionBoard(true);
        Binding binding = CantGainLifeExchange.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("permanent"));
        assertEquals(board.you, binding.player("player"));
    }

    @Test
    void cantGainLifeExchangeBindsNothingWhenThePlayerCanGainLife() {
        assertTrue(CantGainLifeExchange.bind(redemptionBoard(false).game).isEmpty());
    }

    @Test
    void cantGainLifeExchangeForksALifeTotalAboveTheToughness() {
        Twin twin = redemptionBoard(true).twin(CantGainLifeExchange.rule());
        assertEquals(Set.of("P0.life"), twin.changes());
        assertEquals(20, twin.fork().player("player").getLife());
    }

    // ── 119.8, an exchange a can't-lose-life effect stops: remove_permanent

    private static Board perditionBoard(int opponentLife) {
        Board board = new Board(PhaseType.MAIN1);
        board.other = board.card(board.you, "Tree of Perdition");
        // Set before the Emperion arrives, which keeps the life total from changing.
        board.opponent.setLife(opponentLife, null);
        board.subject = board.card(board.opponent, "Platinum Emperion");
        return board;
    }

    @Test
    void cantLoseLifeExchangeBindsTheTreeTheOpponentAndTheEmperion() {
        Board board = perditionBoard(20);
        Binding binding = CantLoseLifeExchange.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("permanent"));
        assertEquals(board.opponent, binding.player("opponent"));
        assertEquals(board.subject, binding.card("source"));
    }

    @Test
    void cantLoseLifeExchangeBindsNothingWhenTheExchangeWouldRaiseTheLife() {
        assertTrue(CantLoseLifeExchange.bind(perditionBoard(10).game).isEmpty());
    }

    @Test
    void cantLoseLifeExchangeForksTheBoardWithoutTheEmperion() {
        Board board = perditionBoard(20);
        Twin twin = board.twin(CantLoseLifeExchange.rule());
        assertOnlyRemoved(twin, board.subject, "source");
        assertTrue(twin.fork().player("opponent").canLoseLife());
    }

    // ── 704.5b (121.4), a draw step's draw from an empty library: move_card

    /** Your upkeep of game turn {@code turn}, {@code library} Forests in your library. */
    private static Board lastCardBoard(int library, PhaseType phase, int turn) {
        Board board = new Board(phase);
        board.game.getPhaseHandler().devModeSet(phase, board.you, turn);
        board.card(board.you, "Island");
        for (int i = 0; i < library; i++) {
            board.subject = board.game.getAction().moveToLibrary(
                    TestBoard.card(board.game, board.you, "Forest"), -1, null, null);
        }
        return board;
    }

    @Test
    void drawFromEmptyLibraryBindsTheLastCardAsTheUpkeepBegins() {
        Board board = lastCardBoard(1, PhaseType.UPKEEP, 3);
        assertEquals(board.subject, DrawFromEmptyLibrary.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void drawFromEmptyLibraryBindsNothingWithTwoCardsLeft() {
        assertTrue(DrawFromEmptyLibrary.bind(lastCardBoard(2, PhaseType.UPKEEP, 3).game).isEmpty());
    }

    @Test
    void drawFromEmptyLibraryBindsNothingWhereNoDrawStepFollows() {
        assertTrue(DrawFromEmptyLibrary.bind(lastCardBoard(1, PhaseType.MAIN1, 3).game).isEmpty(),
                "past the draw step");
        assertTrue(DrawFromEmptyLibrary.bind(lastCardBoard(1, PhaseType.UPKEEP, 1).game).isEmpty(),
                "the first turn skips its draw step");
    }

    @Test
    void drawFromEmptyLibraryForksAnEmptyLibrary() {
        Board board = lastCardBoard(1, PhaseType.UPKEEP, 3);
        Twin twin = board.twin(DrawFromEmptyLibrary.rule());
        String forest = id(board.subject);
        assertTrue(twin.changes().containsAll(Set.of("P0.library", "P0.graveyard")), twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(Set.of("P0.library", "P0.graveyard").contains(change) || change.startsWith(forest + "."),
                    "moving changed " + change + "; all: " + twin.changes());
        }
        assertTrue(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library).isEmpty());
    }

    // ── 704.5v (122.1g), a battle's last defense counter: counters ──────

    private static Board battleBoard(int defense) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Mountain");
        board.other = TestBoard.inHand(board.game, board.you, "Lightning Bolt");
        board.subject = board.card(board.you, "Invasion of Zendikar");
        board.subject.setCounters(CounterEnumType.DEFENSE, defense);
        board.subject.setProtectingPlayer(board.opponent);
        return board;
    }

    @Test
    void battleDefenseBindsTheBattleTheSpellWouldDefeat() {
        Board board = battleBoard(3);
        Binding binding = BattleDefense.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("battle"));
        assertEquals(board.other, binding.card("spell"));
    }

    @Test
    void battleDefenseBindsNothingWhenTheSpellLeavesADefenseCounter() {
        assertTrue(BattleDefense.bind(battleBoard(4).game).isEmpty());
    }

    @Test
    void battleDefenseForksOneDefenseCounterMoreThanTheDamage() {
        Board board = battleBoard(2);
        Twin twin = board.twin(BattleDefense.rule());
        assertEquals(Set.of(id(board.subject) + ".counters"), twin.changes());
        assertEquals(4, twin.fork().card("battle").getCounters(CounterEnumType.DEFENSE));
    }

    // ── 205.1a, a type-setting effect: counters ─────────────────────────

    private static Board dryadsBoard(int forests) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < forests; i++) {
            board.card(board.you, "Forest");
        }
        board.other = TestBoard.inHand(board.game, board.you, "Song of the Dryads");
        board.subject = board.card(board.opponent, "Grizzly Bears");
        return board;
    }

    @Test
    void typeSettingAuraBindsTheCreatureAndTheAura() {
        Board board = dryadsBoard(3);
        Binding binding = TypeSettingAura.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("creature"));
        assertEquals(board.other, binding.card("aura"));
    }

    @Test
    void typeSettingAuraBindsNothingWithoutTheManaToCastIt() {
        assertTrue(TypeSettingAura.bind(dryadsBoard(2).game).isEmpty());
    }

    @Test
    void typeSettingAuraForksTheCreatureWithACounter() {
        Board board = dryadsBoard(3);
        Twin twin = board.twin(TypeSettingAura.rule());
        String bears = id(board.subject);
        assertTrue(twin.changes().contains(bears + ".counters"), twin.changes().toString());
        assertTrue(Set.of(bears + ".counters", bears + ".pt_boosts").containsAll(twin.changes()),
                twin.changes().toString());
        assertEquals(1, twin.fork().card("creature").getCounters(CounterEnumType.P1P1));
    }

    // ── 405.6d, a special action: tap ───────────────────────────────────

    /** A's face-down Exalted Angel (morph {2}{W}{W}) beside {@code plains} untapped Plains; the first is {@code other}. */
    private static Board angelBoard(int plains) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < plains; i++) {
            Card land = board.card(board.you, "Plains");
            if (board.other == null) {
                board.other = land;
            }
        }
        Card angel = TestBoard.card(board.game, board.you, "Exalted Angel");
        angel.turnFaceDownNoUpdate();
        board.you.getZone(ZoneType.Battlefield).add(angel);
        angel.setSickness(false);
        angel.setGameTimestamp(board.game.getNextTimestamp());
        board.subject = angel;
        return board;
    }

    @Test
    void morphFaceUpBindsTheFaceDownPermanentAndALand() {
        Board board = angelBoard(4);
        Binding binding = MorphFaceUp.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("permanent"));
        assertEquals(board.other, binding.card("land"));
    }

    @Test
    void morphFaceUpBindsNothingWithALandToSpare() {
        assertTrue(MorphFaceUp.bind(angelBoard(5).game).isEmpty());
    }

    @Test
    void morphFaceUpForksTheLandTapped() {
        Board board = angelBoard(4);
        Twin twin = board.twin(MorphFaceUp.rule());
        String land = id(board.other);
        assertTrue(twin.changes().contains(land + ".tapped"), twin.changes().toString());
        assertTrue(Set.of(land + ".tapped", "P0.untapped_production").containsAll(twin.changes()),
                twin.changes().toString());
        assertTrue(twin.fork().card("land").isTapped());
    }

    // ── 601.2c, one target chosen once: add_permanent ───────────────────

    private static Board undoBoard() {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Island");
        }
        board.subject = board.card(board.opponent, "Grizzly Bears");
        board.other = TestBoard.inHand(board.game, board.you, "Undo");
        return board;
    }

    @Test
    void distinctTargetsBindsTheSpellAndTheOpponent() {
        Board board = undoBoard();
        Binding binding = DistinctTargets.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("spell"));
        assertEquals(board.opponent, binding.player("opponent"));
    }

    @Test
    void distinctTargetsBindsNothingWithTwoCreatures() {
        Board board = undoBoard();
        board.card(board.opponent, "Hill Giant");
        assertTrue(DistinctTargets.bind(board.game).isEmpty());
    }

    @Test
    void distinctTargetsForksASecondCreature() {
        Board board = undoBoard();
        Twin twin = board.twin(DistinctTargets.rule());
        assertOnlyAdded(twin, twin.fork().player("opponent"), DistinctTargets.WURM);
    }

    // ── 607.2k, champion: remove_permanent ──────────────────────────────

    private static Board championBoard() {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 5; i++) {
            board.card(board.you, "Plains");
        }
        board.subject = board.card(board.you, "Grizzly Bears");
        board.other = TestBoard.inHand(board.game, board.you, "Changeling Hero");
        return board;
    }

    @Test
    void championBindsTheOnlyCreatureToChampionAndTheCard() {
        Board board = championBoard();
        Binding binding = Champion.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("creature"));
        assertEquals(board.other, binding.card("card"));
    }

    @Test
    void championBindsNothingWithTwoCreaturesToChampion() {
        Board board = championBoard();
        board.card(board.you, "Hill Giant");
        assertTrue(Champion.bind(board.game).isEmpty());
    }

    @Test
    void championForksTheBoardWithoutTheCreature() {
        Board board = championBoard();
        assertOnlyRemoved(board.twin(Champion.rule()), board.subject, "creature");
    }

    // ── 608.2p, a trigger on an ability resolving: counters ─────────────

    private static Board benaliaBoard(int lore) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Narci, Fable Singer");
        board.subject = board.card(board.you, "History of Benalia");
        board.subject.setCounters(CounterEnumType.LORE, lore);
        return board;
    }

    @Test
    void finalChapterResolvesBindsTheSagaOneCounterShortOfItsLastChapter() {
        Board board = benaliaBoard(2);
        assertEquals(board.subject, FinalChapterResolves.bind(board.game).orElseThrow().card("saga"));
    }

    @Test
    void finalChapterResolvesBindsNothingTwoCountersShort() {
        assertTrue(FinalChapterResolves.bind(benaliaBoard(1).game).isEmpty());
    }

    @Test
    void finalChapterResolvesForksTheSagaWithACounterFewer() {
        Board board = benaliaBoard(2);
        Twin twin = board.twin(FinalChapterResolves.rule());
        String saga = id(board.subject);
        assertEquals(Set.of(saga + ".counters"), twin.changes());
        assertEquals(1, twin.fork().card("saga").getCounters(CounterEnumType.LORE));
    }

    // ── 614.17d, a permanent that can't enter: remove_permanent ─────────

    private static Board cageBoard(boolean creatureInGraveyard) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 4; i++) {
            board.card(board.you, "Swamp");
        }
        board.subject = board.card(board.you, "Grafdigger's Cage");
        TestBoard.inHand(board.game, board.you, "Zombify");
        if (creatureInGraveyard) {
            TestBoard.inGraveyard(board.game, board.you, "Grizzly Bears");
        }
        return board;
    }

    @Test
    void cantEnterBattlefieldBindsTheCage() {
        Board board = cageBoard(true);
        assertEquals(board.subject, CantEnterBattlefield.bind(board.game).orElseThrow().card("permanent"));
    }

    @Test
    void cantEnterBattlefieldBindsNothingWithNoCreatureCardToReturn() {
        assertTrue(CantEnterBattlefield.bind(cageBoard(false).game).isEmpty());
    }

    @Test
    void cantEnterBattlefieldForksTheBoardWithoutTheCage() {
        Board board = cageBoard(true);
        assertOnlyRemoved(board.twin(CantEnterBattlefield.rule()), board.subject, "permanent");
    }

    // ── 701.12e, Aura swap: move_card ───────────────────────────────────

    private static Board auraSwapBoard(String... hand) {
        Board board = new Board(PhaseType.MAIN1);
        Card bears = board.card(board.you, "Grizzly Bears");
        board.other = board.card(board.you, "Arcanum Wings");
        board.other.attachToEntity(bears, null);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Island");
        }
        for (String name : hand) {
            board.subject = TestBoard.inHand(board.game, board.you, name);
        }
        return board;
    }

    @Test
    void auraSwapBindsTheAuraInHandAndTheAttachedAura() {
        Board board = auraSwapBoard("Holy Strength");
        Binding binding = AuraSwap.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("aura"));
    }

    @Test
    void auraSwapBindsNothingWithTwoAurasInHand() {
        assertTrue(AuraSwap.bind(auraSwapBoard("Holy Strength", "Pacifism").game).isEmpty());
    }

    @Test
    void auraSwapBindsNothingForAnAuraThatCannotEnchantTheHost() {
        assertTrue(AuraSwap.bind(auraSwapBoard("Wild Growth").game).isEmpty());
    }

    @Test
    void auraSwapForksAGiantGrowthInTheAurasPlace() {
        Board board = auraSwapBoard("Holy Strength");
        Twin twin = board.twin(AuraSwap.rule());
        List<Card> hand = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand));
        assertEquals(List.of(AuraSwap.GIANT_GROWTH), hand.stream().map(Card::getName).toList());
        assertOnlySwapped(twin, board.subject, hand.get(0));
    }

    // ── 701.12h, a text box exchange: replace_card ──────────────────────

    private static Board exchangeBoard(String opponentCreature) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Grizzly Bears");
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Island");
        }
        board.subject = board.card(board.opponent, opponentCreature);
        TestBoard.inHand(board.game, board.you, "Exchange of Words");
        return board;
    }

    @Test
    void textBoxExchangeBindsTheCreatureWithText() {
        Board board = exchangeBoard("Typhoid Rats");
        assertEquals(board.subject, TextBoxExchange.bind(board.game).orElseThrow().card("creature"));
    }

    @Test
    void textBoxExchangeBindsNothingWhenBothCreaturesAreVanilla() {
        assertTrue(TextBoxExchange.bind(exchangeBoard("Hill Giant").game).isEmpty());
    }

    @Test
    void textBoxExchangeForksAVanillaCreatureInItsPlace() {
        Board board = exchangeBoard("Typhoid Rats");
        Card giant = assertOnlyReplaced(board.twin(TextBoxExchange.rule()), board.subject, TextBoxExchange.GIANT);
        assertFalse(TextBoxExchange.hasRulesText(giant));
    }

    // ── 701.23f, a limited search: remove_permanent ─────────────────────

    private static Board mindcensorBoard(boolean swampUntapped) {
        Board board = new Board(PhaseType.MAIN1);
        board.subject = board.card(board.you, "Aven Mindcensor");
        board.card(board.you, "Archivist of Oghma");
        board.card(board.opponent, "Swamp").setTapped(!swampUntapped);
        TestBoard.inHand(board.game, board.opponent, "Vampiric Tutor");
        return board;
    }

    @Test
    void limitedSearchBindsTheLimitingPermanent() {
        Board board = mindcensorBoard(true);
        assertEquals(board.subject, LimitedSearch.bind(board.game).orElseThrow().card("permanent"));
    }

    @Test
    void limitedSearchBindsNothingWhenTheOpponentCannotSearch() {
        assertTrue(LimitedSearch.bind(mindcensorBoard(false).game).isEmpty());
    }

    @Test
    void limitedSearchForksTheBoardWithoutTheLimitingPermanent() {
        Board board = mindcensorBoard(true);
        assertOnlyRemoved(board.twin(LimitedSearch.rule()), board.subject, "permanent");
    }

    // ── 701.24e, shuffling a library of one card: move_card ─────────────

    /** The opponent's Cosi's Trickster; your Demonic Tutor over {@code library}, top card first. */
    private static Board tricksterBoard(String... library) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.opponent, "Cosi's Trickster");
        board.card(board.you, "Swamp");
        board.card(board.you, "Swamp");
        TestBoard.inHand(board.game, board.you, "Demonic Tutor");
        for (String name : library) {
            board.subject = board.game.getAction().moveToLibrary(
                    TestBoard.card(board.game, board.you, name), -1, null, null);
        }
        return board;
    }

    @Test
    void smallLibraryShuffleBindsTheBottomCardOfATwoCardLibrary() {
        Board board = tricksterBoard("Forest", "Grizzly Bears");
        assertEquals(board.subject, SmallLibraryShuffle.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void smallLibraryShuffleBindsNothingWithThreeCards() {
        assertTrue(SmallLibraryShuffle.bind(tricksterBoard("Forest", "Grizzly Bears", "Island").game).isEmpty());
    }

    @Test
    void smallLibraryShuffleForksTheCardInTheGraveyard() {
        Board board = tricksterBoard("Forest", "Grizzly Bears");
        Twin twin = board.twin(SmallLibraryShuffle.rule());
        String bears = id(board.subject);
        // A graveyard card is an entity of the snapshot, a library card is not.
        assertTrue(twin.changes().containsAll(Set.of("P0.library", "P0.graveyard", "+" + bears)),
                twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(Set.of("P0.library", "P0.graveyard", "+" + bears).contains(change)
                            || change.startsWith("+link ") && change.contains("=" + bears),
                    "moving changed " + change + "; all: " + twin.changes());
        }
        assertTrue(twin.fork().card("card").getZone().is(ZoneType.Graveyard));
        assertEquals(1, twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library).size());
    }

    // ── 701.24g, a shuffle and a put-on-top at once: replace_card ───────

    private static Board barterBoard(String... opponentCreatures) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Cosi's Trickster");
        board.card(board.you, "Grizzly Bears");
        board.card(board.you, "Grizzly Bears");
        for (int i = 0; i < 4; i++) {
            board.card(board.you, "Swamp");
        }
        TestBoard.inHand(board.game, board.you, "Barter in Blood");
        for (String name : opponentCreatures) {
            Card creature = board.card(board.opponent, name);
            if (board.subject == null) {
                board.subject = creature;
            }
        }
        return board;
    }

    @Test
    void shuffleAndPlaceOnTopBindsTheShuffledCreature() {
        Board board = barterBoard("Darksteel Colossus", "Gravebane Zombie");
        assertEquals(board.subject, ShuffleAndPlaceOnTop.bind(board.game).orElseThrow().card("creature"));
    }

    @Test
    void shuffleAndPlaceOnTopBindsNothingWithoutTheShuffledCreature() {
        assertTrue(ShuffleAndPlaceOnTop.bind(barterBoard("Grizzly Bears", "Gravebane Zombie").game).isEmpty());
    }

    @Test
    void shuffleAndPlaceOnTopForksGrizzlyBearsInTheColossussPlace() {
        Board board = barterBoard("Darksteel Colossus", "Gravebane Zombie");
        Card bears = assertOnlyReplaced(board.twin(ShuffleAndPlaceOnTop.rule()), board.subject, "Grizzly Bears");
        assertTrue(bears.getReplacementEffects().isEmpty(), "nothing shuffles the Bears anywhere");
    }

    // ── 701.39a, bolster: remove_permanent ──────────────────────────────

    /** A's Cached Defenses over three Forests, beside A's {@code creatures}; the first is the subject. */
    private static Board bolsterBoard(String... creatures) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Forest");
        }
        for (String name : creatures) {
            Card creature = board.card(board.you, name);
            if (board.subject == null) {
                board.subject = creature;
            }
        }
        board.card(board.opponent, "Savannah Lions");
        TestBoard.inHand(board.game, board.you, "Cached Defenses");
        return board;
    }

    @Test
    void bolsterBindsTheOnlyCreatureOfTheBolsteringPlayer() {
        Board board = bolsterBoard("Grizzly Bears");
        assertEquals(board.subject, Bolster.bind(board.game).orElseThrow().card("creature"));
    }

    @Test
    void bolsterBindsNothingWithTwoCreatures() {
        assertTrue(Bolster.bind(bolsterBoard("Grizzly Bears", "Wall of Wood").game).isEmpty());
    }

    @Test
    void bolsterForksTheBoardWithoutTheCreature() {
        Board board = bolsterBoard("Grizzly Bears");
        assertOnlyRemoved(board.twin(Bolster.rule()), board.subject, "creature");
    }

    // ── 701.42a-b, meld: replace_card ───────────────────────────────────

    /**
     * A's {@code host} beside A's {@code partner}, a card or a token copy of
     * one (built as a game copy builds a token: the card's copiable values,
     * its front face alone); the partner is the subject. The copy calls the
     * deprecated {@code copyCopiableCharacteristics} because the game copy does,
     * and Forge offers nothing in its place.
     */
    @SuppressWarnings("deprecation")
    private static Board meldBoard(String host, String partner, boolean token, int mountains) {
        Board board = new Board(PhaseType.MAIN1);
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
    void meldBindsThePartnerCard() {
        Board board = meldBoard("Graf Rats", "Midnight Scavengers", false, 0);
        assertEquals(board.subject, Meld.bind(board.game).orElseThrow().card("partner"));
    }

    @Test
    void meldBindsNothingWithoutAPartner() {
        assertTrue(Meld.bind(meldBoard("Graf Rats", "Hill Giant", false, 0).game).isEmpty());
        assertTrue(Meld.bind(meldBoard("Graf Rats", "Midnight Scavengers", true, 0).game).isEmpty());
    }

    @Test
    void meldBindsNothingInTheSecondMainPhase() {
        Board board = meldBoard("Graf Rats", "Midnight Scavengers", false, 0);
        board.game.getPhaseHandler().devModeSet(PhaseType.MAIN2, board.you);
        assertTrue(Meld.bind(board.game).isEmpty());
    }

    @Test
    void meldForksHillGiantInThePartnersPlace() {
        Board board = meldBoard("Graf Rats", "Midnight Scavengers", false, 0);
        assertOnlyReplaced(board.twin(Meld.rule()), board.subject, Meld.GIANT);
    }

    @Test
    void tokenMeldPartnerBindsTheTokenPartnerOfATriggeredMeld() {
        Board board = meldBoard("Graf Rats", "Midnight Scavengers", true, 0);
        assertEquals(board.subject, TokenMeldPartner.bind(board.game).orElseThrow().card("partner"));
    }

    @Test
    void tokenMeldPartnerBindsNothingForACardPartnerOrAnActivatedMeld() {
        assertTrue(TokenMeldPartner.bind(meldBoard("Graf Rats", "Midnight Scavengers", false, 0).game).isEmpty());
        assertTrue(TokenMeldPartner.bind(
                meldBoard("Hanweir Battlements", "Hanweir Garrison", true, 5).game).isEmpty());
    }

    @Test
    void tokenMeldPartnerForksTheCardInTheTokensPlace() {
        Board board = meldBoard("Graf Rats", "Midnight Scavengers", true, 0);
        Card card = assertOnlyReplaced(board.twin(TokenMeldPartner.rule()), board.subject, "Midnight Scavengers",
                "token_script_id");
        assertFalse(card.isToken());
    }

    // ── 701.44b, exploring with an empty library: move_card ─────────────

    /** A's Merfolk Branchwalker in hand over two Forests, Wildgrowth Walker, a Forest in A's graveyard. */
    private static Board branchwalkerBoard() {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Forest");
        board.card(board.you, "Forest");
        board.card(board.you, "Wildgrowth Walker");
        TestBoard.inHand(board.game, board.you, "Merfolk Branchwalker");
        board.subject = TestBoard.inGraveyard(board.game, board.you, "Forest");
        return board;
    }

    @Test
    void exploreEmptyLibraryBindsTheLandInTheGraveyard() {
        Board board = branchwalkerBoard();
        assertEquals(board.subject, ExploreEmptyLibrary.bind(board.game).orElseThrow().card("land"));
    }

    @Test
    void exploreEmptyLibraryBindsNothingWithACardInTheLibrary() {
        Board board = branchwalkerBoard();
        board.library(board.you, "Island");
        assertTrue(ExploreEmptyLibrary.bind(board.game).isEmpty());
    }

    @Test
    void exploreEmptyLibraryForksTheLandOnTopOfTheLibrary() {
        Board board = branchwalkerBoard();
        Twin twin = board.twin(ExploreEmptyLibrary.rule());
        assertOnlyChanged(twin, board.subject, board.subject, "P0.library", "P0.graveyard");
        assertEquals(List.of(twin.fork().card("land")),
                List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library)));
    }

    // ── 701.44d, several permanents exploring: replace_card ─────────────

    /** Hakbal of the Surging Soul beside A's {@code other}, over a library of Forest then Shock. */
    private static Board hakbalBoard(String other) {
        Board board = new Board(PhaseType.MAIN1);
        board.other = board.card(board.you, "Hakbal of the Surging Soul");
        board.subject = board.card(board.you, other);
        board.library(board.you, "Forest", "Shock");
        return board;
    }

    @Test
    void severalExploreBindsTheOtherMerfolk() {
        Board board = hakbalBoard("Merfolk Branchwalker");
        assertEquals(board.subject, SeveralExplore.bind(board.game).orElseThrow().card("explorer"));
    }

    @Test
    void severalExploreBindsNothingWithOneMerfolk() {
        assertTrue(SeveralExplore.bind(hakbalBoard("Grizzly Bears").game).isEmpty());
    }

    @Test
    void severalExploreForksGrizzlyBearsInTheMerfolksPlace() {
        Board board = hakbalBoard("Merfolk Branchwalker");
        assertOnlyReplaced(board.twin(SeveralExplore.rule()), board.subject, SeveralExplore.BEARS);
    }

    // ── 701.50f, conniving when drawing is impossible: move_card ────────

    /**
     * A has drawn its one card this turn (a Forest), and holds Echo Inspector
     * and the Forest over four Islands and Glorious Purpose; B's Spirit of the
     * Labyrinth, when {@code spirit}.
     */
    private static Board inspectorBoard(boolean spirit) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 4; i++) {
            board.card(board.you, "Island");
        }
        board.card(board.you, "Glorious Purpose");
        if (spirit) {
            board.card(board.opponent, "Spirit of the Labyrinth");
        }
        TestBoard.inHand(board.game, board.you, "Echo Inspector");
        board.library(board.you, "Forest");
        board.subject = board.you.drawCard().get(0);
        return board;
    }

    @Test
    void conniveWithoutDrawBindsTheOtherCardInHand() {
        Board board = inspectorBoard(true);
        assertEquals(board.subject, ConniveWithoutDraw.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void conniveWithoutDrawBindsNothingWhenThePlayerCanDraw() {
        assertTrue(ConniveWithoutDraw.bind(inspectorBoard(false).game).isEmpty());
    }

    @Test
    void conniveWithoutDrawForksTheCardAtTheBottomOfTheLibrary() {
        Board board = inspectorBoard(true);
        Twin twin = board.twin(ConniveWithoutDraw.rule());
        assertOnlyChanged(twin, board.subject, board.subject, "P0.hand", "P0.library");
        assertTrue(twin.fork().card("card").getZone().is(ZoneType.Library));
        assertEquals(1, twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand).size());
    }

    // ── 701.57b, a discovery that finds nothing: move_card ──────────────

    /** A's Geological Appraiser over four Mountains and Val, Marooned Surveyor; A's library top card first. */
    private static Board appraiserBoard(String... library) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 4; i++) {
            board.card(board.you, "Mountain");
        }
        board.card(board.you, "Val, Marooned Surveyor");
        TestBoard.inHand(board.game, board.you, "Geological Appraiser");
        List<Card> cards = board.library(board.you, library);
        board.subject = cards.stream().filter(card -> !card.isLand()).findFirst().orElse(null);
        return board;
    }

    @Test
    void discoverNothingFoundBindsTheFirstNonlandCard() {
        Board board = appraiserBoard("Forest", "Hill Giant", "Island");
        assertEquals(board.subject, DiscoverNothingFound.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void discoverNothingFoundBindsNothingWhenACardWouldBeFound() {
        assertTrue(DiscoverNothingFound.bind(appraiserBoard("Forest", "Grizzly Bears", "Island").game).isEmpty());
    }

    @Test
    void discoverNothingFoundForksGrizzlyBearsInTheCardsPlace() {
        Board board = appraiserBoard("Forest", "Hill Giant", "Island");
        Twin twin = board.twin(DiscoverNothingFound.rule());
        List<Card> library = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library));
        assertEquals(List.of("Forest", DiscoverNothingFound.BEARS, "Island"),
                library.stream().map(Card::getName).toList());
        // A library card the board shows by count alone: the swap the list
        // above shows changes no visible member.
        assertTrue(twin.changes().isEmpty(), "visible changes: " + twin.changes());
    }

    // ── 702.1c, keywords shared by "the same is true for": replace_card ─

    /** A's Concerted Effort beside A's {@code creatures} as the upkeep begins; the first is the subject. */
    private static Board concertedBoard(String... creatures) {
        Board board = new Board(PhaseType.UPKEEP);
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
        Board board = concertedBoard("White Knight", "Grizzly Bears");
        assertEquals(board.subject, SharedKeywords.bind(board.game).orElseThrow().card("creature"));
    }

    @Test
    void sharedKeywordsBindsNothingWithTwoHoldersOrNoOtherCreature() {
        assertTrue(SharedKeywords.bind(concertedBoard("White Knight", "Youthful Knight").game).isEmpty());
        assertTrue(SharedKeywords.bind(concertedBoard("White Knight").game).isEmpty());
    }

    @Test
    void sharedKeywordsForksHillGiantInTheKnightsPlace() {
        Board board = concertedBoard("White Knight", "Grizzly Bears");
        assertOnlyReplaced(board.twin(SharedKeywords.rule()), board.subject, SharedKeywords.GIANT);
    }

    // ── 702.35a, madness: move_card ─────────────────────────────────────

    /** A's Tormenting Voice and Fiery Temper in hand over {@code mountains} Mountains. */
    private static Board madnessBoard(int mountains) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < mountains; i++) {
            board.card(board.you, "Mountain");
        }
        TestBoard.inHand(board.game, board.you, "Tormenting Voice");
        board.subject = TestBoard.inHand(board.game, board.you, "Fiery Temper");
        return board;
    }

    @Test
    void madnessBindsTheMadnessCard() {
        Board board = madnessBoard(3);
        assertEquals(board.subject, Madness.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void madnessBindsNothingWithoutTheManaForTheDiscardOutlet() {
        assertTrue(Madness.bind(madnessBoard(1).game).isEmpty());
    }

    @Test
    void madnessForksAShockInTheMadnessCardsPlace() {
        Board board = madnessBoard(3);
        Twin twin = board.twin(Madness.rule());
        Card shock = twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand).stream()
                .filter(card -> card.getName().equals(Madness.SHOCK)).findFirst().orElseThrow();
        assertOnlySwapped(twin, board.subject, shock);
    }

    // ── 702.53a, transmute: move_card ───────────────────────────────────

    /** A's Dimir Infiltrator in hand over an Island, a Swamp and a Forest; A's library top card first. */
    private static Board infiltratorBoard(String... library) {
        Board board = new Board(PhaseType.MAIN1);
        for (String land : List.of("Island", "Swamp", "Forest")) {
            board.card(board.you, land);
        }
        board.other = TestBoard.inHand(board.game, board.you, "Dimir Infiltrator");
        board.subject = board.library(board.you, library).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElse(null);
        return board;
    }

    @Test
    void transmuteBindsTheTransmuterAndTheOneCardOfItsManaValue() {
        Board board = infiltratorBoard("Forest", "Shock", "Grizzly Bears", "Island");
        Binding binding = Transmute.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("transmuter"));
        assertEquals(board.subject, binding.card("card"));
    }

    @Test
    void transmuteBindsNothingWithTwoCardsOfItsManaValue() {
        assertTrue(Transmute.bind(infiltratorBoard("Grizzly Bears", "Grizzly Bears").game).isEmpty());
    }

    @Test
    void transmuteForksACentaurCourserInTheCardsPlace() {
        Board board = infiltratorBoard("Forest", "Shock", "Grizzly Bears", "Island");
        Twin twin = board.twin(Transmute.rule());
        List<Card> library = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library));
        assertEquals(List.of("Forest", "Shock", Transmute.COURSER, "Island"),
                library.stream().map(Card::getName).toList());
        // A library card the board shows by count alone: the swap the list
        // above shows changes no visible member.
        assertTrue(twin.changes().isEmpty(), "visible changes: " + twin.changes());
    }

    // ── 702.62c, a prohibition stopping a suspend: remove_permanent ─────

    private static Board balothBoard(boolean golem) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Forest");
        if (golem) {
            board.subject = board.card(board.you, "Steel Golem");
        }
        TestBoard.inHand(board.game, board.you, "Durkwood Baloth");
        return board;
    }

    @Test
    void suspendProhibitedBindsTheProhibitingPermanent() {
        Board board = balothBoard(true);
        assertEquals(board.subject, SuspendProhibited.bind(board.game).orElseThrow().card("permanent"));
    }

    @Test
    void suspendProhibitedBindsNothingWithoutAProhibition() {
        assertTrue(SuspendProhibited.bind(balothBoard(false).game).isEmpty());
    }

    @Test
    void suspendProhibitedForksTheBoardWithoutTheProhibitingPermanent() {
        Board board = balothBoard(true);
        assertOnlyRemoved(board.twin(SuspendProhibited.rule()), board.subject, "permanent");
    }

    // ── 702.65a, Aura swap's exchange: move_card ────────────────────────

    @Test
    void auraSwapExchangeBindsTheAuraInHandAndTheAttachedAura() {
        Board board = auraSwapBoard("Holy Strength");
        Binding binding = AuraSwapExchange.rule().precondition(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("aura"));
    }

    @Test
    void auraSwapExchangeBindsNothingForAnAuraThatCannotEnchantTheHost() {
        assertTrue(AuraSwapExchange.rule().precondition(auraSwapBoard("Wild Growth").game).isEmpty());
    }

    @Test
    void auraSwapExchangeForksGrizzlyBearsInTheAurasPlace() {
        Board board = auraSwapBoard("Holy Strength");
        Twin twin = board.twin(AuraSwapExchange.rule());
        List<Card> hand = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand));
        assertEquals(List.of(AuraSwapExchange.BEARS), hand.stream().map(Card::getName).toList());
        assertOnlySwapped(twin, board.subject, hand.get(0));
    }

    // ── 702.65b, an Aura swap whose Aura cannot enchant: move_card ──────

    @Test
    void auraSwapCantEnchantBindsAnAuraInHandThatCannotEnchantTheHost() {
        Board board = auraSwapBoard("Wild Growth");
        Binding binding = AuraSwapCantEnchant.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("aura"));
    }

    @Test
    void auraSwapCantEnchantBindsNothingForAnAuraThatCanEnchant() {
        assertTrue(AuraSwapCantEnchant.bind(auraSwapBoard("Holy Strength").game).isEmpty());
    }

    @Test
    void auraSwapCantEnchantForksHolyStrengthInTheAurasPlace() {
        Board board = auraSwapBoard("Wild Growth");
        Twin twin = board.twin(AuraSwapCantEnchant.rule());
        List<Card> hand = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand));
        assertEquals(List.of(AuraSwapCantEnchant.HOLY_STRENGTH), hand.stream().map(Card::getName).toList());
        assertOnlySwapped(twin, board.subject, hand.get(0));
    }

    // ── 702.70b, poisonous instances: remove_permanent ──────────────────

    /** A's Virulent Sliver attacks B; {@code slivers} more Virulent Slivers stay home; no blocker declared yet. */
    private static Board sliverBoard(int slivers) {
        Board board = new Board(PhaseType.COMBAT_DECLARE_BLOCKERS);
        board.other = board.card(board.you, "Virulent Sliver");
        for (int i = 0; i < slivers; i++) {
            board.subject = board.card(board.you, "Virulent Sliver");
        }
        Combat combat = new Combat(board.you);
        combat.addAttacker(board.other, board.opponent);
        board.game.getPhaseHandler().setCombat(combat);
        board.game.getAction().checkStaticAbilities();
        return board;
    }

    @Test
    void poisonousInstancesBindsTheOtherGrantingSliver() {
        Board board = sliverBoard(1);
        assertEquals(2, board.other.getKeywords(Keyword.POISONOUS).size());
        assertEquals(board.subject, PoisonousInstances.bind(board.game).orElseThrow().card("permanent"));
    }

    @Test
    void poisonousInstancesBindsNothingOnOneInstance() {
        assertTrue(PoisonousInstances.bind(sliverBoard(0).game).isEmpty());
    }

    @Test
    void poisonousInstancesForksTheBoardWithoutTheGrantingSliver() {
        Board board = sliverBoard(1);
        Twin twin = board.twin(PoisonousInstances.rule());
        String gone = id(board.subject);
        String attacker = id(board.other);
        assertTrue(twin.changes().contains("-" + gone), twin.changes().toString());
        for (String change : twin.changes()) {
            // The attacker loses the instance the removed Sliver granted it.
            assertTrue(change.equals("-" + gone) || change.endsWith(".order_rank")
                            || change.equals(attacker + ".lines") || change.equals(attacker + ".granted_temporary")
                            || change.startsWith("-link ") && change.contains("=" + gone),
                    "removing changed " + change + "; all: " + twin.changes());
        }
        Card forkedAttacker = twin.forkGame().getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> id(card).equals(attacker)).findFirst().orElseThrow();
        assertEquals(1, forkedAttacker.getKeywords(Keyword.POISONOUS).size());
    }

    // ── 702.71a, transfigure: move_card ─────────────────────────────────

    /** A's Fleshwrither (mana value 4) over three Swamps; A's library top card first. */
    private static Board fleshwritherBoard(String... library) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Swamp");
        }
        board.other = board.card(board.you, "Fleshwrither");
        board.subject = board.library(board.you, library).get(0);
        return board;
    }

    @Test
    void transfigureBindsTheCreatureAndTheOneCreatureCardOfItsManaValue() {
        Board board = fleshwritherBoard("Hill Giant", "Grizzly Bears");
        Binding binding = Transfigure.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("creature"));
        assertEquals(board.subject, binding.card("card"));
    }

    @Test
    void transfigureBindsNothingWithTwoCreatureCardsOfItsManaValue() {
        assertTrue(Transfigure.bind(fleshwritherBoard("Hill Giant", "Hill Giant").game).isEmpty());
    }

    @Test
    void transfigureForksACrawWurmInTheCardsPlace() {
        Board board = fleshwritherBoard("Hill Giant", "Grizzly Bears");
        Twin twin = board.twin(Transfigure.rule());
        List<Card> library = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library));
        assertEquals(List.of(Transfigure.WURM, "Grizzly Bears"), library.stream().map(Card::getName).toList());
        assertOnlyChanged(twin, board.subject, library.get(0));
    }

    // ── 702.84a, unearth from the graveyard only: move_card ─────────────

    private static Board dregscapeBoard(boolean swampUntapped) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Swamp").setTapped(!swampUntapped);
        board.subject = TestBoard.inGraveyard(board.game, board.you, "Dregscape Zombie");
        return board;
    }

    @Test
    void unearthBindsTheCardInTheGraveyard() {
        Board board = dregscapeBoard(true);
        assertEquals(board.subject, Unearth.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void unearthBindsNothingWithoutTheManaToUnearth() {
        assertTrue(Unearth.bind(dregscapeBoard(false).game).isEmpty());
    }

    @Test
    void unearthForksTheCardInHand() {
        Board board = dregscapeBoard(true);
        Twin twin = board.twin(Unearth.rule());
        assertOnlyChanged(twin, board.subject, board.subject, "P0.hand", "P0.graveyard");
        assertTrue(twin.fork().card("card").getZone().is(ZoneType.Hand));
    }

    // ── 702.85a, cascade: move_card ─────────────────────────────────────

    /** A's Bloodbraid Elf over two Mountains and two Forests; A's library top card first. */
    private static Board bloodbraidBoard(String... library) {
        Board board = new Board(PhaseType.MAIN1);
        for (String land : List.of("Mountain", "Mountain", "Forest", "Forest")) {
            board.card(board.you, land);
        }
        board.card(board.opponent, "Grizzly Bears");
        TestBoard.inHand(board.game, board.you, "Bloodbraid Elf");
        board.subject = board.library(board.you, library).get(1);
        return board;
    }

    @Test
    void cascadeBindsTheOneCheaperNonlandCard() {
        Board board = bloodbraidBoard("Mountain", "Shock", "Hill Giant");
        assertEquals(board.subject, Cascade.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void cascadeBindsNothingWithTwoCheaperCards() {
        assertTrue(Cascade.bind(bloodbraidBoard("Mountain", "Shock", "Grizzly Bears").game).isEmpty());
    }

    @Test
    void cascadeForksACrawWurmInTheCardsPlace() {
        Board board = bloodbraidBoard("Mountain", "Shock", "Hill Giant");
        Twin twin = board.twin(Cascade.rule());
        List<Card> library = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library));
        assertEquals(List.of("Mountain", Cascade.WURM, "Hill Giant"), library.stream().map(Card::getName).toList());
        // A library card the board shows by count alone: the swap the list
        // above shows changes no visible member.
        assertTrue(twin.changes().isEmpty(), "visible changes: " + twin.changes());
    }

    // ── 702.98a, unleash can't block: counters ──────────────────────────

    /** Grizzly Bears attacks B, who controls {@code blocker} with {@code counters} +1/+1 counters; no blocker declared yet. */
    private static Board unleashBoard(String blocker, int counters) {
        Board board = new Board(PhaseType.COMBAT_DECLARE_BLOCKERS);
        board.other = board.card(board.you, "Grizzly Bears");
        board.subject = board.card(board.opponent, blocker);
        board.subject.setCounters(CounterEnumType.P1P1, counters);
        Combat combat = new Combat(board.you);
        combat.addAttacker(board.other, board.opponent);
        board.game.getPhaseHandler().setCombat(combat);
        return board;
    }

    @Test
    void unleashBindsTheAttackerAndAnUnleashedCreatureWithACounter() {
        Board board = unleashBoard("Rakdos Cackler", 1);
        Binding binding = UnleashCantBlock.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("attacker"));
        assertEquals(board.subject, binding.card("blocker"));
    }

    @Test
    void unleashBindsNothingWithoutTheCounterOrTheKeyword() {
        assertTrue(UnleashCantBlock.bind(unleashBoard("Rakdos Cackler", 0).game).isEmpty());
        assertTrue(UnleashCantBlock.bind(unleashBoard("Runeclaw Bear", 1).game).isEmpty());
    }

    @Test
    void unleashForksTheCreatureWithoutItsCounter() {
        Board board = unleashBoard("Rakdos Cackler", 1);
        Twin twin = board.twin(UnleashCantBlock.rule());
        String cackler = id(board.subject);
        assertTrue(twin.changes().contains(cackler + ".counters"), twin.changes().toString());
        assertTrue(Set.of(cackler + ".counters", cackler + ".pt_boosts").containsAll(twin.changes()),
                twin.changes().toString());
        Card forked = twin.fork().card("blocker");
        assertEquals(0, forked.getCounters(CounterEnumType.P1P1));
        assertFalse(CombatUtil.canBlock(board.subject), "the live creature may not block");
        assertTrue(CombatUtil.canBlock(forked), "with no counter it may");
    }

    // ── 702.102c, a fused split spell's total cost: remove_permanent ────

    /** A's Give // Take in hand over three Forests, {@code islands} Islands and Grizzly Bears; the first Forest is the subject. */
    private static Board fuseBoard(int islands) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            Card forest = board.card(board.you, "Forest");
            if (board.subject == null) {
                board.subject = forest;
            }
        }
        for (int i = 0; i < islands; i++) {
            board.card(board.you, "Island");
        }
        board.card(board.you, "Grizzly Bears");
        TestBoard.inHand(board.game, board.you, "Give // Take");
        return board;
    }

    @Test
    void fusedCostBindsALandTheFusedCastNeeds() {
        Board board = fuseBoard(3);
        assertEquals(board.subject, FusedCost.bind(board.game).orElseThrow().card("land"));
    }

    @Test
    void fusedCostBindsNothingWithALandToSpare() {
        assertTrue(FusedCost.bind(fuseBoard(4).game).isEmpty());
    }

    @Test
    void fusedCostForksTheBoardWithoutTheLand() {
        Board board = fuseBoard(3);
        assertOnlyRemoved(board.twin(FusedCost.rule()), board.subject, "land");
    }

    // ── 702.103e, a bestowed Aura whose target is gone: replace_card ────

    /** A's Hopeful Eidolon in hand over four Plains and its one creature {@code creature}. */
    private static Board eidolonBoard(String creature) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 4; i++) {
            board.card(board.you, "Plains");
        }
        board.subject = board.card(board.you, creature);
        TestBoard.inHand(board.game, board.you, "Hopeful Eidolon");
        return board;
    }

    @Test
    void bestowIllegalTargetBindsTheOneCreatureTheAuraWouldEnchant() {
        Board board = eidolonBoard("Grizzly Bears");
        assertEquals(board.subject, BestowIllegalTarget.bind(board.game).orElseThrow().card("creature"));
    }

    @Test
    void bestowIllegalTargetBindsNothingForACreatureAShockLeavesAlive() {
        assertTrue(BestowIllegalTarget.bind(eidolonBoard("Centaur Courser").game).isEmpty());
    }

    @Test
    void bestowIllegalTargetForksACentaurCourserInTheCreaturesPlace() {
        Board board = eidolonBoard("Grizzly Bears");
        assertOnlyReplaced(board.twin(BestowIllegalTarget.rule()), board.subject, BestowIllegalTarget.COURSER);
    }

    // ── 702.124j, partner with's search: move_card ──────────────────────

    /** A's Pir, Imaginative Rascal in hand over three Forests; A's library top card first. */
    private static Board pirBoard(String... library) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Forest");
        }
        TestBoard.inHand(board.game, board.you, "Pir, Imaginative Rascal");
        board.subject = board.library(board.you, library).stream()
                .filter(card -> card.getName().equals("Toothy, Imaginary Friend")).findFirst().orElse(null);
        return board;
    }

    @Test
    void partnerWithSearchBindsTheNamedCardInTheLibrary() {
        Board board = pirBoard("Forest", "Toothy, Imaginary Friend", "Forest");
        assertEquals(board.subject, PartnerWithSearch.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void partnerWithSearchBindsNothingWithoutTheNamedCard() {
        assertTrue(PartnerWithSearch.bind(pirBoard("Forest", "Forest").game).isEmpty());
    }

    @Test
    void partnerWithSearchForksAForestInTheNamedCardsPlace() {
        Board board = pirBoard("Forest", "Toothy, Imaginary Friend", "Forest");
        Twin twin = board.twin(PartnerWithSearch.rule());
        List<Card> library = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library));
        assertEquals(List.of("Forest", PartnerWithSearch.FOREST, "Forest"),
                library.stream().map(Card::getName).toList());
        // A library card the board shows by count alone: the swap the list
        // above shows changes no visible member.
        assertTrue(twin.changes().isEmpty(), "visible changes: " + twin.changes());
    }

    // ── 702.126a, improvise: tap ────────────────────────────────────────

    /** A's Reverse Engineer ({3}{U}{U}) in hand over two Islands and {@code ornithopters} Ornithopters; the first is the subject. */
    private static Board engineerBoard(int ornithopters) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Island");
        board.card(board.you, "Island");
        for (int i = 0; i < ornithopters; i++) {
            Card ornithopter = board.card(board.you, "Ornithopter");
            if (board.subject == null) {
                board.subject = ornithopter;
            }
        }
        TestBoard.inHand(board.game, board.you, "Reverse Engineer");
        return board;
    }

    @Test
    void improviseBindsAnArtifactTheCastNeeds() {
        Board board = engineerBoard(3);
        assertEquals(board.subject, Improvise.bind(board.game).orElseThrow().card("artifact"));
    }

    @Test
    void improviseBindsNothingWithAnArtifactShortOrToSpare() {
        assertTrue(Improvise.bind(engineerBoard(2).game).isEmpty());
        assertTrue(Improvise.bind(engineerBoard(4).game).isEmpty());
    }

    @Test
    void improviseForksTheArtifactTapped() {
        Board board = engineerBoard(3);
        Twin twin = board.twin(Improvise.rule());
        assertEquals(Set.of(id(board.subject) + ".tapped"), twin.changes());
        assertTrue(twin.fork().card("artifact").isTapped());
    }

    // ── 702.133a, jump-start's discard: move_card ───────────────────────

    /** A's Radical Idea in the graveyard over two Islands, with {@code hand} in hand; the first is the subject. */
    private static Board radicalIdeaBoard(String... hand) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Island");
        board.card(board.you, "Island");
        TestBoard.inGraveyard(board.game, board.you, "Radical Idea");
        for (String name : hand) {
            Card card = TestBoard.inHand(board.game, board.you, name);
            if (board.subject == null) {
                board.subject = card;
            }
        }
        return board;
    }

    @Test
    void jumpStartBindsTheOnlyCardToDiscard() {
        Board board = radicalIdeaBoard("Forest");
        assertEquals(board.subject, JumpStart.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void jumpStartBindsNothingWithTwoCardsToDiscard() {
        assertTrue(JumpStart.bind(radicalIdeaBoard("Forest", "Plains").game).isEmpty());
    }

    @Test
    void jumpStartForksTheCardAtTheBottomOfTheLibrary() {
        Board board = radicalIdeaBoard("Forest");
        Twin twin = board.twin(JumpStart.rule());
        assertOnlyChanged(twin, board.subject, board.subject, "P0.hand", "P0.library");
        assertTrue(twin.fork().card("card").getZone().is(ZoneType.Library));
        assertTrue(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand).isEmpty());
    }

    // ── 702.167a, craft's materials: remove_permanent ───────────────────

    /** A's Braided Net over two Islands, with {@code materials} beside it; the first is the subject. */
    private static Board braidedNetBoard(String... materials) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Island");
        board.card(board.you, "Island");
        board.other = board.card(board.you, "Braided Net");
        for (String name : materials) {
            Card card = board.card(board.you, name);
            if (board.subject == null) {
                board.subject = card;
            }
        }
        return board;
    }

    @Test
    void craftMaterialBindsTheOnlyMaterial() {
        Board board = braidedNetBoard("Short Sword");
        assertEquals(board.subject, CraftMaterial.bind(board.game).orElseThrow().card("material"));
    }

    @Test
    void craftMaterialBindsNothingWithAMaterialToSpare() {
        assertTrue(CraftMaterial.bind(braidedNetBoard("Short Sword", "Ornithopter").game).isEmpty());
        assertTrue(CraftMaterial.bind(braidedNetBoard().game).isEmpty());
    }

    @Test
    void craftMaterialForksTheBoardWithoutTheMaterial() {
        Board board = braidedNetBoard("Short Sword");
        assertOnlyRemoved(board.twin(CraftMaterial.rule()), board.subject, "material");
    }

    // ── 702.187b, mayhem from a card discarded this turn: remove_permanent

    /** A's Electro's Bolt (mayhem {1}{R}) in the graveyard, discarded this turn when {@code discarded}, over {@code mountains} Mountains. */
    private static Board boltBoard(boolean discarded, int mountains) {
        Board board = new Board(PhaseType.MAIN2);
        for (int i = 0; i < mountains; i++) {
            Card mountain = board.card(board.you, "Mountain");
            if (board.subject == null) {
                board.subject = mountain;
            }
        }
        board.other = TestBoard.inGraveyard(board.game, board.you, "Electro's Bolt");
        board.other.setDiscarded(discarded);
        return board;
    }

    @Test
    void mayhemBindsALandTheMayhemCostNeeds() {
        Board board = boltBoard(true, 2);
        assertEquals(board.subject, Mayhem.bind(board.game).orElseThrow().card("land"));
    }

    @Test
    void mayhemBindsNothingForACardNotDiscardedOrWithALandToSpare() {
        assertTrue(Mayhem.bind(boltBoard(false, 2).game).isEmpty());
        assertTrue(Mayhem.bind(boltBoard(true, 3).game).isEmpty());
    }

    @Test
    void mayhemForksTheBoardWithoutTheLand() {
        Board board = boltBoard(true, 2);
        Twin twin = board.twin(Mayhem.rule());
        assertOnlyRemoved(twin, board.subject, "land");
        Card bolt = twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Graveyard).get(0);
        assertTrue(bolt.wasDiscarded() && bolt.enteredThisTurn(), "the fork's Bolt was discarded this turn");
    }

    // ── 702.188a, web-slinging's tapped creature: tap ───────────────────

    /** A's Spider-Man, Web-Slinger in hand over a Plains and Grizzly Bears, tapped when {@code tapped}. */
    private static Board spiderManBoard(boolean tapped) {
        Board board = new Board(PhaseType.MAIN2);
        board.card(board.you, "Plains");
        board.subject = board.card(board.you, "Grizzly Bears");
        board.subject.setTapped(tapped);
        TestBoard.inHand(board.game, board.you, "Spider-Man, Web-Slinger");
        return board;
    }

    @Test
    void webSlingingBindsTheOnlyTappedCreature() {
        Board board = spiderManBoard(true);
        assertEquals(board.subject, WebSlinging.bind(board.game).orElseThrow().card("creature"));
    }

    @Test
    void webSlingingBindsNothingWithoutATappedCreature() {
        assertTrue(WebSlinging.bind(spiderManBoard(false).game).isEmpty());
    }

    @Test
    void webSlingingForksTheCreatureUntapped() {
        Board board = spiderManBoard(true);
        Twin twin = board.twin(WebSlinging.rule());
        assertEquals(Set.of(id(board.subject) + ".tapped"), twin.changes());
        assertFalse(twin.fork().card("creature").isTapped());
    }

    // ── 704.7, one replacement for every loss: remove_permanent ─────────

    /** A at {@code life} with Lich's Mirror, A's library {@code library}. */
    private static Board mirrorBoard(int life, String... library) {
        Board board = new Board(PhaseType.MAIN1);
        board.subject = board.card(board.you, "Lich's Mirror");
        board.you.setLife(life, null);
        board.library(board.you, library);
        return board;
    }

    @Test
    void gameLossReplacedBindsTheReplacingPermanentOfAPlayerCloseToLosing() {
        Board board = mirrorBoard(1, "Plains", "Plains");
        assertEquals(board.subject, GameLossReplaced.bind(board.game).orElseThrow().card("permanent"));
        board = mirrorBoard(20, "Plains");
        assertEquals(board.subject, GameLossReplaced.bind(board.game).orElseThrow().card("permanent"));
    }

    @Test
    void gameLossReplacedBindsNothingForAPlayerFarFromLosing() {
        assertTrue(GameLossReplaced.bind(mirrorBoard(20, "Plains", "Plains").game).isEmpty());
    }

    @Test
    void gameLossReplacedForksTheBoardWithoutTheReplacingPermanent() {
        Board board = mirrorBoard(1, "Plains");
        assertOnlyRemoved(board.twin(GameLossReplaced.rule()), board.subject, "permanent");
    }

    // ── 707.10d, a copy for each legal target: replace_card ─────────────

    /** A's Radiate and {@code hand} in hand over six Mountains; B's Grizzly Bears and Abbey Gargoyles. */
    private static Board radiateBoard(String... hand) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 6; i++) {
            board.card(board.you, "Mountain");
        }
        board.card(board.opponent, "Grizzly Bears");
        board.subject = board.card(board.opponent, "Abbey Gargoyles");
        TestBoard.inHand(board.game, board.you, "Radiate");
        for (String name : hand) {
            TestBoard.inHand(board.game, board.you, name);
        }
        return board;
    }

    @Test
    void copyForEachTargetBindsTheProtectedCreature() {
        Board board = radiateBoard("Shock");
        assertEquals(board.subject, CopyForEachTarget.bind(board.game).orElseThrow().card("creature"));
    }

    @Test
    void copyForEachTargetBindsNothingWithoutASpellToCopy() {
        assertTrue(CopyForEachTarget.bind(radiateBoard().game).isEmpty());
    }

    @Test
    void copyForEachTargetForksHillGiantInTheProtectedCreaturesPlace() {
        Board board = radiateBoard("Shock");
        assertOnlyReplaced(board.twin(CopyForEachTarget.rule()), board.subject, CopyForEachTarget.GIANT);
    }

    // ── 712.4a and 712.14c, melding onto the back faces: replace_card ───

    @Test
    void meldPairAndMeldedEntryBindThePartnerCard() {
        Board board = meldBoard("Graf Rats", "Midnight Scavengers", false, 0);
        assertEquals(board.subject, MeldPair.rule().precondition(board.game).orElseThrow().card("partner"));
        assertEquals(board.subject, MeldedEntry.rule().precondition(board.game).orElseThrow().card("partner"));
    }

    @Test
    void meldPairAndMeldedEntryBindNothingWithoutAPartner() {
        Game game = meldBoard("Graf Rats", "Hill Giant", false, 0).game;
        assertTrue(MeldPair.rule().precondition(game).isEmpty());
        assertTrue(MeldedEntry.rule().precondition(game).isEmpty());
    }

    @Test
    void meldPairAndMeldedEntryForkHillGiantInThePartnersPlace() {
        for (DirectedRule rule : List.of(MeldPair.rule(), MeldedEntry.rule())) {
            Board board = meldBoard("Graf Rats", "Midnight Scavengers", false, 0);
            assertOnlyReplaced(board.twin(rule), board.subject, Meld.GIANT);
        }
    }

    // ── 712.11d, casting transformed from the graveyard: move_card ──────

    /** A's Beloved Beggar (disturb {4}{W}{W}) in the graveyard over {@code plains} Plains. */
    private static Board beggarBoard(int plains) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < plains; i++) {
            board.card(board.you, "Plains");
        }
        board.subject = TestBoard.inGraveyard(board.game, board.you, "Beloved Beggar");
        return board;
    }

    @Test
    void disturbBindsTheCardInTheGraveyard() {
        Board board = beggarBoard(6);
        assertEquals(board.subject, Disturb.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void disturbBindsNothingWithoutTheManaForTheDisturbCost() {
        assertTrue(Disturb.bind(beggarBoard(5).game).isEmpty());
    }

    @Test
    void disturbForksHanweirWatchkeepInTheCardsPlace() {
        Board board = beggarBoard(6);
        Twin twin = board.twin(Disturb.rule());
        List<Card> graveyard = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Graveyard));
        assertEquals(List.of(Disturb.WATCHKEEP), graveyard.stream().map(Card::getName).toList());
        assertOnlySwapped(twin, board.subject, graveyard.get(0));
    }

    // ── 718.3a, a prototype's cost alone: remove_permanent ──────────────

    /** A's Blitz Automaton (prototype {2}{R}, mana value 7) in hand over {@code mountains} Mountains; the first is the subject. */
    private static Board automatonBoard(int mountains) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < mountains; i++) {
            Card mountain = board.card(board.you, "Mountain");
            if (board.subject == null) {
                board.subject = mountain;
            }
        }
        TestBoard.inHand(board.game, board.you, "Blitz Automaton");
        return board;
    }

    @Test
    void prototypeBindsALandThePrototypeCostNeeds() {
        Board board = automatonBoard(3);
        assertEquals(board.subject, Prototype.bind(board.game).orElseThrow().card("land"));
    }

    @Test
    void prototypeBindsNothingWithALandToSpare() {
        assertTrue(Prototype.bind(automatonBoard(4).game).isEmpty());
    }

    @Test
    void prototypeForksTheBoardWithoutTheLand() {
        Board board = automatonBoard(3);
        assertOnlyRemoved(board.twin(Prototype.rule()), board.subject, "land");
    }

    // ── 730.2g, a face-down permanent holding an instant: move_card ─────

    /** A's Soul Summons in hand over two Plains; A's library top card first. */
    private static Board summonsBoard(String... library) {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Plains");
        board.card(board.you, "Plains");
        TestBoard.inHand(board.game, board.you, "Soul Summons");
        board.subject = board.library(board.you, library).get(0);
        return board;
    }

    @Test
    void unturnableFaceDownBindsTheInstantOnTopOfTheLibrary() {
        Board board = summonsBoard("Shock", "Mountain");
        assertEquals(board.subject, UnturnableFaceDown.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void unturnableFaceDownBindsNothingUnderACreatureCard() {
        assertTrue(UnturnableFaceDown.bind(summonsBoard("Grizzly Bears", "Mountain").game).isEmpty());
    }

    @Test
    void unturnableFaceDownForksGrizzlyBearsOnTopOfTheLibrary() {
        Board board = summonsBoard("Shock", "Mountain");
        Twin twin = board.twin(UnturnableFaceDown.rule());
        List<Card> library = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library));
        assertEquals(List.of(UnturnableFaceDown.BEARS, "Mountain"), library.stream().map(Card::getName).toList());
        assertOnlyChanged(twin, board.subject, library.get(0));
    }

    // ── 104.4a, a draw when every player loses at once: life ────────────

    /** Your Plague Spitter as your upkeep begins, you at {@code yours} life and the opponent at {@code theirs}. */
    private static Board spitterBoard(int yours, int theirs) {
        Board board = new Board(PhaseType.UPKEEP);
        board.subject = board.card(board.you, "Plague Spitter");
        board.you.setLife(yours, null);
        board.opponent.setLife(theirs, null);
        return board;
    }

    @Test
    void simultaneousLossBindsAnUpkeepTriggerTakingEveryPlayersLastLife() {
        Board board = spitterBoard(1, 1);
        Binding binding = SimultaneousLoss.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("permanent"));
        assertEquals(board.opponent, binding.player("opponent"));
    }

    @Test
    void simultaneousLossBindsNothingWhenAPlayerWouldStayInTheGame() {
        assertTrue(SimultaneousLoss.bind(spitterBoard(1, 2).game).isEmpty());
        assertTrue(SimultaneousLoss.bind(spitterBoard(2, 1).game).isEmpty());
    }

    @Test
    void simultaneousLossBindsNothingForTheOtherPlayersTrigger() {
        Board board = new Board(PhaseType.UPKEEP);
        board.card(board.opponent, "Plague Spitter");
        board.you.setLife(1, null);
        board.opponent.setLife(1, null);
        assertTrue(SimultaneousLoss.bind(board.game).isEmpty());
    }

    /** Stormfist Crusader's chain draws, then takes 1 life from each player. */
    @Test
    void simultaneousLossReadsTheLifeLossBelowADraw() {
        Board board = new Board(PhaseType.UPKEEP);
        board.card(board.you, "Stormfist Crusader");
        board.you.setLife(1, null);
        board.opponent.setLife(1, null);
        assertTrue(SimultaneousLoss.bind(board.game).isPresent());
    }

    @Test
    void simultaneousLossBindsNothingAtAnotherBoundary() {
        Board board = new Board(PhaseType.DRAW);
        board.card(board.you, "Plague Spitter");
        board.you.setLife(1, null);
        board.opponent.setLife(1, null);
        assertTrue(SimultaneousLoss.bind(board.game).isEmpty());
    }

    /** Platinum Emperion keeps its controller's life from changing: the opponent stays at 1. */
    @Test
    void simultaneousLossBindsNothingWhenAPlayerCannotLoseLife() {
        Board board = spitterBoard(1, 1);
        board.card(board.opponent, "Platinum Emperion");
        assertTrue(SimultaneousLoss.bind(board.game).isEmpty());
    }

    /** Platinum Angel's controller cannot lose the game: the 0 life loses it nothing. */
    @Test
    void simultaneousLossBindsNothingWhenAPlayerCannotLoseForZeroLife() {
        Board board = spitterBoard(1, 1);
        board.card(board.opponent, "Platinum Angel");
        assertTrue(SimultaneousLoss.bind(board.game).isEmpty());
    }

    /** A trigger asking a condition of its own, or its controller whether it happens, is no engine certainty. */
    @Test
    void simultaneousLossBindsNothingForATriggerWithAConditionOrAChoice() {
        for (Map.Entry<String, String> param : Map.of("IsPresent", "Card.Self", "CheckSVar", "X",
                "OptionalDecider", "You").entrySet()) {
            Board board = spitterBoard(1, 1);
            upkeepTrigger(board.subject).putParam(param.getKey(), param.getValue());
            assertTrue(SimultaneousLoss.bind(board.game).isEmpty(), param.getKey());
        }
    }

    /** A chain segment with a condition, an unless cost or an optional part, or another API, decides nothing for sure. */
    @Test
    void simultaneousLossBindsNothingForAChainWithAConditionOrAnotherApi() {
        String damage = "DB$ DamageAll | ValidCards$ Creature | ValidPlayers$ Player | NumDmg$ 1";
        for (String extra : List.of(" | ConditionCheckSVar$ X", " | UnlessCost$ 1", " | Optional$ True",
                " | SubAbility$ DBDestroy")) {
            Board board = spitterBoard(1, 1);
            board.subject.setSVar("TrigDamageAll", damage + extra);
            board.subject.setSVar("DBDestroy", "DB$ Destroy | Defined$ Self");
            assertTrue(SimultaneousLoss.bind(board.game).isEmpty(), extra);
        }
        Board plain = spitterBoard(1, 1);
        plain.subject.setSVar("TrigDamageAll", damage);
        assertTrue(SimultaneousLoss.bind(plain.game).isPresent(), "the plain chain binds");
    }

    private static forge.game.trigger.Trigger upkeepTrigger(Card card) {
        return card.getTriggers().stream()
                .filter(trigger -> "Upkeep".equals(trigger.getParam("Phase"))).findFirst().orElseThrow();
    }

    @Test
    void simultaneousLossForksTheOpponentOneLifeAbove() {
        Twin twin = spitterBoard(1, 1).twin(SimultaneousLoss.rule());
        assertEquals(Set.of("P1.life"), twin.changes());
        assertEquals(2, twin.fork().player("opponent").getLife());
    }

    // ── 702.19c, trample over planeswalkers: replace_card ───────────────

    /** Your {@code attacker} attacks B's Jace Beleren of {@code loyalty} unblocked; the damage step begins. */
    private static Board planeswalkerBoard(String attacker, int loyalty) {
        return planeswalkerBoard(attacker, loyalty, PhaseType.COMBAT_DAMAGE);
    }

    /** {@link #planeswalkerBoard(String, int)} as {@code step} begins. */
    private static Board planeswalkerBoard(String attacker, int loyalty, PhaseType step) {
        Board board = new Board(step);
        board.subject = board.card(board.you, attacker);
        board.other = board.card(board.opponent, "Jace Beleren");
        board.other.setCounters(CounterEnumType.LOYALTY, loyalty);
        Combat combat = new Combat(board.you);
        combat.addAttacker(board.subject, board.other);
        combat.setBlocked(board.subject, false);
        board.game.getPhaseHandler().setCombat(combat);
        return board;
    }

    @Test
    void trampleOverPlaneswalkersBindsAnUnblockedAttackerAboveTheLoyalty() {
        Board board = planeswalkerBoard("Thrasta, Tempest's Roar", 3);
        assertEquals(board.subject, TrampleOverPlaneswalkers.bind(board.game).orElseThrow().card("attacker"));
    }

    @Test
    void trampleOverPlaneswalkersBindsNothingWithoutTheKeyword() {
        assertTrue(TrampleOverPlaneswalkers.bind(planeswalkerBoard("Craw Wurm", 3).game).isEmpty());
    }

    /** Thrasta's 7 tops a loyalty of 6, the Dreadmaw's 6 does not: the fork's attack would have no excess. */
    @Test
    void trampleOverPlaneswalkersBindsNothingWhenTheDreadmawWouldHaveNoExcess() {
        assertTrue(TrampleOverPlaneswalkers.bind(planeswalkerBoard("Thrasta, Tempest's Roar", 6).game).isEmpty());
    }

    @Test
    void trampleOverPlaneswalkersBindsNothingOnABlockedAttacker() {
        Board board = planeswalkerBoard("Thrasta, Tempest's Roar", 3);
        Card bears = board.card(board.opponent, "Grizzly Bears");
        board.game.getCombat().addBlocker(board.subject, bears);
        board.game.getCombat().setBlocked(board.subject, true);
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isEmpty());
    }

    @Test
    void trampleOverPlaneswalkersBindsNothingAtAnotherBoundary() {
        assertTrue(TrampleOverPlaneswalkers.bind(planeswalkerBoard("Thrasta, Tempest's Roar", 3,
                PhaseType.COMBAT_DECLARE_BLOCKERS).game).isEmpty());
    }

    /** A granted trample over planeswalkers would stay on the Dreadmaw put in the Wurm's place. */
    @Test
    void trampleOverPlaneswalkersBindsNothingForAGrantedKeyword() {
        Board board = planeswalkerBoard("Craw Wurm", 3);
        board.subject.addChangedCardKeywords(List.of("Trample:Planeswalker"), null, false,
                board.game.getNextTimestamp(), null);
        assertTrue(Boards.tramplesOverPlaneswalkers(board.subject));
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isEmpty());
    }

    /** Thrasta's 6 marked damage would kill a 6/6 Dreadmaw. */
    @Test
    void trampleOverPlaneswalkersBindsNothingWhenTheDreadmawWouldDie() {
        Board board = planeswalkerBoard("Thrasta, Tempest's Roar", 3);
        board.subject.setDamage(6);
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isEmpty());
    }

    /** A first striker deals its damage in the first-strike step, not in the regular one the rule forks at. */
    @Test
    void trampleOverPlaneswalkersBindsNothingForAFirstStriker() {
        Board board = planeswalkerBoard("Thrasta, Tempest's Roar", 3);
        board.subject.addChangedCardKeywords(List.of("First Strike"), null, false,
                board.game.getNextTimestamp(), null);
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isEmpty());
    }

    @Test
    void trampleOverPlaneswalkersForksAColossalDreadmawAttackingTheSamePlaneswalker() {
        Board board = planeswalkerBoard("Thrasta, Tempest's Roar", 3);
        Twin twin = board.twin(TrampleOverPlaneswalkers.rule());
        Card dreadmaw = twin.forkGame().getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(TrampleOverPlaneswalkers.DREADMAW)).findFirst().orElseThrow();
        // Thrasta's own hexproof static, which affects Thrasta alone, goes
        // with it; a Dreadmaw is no legend.
        String thrastaLink = "-link {from={entity=" + id(dreadmaw) + ", key={face=0, trait_kind=static";
        Set<String> identity = Set.of("name", "colors", "subtypes", "supertypes", "mana_value", "mana_cost",
                "base_pt", "lines", "granted_temporary");
        for (String change : changes(renamed(twin.real(), id(board.subject), id(dreadmaw)), twin.forked())) {
            assertTrue(change.startsWith(thrastaLink) && change.endsWith("to=" + id(dreadmaw) + ", type=affects}")
                            || change.startsWith(id(dreadmaw) + ".")
                            && identity.contains(change.substring(change.indexOf('.') + 1)),
                    "replacing changed " + change);
        }
        assertTrue(dreadmaw.hasKeyword(Keyword.TRAMPLE));
        assertFalse(Boards.tramplesOverPlaneswalkers(dreadmaw));
        assertEquals(twin.forkGame().getCombat().getDefenderByAttacker(dreadmaw).getId(), board.other.getId());
    }

    // ── 702.19e and 702.19f, a planeswalker gone from combat: replace_card

    /**
     * Your {@code attacker} attacks B's Jace Beleren unblocked, and Jace has
     * left the battlefield, so the attacker attacks Forge's placeholder
     * (506.4c), while your Grizzly Bears attacks B; {@code step} begins.
     */
    private static Board goneWalkerBoard(String attacker, PhaseType step) {
        Board board = planeswalkerBoard(attacker, 3, step);
        Card bears = board.card(board.you, "Grizzly Bears");
        board.game.getCombat().addAttacker(bears, board.opponent);
        board.game.getCombat().setBlocked(bears, false);
        board.game.getAction().moveToGraveyard(board.other, null);
        assertTrue(Combat.isRemovedDefender(board.game.getCombat().getDefenderByAttacker(board.subject)));
        return board;
    }

    private static Board goneWalkerBoard(String attacker) {
        return goneWalkerBoard(attacker, PhaseType.COMBAT_DAMAGE);
    }

    @Test
    void aGoneWalkersTramplerBindsWithItsOwnTrampleOverPlaneswalkers() {
        Board board = goneWalkerBoard("Thrasta, Tempest's Roar");
        assertEquals(board.subject, TrampleOverAGonePlaneswalker.bind(board.game).orElseThrow().card("attacker"));
        assertTrue(NoDamageToAGonePlaneswalker.bind(board.game).isEmpty());
    }

    @Test
    void aGoneWalkersAttackerWithoutTheKeywordBindsTheNoDamageRule() {
        Board board = goneWalkerBoard("Craw Wurm");
        assertEquals(board.subject, NoDamageToAGonePlaneswalker.bind(board.game).orElseThrow().card("attacker"));
        assertTrue(TrampleOverAGonePlaneswalker.bind(board.game).isEmpty());
    }

    @Test
    void neitherGoneWalkerRuleBindsWhileThePlaneswalkerStays() {
        for (String attacker : List.of("Thrasta, Tempest's Roar", "Craw Wurm")) {
            Board board = planeswalkerBoard(attacker, 3);
            assertTrue(TrampleOverAGonePlaneswalker.bind(board.game).isEmpty(), attacker);
            assertTrue(NoDamageToAGonePlaneswalker.bind(board.game).isEmpty(), attacker);
        }
    }

    /** Alone in the step, the attacker would deal no damage in one of the two games, which then writes no combat record. */
    @Test
    void neitherGoneWalkerRuleBindsWithoutAnotherCombatantsDamage() {
        for (String attacker : List.of("Thrasta, Tempest's Roar", "Craw Wurm")) {
            Board board = goneWalkerBoard(attacker);
            Card bears = board.game.getCombat().getAttackers().stream()
                    .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
            board.game.getCombat().removeFromCombat(bears);
            assertTrue(TrampleOverAGonePlaneswalker.bind(board.game).isEmpty(), attacker);
            assertTrue(NoDamageToAGonePlaneswalker.bind(board.game).isEmpty(), attacker);
        }
    }

    @Test
    void neitherGoneWalkerRuleBindsAtAnotherBoundary() {
        assertTrue(TrampleOverAGonePlaneswalker.bind(goneWalkerBoard("Thrasta, Tempest's Roar",
                PhaseType.COMBAT_DECLARE_BLOCKERS).game).isEmpty());
        assertTrue(NoDamageToAGonePlaneswalker.bind(goneWalkerBoard("Craw Wurm",
                PhaseType.COMBAT_DECLARE_BLOCKERS).game).isEmpty());
    }

    @Test
    void aGoneWalkersTramplerBindsNothingForAGrantedKeywordOrADyingDreadmaw() {
        Board granted = goneWalkerBoard("Craw Wurm");
        granted.subject.addChangedCardKeywords(List.of("Trample:Planeswalker"), null, false,
                granted.game.getNextTimestamp(), null);
        assertTrue(TrampleOverAGonePlaneswalker.bind(granted.game).isEmpty(), "granted");
        Board damaged = goneWalkerBoard("Thrasta, Tempest's Roar");
        damaged.subject.setDamage(6);
        assertTrue(TrampleOverAGonePlaneswalker.bind(damaged.game).isEmpty(), "a 6/6 with 6 damage");
    }

    /** Ghalta's 8 marked damage would kill a 7/7 Thrasta. */
    @Test
    void aGoneWalkersAttackerBindsNothingWhenThrastaWouldDie() {
        Board board = goneWalkerBoard("Ghalta, Primal Hunger");
        assertTrue(NoDamageToAGonePlaneswalker.bind(board.game).isPresent());
        board.subject.setDamage(8);
        assertTrue(NoDamageToAGonePlaneswalker.bind(board.game).isEmpty());
    }

    /** The fork's {@code name} attacks the copy's own placeholder; nothing else changed but the new card. */
    private static void assertReplacedAttackingTheGoneWalker(Board board, Twin twin, String name) {
        Card replacement = twin.forkGame().getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(name)).findFirst().orElseThrow();
        Combat combat = twin.forkGame().getCombat();
        assertTrue(combat.isAttacking(replacement));
        assertTrue(Combat.isRemovedDefender(combat.getDefenderByAttacker(replacement)));
        assertEquals(board.opponent.getId(), combat.getDefenderPlayerByAttacker(replacement).getId());
        for (String change : changes(renamed(twin.real(), id(board.subject), id(replacement)), twin.forked())) {
            assertTrue(change.startsWith(id(replacement) + ".") || change.contains(id(replacement)),
                    "replacing changed " + change);
        }
    }

    @Test
    void aGoneWalkersTramplerForksAColossalDreadmaw() {
        Board board = goneWalkerBoard("Thrasta, Tempest's Roar");
        assertReplacedAttackingTheGoneWalker(board, board.twin(TrampleOverAGonePlaneswalker.rule()),
                TrampleOverAGonePlaneswalker.DREADMAW);
    }

    @Test
    void aGoneWalkersAttackerForksThrasta() {
        Board board = goneWalkerBoard("Craw Wurm");
        assertReplacedAttackingTheGoneWalker(board, board.twin(NoDamageToAGonePlaneswalker.rule()),
                NoDamageToAGonePlaneswalker.THRASTA);
    }

    // ── 702.118b, skulk: replace_card ───────────────────────────────────

    /** Furtive Homunculus attacks B, who controls {@code blocker}; no blocker declared yet. */
    private static Board skulkBoard(String blocker) {
        return skulkBoard(blocker, PhaseType.COMBAT_DECLARE_BLOCKERS);
    }

    /** {@link #skulkBoard(String)} as {@code step} begins. */
    private static Board skulkBoard(String blocker, PhaseType step) {
        Board board = new Board(step);
        board.other = board.card(board.you, "Furtive Homunculus");
        board.subject = board.card(board.opponent, blocker);
        Combat combat = new Combat(board.you);
        combat.addAttacker(board.other, board.opponent);
        board.game.getPhaseHandler().setCombat(combat);
        return board;
    }

    @Test
    void skulkBlockerBindsTheAttackerAndTheCreatureWithGreaterPower() {
        Board board = skulkBoard("Centaur Courser");
        Binding binding = SkulkBlocker.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("attacker"));
        assertEquals(board.subject, binding.card("blocker"));
    }

    @Test
    void skulkBlockerBindsNothingWhenTheCreatureMayBlock() {
        assertTrue(SkulkBlocker.bind(skulkBoard("Grizzly Bears").game).isEmpty());
    }

    /** A +1/+1 counter the Elves would carry leaves them at the attacker's power 2, not below it. */
    @Test
    void skulkBlockerBindsNothingWhenTheElvesWouldNotStayBelowTheAttacker() {
        Board board = skulkBoard("Centaur Courser");
        board.subject.setCounters(CounterEnumType.P1P1, 1);
        assertTrue(SkulkBlocker.bind(board.game).isEmpty());
    }

    @Test
    void skulkBlockerBindsNothingAtAnotherBoundary() {
        assertTrue(SkulkBlocker.bind(skulkBoard("Centaur Courser", PhaseType.COMBAT_DAMAGE).game).isEmpty());
    }

    /** Phantom Warrior cannot be blocked at all, the Elves no more than the Courser: no skulk to fork away. */
    @Test
    void skulkBlockerBindsNothingWithoutSkulk() {
        Board board = new Board(PhaseType.COMBAT_DECLARE_BLOCKERS);
        board.other = board.card(board.you, "Phantom Warrior");
        board.subject = board.card(board.opponent, "Centaur Courser");
        Combat combat = new Combat(board.you);
        combat.addAttacker(board.other, board.opponent);
        board.game.getPhaseHandler().setCombat(combat);
        assertFalse(CombatUtil.canBlock(board.other, board.subject));
        assertTrue(SkulkBlocker.bind(board.game).isEmpty());
    }

    /** An evasion besides skulk (flying), or any protection (from green, from creatures), would keep the Elves out too. */
    @Test
    void skulkBlockerBindsNothingForAnotherEvasionOrAProtection() {
        for (String keyword : List.of("Flying", "Protection from green", "Protection:Creature:creatures",
                "Protection from blue")) {
            Board board = skulkBoard("Centaur Courser");
            board.other.addChangedCardKeywords(List.of(keyword), null, false, board.game.getNextTimestamp(), null);
            assertTrue(SkulkBlocker.bind(board.game).isEmpty(), keyword);
        }
    }

    /** Protection from green keeps Llanowar Elves out of the block as well, the case the list is for. */
    @Test
    void protectionFromGreenWouldKeepTheElvesOut() {
        Board board = skulkBoard("Centaur Courser");
        board.other.addChangedCardKeywords(List.of("Protection from green"), null, false,
                board.game.getNextTimestamp(), null);
        Card elves = board.card(board.opponent, SkulkBlocker.ELVES);
        assertFalse(CombatUtil.canBlock(board.other, elves));
        assertTrue(SkulkBlocker.otherEvasions().contains(Keyword.PROTECTION));
    }

    @Test
    void skulkBlockerBindsNothingForATappedCreatureOrOneThatCannotBlock() {
        Board tapped = skulkBoard("Centaur Courser");
        tapped.subject.setTapped(true);
        assertTrue(SkulkBlocker.bind(tapped.game).isEmpty(), "tapped");
        assertTrue(SkulkBlocker.bind(skulkBoard("Craven Giant").game).isEmpty(), "can't block");
    }

    /** One marked damage would kill the 1/1 Elves. */
    @Test
    void skulkBlockerBindsNothingWhenTheElvesWouldDie() {
        Board board = skulkBoard("Centaur Courser");
        board.subject.setDamage(1);
        assertTrue(SkulkBlocker.bind(board.game).isEmpty());
    }

    /** Attacking a battle no player protects, the attacker has no defending player to block it. */
    @Test
    void skulkBlockerBindsNothingWithoutADefendingPlayer() {
        Board board = new Board(PhaseType.COMBAT_DECLARE_BLOCKERS);
        board.other = board.card(board.you, "Furtive Homunculus");
        board.subject = board.card(board.opponent, "Centaur Courser");
        Card battle = board.card(board.opponent, "Invasion of Amonkhet");
        battle.setProtectingPlayer(board.opponent);
        Combat combat = new Combat(board.you);
        combat.addAttacker(board.other, battle);
        board.game.getPhaseHandler().setCombat(combat);
        battle.setProtectingPlayer(null);
        assertNull(combat.getDefenderPlayerByAttacker(board.other));
        assertTrue(SkulkBlocker.bind(board.game).isEmpty());
    }

    @Test
    void skulkBlockerForksLlanowarElvesAmongTheLegalBlockers() {
        Board board = skulkBoard("Centaur Courser");
        Twin twin = board.twin(SkulkBlocker.rule());
        Card elves = twin.forkGame().getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(SkulkBlocker.ELVES)).findFirst().orElseThrow();
        // The Elves' mana ability adds to what B can make untapped.
        Set<String> identity = Set.of("name", "colors", "subtypes", "mana_value", "mana_cost",
                "base_pt", "lines", "granted_temporary");
        for (String change : changes(renamed(twin.real(), id(board.subject), id(elves)), twin.forked())) {
            assertTrue(change.equals("P1.untapped_production") || change.startsWith(id(elves) + ".")
                            && identity.contains(change.substring(change.indexOf('.') + 1)),
                    "replacing changed " + change);
        }
        assertTrue(CombatUtil.canBlock(twin.fork().card("attacker"), elves));
        assertFalse(CombatUtil.canBlock(board.other, board.subject), "the live Courser still cannot block");
    }

    // ── 702.131a, ascend: remove_permanent ──────────────────────────────

    /** Nine Islands and Grizzly Bears of yours, Secrets of the Golden City in your hand, then {@code inHand}. */
    private static Board ascendBoard(String... inHand) {
        return ascendBoard(PhaseType.MAIN1, "Secrets of the Golden City", inHand);
    }

    /** {@link #ascendBoard(String...)} as {@code step} begins, with {@code spell} for Secrets. */
    private static Board ascendBoard(PhaseType step, String spell, String... inHand) {
        Board board = new Board(step);
        for (int i = 0; i < 9; i++) {
            board.card(board.you, "Island");
        }
        board.subject = board.card(board.you, "Grizzly Bears");
        board.other = TestBoard.inHand(board.game, board.you, spell);
        for (String name : inHand) {
            TestBoard.inHand(board.game, board.you, name);
        }
        return board;
    }

    @Test
    void ascendBlessingBindsASpareIslandAtTenPermanents() {
        Board board = ascendBoard();
        Binding binding = AscendBlessing.bind(board.game).orElseThrow();
        assertEquals("Island", binding.card("permanent").getName());
        assertEquals(board.other, binding.card("spell"));
    }

    /** Three Islands the spell needs and seven Grizzly Bears: with no land to spare, a Bears, which makes no mana, goes. */
    @Test
    void ascendBlessingTakesAPermanentThatMakesNoManaWhenNoLandIsSpare() {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Island");
        }
        for (int i = 0; i < 7; i++) {
            board.card(board.you, "Grizzly Bears");
        }
        TestBoard.inHand(board.game, board.you, "Secrets of the Golden City");
        assertEquals("Grizzly Bears", AscendBlessing.bind(board.game).orElseThrow().card("permanent").getName());
    }

    @Test
    void ascendBlessingBindsNothingAtElevenPermanents() {
        Board board = ascendBoard();
        board.card(board.you, "Island");
        assertTrue(AscendBlessing.bind(board.game).isEmpty());
    }

    /** A land to play or a creature to cast would bring the fork's nine back to ten before the spell. */
    @Test
    void ascendBlessingBindsNothingWhenAPermanentOfItsOwnCouldMakeTen() {
        assertTrue(AscendBlessing.bind(ascendBoard("Island").game).isEmpty());
        assertTrue(AscendBlessing.bind(ascendBoard("Hill Giant").game).isEmpty());
    }

    @Test
    void ascendBlessingBindsNothingAtAnotherBoundary() {
        assertTrue(AscendBlessing.bind(ascendBoard(PhaseType.COMBAT_DECLARE_ATTACKERS,
                "Secrets of the Golden City").game).isEmpty());
    }

    /** The AI holds an ascend instant for the opponent's turn, after the fork stops at this one's end. */
    @Test
    void ascendBlessingBindsNothingForAnInstant() {
        assertTrue(AscendBlessing.bind(ascendBoard(PhaseType.MAIN1, "Expel from Orazca").game).isEmpty());
    }

    @Test
    void ascendBlessingBindsNothingWithTheBlessingAlready() {
        Board board = ascendBoard();
        board.you.setBlessing(true, "RIX");
        assertTrue(AscendBlessing.bind(board.game).isEmpty());
    }

    @Test
    void ascendBlessingBindsNothingWithoutTheLandsForTheSpell() {
        Board board = ascendBoard();
        board.you.getLandsInPlay().stream().limit(7).forEach(land -> land.setTapped(true));
        assertTrue(AscendBlessing.bind(board.game).isEmpty());
    }

    /** A tapped land goes before anything else: it plays no part in the turn. */
    @Test
    void ascendBlessingTakesATappedLandFirst() {
        Board board = ascendBoard();
        Card tapped = board.you.getLandsInPlay().get(0);
        tapped.setTapped(true);
        assertEquals(tapped, AscendBlessing.bind(board.game).orElseThrow().card("permanent"));
    }

    /** A spare untapped land goes before the Bears, whose absence would change the combat too. */
    @Test
    void ascendBlessingTakesASpareLandBeforeACreature() {
        Card taken = AscendBlessing.bind(ascendBoard().game).orElseThrow().card("permanent");
        assertTrue(taken.isLand(), "took " + taken);
    }

    /** Three Islands for the spell and seven Llanowar Elves: every permanent makes mana the spell may need. */
    @Test
    void ascendBlessingBindsNothingWhenNothingIsSpare() {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Island");
        }
        for (int i = 0; i < 7; i++) {
            board.card(board.you, "Llanowar Elves");
        }
        TestBoard.inHand(board.game, board.you, "Secrets of the Golden City");
        assertTrue(AscendBlessing.bind(board.game).isEmpty());
    }

    /** Forge counts a phased-out permanent toward ten (its battlefield zone), so the rule does too. */
    @Test
    void ascendBlessingCountsAPhasedOutPermanentAsForgeDoes() {
        Board board = ascendBoard();
        board.card(board.you, "Island").setPhasedOut(board.you);
        assertEquals(11, board.you.getZone(ZoneType.Battlefield).size());
        assertTrue(AscendBlessing.bind(board.game).isEmpty(), "eleven with the phased-out one");
        board.subject.getGame().getAction().exile(board.subject, null, null);
        Card taken = AscendBlessing.bind(board.game).orElseThrow().card("permanent");
        assertFalse(taken.isPhasedOut());
    }

    /** With every permanent an untapped land, only one beyond the spell's three goes. */
    @Test
    void ascendBlessingTakesAnUntappedLandOnlyBeyondTheSpellsCost() {
        Board board = ascendBoard();
        board.card(board.you, "Island");
        board.subject.getGame().getAction().exile(board.subject, null, null);
        Card land = AscendBlessing.bind(board.game).orElseThrow().card("permanent");
        assertTrue(land.isLand());
    }

    @Test
    void ascendBlessingForksNinePermanents() {
        Board board = ascendBoard();
        Card land = AscendBlessing.bind(board.game).orElseThrow().card("permanent");
        Twin twin = board.twin(AscendBlessing.rule());
        assertOnlyRemoved(twin, land, "permanent");
        assertEquals(9, twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Battlefield).size());
    }

    // ── 702.170c, a card becoming plotted: move_card ────────────────────

    /** A Forest, a Plains and an Island of yours, Kellan Joins Up in your hand, then {@code inHand}. */
    private static Board kellanBoard(String... inHand) {
        return kellanBoard(PhaseType.MAIN1, inHand);
    }

    /** {@link #kellanBoard(String...)} as {@code step} begins. */
    private static Board kellanBoard(PhaseType step, String... inHand) {
        Board board = new Board(step);
        for (String land : List.of("Forest", "Plains", "Island")) {
            board.card(board.you, land);
        }
        board.other = TestBoard.inHand(board.game, board.you, PlottedFromHand.KELLAN);
        for (String name : inHand) {
            Card card = TestBoard.inHand(board.game, board.you, name);
            if (board.subject == null) {
                board.subject = card;
            }
        }
        return board;
    }

    @Test
    void plottedFromHandBindsTheOneCardTheTriggerCouldExile() {
        Board board = kellanBoard("Grizzly Bears", "Hill Giant", "Forest");
        Binding binding = PlottedFromHand.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("spell"));
    }

    @Test
    void plottedFromHandBindsNothingWithTwoCardsTheTriggerCouldExile() {
        assertTrue(PlottedFromHand.bind(kellanBoard("Grizzly Bears", "Shock").game).isEmpty());
    }

    @Test
    void plottedFromHandBindsNothingWithoutTheLandsForKellan() {
        Board board = kellanBoard("Grizzly Bears");
        board.you.getLandsInPlay().forEach(land -> land.setTapped(true));
        assertTrue(PlottedFromHand.bind(board.game).isEmpty());
    }

    @Test
    void plottedFromHandBindsNothingAtAnotherBoundary() {
        assertTrue(PlottedFromHand.bind(kellanBoard(PhaseType.COMBAT_DECLARE_ATTACKERS, "Grizzly Bears").game)
                .isEmpty());
    }

    /**
     * Five lands cast Kellan and the Bears both: the AI casts the Bears
     * first, and Kellan's trigger finds nothing to exile on either side.
     * Four leave one of them uncast.
     */
    @Test
    void plottedFromHandBindsNothingWhenBothCanBeCast() {
        Board board = kellanBoard("Grizzly Bears");
        board.card(board.you, "Forest");
        assertTrue(PlottedFromHand.bind(board.game).isPresent(), "four lands");
        board.card(board.you, "Forest");
        assertTrue(PlottedFromHand.bind(board.game).isEmpty(), "five lands");
    }

    @Test
    void plottedFromHandForksAHillGiantInTheCardsPlace() {
        Board board = kellanBoard("Grizzly Bears");
        Twin twin = board.twin(PlottedFromHand.rule());
        Card giant = twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand).stream()
                .filter(card -> card.getName().equals(PlottedFromHand.HILL_GIANT)).findFirst().orElseThrow();
        assertOnlyChanged(twin, board.subject, giant);
    }

    // ── what makes a variant come undone (review C4, V3 #9-10) ──────────

    /**
     * C4 #2: a land in the holder's hand would be played as the turn goes on
     * and give back the mana a "one land short" variant takes, so the rule
     * does not bind; with the turn's land drop used it binds again.
     */
    private static void assertALandToPlayStopsTheBinding(Board board, DirectedRule rule) {
        assertTrue(rule.precondition(board.game).isPresent(), rule.rule() + ": the base binds");
        TestBoard.inHand(board.game, board.you, "Forest");
        assertTrue(rule.precondition(board.game).isEmpty(), rule.rule() + ": binds with a land to play");
        board.you.setLandsPlayedThisTurn(1);
        assertTrue(rule.precondition(board.game).isPresent(), rule.rule() + ": the land drop is used");
    }

    @Test
    void fusedCostBindsNothingWithALandToPlay() {
        assertALandToPlayStopsTheBinding(fuseBoard(3), FusedCost.rule());
    }

    @Test
    void prototypeBindsNothingWithALandToPlay() {
        assertALandToPlayStopsTheBinding(automatonBoard(3), Prototype.rule());
    }

    @Test
    void mayhemBindsNothingWithALandToPlay() {
        assertALandToPlayStopsTheBinding(boltBoard(true, 2), Mayhem.rule());
    }

    @Test
    void morphFaceUpBindsNothingWithALandToPlay() {
        assertALandToPlayStopsTheBinding(angelBoard(4), MorphFaceUp.rule());
    }

    @Test
    void mutagenTokenBindsNothingWithALandToPlay() {
        assertALandToPlayStopsTheBinding(mutagenBoard(false), MutagenToken.rule());
    }

    @Test
    void sorcererRoleBindsNothingWithALandToPlay() {
        assertALandToPlayStopsTheBinding(vendorBoard(1), SorcererRole.rule());
    }

    @Test
    void improviseBindsNothingWithALandToPlay() {
        assertALandToPlayStopsTheBinding(engineerBoard(3), Improvise.rule());
    }

    /** A modal double-faced card whose back is a land is a land to play too. */
    @Test
    void aModalCardWithALandBackIsALandToPlay() {
        Board board = fuseBoard(3);
        TestBoard.inHand(board.game, board.you, "Turntimber Symbiosis");
        assertTrue(FusedCost.bind(board.game).isEmpty());
    }

    /**
     * C4 #3: Lich's Mirror's fork is taken only where its loss can still
     * happen in the turn the fork plays: not as the untap step begins, nor at
     * the end step or cleanup, after the turn's last draw and damage.
     */
    @Test
    void gameLossReplacedBindsNothingOutsideTheTurnsDecisiveSteps() {
        for (PhaseType phase : List.of(PhaseType.UNTAP, PhaseType.END_OF_TURN, PhaseType.CLEANUP)) {
            Board board = mirrorBoard(1, "Plains", "Plains");
            board.game.getPhaseHandler().devModeSet(phase, board.you);
            assertTrue(GameLossReplaced.bind(board.game).isEmpty(), phase.toString());
        }
        for (PhaseType phase : List.of(PhaseType.UPKEEP, PhaseType.DRAW, PhaseType.MAIN1,
                PhaseType.COMBAT_DECLARE_ATTACKERS, PhaseType.MAIN2)) {
            Board board = mirrorBoard(1, "Plains", "Plains");
            board.game.getPhaseHandler().devModeSet(phase, board.you);
            assertTrue(GameLossReplaced.bind(board.game).isPresent(), phase.toString());
        }
    }

    /** C4 #4: a second permanent doing the same thing would still do it once the bound one is removed. */
    @Test
    void gameLossReplacedBindsNothingWithASecondReplacingPermanent() {
        Board board = mirrorBoard(1, "Plains", "Plains");
        board.card(board.you, "Lich's Mirror");
        assertTrue(GameLossReplaced.bind(board.game).isEmpty());
    }

    /** Platinum Angel stops the loss too, by a static rather than a replacement. */
    @Test
    void gameLossReplacedBindsNothingForAPlayerWhoCannotLose() {
        Board board = mirrorBoard(1, "Plains", "Plains");
        board.card(board.you, "Platinum Angel");
        board.game.getAction().checkStaticAbilities();
        assertTrue(GameLossReplaced.bind(board.game).isEmpty());
    }

    @Test
    void graveyardReplacementBindsNothingWithASecondReplacingPermanent() {
        Board board = restInPeaceBoard("Rest in Peace");
        board.card(board.opponent, "Leyline of the Void");
        assertTrue(GraveyardReplacement.bind(board.game).isEmpty());
    }

    @Test
    void cantEnterBattlefieldBindsNothingWithASecondStoppingPermanent() {
        Board board = cageBoard(true);
        board.card(board.opponent, "Grafdigger's Cage");
        assertTrue(CantEnterBattlefield.bind(board.game).isEmpty());
    }

    @Test
    void limitedSearchBindsNothingWithASecondLimitingPermanent() {
        Board board = mindcensorBoard(true);
        board.card(board.you, "Aven Mindcensor");
        assertTrue(LimitedSearch.bind(board.game).isEmpty());
    }

    /**
     * C4 #4: a keyword the bound creature has from an Equipment stays on the
     * card {@code replace_card} puts in its place, since the edit relinks the
     * attachment; so does one a resolved effect gave it.
     */
    /** Sword of Fire and Ice (protection from red and from blue) on {@code creature}. */
    private static void equipSword(Board board, Card creature) {
        Card sword = board.card(creature.getController(), "Sword of Fire and Ice");
        sword.attachToEntity(creature, null);
        board.game.getAction().checkStaticAbilities();
        assertTrue(creature.hasKeyword(Keyword.PROTECTION));
    }

    @Test
    void sharedKeywordsBindsNothingWhenTheKeywordComesFromAnAttachment() {
        Board board = concertedBoard("Grizzly Bears", "Hill Giant");
        equipSword(board, board.subject);
        assertTrue(SharedKeywords.bind(board.game).isEmpty());
    }

    @Test
    void sharedKeywordsBindsNothingWhenTheKeywordComesFromAResolvedEffect() {
        Board board = concertedBoard("Grizzly Bears", "Hill Giant");
        KeywordEdit.add(board.game, board.subject, "First Strike");
        assertTrue(board.subject.hasFirstStrike());
        assertTrue(SharedKeywords.bind(board.game).isEmpty());
    }

    @Test
    void copyForEachTargetBindsNothingWhenTheProtectionComesFromAnAttachment() {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 6; i++) {
            board.card(board.you, "Mountain");
        }
        TestBoard.inHand(board.game, board.you, "Radiate");
        TestBoard.inHand(board.game, board.you, "Shock");
        board.subject = board.card(board.opponent, "Grizzly Bears");
        equipSword(board, board.subject);
        assertTrue(CopyForEachTarget.bind(board.game).isEmpty());
    }

    /**
     * C4 #5: a sorcery in the non-active player's hand cannot be cast this
     * turn, so the fork would be spent on nothing. An instant can
     * ({@code toughnessZeroBindsTheCreatureTheSpellWouldKill}'s Disfigure,
     * {@code limitedSearchBindsTheLimitingPermanent}'s Vampiric Tutor).
     */
    @Test
    void toughnessZeroBindsNothingForASorceryInTheNonActiveHand() {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Drudge Skeletons");
        for (int i = 0; i < 3; i++) {
            board.card(board.opponent, "Swamp");
        }
        Card fatigue = TestBoard.inHand(board.game, board.opponent, "Crippling Fatigue");
        assertEquals(2, ToughnessZero.toughnessLoss(fatigue));
        assertTrue(ToughnessZero.bind(board.game).isEmpty());

        Board instant = new Board(PhaseType.MAIN1);
        instant.card(instant.you, "Drudge Skeletons");
        instant.card(instant.opponent, "Swamp");
        TestBoard.inHand(instant.game, instant.opponent, "Disfigure");
        assertTrue(ToughnessZero.bind(instant.game).isPresent(), "Disfigure is an instant");
    }

    @Test
    void smallLibraryShuffleBindsNothingForASorceryInTheNonActiveHand() {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Cosi's Trickster");
        board.card(board.opponent, "Swamp");
        board.card(board.opponent, "Swamp");
        TestBoard.inHand(board.game, board.opponent, "Demonic Tutor");
        board.library(board.opponent, "Forest", "Grizzly Bears");
        assertTrue(SmallLibraryShuffle.bind(board.game).isEmpty());
    }

    @Test
    void limitedSearchBindsNothingForASorceryInTheNonActiveHand() {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Aven Mindcensor");
        board.card(board.you, "Archivist of Oghma");
        board.card(board.opponent, "Swamp");
        board.card(board.opponent, "Swamp");
        TestBoard.inHand(board.game, board.opponent, "Demonic Tutor");
        assertTrue(LimitedSearch.bind(board.game).isEmpty());
        // Vampiric Tutor, an instant, binds there (limitedSearchBindsTheLimitingPermanent).
    }

    /**
     * C4 #1, V3 #9 (b): {@code replace_card} carries the permanent's marked
     * damage and counters to the card put in its place, so a smaller card
     * dies of them at the fork's first state-based check.
     */
    @Test
    void simultaneousDamageBindsNothingWhenTheBlockerCarriesDamage() {
        Board board = blockedBoard("Grizzly Bears");
        board.subject.setDamage(1);
        assertTrue(SimultaneousDamage.bind(board.game).isEmpty(), "Youthful Knight is a 2/1");
    }

    @Test
    void shuffleAndPlaceOnTopBindsNothingWhenTheColossusCarriesDamage() {
        Board board = barterBoard("Darksteel Colossus", "Gravebane Zombie");
        board.subject.setDamage(3);
        assertTrue(ShuffleAndPlaceOnTop.bind(board.game).isEmpty());
    }

    @Test
    void copyForEachTargetBindsNothingWhenTheCreatureCarriesAMinusCounter() {
        Board board = radiateBoard("Shock");
        board.subject.setCounters(CounterEnumType.M1M1, 3);
        assertTrue(CopyForEachTarget.bind(board.game).isEmpty(), "Hill Giant is a 3/3");
    }

    @Test
    void textBoxExchangeBindsNothingWhenTheCreatureCarriesDamage() {
        Board board = exchangeBoard("Giant Spider");
        assertTrue(TextBoxExchange.bind(board.game).isPresent(), "Giant Spider has reach");
        board.subject.setDamage(3);
        assertTrue(TextBoxExchange.bind(board.game).isEmpty(), "Hill Giant is a 3/3");
    }

    /**
     * The precondition asks whether the card put in the permanent's place
     * dies carrying what the edit carries: 1 damage on a Giant Spider (2/4)
     * leaves the Hill Giant (3/3) put in its place alive, so the board binds.
     */
    @Test
    void textBoxExchangeBindsWhenTheReplacingCardSurvivesTheCarriedDamage() {
        Board board = exchangeBoard("Giant Spider");
        board.subject.setDamage(1);
        assertTrue(TextBoxExchange.bind(board.game).isPresent(), "Hill Giant lives with 1 damage");
        Twin twin = board.twin(TextBoxExchange.rule());
        assertEquals(1, twin.fork().card("creature").getDamage(), "the edit carried the damage");
    }

    /**
     * A resolved -3/-3 until end of turn leaves Giant Spider at 1 toughness,
     * alive, and would leave the Hill Giant put in its place at 0: the board
     * binds nothing, so no fork is spent on the edit's refusal.
     */
    @Test
    void textBoxExchangeBindsNothingWhenACarriedBoostKillsTheReplacingCard() {
        Board board = exchangeBoard("Giant Spider");
        board.subject.addPTBoost(-3, -3, board.game.getNextTimestamp(), 0);
        assertEquals(1, board.subject.getNetToughness());
        assertTrue(TextBoxExchange.bind(board.game).isEmpty(), "Hill Giant would be a 0/0");
    }

    /** Serra Angel (4/4) lives with 3 damage; the Hill Giant (3/3) put in its place would not. */
    @Test
    void sharedKeywordsBindsNothingWhenTheHolderCarriesDamage() {
        Board board = concertedBoard("Serra Angel", "Grizzly Bears");
        assertTrue(SharedKeywords.bind(board.game).isPresent(), "Serra Angel has flying and vigilance");
        board.subject.setDamage(3);
        assertTrue(SharedKeywords.bind(board.game).isEmpty(), "Hill Giant is a 3/3");
    }

    /** C4 #6: a +1/+1 counter set beside a -1/-1 counter is annihilated with it (704.5q). */
    @Test
    void typeSettingAuraBindsNothingOnACreatureWithAMinusCounter() {
        Board board = dryadsBoard(3);
        board.subject.setCounters(CounterEnumType.M1M1, 1);
        assertTrue(TypeSettingAura.bind(board.game).isEmpty());
    }

    /** C4 #6: taking the counter away kills a creature whose toughness rests on it. */
    @Test
    void unleashBindsNothingWhenTheCreatureLivesByItsCounter() {
        Board board = unleashBoard("Rakdos Cackler", 1);
        board.subject.setDamage(2);
        assertTrue(UnleashCantBlock.bind(board.game).isEmpty(), "a 2/2 Cackler with 2 damage dies at 1 toughness");
    }

    /** C4 #7: Leyline of the Void exiles only opponents' cards, so it does not replace its controller's sacrifice. */
    @Test
    void replacedSacrificeCostBindsNothingWhenTheReplacementSparesTheSacrifice() {
        Board board = bombardmentBoard(false);
        board.card(board.you, "Leyline of the Void");
        assertTrue(ReplacedSacrificeCost.bind(board.game).isEmpty());
        Board opposing = bombardmentBoard(false);
        opposing.card(opposing.opponent, "Leyline of the Void");
        assertTrue(ReplacedSacrificeCost.bind(opposing.game).isPresent());
    }

    /**
     * C4 #8: the defender cut to one life under the most anyone else has
     * must not be left in reach of the attack the fork's point is.
     */
    @Test
    void dethroneBindsNothingWhenTheCutWouldLeaveTheDefenderInReach() {
        Board board = dethroneBoard(4);
        board.you.setLife(4, null);
        assertTrue(Dethrone.bind(board.game).isEmpty(), "3 life against Enraged Revolutionary's 2 power and its dethrone counter");
        Board safe = dethroneBoard(20);
        safe.you.setLife(20, null);
        assertTrue(Dethrone.bind(safe.game).isPresent());
    }

    // ── what the twin itself catches ────────────────────────────────────

    /** A rule of one variant on the active player, for the twin's own tests. */
    private static DirectedRule onePlayerRule(ForkOperation operation,
                                              Function<Binding, BoardEdit> edit) {
        return DirectedRule.of("0.0", "0.0-1",
                live -> Optional.of(Binding.empty().with("player", Boards.active(live))),
                "v", new Application(operation, edit));
    }

    /** TC3: the first check ending the fork's game moves no card, and the twin still sees it. */
    @Test
    void theTwinCatchesAnEditTheFirstCheckLosesAPlayerWith() {
        Board board = new Board(PhaseType.MAIN1);
        DirectedRule zeroLife = onePlayerRule(ForkOperation.LIFE, fork -> new LifeOperation(fork.player("player"), 0));
        AssertionError error = assertThrows(AssertionError.class,
                () -> board.twin(zeroLife));
        assertTrue(error.getMessage().contains("the first state-based check"), error.getMessage());
    }

    @Test
    void theTwinCatchesTenPoisonCounters() {
        Board board = new Board(PhaseType.MAIN1);
        DirectedRule poison = onePlayerRule(ForkOperation.COUNTERS,
                fork -> new PoisonCounters(fork.player("player")));
        AssertionError error = assertThrows(AssertionError.class,
                () -> board.twin(poison));
        assertTrue(error.getMessage().contains("the first state-based check"), error.getMessage());
    }

    /**
     * Only a player's Keyword Effects card is the engine's to tidy at the
     * first check: a change naming another effect card or an emblem, and a
     * floating effect gone that no Keyword Effects card made, stay changes.
     */
    @Test
    void theTwinForgivesOnlyAKeywordEffectsCardTheCheckTidies() {
        Map<String, Object> edited = Map.of("global", Map.of("floating_effects",
                List.of(Map.of("source", "E3"), Map.of("source", "E9"))));
        Map<String, Object> keywordCardGone = Map.of("global", Map.of("floating_effects",
                List.of(Map.of("source", "E9"))));
        Map<String, Object> otherGone = Map.of("global", Map.of("floating_effects",
                List.of(Map.of("source", "E3"))));
        Set<String> tidied = Set.of("E3");
        assertEquals(Set.of(), Board.withoutTidied(
                Set.of("-E3", "E3.lines", "global.floating_effects"), tidied, edited, keywordCardGone));
        assertEquals(Set.of("-E9", "global.floating_effects"), Board.withoutTidied(
                Set.of("-E9", "global.floating_effects"), tidied, edited, otherGone));
    }

    /** A +1/+1 counter beside a -1/-1 one moves no card, but the check annihilates the pair. */
    @Test
    void theTwinCatchesCountersTheFirstCheckAnnihilates() {
        Board board = new Board(PhaseType.MAIN1);
        board.subject = board.card(board.you, "Hill Giant");
        board.subject.setCounters(CounterEnumType.M1M1, 1);
        Card giant = board.subject;
        DirectedRule counter = DirectedRule.of("0.0", "0.0-1",
                live -> Optional.of(Binding.empty().with("creature", giant)), "v",
                new Application(ForkOperation.COUNTERS,
                        fork -> new CountersOperation(fork.card("creature"), CounterEnumType.P1P1, 1)));
        AssertionError error = assertThrows(AssertionError.class,
                () -> board.twin(counter));
        assertTrue(error.getMessage().contains("changed the edited board"), error.getMessage());
    }

    /** Ten poison counters on a player, as a {@code counters} edit for the twin's own test. */
    private record PoisonCounters(Player player) implements BoardEdit {
        @Override
        public ForkOperation operation() {
            return ForkOperation.COUNTERS;
        }

        @Override
        public void apply(Game fork) {
            player.setPoisonCounters(10, null);
            BoardEdit.settle(fork);
        }
    }

    // ── shared checks ───────────────────────────────────────────────────

    /**
     * The fork differs from the live board by {@code removed} alone: it is
     * gone, with its links, the battlefield's ranks after it and the mana its
     * controller has untapped.
     */
    private static void assertOnlyRemoved(Twin twin, Card removed, String role) {
        String gone = id(removed);
        assertTrue(twin.changes().contains("-" + gone), twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(change.equals("-" + gone) || change.endsWith(".order_rank")
                            || change.endsWith(".untapped_production")
                            || change.startsWith("-link ") && change.contains("=" + gone),
                    "removing changed " + change + "; all: " + twin.changes());
        }
        assertNull(twin.fork().card(role).getZone());
    }

    /**
     * The fork differs from the live board by a new {@code name} under
     * {@code player} alone: it is there, with its links.
     */
    private static void assertOnlyAdded(Twin twin, Player player, String name) {
        Card card = player.getCardsIn(ZoneType.Battlefield).stream()
                .filter(c -> c.getName().equals(name)).findFirst().orElseThrow();
        String added = id(card);
        assertTrue(twin.changes().contains("+" + added), twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(change.equals("+" + added) || change.startsWith("+link ") && change.contains("=" + added),
                    "adding changed " + change + "; all: " + twin.changes());
        }
    }

    /**
     * The fork differs from the live board by a {@code name} in
     * {@code replaced}'s place alone: read with the new card's id wherever the
     * old one's stood, what is left differs only in what makes the card that
     * card, and in the {@code alsoChanged} members the variant names besides
     * (a token's {@code token_script_id}, where a card takes its place).
     * Returns the new card.
     */
    private static Card assertOnlyReplaced(Twin twin, Card replaced, String name, String... alsoChanged) {
        Card card = twin.forkGame().getCardsIn(ZoneType.Battlefield).stream()
                .filter(c -> c.getName().equals(name) && twin.changes().contains("+" + id(c)))
                .findFirst().orElseThrow(() -> new AssertionError("no new " + name + "; all: " + twin.changes()));
        Set<String> left = changes(renamed(twin.real(), id(replaced), id(card)), twin.forked());
        Set<String> identity = new TreeSet<>(List.of("name", "colors", "types", "subtypes", "mana_value",
                "mana_cost", "base_pt", "lines", "granted_temporary"));
        identity.addAll(List.of(alsoChanged));
        for (String change : left) {
            assertTrue(change.startsWith(id(card) + ".")
                            && identity.contains(change.substring(change.indexOf('.') + 1)),
                    "replacing changed " + change + "; all: " + left);
        }
        return card;
    }

    /**
     * The fork differs from the live board by {@code fresh} in {@code old}'s
     * place in a hand or library alone, with their links.
     */
    private static void assertOnlySwapped(Twin twin, Card old, Card fresh) {
        String gone = id(old);
        String added = id(fresh);
        assertTrue(twin.changes().containsAll(Set.of("-" + gone, "+" + added)), twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(change.equals("-" + gone) || change.equals("+" + added)
                            || change.startsWith("-link ") && change.contains("=" + gone)
                            || change.startsWith("+link ") && change.contains("=" + added),
                    "replacing changed " + change + "; all: " + twin.changes());
        }
    }

    /**
     * The fork differs from the live board by {@code gone} leaving the
     * snapshot and {@code added} joining it alone (the same card for a move),
     * with their links and own members, and the player members named.
     */
    private static void assertOnlyChanged(Twin twin, Card gone, Card added, String... members) {
        Set<String> ids = new TreeSet<>(List.of(id(gone), id(added)));
        java.util.function.Predicate<String> own = change -> ids.stream().anyMatch(
                card -> change.equals("-" + card) || change.equals("+" + card)
                        || change.startsWith(card + ".")
                        || change.startsWith("-link ") && change.contains("=" + card)
                        || change.startsWith("+link ") && change.contains("=" + card));
        // Something of the edited cards changed: an edit that did nothing
        // passes every "only" check below.
        assertTrue(twin.changes().stream().anyMatch(own),
                "the edit changed nothing of " + ids + "; all: " + twin.changes());
        for (String change : twin.changes()) {
            assertTrue(own.test(change) || List.of(members).contains(change),
                    "changed " + change + "; all: " + twin.changes());
        }
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

        /** {@code names} at the bottom of {@code owner}'s library in order, so the first listed is the topmost of them. */
        List<Card> library(Player owner, String... names) {
            List<Card> cards = new ArrayList<>();
            for (String name : names) {
                cards.add(game.getAction().moveToLibrary(TestBoard.card(game, owner, name), -1, null, null));
            }
            return cards;
        }

        /**
         * Copy the game as a directed fork does, score-check the copy as every
         * fork is checked before it is changed, map the precondition's
         * binding into the copy and apply the rule's one variant, then compare
         * the copy with the live board. The live board is left as it was.
         *
         * <p>The edited copy is then put through the state-based check the
         * fork's first priority makes, which must move no card: an edit that
         * the check undoes (a creature dying of the toughness it was given,
         * an Aura falling off, the legend rule) would fork a board the rule's
         * variant does not describe.
         */
        Twin twin(DirectedRule rule) {
            assertEquals(1, rule.variants().size(), rule.rule() + " registers one variant");
            Binding binding = rule.precondition(game).orElseThrow(
                    () -> new AssertionError(rule.rule() + ": the precondition does not hold"));
            Map<String, Object> real = BoardStates.of(game);
            GameCopier copier = new GameCopier(game);
            Game fork = copier.makeCopy();
            assertEquals(Set.of(), changes(real, BoardStates.of(fork)), "the unedited copy differs");
            assertTrue(TestForks.scoreChecks(game, copier, fork), rule.rule() + ": the copy fails its score check");

            Binding forked = binding.inFork(copier);
            rule.variants().values().iterator().next().edit(forked).apply(fork);

            Map<String, Object> edited = BoardStates.of(fork);
            assertEquals(real, BoardStates.of(game), "the edit reached the live game");
            Map<String, String> placed = placed(fork);
            Set<String> tidied = keywordEffectCards(fork);
            fork.getAction().checkStateEffects(true);
            assertEquals(placed, placed(fork),
                    rule.rule() + ": the first state-based check moved a card of the edited board");
            Map<String, Object> checked = BoardStates.of(fork);
            assertEquals(Set.of(), withoutTidied(changes(edited, checked), tidied, edited, checked),
                    rule.rule() + ": the first state-based check changed the edited board");
            assertFalse(fork.isGameOver(), rule.rule() + ": the first state-based check ended the fork's game");
            for (Player player : fork.getRegisteredPlayers()) {
                assertFalse(player.hasLost(), rule.rule() + ": the first state-based check lost " + player);
            }
            assertEquals(real, BoardStates.of(game), "the check reached the live game");
            return new Twin(changes(real, edited), real, edited, forked, fork);
        }

        /**
         * The effect card the engine tidies away at a state-based check once
         * its source is gone: a player's {@code Keyword Effects} card
         * ({@code Player.getKeywordCard}), which holds the keywords statics
         * give the player. No other effect card, emblem or command-zone card
         * leaves at a check, and a change to one fails the twin.
         */
        static final String KEYWORD_EFFECTS = "Keyword Effects";

        /** Whether {@code card} is a player's {@link #KEYWORD_EFFECTS} card. */
        private static boolean keywordEffectCard(Card card) {
            return card.isImmutable() && !card.isEmblem() && KEYWORD_EFFECTS.equals(card.getName());
        }

        /** The snapshot ids of the players' {@link #KEYWORD_EFFECTS} cards in {@code game} ({@code E<id>}). */
        private static Set<String> keywordEffectCards(Game game) {
            Set<String> ids = new TreeSet<>();
            for (Card card : game.getCardsInGame()) {
                if (keywordEffectCard(card)) {
                    ids.add("E" + card.getId());
                }
            }
            return ids;
        }

        /** Whether a change names one of {@code ids}: the entity itself, its fields, or a link to it. */
        private static boolean names(String change, Set<String> ids) {
            return ids.stream().anyMatch(id -> change.equals("-" + id) || change.equals("+" + id)
                    || change.startsWith(id + ".") || change.contains("=" + id + ",")
                    || change.contains("=" + id + "}"));
        }

        /**
         * {@code changes} without what names one of the {@code tidied}
         * {@link #KEYWORD_EFFECTS} cards, and without {@code global.floating_effects}
         * when the two snapshots' floating effects differ only by entries
         * naming one of them.
         */
        static Set<String> withoutTidied(Set<String> changes, Set<String> tidied,
                                         Map<String, Object> edited, Map<String, Object> checked) {
            Set<String> kept = new TreeSet<>();
            for (String change : changes) {
                if (!names(change, tidied)) {
                    kept.add(change);
                }
            }
            if (kept.contains("global.floating_effects")
                    && floatingWithout(edited, tidied).equals(floatingWithout(checked, tidied))) {
                kept.remove("global.floating_effects");
            }
            return kept;
        }

        /** The snapshot's floating effects, as text, less each entry naming one of {@code ids}. */
        private static List<String> floatingWithout(Map<String, Object> snapshot, Set<String> ids) {
            Object global = snapshot.get("global");
            Object floating = global instanceof Map<?, ?> map ? map.get("floating_effects") : null;
            List<String> kept = new ArrayList<>();
            if (floating instanceof List<?> entries) {
                for (Object entry : entries) {
                    String text = String.valueOf(entry);
                    if (ids.stream().noneMatch(id -> text.matches(".*\\b" + id + "\\b.*"))) {
                        kept.add(text);
                    }
                }
            }
            return kept;
        }

        /**
         * Every card of {@code game} by id, to its zone and what it is
         * attached to; not a player's {@link #KEYWORD_EFFECTS} card, which the
         * check tidies away once its source is gone.
         */
        private static Map<String, String> placed(Game game) {
            Map<String, String> placed = new java.util.TreeMap<>();
            for (Card card : game.getCardsInGame()) {
                if (keywordEffectCard(card)) {
                    continue;
                }
                placed.put(card.getName() + "#" + card.getId(),
                        card.getZone() + " on " + card.getEntityAttachedTo());
            }
            return placed;
        }
    }
}
