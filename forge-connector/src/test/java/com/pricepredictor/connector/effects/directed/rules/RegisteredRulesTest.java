package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.BoardStates;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.DirectedRules;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.KeywordEdit;
import forge.StaticData;
import forge.ai.simulation.GameCopier;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardFactory;
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

    // ── 121.4, a draw from an empty library: move_card ──────────────────

    private static Board divinationBoard(int library) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Island");
        }
        board.other = TestBoard.inHand(board.game, board.you, "Divination");
        for (int i = 0; i < library; i++) {
            board.subject = board.game.getAction().moveToLibrary(
                    TestBoard.card(board.game, board.you, "Forest"), -1, null, null);
        }
        return board;
    }

    @Test
    void drawFromEmptyLibraryBindsTheLastCardAndTheDrawSpell() {
        Board board = divinationBoard(1);
        Binding binding = DrawFromEmptyLibrary.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("spell"));
    }

    @Test
    void drawFromEmptyLibraryBindsNothingWithTwoCardsLeft() {
        assertTrue(DrawFromEmptyLibrary.bind(divinationBoard(2).game).isEmpty());
    }

    @Test
    void drawFromEmptyLibraryForksAnEmptyLibrary() {
        Board board = divinationBoard(1);
        Twin twin = board.twin(DrawFromEmptyLibrary.rule());
        String forest = id(board.subject);
        assertTrue(twin.changes().containsAll(Set.of("P0.library", "P0.graveyard")), twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(Set.of("P0.library", "P0.graveyard").contains(change) || change.startsWith(forest + "."),
                    "moving changed " + change + "; all: " + twin.changes());
        }
        assertTrue(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library).isEmpty());
    }

    // ── 122.1g, a battle's last defense counter: counters ───────────────

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

    // ── 122.1h, a finality counter: counters ────────────────────────────

    private static Board finalityBoard(int finality) {
        Board board = new Board(PhaseType.MAIN1);
        board.subject = board.card(board.opponent, "Grizzly Bears");
        board.subject.setCounters(CounterEnumType.FINALITY, finality);
        return board;
    }

    @Test
    void finalityCounterBindsTheCreatureWithOne() {
        Board board = finalityBoard(1);
        assertEquals(board.subject, FinalityCounter.bind(board.game).orElseThrow().card("creature"));
    }

    @Test
    void finalityCounterBindsNothingWithoutOne() {
        assertTrue(FinalityCounter.bind(finalityBoard(0).game).isEmpty());
    }

    @Test
    void finalityCounterForksTheCreatureWithoutIt() {
        Board board = finalityBoard(1);
        Twin twin = board.twin(FinalityCounter.rule());
        String bears = id(board.subject);
        assertTrue(twin.changes().contains(bears + ".counters"), twin.changes().toString());
        assertTrue(Set.of(bears + ".counters", bears + ".lines").containsAll(twin.changes()),
                twin.changes().toString());
        assertEquals(0, twin.fork().card("creature").getCounters(CounterEnumType.FINALITY));
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

    // ── 505.5, a roll to visit Attractions: remove_permanent ────────────

    /** A's Balloon Stand, an Attraction: Forge files Attractions with the variant cards, not the common ones. */
    private static Board balloonBoard(PhaseType phase) {
        Board board = new Board(phase);
        Card stand = CardFactory.getCard(
                StaticData.instance().getVariantCards().getCard("Balloon Stand"), board.you, board.game);
        board.subject = board.game.getAction().moveToPlay(stand, board.you, null, null);
        board.subject.setGameTimestamp(board.game.getNextTimestamp());
        return board;
    }

    @Test
    void attractionVisitBindsTheAttractionAsThePrecombatMainPhaseBegins() {
        Board board = balloonBoard(PhaseType.MAIN1);
        assertEquals(board.subject, AttractionVisit.bind(board.game).orElseThrow().card("attraction"));
    }

    @Test
    void attractionVisitBindsNothingInTheSecondMainPhase() {
        assertTrue(AttractionVisit.bind(balloonBoard(PhaseType.MAIN2).game).isEmpty());
    }

    @Test
    void attractionVisitForksTheBoardWithoutTheAttraction() {
        Board board = balloonBoard(PhaseType.MAIN1);
        assertOnlyRemoved(board.twin(AttractionVisit.rule()), board.subject, "attraction");
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

    // ── 614.10a, a skipped upkeep's trigger: remove_permanent ───────────

    private static Board eonHubBoard(boolean arena) {
        Board board = new Board(PhaseType.UNTAP);
        board.subject = board.card(board.you, "Eon Hub");
        if (arena) {
            board.card(board.you, "Phyrexian Arena");
        }
        return board;
    }

    @Test
    void skippedUpkeepTriggerBindsTheSkippingPermanent() {
        Board board = eonHubBoard(true);
        assertEquals(board.subject, SkippedUpkeepTrigger.bind(board.game).orElseThrow().card("permanent"));
    }

    @Test
    void skippedUpkeepTriggerBindsNothingWithoutAnUpkeepTrigger() {
        assertTrue(SkippedUpkeepTrigger.bind(eonHubBoard(false).game).isEmpty());
    }

    @Test
    void skippedUpkeepTriggerForksTheBoardWithoutTheSkippingPermanent() {
        Board board = eonHubBoard(true);
        assertOnlyRemoved(board.twin(SkippedUpkeepTrigger.rule()), board.subject, "permanent");
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

    // ── 701.24f, two shuffles at once: replace_card ─────────────────────

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
    void simultaneousShufflesBindsOneOfTheTwoColossi() {
        Board board = barterBoard("Darksteel Colossus", "Darksteel Colossus");
        Card bound = SimultaneousShuffles.bind(board.game).orElseThrow().card("creature");
        assertEquals("Darksteel Colossus", bound.getName());
        assertEquals(board.opponent, bound.getController());
    }

    @Test
    void simultaneousShufflesBindsNothingWithOneColossus() {
        assertTrue(SimultaneousShuffles.bind(barterBoard("Darksteel Colossus", "Grizzly Bears").game).isEmpty());
    }

    @Test
    void simultaneousShufflesForksGrizzlyBearsInAColossussPlace() {
        Board board = barterBoard("Darksteel Colossus", "Darksteel Colossus");
        Card bound = SimultaneousShuffles.bind(board.game).orElseThrow().card("creature");
        Card bears = assertOnlyReplaced(board.twin(SimultaneousShuffles.rule()), bound, SimultaneousShuffles.BEARS);
        assertFalse(SimultaneousShuffles.shufflesIntoLibrary(bears));
    }

    // ── 701.24g, a shuffle and a put-on-top at once: replace_card ───────

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
        Card bears = assertOnlyReplaced(board.twin(ShuffleAndPlaceOnTop.rule()), board.subject,
                SimultaneousShuffles.BEARS);
        assertFalse(SimultaneousShuffles.shufflesIntoLibrary(bears));
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
     * card. Returns the new card.
     */
    private static Card assertOnlyReplaced(Twin twin, Card replaced, String name) {
        Card card = twin.forkGame().getCardsIn(ZoneType.Battlefield).stream()
                .filter(c -> c.getName().equals(name) && twin.changes().contains("+" + id(c)))
                .findFirst().orElseThrow(() -> new AssertionError("no new " + name + "; all: " + twin.changes()));
        Set<String> left = changes(renamed(twin.real(), id(replaced), id(card)), twin.forked());
        Set<String> identity = Set.of("name", "colors", "types", "subtypes", "mana_value", "mana_cost",
                "base_pt", "lines", "granted_temporary");
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
