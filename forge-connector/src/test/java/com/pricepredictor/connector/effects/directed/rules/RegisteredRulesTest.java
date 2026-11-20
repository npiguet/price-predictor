package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.BoardStates;
import com.pricepredictor.connector.effects.ForcedPlayRun;
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
 * variant's one fact (story 3, acceptance 3).
 *
 * <p>The twin comparison reads the board-only snapshot a record would carry,
 * member by member ({@code BoardChanges}), and checks first that the unedited
 * copy reads as the live board does, so what differs afterwards is the edit's.
 */
@ExtendWith(ForgeExtension.class)
class RegisteredRulesTest {

    // ── the registry ────────────────────────────────────────────────────

    /**
     * Every operation but {@code keyword} and {@code colour_type} is a
     * registered rule's: their rules (702.63c, 702.13b) were registered for
     * no listed run and left the registry. The protection forks edit through
     * {@code keyword}, and {@code ForkOperationsTest} checks each operation's
     * one fact.
     */
    @Test
    void theRegisteredRulesCoverEveryOperationButKeywordAndColourType() {
        Set<ForkOperation> covered = EnumSet.noneOf(ForkOperation.class);
        for (DirectedRule rule : DirectedRules.registered().rules()) {
            rule.variants().values().forEach(application -> covered.add(application.operation()));
        }
        assertEquals(EnumSet.complementOf(EnumSet.of(ForkOperation.KEYWORD, ForkOperation.COLOUR_TYPE)), covered);
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

    // ── 701.12g, a life exchange: life ──────────────────────────────────

    private static Board treeBoard(int opponentLife) {
        Board board = new Board(PhaseType.MAIN1);
        board.subject = board.card(board.you, "Tree of Perdition");
        board.opponent.setLife(opponentLife, null);
        return board;
    }

    @Test
    void lifeExchangeBindsTheTreeAndAnOpponentItWouldNotLower() {
        for (int life : List.of(7, 13)) {
            Board board = treeBoard(life);
            Binding binding = LifeExchange.bind(board.game).orElseThrow();
            assertEquals(board.subject, binding.card("permanent"));
            assertEquals(board.opponent, binding.player("opponent"));
        }
    }

    @Test
    void lifeExchangeBindsNothingWhenTheExchangeWouldAlreadyLowerTheLife() {
        assertTrue(LifeExchange.bind(treeBoard(20).game).isEmpty());
    }

    /** Under Platinum Emperion (its controller's life total can't change) the exchange does not happen. */
    @Test
    void lifeExchangeBindsNothingWhenTheOpponentCannotLoseLife() {
        Board board = treeBoard(7);
        board.card(board.opponent, "Platinum Emperion");
        assertTrue(LifeExchange.bind(board.game).isEmpty());
    }

    @Test
    void lifeExchangeForksTheOpponentAboveTheToughness() {
        Board board = treeBoard(7);
        Twin twin = board.twin(LifeExchange.rule());
        assertEquals(Set.of("P1.life"), twin.changes());
        assertEquals(20, twin.fork().player("opponent").getLife());
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
    void dethroneBindsTheCreatureAndTheOpponentWithMoreLifeThanAnyOther() {
        Board board = dethroneBoard(20);
        Binding binding = Dethrone.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("attacker"));
        assertEquals(board.opponent, binding.player("opponent"));
    }

    /** Tied already, the fork's cut would change nothing; with less life, dethrone would not fire. */
    @Test
    void dethroneBindsNothingAgainstAnOpponentWithNoMoreLifeThanAnother() {
        assertTrue(Dethrone.bind(dethroneBoard(15).game).isEmpty(), "tied");
        assertTrue(Dethrone.bind(dethroneBoard(14).game).isEmpty(), "less");
    }

    /**
     * You at 7 life beside Enraged Revolutionary (2 power, 1 more from its
     * dethrone counter) and a 2-power double striker: counted once the
     * reach is 5, below the 7 the cut opponent keeps, but the double striker
     * deals its 2 twice, and 7 is lethal.
     */
    @Test
    void dethroneCountsADoubleStrikersDamageTwice() {
        Board board = dethroneBoard(20);
        board.you.setLife(7, null);
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
    void dethroneForksTheOpponentTiedForTheMostLife() {
        Board board = dethroneBoard(20);
        Twin twin = board.twin(Dethrone.rule());
        assertEquals(Set.of("P1.life"), twin.changes());
        assertEquals(15, twin.fork().player("opponent").getLife());
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
        assertTrue(SimultaneousDamage.bind(blockedBoard(SimultaneousDamage.KNIGHT).game).isEmpty());
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
    void lifeTotalWinBindsTheSovereignBelowFortyLife() {
        Board board = sovereignBoard(39);
        Binding binding = LifeTotalWin.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("permanent"));
        assertEquals(board.you, binding.player("player"));
    }

    @Test
    void lifeTotalWinBindsNothingAtFortyLife() {
        assertTrue(LifeTotalWin.bind(sovereignBoard(40).game).isEmpty());
    }

    /** Platinum Emperion stops the life change the variant makes. */
    @Test
    void lifeTotalWinBindsNothingWhenTheLifeTotalCannotChange() {
        Board board = sovereignBoard(39);
        board.card(board.you, "Platinum Emperion");
        assertTrue(LifeTotalWin.bind(board.game).isEmpty());
    }

    @Test
    void lifeTotalWinForksThePlayerAtFortyLife() {
        Twin twin = sovereignBoard(12).twin(LifeTotalWin.rule());
        assertEquals(Set.of("P0.life"), twin.changes());
        assertEquals(40, twin.fork().player("player").getLife());
    }

    // ── 104.3d, ten poison counters: counters ───────────────────────────

    /** B at {@code poison} counters; A holds Ichor Rats over three Swamps. */
    private static Board poisonBoard(int poison) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Swamp");
        }
        board.other = TestBoard.inHand(board.game, board.you, "Ichor Rats");
        board.opponent.setCounters(CounterEnumType.POISON, poison, null, false);
        return board;
    }

    @Test
    void poisonLossBindsAnOpponentAtAnyPoisonBelowNineAndThePoisonSource() {
        Board board = poisonBoard(0);
        Binding binding = PoisonLoss.bind(board.game).orElseThrow();
        assertEquals(board.opponent, binding.player("player"));
        assertEquals(board.other, binding.card("card"));
        assertEquals(board.you, binding.player("caster"));
    }

    @Test
    void poisonLossBindsNothingWithoutAPoisonSourceToCast() {
        Board board = poisonBoard(0);
        board.you.getLandsInPlay().forEach(land -> land.setTapped(true));
        assertTrue(PoisonLoss.bind(board.game).isEmpty());
    }

    @Test
    void poisonLossBindsNothingOneCounterShort() {
        assertTrue(PoisonLoss.bind(poisonBoard(9).game).isEmpty());
    }

    /** The caster's own poison is no matter: the opponent's is the one the edit sets. */
    @Test
    void poisonLossBindsTheOpponentNotTheCaster() {
        Board board = poisonBoard(3);
        board.you.setCounters(CounterEnumType.POISON, 5, null, false);
        assertEquals(board.opponent, PoisonLoss.bind(board.game).orElseThrow().player("player"));
    }

    @Test
    void poisonLossForksThePlayerWithNineCounters() {
        Twin twin = poisonBoard(0).twin(PoisonLoss.rule());
        assertEquals(Set.of("P1.poison"), twin.changes());
        assertEquals(9, twin.fork().player("player").getPoisonCounters());
    }

    // ── 111.10v, a Mutagen token: add_permanent, inverted ───────────────

    /** A Mutagen token beside Crustacean Commando, A's one Island {@code islandTapped}. */
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
    void mutagenTokenBindsTheTokenWithNoManaForIt() {
        Board board = mutagenBoard(true);
        Binding binding = MutagenToken.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("token"));
        assertEquals(board.you, binding.player("player"));
    }

    @Test
    void mutagenTokenBindsNothingWithTheManaForIt() {
        assertTrue(MutagenToken.bind(mutagenBoard(false).game).isEmpty());
    }

    @Test
    void mutagenTokenForksAnUntappedIsland() {
        Board board = mutagenBoard(true);
        Twin twin = board.twin(MutagenToken.rule());
        Card island = twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(Cards.ISLAND) && !card.isTapped())
                .findFirst().orElseThrow();
        for (String change : twin.changes()) {
            assertTrue(change.equals("+" + id(island)) || change.startsWith("+link ") && change.contains("=" + id(island))
                            || change.endsWith(".untapped_production"),
                    "adding changed " + change + "; all: " + twin.changes());
        }
    }

    // ── 118.12b, an optional search: add_permanent ──────────────────────

    /** A's upkeep begins over A's library, listed top card first. */
    private static Board upkeepBoard(String... library) {
        Board board = new Board(PhaseType.UPKEEP);
        board.library(board.you, library);
        return board;
    }

    @Test
    void optionalSearchBindsTheActivePlayerWithABasicLandInItsLibrary() {
        Board board = upkeepBoard("Shock", "Forest", "Forest");
        assertEquals(board.you, OptionalSearch.bind(board.game).orElseThrow().player("player"));
    }

    @Test
    void optionalSearchBindsNothingWithoutABasicLandToFind() {
        assertTrue(OptionalSearch.bind(upkeepBoard("Shock", "Lightning Bolt").game).isEmpty());
    }

    /** Elfhame Sanctuary finds a basic land card: a nonbasic land in the library is none. */
    @Test
    void optionalSearchBindsNothingWithOnlyANonbasicLandToFind() {
        assertTrue(OptionalSearch.bind(upkeepBoard("Shock", "Evolving Wilds").game).isEmpty());
    }

    @Test
    void optionalSearchBindsNothingAtAnotherBoundary() {
        Board board = new Board(PhaseType.MAIN1);
        board.library(board.you, "Forest");
        assertTrue(OptionalSearch.bind(board.game).isEmpty());
    }

    /** The added Sanctuary is bound on the copy for the forced answer, which no live card maps to. */
    @Test
    void optionalSearchForksElfhameSanctuaryAndBindsItForTheAnswer() {
        Board board = upkeepBoard("Shock", "Forest");
        Twin twin = board.twin(OptionalSearch.rule());
        assertOnlyAdded(twin, twin.fork().player("player"), OptionalSearch.SANCTUARY);
        Application application = OptionalSearch.rule().variants().get(OptionalSearch.SANCTUARY_ADDED);
        Card sanctuary = application.afterEdit(twin.fork(), twin.forkGame()).card("sanctuary");
        assertEquals(OptionalSearch.SANCTUARY, sanctuary.getName());
        assertTrue(sanctuary.isInPlay());
    }

    // ── 111.10p, a Virtuous Role: add_permanent ─────────────────────────

    /** A's Grizzly Bears and four lands, Ellivere of the Wild Court in hand, as a main phase begins. */
    private static Board elliveresBoard(PhaseType phase, String... lands) {
        Board board = new Board(phase);
        for (String land : lands) {
            board.card(board.you, land);
        }
        board.other = board.card(board.you, "Grizzly Bears");
        board.subject = TestBoard.inHand(board.game, board.you, "Ellivere of the Wild Court");
        return board;
    }

    private static Board elliveresBoard() {
        return elliveresBoard(PhaseType.MAIN1, "Forest", "Forest", "Forest", "Plains");
    }

    @Test
    void virtuousRoleBindsTheCasterAndEllivere() {
        Board board = elliveresBoard();
        Binding binding = VirtuousRole.bind(board.game).orElseThrow();
        assertEquals(board.you, binding.player("caster"));
        assertEquals(board.subject, binding.card("card"));
    }

    @Test
    void virtuousRoleBindsNothingWithoutTheLands() {
        assertTrue(VirtuousRole.bind(elliveresBoard(PhaseType.MAIN1, "Forest", "Plains").game).isEmpty());
    }

    @Test
    void virtuousRoleBindsNothingWithoutACreatureForTheRole() {
        Board board = elliveresBoard();
        board.game.getAction().moveToGraveyard(board.other, null);
        assertTrue(VirtuousRole.bind(board.game).isEmpty());
    }

    @Test
    void virtuousRoleBindsNothingForACreatureWithoutTheRoleTrigger() {
        Board board = new Board(PhaseType.MAIN1);
        for (String land : List.of("Forest", "Forest", "Forest", "Plains")) {
            board.card(board.you, land);
        }
        board.card(board.you, "Grizzly Bears");
        // Centaur Courser is affordable: only the missing role trigger refuses it.
        TestBoard.inHand(board.game, board.you, "Centaur Courser");
        assertTrue(VirtuousRole.bind(board.game).isEmpty());
    }

    @Test
    void virtuousRoleBindsNothingOnTheOpponentsTurn() {
        assertTrue(VirtuousRole.bind(elliveresBoard(PhaseType.UPKEEP, "Forest", "Forest", "Forest", "Plains").game)
                .isEmpty());
    }

    @Test
    void virtuousRoleForksSealOfFire() {
        Board board = elliveresBoard();
        Twin twin = board.twin(VirtuousRole.rule());
        assertOnlyAdded(twin, twin.fork().player("caster"), VirtuousRole.SEAL_OF_FIRE);
    }

    // ── 205.1a, a type-setting Aura: counters ───────────────────────────

    /** A's three Forests and Song of the Dryads in hand, B's {@code creature}, as A's first main phase begins. */
    private static Board songBoard(String creature, int forests) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < forests; i++) {
            board.card(board.you, "Forest");
        }
        board.subject = board.card(board.opponent, creature);
        board.other = TestBoard.inHand(board.game, board.you, "Song of the Dryads");
        return board;
    }

    @Test
    void typeSettingAuraBindsTheAuraAndTheCreature() {
        Board board = songBoard("Grizzly Bears", 3);
        Binding binding = TypeSettingAura.bind(board.game).orElseThrow();
        assertEquals(board.you, binding.player("caster"));
        assertEquals(board.other, binding.card("aura"));
        assertEquals(board.subject, binding.card("creature"));
    }

    @Test
    void typeSettingAuraBindsNothingWithoutTheLands() {
        assertTrue(TypeSettingAura.bind(songBoard("Grizzly Bears", 2).game).isEmpty());
    }

    /** A creature that already has a +1/+1 counter is no board the variant's one counter describes. */
    @Test
    void typeSettingAuraBindsNothingOnACreatureWithACounter() {
        Board board = songBoard("Grizzly Bears", 3);
        board.subject.setCounters(CounterEnumType.P1P1, 1);
        assertTrue(TypeSettingAura.bind(board.game).isEmpty());
    }

    /** The battlefield's one creature has hexproof: Song of the Dryads, cast by its opponent, cannot target it. */
    @Test
    void typeSettingAuraBindsNothingOnACreatureTheAuraCannotTarget() {
        assertTrue(TypeSettingAura.bind(songBoard("Gladecover Scout", 3).game).isEmpty());
    }

    /** Holy Strength adds to its creature's power and toughness and sets no card type. */
    @Test
    void typeSettingAuraBindsNothingForAnAuraThatSetsNoType() {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Plains");
        board.card(board.opponent, "Grizzly Bears");
        TestBoard.inHand(board.game, board.you, Cards.HOLY_STRENGTH);
        assertTrue(TypeSettingAura.bind(board.game).isEmpty());
    }

    @Test
    void typeSettingAuraForksTheCreatureWithACounter() {
        Board board = songBoard("Grizzly Bears", 3);
        Twin twin = board.twin(TypeSettingAura.rule());
        for (String change : twin.changes()) {
            assertTrue(change.startsWith(id(board.subject) + "."), "the edit changed " + change);
        }
        assertEquals(1, twin.fork().card("creature").getCounters(CounterEnumType.P1P1));
    }

    // ── 701.23f, a search limited to the top cards: add_permanent ───────

    /** A's Archivist of Oghma and B's untapped {@code permanent}, as A's first main phase begins. */
    private static Board archivistBoard(String watcher, String permanent) {
        Board board = new Board(PhaseType.MAIN1);
        board.other = board.card(board.you, watcher);
        board.subject = board.card(board.opponent, permanent);
        board.library(board.opponent, "Grizzly Bears", "Forest");
        return board;
    }

    @Test
    void searchUnderMindcensorBindsTheWatcherTheSearcherAndTheFetch() {
        Board board = archivistBoard("Archivist of Oghma", "Evolving Wilds");
        Binding binding = SearchUnderMindcensor.bind(board.game).orElseThrow();
        assertEquals(board.you, binding.player("watcher"));
        assertEquals(board.opponent, binding.player("searcher"));
        assertEquals(board.subject, binding.card("permanent"));
    }

    @Test
    void searchUnderMindcensorBindsNothingWithoutASearchToWatch() {
        assertTrue(SearchUnderMindcensor.bind(archivistBoard("Grizzly Bears", "Evolving Wilds").game).isEmpty());
    }

    @Test
    void searchUnderMindcensorBindsNothingWithoutASearchingPermanent() {
        assertTrue(SearchUnderMindcensor.bind(archivistBoard("Archivist of Oghma", "Forest").game).isEmpty());
    }

    /**
     * Prime Speaker Vannifar's search is activated only as a sorcery: on its
     * controller's opponent's turn the forced activation could never be made.
     */
    @Test
    void searchUnderMindcensorBindsNothingForASorcerySpeedSearch() {
        Board board = archivistBoard("Archivist of Oghma", "Prime Speaker Vannifar");
        board.card(board.opponent, "Grizzly Bears");
        assertTrue(SearchUnderMindcensor.bind(board.game).isEmpty());
    }

    /** A tapped Evolving Wilds cannot pay its own tap, whatever other lands its controller has. */
    @Test
    void searchUnderMindcensorBindsNothingForATappedFetch() {
        Board board = archivistBoard("Archivist of Oghma", "Evolving Wilds");
        board.card(board.opponent, "Forest");
        board.subject.setTapped(true);
        assertTrue(SearchUnderMindcensor.bind(board.game).isEmpty());
    }

    /** Archivist watches the opponents' searches only: its controller's own fetch is no search it watches. */
    @Test
    void searchUnderMindcensorBindsNothingForTheWatchersOwnFetch() {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Archivist of Oghma");
        board.card(board.you, "Evolving Wilds");
        assertTrue(SearchUnderMindcensor.bind(board.game).isEmpty());
    }

    /**
     * Besides the Mindcensor, the searcher's Keyword Effects card holds the
     * limit the Mindcensor's static gives it ({@code LimitSearchLibrary}).
     */
    @Test
    void searchUnderMindcensorForksAvenMindcensor() {
        Board board = archivistBoard("Archivist of Oghma", "Evolving Wilds");
        Twin twin = board.twin(SearchUnderMindcensor.rule());
        Card mindcensor = twin.fork().player("watcher").getCardsIn(ZoneType.Battlefield).stream()
                .filter(c -> c.getName().equals(SearchUnderMindcensor.MINDCENSOR)).findFirst().orElseThrow();
        Card effects = twin.fork().player("searcher").getKeywordCard();
        Set<String> named = Set.of(id(mindcensor), id(effects));
        for (String change : twin.changes()) {
            assertTrue(change.equals("global.floating_effects")
                            || named.stream().anyMatch(id -> change.equals("+" + id) || change.contains("=" + id)),
                    "adding changed " + change + "; all: " + twin.changes());
        }
        assertTrue(twin.changes().contains("+" + id(mindcensor)));
        assertTrue(twin.fork().player("searcher").hasKeyword("LimitSearchLibrary"));
    }

    // ── 702.133a, jump-start: move_card ─────────────────────────────────

    /** Radical Idea in A's graveyard and {@code islands} Islands, as A's first main phase begins. */
    private static Board jumpStartBoard(String card, int islands) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < islands; i++) {
            board.card(board.you, "Island");
        }
        board.subject = board.game.getAction().moveToGraveyard(TestBoard.card(board.game, board.you, card), null);
        return board;
    }

    @Test
    void jumpStartFromHandBindsTheCardInTheGraveyard() {
        Board board = jumpStartBoard("Radical Idea", 2);
        Binding binding = JumpStartFromHand.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.you, binding.player("caster"));
    }

    @Test
    void jumpStartFromHandBindsNothingWithoutTheLands() {
        assertTrue(JumpStartFromHand.bind(jumpStartBoard("Radical Idea", 1).game).isEmpty());
    }

    @Test
    void jumpStartFromHandBindsNothingForACardWithoutJumpStart() {
        assertTrue(JumpStartFromHand.bind(jumpStartBoard("Divination", 3).game).isEmpty());
    }

    /** Quasiduplicate needs a target, which the forced cast's clause would draw: a spell with one is no board the rule binds. */
    @Test
    void jumpStartFromHandBindsNothingForASpellThatTargets() {
        assertTrue(JumpStartFromHand.bind(jumpStartBoard("Quasiduplicate", 3).game).isEmpty());
    }

    @Test
    void jumpStartFromHandForksTheCardInTheHand() {
        Board board = jumpStartBoard("Radical Idea", 2);
        Twin twin = board.twin(JumpStartFromHand.rule());
        assertTrue(twin.fork().player("caster").getCardsIn(ZoneType.Hand).stream()
                .anyMatch(card -> card.getName().equals("Radical Idea")));
        assertTrue(twin.fork().player("caster").getCardsIn(ZoneType.Graveyard).isEmpty());
    }

    // ── 119.7, an exchange a can't-gain-life effect stops: add_permanent

    private static Board redemptionBoard(boolean leyline) {
        Board board = new Board(PhaseType.MAIN1);
        board.subject = board.card(board.you, "Tree of Redemption");
        if (leyline) {
            board.card(board.you, CantGainLifeExchange.LEYLINE_OF_PUNISHMENT);
        }
        board.you.setLife(5, null);
        return board;
    }

    @Test
    void cantGainLifeExchangeBindsTheTreeAndItsController() {
        Board board = redemptionBoard(false);
        Binding binding = CantGainLifeExchange.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("permanent"));
        assertEquals(board.you, binding.player("player"));
    }

    @Test
    void cantGainLifeExchangeBindsNothingWhenThePlayerCannotGainLife() {
        assertTrue(CantGainLifeExchange.bind(redemptionBoard(true).game).isEmpty());
    }

    @Test
    void cantGainLifeExchangeBindsNothingWhenTheExchangeWouldLowerTheLife() {
        Board board = redemptionBoard(false);
        board.you.setLife(13, null);
        assertTrue(CantGainLifeExchange.bind(board.game).isEmpty());
    }

    @Test
    void cantGainLifeExchangeForksLeylineOfPunishment() {
        Board board = redemptionBoard(false);
        Twin twin = board.twin(CantGainLifeExchange.rule());
        assertOnlyAdded(twin, twin.fork().player("player"), CantGainLifeExchange.LEYLINE_OF_PUNISHMENT);
        assertFalse(twin.fork().player("player").canGainLife());
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
        assertOnlyAdded(twin, twin.fork().player("opponent"), Cards.WURM);
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
    void finalChapterResolvesBindsTheSagaTwoCountersShortOfItsLastChapter() {
        Board board = benaliaBoard(1);
        assertEquals(board.subject, FinalChapterResolves.bind(board.game).orElseThrow().card("saga"));
    }

    @Test
    void finalChapterResolvesBindsNothingWhenTheNextCounterReachesTheLastChapter() {
        assertTrue(FinalChapterResolves.bind(benaliaBoard(2).game).isEmpty());
    }

    @Test
    void finalChapterResolvesForksTheSagaWithACounterMore() {
        Board board = benaliaBoard(1);
        Twin twin = board.twin(FinalChapterResolves.rule());
        String saga = id(board.subject);
        assertEquals(Set.of(saga + ".counters"), twin.changes());
        assertEquals(2, twin.fork().card("saga").getCounters(CounterEnumType.LORE));
    }

    // ── 614.17d, a permanent that can't enter: add_permanent ────────────

    private static Board zombifyBoard(boolean creatureInGraveyard) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 4; i++) {
            board.card(board.you, "Swamp");
        }
        TestBoard.inHand(board.game, board.you, "Zombify");
        if (creatureInGraveyard) {
            TestBoard.inGraveyard(board.game, board.you, "Grizzly Bears");
        }
        return board;
    }

    @Test
    void cantEnterBattlefieldBindsThePlayerWithACreatureToReturn() {
        Board board = zombifyBoard(true);
        assertEquals(board.you, CantEnterBattlefield.bind(board.game).orElseThrow().player("player"));
    }

    @Test
    void cantEnterBattlefieldBindsNothingWithNoCreatureCardToReturn() {
        assertTrue(CantEnterBattlefield.bind(zombifyBoard(false).game).isEmpty());
    }

    @Test
    void cantEnterBattlefieldBindsNothingWhenACageIsOut() {
        Board board = zombifyBoard(true);
        board.card(board.opponent, CantEnterBattlefield.GRAFDIGGERS_CAGE);
        assertTrue(CantEnterBattlefield.bind(board.game).isEmpty());
    }

    @Test
    void cantEnterBattlefieldForksTheBoardWithTheCage() {
        Board board = zombifyBoard(true);
        Twin twin = board.twin(CantEnterBattlefield.rule());
        assertOnlyAdded(twin, twin.fork().player("player"), CantEnterBattlefield.GRAFDIGGERS_CAGE);
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
    void auraSwapBindsTheHandsCardAndTheAttachedAura() {
        Board board = auraSwapBoard("Giant Growth");
        Binding binding = AuraSwap.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("aura"));
    }

    @Test
    void auraSwapBindsNothingWithAnAuraInHand() {
        assertTrue(AuraSwap.bind(auraSwapBoard("Giant Growth", "Wild Growth").game).isEmpty());
    }

    @Test
    void auraSwapBindsNothingWithAnEmptyHand() {
        assertTrue(AuraSwap.bind(auraSwapBoard().game).isEmpty());
    }

    /** Holy Strength is white: Black Knight, with protection from white, could not take it. */
    @Test
    void auraSwapBindsNothingOnAHostWithProtection() {
        Board board = new Board(PhaseType.MAIN1);
        Card knight = board.card(board.you, "Black Knight");
        board.card(board.you, "Arcanum Wings").attachToEntity(knight, null);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Island");
        }
        TestBoard.inHand(board.game, board.you, "Giant Growth");
        assertTrue(AuraSwap.bind(board.game).isEmpty());
    }

    @Test
    void auraSwapForksAHolyStrengthInTheCardsPlace() {
        Board board = auraSwapBoard("Giant Growth");
        Twin twin = board.twin(AuraSwap.rule());
        List<Card> hand = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand));
        assertEquals(List.of(Cards.HOLY_STRENGTH), hand.stream().map(Card::getName).toList());
        assertOnlySwapped(twin, board.subject, hand.get(0));
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

    /** The searcher binds over a library of common cards, whatever its size: the edit empties it. */
    @Test
    void smallLibraryShuffleBindsTheSearcherOverAnyLibraryOfOneCardOrMore() {
        for (String[] library : new String[][]{{"Forest"}, {"Forest", "Grizzly Bears", "Island"}}) {
            Board board = tricksterBoard(library);
            Binding binding = SmallLibraryShuffle.bind(board.game).orElseThrow();
            assertEquals(board.you, binding.player("searcher"));
            assertEquals("Demonic Tutor", binding.card("spell").getName());
        }
    }

    /** An empty library is the rare board itself: the edit would add nothing to it. */
    @Test
    void smallLibraryShuffleBindsNothingOverAnEmptyLibrary() {
        assertTrue(SmallLibraryShuffle.bind(tricksterBoard().game).isEmpty());
    }

    @Test
    void smallLibraryShuffleForksTheLibraryEmptyAndTheGraveyardAsItWas() {
        Board board = tricksterBoard("Forest", "Grizzly Bears", "Island");
        Card top = board.you.getCardsIn(ZoneType.Library).get(0);
        Twin twin = board.twin(SmallLibraryShuffle.rule());
        // The snapshot shows a library by its count and its top card alone.
        assertOnlyChanged(twin, top, top, "P0.library");
        Player searcher = twin.fork().player("searcher");
        assertTrue(searcher.getCardsIn(ZoneType.Library).isEmpty());
        assertTrue(searcher.getCardsIn(ZoneType.Graveyard).isEmpty(), "no card went to the graveyard");
    }

    // ── 701.39a, bolster: add_permanent ─────────────────────────────────

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
    void bolsterBindsTheBolsteringCardOfAPlayerWithNoCreature() {
        Board board = bolsterBoard();
        Card defenses = board.you.getCardsIn(ZoneType.Hand).get(0);
        assertEquals(defenses, Bolster.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void bolsterBindsNothingWhenThePlayerControlsACreature() {
        assertTrue(Bolster.bind(bolsterBoard("Grizzly Bears").game).isEmpty());
    }

    @Test
    void bolsterForksGrizzlyBearsUnderThePlayer() {
        Board board = bolsterBoard();
        Twin twin = board.twin(Bolster.rule());
        assertOnlyAdded(twin, twin.fork().player("player"), Cards.BEARS);
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
    void meldBindsTheHostAndACreatureThePartnerCanReplace() {
        Board board = meldBoard("Graf Rats", "Hill Giant", false, 0);
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
        Board board = meldBoard("Graf Rats", "Hill Giant", false, 0);
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
        Board board = meldBoard("Graf Rats", "Hill Giant", false, 0);
        assertOnlyReplaced(board.twin(Meld.rule()), board.subject, "Midnight Scavengers");
    }

    // ── 701.44b, exploring with an empty library: move_card, inverted ──

    /** A's Merfolk Branchwalker in hand over two Forests, Wildgrowth Walker, A's library {@code library}. */
    private static Board branchwalkerBoard(String... library) {
        Board board = new Board(PhaseType.MAIN1);
        board.cards(board.you, "Forest", 2);
        return watchedCast(board, "Wildgrowth Walker", SeveralExplore.BRANCHWALKER, library);
    }

    /**
     * {@code board} with {@code card} the subject in your hand beside your
     * {@code watcher}, your library {@code library}, top card first.
     */
    private static Board watchedCast(Board board, String watcher, String card, String... library) {
        board.card(board.you, watcher);
        board.subject = TestBoard.inHand(board.game, board.you, card);
        board.library(board.you, library);
        return board;
    }

    @Test
    void exploreEmptyLibraryBindsTheExplorerOverALibraryOfCommonCards() {
        Board board = branchwalkerBoard("Forest", "Grizzly Bears");
        Binding binding = ExploreEmptyLibrary.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.you, binding.player("player"));
    }

    /** An empty library is the rare board itself: the edit would add nothing to it. */
    @Test
    void exploreEmptyLibraryBindsNothingOverAnEmptyLibrary() {
        assertTrue(ExploreEmptyLibrary.bind(branchwalkerBoard().game).isEmpty());
    }

    @Test
    void exploreEmptyLibraryForksTheLibraryEmpty() {
        Board board = branchwalkerBoard("Forest", "Grizzly Bears");
        Card top = board.you.getCardsIn(ZoneType.Library).get(0);
        Twin twin = board.twin(ExploreEmptyLibrary.rule());
        // The snapshot shows a library by its count and its top card alone.
        assertOnlyChanged(twin, top, top, "P0.library");
        assertTrue(twin.fork().player("player").getCardsIn(ZoneType.Library).isEmpty());
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
    void severalExploreBindsTheCreatureBesideTheOnlyMerfolk() {
        Board board = hakbalBoard("Grizzly Bears");
        assertEquals(board.subject, SeveralExplore.bind(board.game).orElseThrow().card("creature"));
    }

    /** Two Merfolk already explore two at a time, the side the counts hold, whatever else is there. */
    @Test
    void severalExploreBindsNothingWithTwoMerfolk() {
        Board board = hakbalBoard(SeveralExplore.BRANCHWALKER);
        assertTrue(SeveralExplore.bind(board.game).isEmpty());
        board.card(board.you, "Grizzly Bears");
        assertTrue(SeveralExplore.bind(board.game).isEmpty(), "beside a creature the Branchwalker could replace");
    }

    @Test
    void severalExploreBindsNothingWithoutACreatureToReplace() {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Hakbal of the Surging Soul");
        board.library(board.you, "Forest", "Shock");
        assertTrue(SeveralExplore.bind(board.game).isEmpty());
    }

    @Test
    void severalExploreBindsNothingWithOneCardInTheLibrary() {
        Board board = new Board(PhaseType.MAIN1);
        board.card(board.you, "Hakbal of the Surging Soul");
        board.card(board.you, "Grizzly Bears");
        board.library(board.you, "Forest");
        assertTrue(SeveralExplore.bind(board.game).isEmpty());
    }

    /** Two damage marked on the Bears would kill a 2/1 Branchwalker in its place. */
    @Test
    void severalExploreBindsNothingWhenTheBranchwalkerWouldDie() {
        Board board = hakbalBoard("Grizzly Bears");
        board.subject.setDamage(1);
        assertTrue(SeveralExplore.bind(board.game).isEmpty());
    }

    @Test
    void severalExploreBindsNothingAtAnotherBoundary() {
        Board board = new Board(PhaseType.MAIN2);
        board.card(board.you, "Hakbal of the Surging Soul");
        board.card(board.you, "Grizzly Bears");
        board.library(board.you, "Forest", "Shock");
        assertTrue(SeveralExplore.bind(board.game).isEmpty());
    }

    @Test
    void severalExploreForksAMerfolkBranchwalkerInTheBearsPlace() {
        Board board = hakbalBoard("Grizzly Bears");
        assertOnlyReplaced(board.twin(SeveralExplore.rule()), board.subject, SeveralExplore.BRANCHWALKER);
    }

    // ── 701.50f, conniving when drawing is impossible: add_permanent, inverted

    /**
     * A holds Echo Inspector over four Islands and Glorious Purpose, and has
     * drawn its one card this turn (a Forest) when {@code drawn}; B's Spirit
     * of the Labyrinth, when {@code spirit}.
     */
    private static Board inspectorBoard(boolean spirit, boolean drawn) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 4; i++) {
            board.card(board.you, "Island");
        }
        board.card(board.you, "Glorious Purpose");
        if (spirit) {
            board.card(board.opponent, ConniveWithoutDraw.SPIRIT);
        }
        board.subject = TestBoard.inHand(board.game, board.you, "Echo Inspector");
        board.library(board.you, "Forest", "Island");
        if (drawn) {
            board.you.drawCard();
        }
        return board;
    }

    @Test
    void conniveWithoutDrawBindsTheConniverOfAPlayerWhoDrewAndCanDrawAgain() {
        Board board = inspectorBoard(false, true);
        Binding binding = ConniveWithoutDraw.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("conniver"));
        assertEquals(board.you, binding.player("player"));
        assertEquals(board.opponent, binding.player("opponent"));
    }

    /** Under the Spirit the board is the rare one already: the edit would add nothing to it. */
    @Test
    void conniveWithoutDrawBindsNothingWhenThePlayerCannotDraw() {
        assertTrue(ConniveWithoutDraw.bind(inspectorBoard(true, true).game).isEmpty());
    }

    /** Before the turn's first draw the Spirit added would stop no draw of the connive. */
    @Test
    void conniveWithoutDrawBindsNothingBeforeTheTurnsFirstDraw() {
        assertTrue(ConniveWithoutDraw.bind(inspectorBoard(false, false).game).isEmpty());
    }

    @Test
    void conniveWithoutDrawForksTheSpiritOfTheLabyrinthUnderTheOpponent() {
        Twin twin = inspectorBoard(false, true).twin(ConniveWithoutDraw.rule());
        assertOnlyAdded(twin, twin.fork().player("opponent"), ConniveWithoutDraw.SPIRIT);
        assertFalse(twin.fork().player("player").canDraw(), "the Spirit stops the connive's draw");
        assertTrue(twin.fork().player("opponent").getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(ConniveWithoutDraw.SPIRIT)).allMatch(Card::isTapped),
                "the Spirit is tapped, so it blocks no attacker of the fork's turn");
    }

    // ── 701.57b, a discovery that finds nothing: move_card, inverted ────

    /** A's Geological Appraiser over four Mountains and Val, Marooned Surveyor; A's library top card first. */
    private static Board appraiserBoard(String... library) {
        Board board = new Board(PhaseType.MAIN1);
        board.cards(board.you, "Mountain", 4);
        return watchedCast(board, "Val, Marooned Surveyor", "Geological Appraiser", library);
    }

    @Test
    void discoverNothingFoundBindsTheDiscovererOverALibraryWithACardToFind() {
        Board board = appraiserBoard("Forest", "Grizzly Bears", "Island");
        Binding binding = DiscoverNothingFound.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("discoverer"));
        assertEquals(board.you, binding.player("player"));
    }

    /** An empty library is the rare board itself: the edit would add nothing to it. */
    @Test
    void discoverNothingFoundBindsNothingOverAnEmptyLibrary() {
        assertTrue(DiscoverNothingFound.bind(appraiserBoard().game).isEmpty());
    }

    @Test
    void discoverNothingFoundForksTheLibraryEmpty() {
        Board board = appraiserBoard("Forest", "Grizzly Bears", "Island");
        Card top = board.you.getCardsIn(ZoneType.Library).get(0);
        Twin twin = board.twin(DiscoverNothingFound.rule());
        // The snapshot shows a library by its count and its top card alone.
        assertOnlyChanged(twin, top, top, "P0.library");
        assertTrue(twin.fork().player("player").getCardsIn(ZoneType.Library).isEmpty());
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
        assertTrue(SharedKeywords.bind(concertedBoard("White Knight", SimultaneousDamage.KNIGHT).game).isEmpty());
        assertTrue(SharedKeywords.bind(concertedBoard("White Knight").game).isEmpty());
    }

    @Test
    void sharedKeywordsForksHillGiantInTheKnightsPlace() {
        Board board = concertedBoard("White Knight", "Grizzly Bears");
        assertOnlyReplaced(board.twin(SharedKeywords.rule()), board.subject, Cards.GIANT);
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
        Binding binding = ManaValueSearch.transmute().precondition(board.game).orElseThrow();
        assertEquals(board.other, binding.card("source"));
        assertEquals(board.subject, binding.card("card"));
    }

    @Test
    void transmuteBindsNothingWithTwoCardsOfItsManaValue() {
        assertTrue(ManaValueSearch.transmute().precondition(infiltratorBoard("Grizzly Bears", "Grizzly Bears").game).isEmpty());
    }

    @Test
    void transmuteForksACentaurCourserInTheCardsPlace() {
        Board board = infiltratorBoard("Forest", "Shock", "Grizzly Bears", "Island");
        Twin twin = board.twin(ManaValueSearch.transmute());
        List<Card> library = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library));
        assertEquals(List.of("Forest", "Shock", Cards.COURSER, "Island"),
                library.stream().map(Card::getName).toList());
        // A library card the board shows by count alone: the swap the list
        // above shows changes no visible member.
        assertTrue(twin.changes().isEmpty(), "visible changes: " + twin.changes());
    }

    // ── 702.65a, Aura swap's exchange, registered with 701.12e: move_card ─

    /** 702.65a-1's base, Grizzly Bears in hand, is 701.12e's registration's board too. */
    @Test
    void auraSwapBindsTheExchangeScenariosHandCard() {
        Board board = auraSwapBoard("Grizzly Bears");
        Binding binding = AuraSwap.rule().precondition(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("aura"));
    }

    @Test
    void auraSwapBindsNothingWithTheAuraTheExchangeWaitsFor() {
        assertTrue(AuraSwap.rule().precondition(auraSwapBoard(Cards.HOLY_STRENGTH).game).isEmpty());
    }

    @Test
    void auraSwapForksAHolyStrengthInTheExchangeScenariosCardsPlace() {
        Board board = auraSwapBoard("Grizzly Bears");
        Twin twin = board.twin(AuraSwap.rule());
        List<Card> hand = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand));
        assertEquals(List.of(Cards.HOLY_STRENGTH), hand.stream().map(Card::getName).toList());
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
        assertTrue(AuraSwapCantEnchant.bind(auraSwapBoard(Cards.HOLY_STRENGTH).game).isEmpty());
    }

    @Test
    void auraSwapCantEnchantForksHolyStrengthInTheAurasPlace() {
        Board board = auraSwapBoard("Wild Growth");
        Twin twin = board.twin(AuraSwapCantEnchant.rule());
        List<Card> hand = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand));
        assertEquals(List.of(Cards.HOLY_STRENGTH), hand.stream().map(Card::getName).toList());
        assertOnlySwapped(twin, board.subject, hand.get(0));
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
        Binding binding = ManaValueSearch.transfigure().precondition(board.game).orElseThrow();
        assertEquals(board.other, binding.card("source"));
        assertEquals(board.subject, binding.card("card"));
    }

    @Test
    void transfigureBindsNothingWithTwoCreatureCardsOfItsManaValue() {
        assertTrue(ManaValueSearch.transfigure().precondition(fleshwritherBoard("Hill Giant", "Hill Giant").game).isEmpty());
    }

    @Test
    void transfigureForksACrawWurmInTheCardsPlace() {
        Board board = fleshwritherBoard("Hill Giant", "Grizzly Bears");
        Twin twin = board.twin(ManaValueSearch.transfigure());
        List<Card> library = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library));
        assertEquals(List.of(Cards.WURM, "Grizzly Bears"), library.stream().map(Card::getName).toList());
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

    // ── 702.98a, unleash can't block: counters ──────────────────────────

    /** Grizzly Bears attacks B, who controls {@code blocker} with {@code counters} +1/+1 counters; no blocker declared yet. */
    private static Board unleashBoard(String blocker, int counters) {
        Board board = new Board(PhaseType.COMBAT_DECLARE_BLOCKERS);
        board.other = board.card(board.you, "Grizzly Bears");
        board.subject = board.card(board.opponent, blocker);
        board.subject.setCounters(CounterEnumType.P1P1, counters);
        board.attacking(board.other);
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

    // ── 702.103e, a bestowed Aura whose target is gone: replace_card ────

    /** A's Hopeful Eidolon in hand over four Plains and its creature {@code creature}, the battlefield's one. */
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
    void bestowIllegalTargetBindsTheOneCreatureTheAuraCouldEnchant() {
        Board board = eidolonBoard("Grizzly Bears");
        assertEquals(board.subject, BestowIllegalTarget.bind(board.game).orElseThrow().card("creature"));
    }

    @Test
    void bestowIllegalTargetBindsNothingWithASecondCreatureToTarget() {
        Board board = eidolonBoard("Grizzly Bears");
        board.card(board.opponent, "Hill Giant");
        assertTrue(BestowIllegalTarget.bind(board.game).isEmpty());
    }

    /** One damage on the Bears would kill the 2/1 Ghost in their place at the fork's first check. */
    @Test
    void bestowIllegalTargetBindsNothingWhenTheGhostWouldNotSurvive() {
        Board board = eidolonBoard("Grizzly Bears");
        board.subject.setDamage(1);
        assertTrue(BestowIllegalTarget.bind(board.game).isEmpty());
    }

    @Test
    void bestowIllegalTargetForksASkulkingGhostInTheCreaturesPlace() {
        Board board = eidolonBoard("Grizzly Bears");
        assertOnlyReplaced(board.twin(BestowIllegalTarget.rule()), board.subject, BestowIllegalTarget.GHOST);
    }

    // ── 702.124j, partner with's search: move_card ──────────────────────

    /** A's Pir, Imaginative Rascal in hand over three Forests; A's library top card first, the top card the subject. */
    private static Board pirBoard(String... library) {
        Board board = new Board(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Forest");
        }
        board.other = TestBoard.inHand(board.game, board.you, "Pir, Imaginative Rascal");
        board.subject = board.library(board.you, library).get(0);
        return board;
    }

    @Test
    void partnerWithSearchBindsTheTopCardOfALibraryWithoutThePartner() {
        Board board = pirBoard("Forest", "Forest", "Forest");
        Binding binding = PartnerWithSearch.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("partner with"));
    }

    @Test
    void partnerWithSearchBindsNothingWithThePartnerInTheLibrary() {
        assertTrue(PartnerWithSearch.bind(pirBoard("Forest", "Toothy, Imaginary Friend", "Forest").game).isEmpty());
    }

    @Test
    void partnerWithSearchForksThePartnerOnTop() {
        Board board = pirBoard("Forest", "Forest", "Forest");
        Twin twin = board.twin(PartnerWithSearch.rule());
        List<Card> library = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library));
        assertEquals(List.of("Toothy, Imaginary Friend", "Forest", "Forest"),
                library.stream().map(Card::getName).toList());
        // The top card is on the board as the library's visible card.
        assertOnlySwapped(twin, board.subject, library.get(0));
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

    // ── 702.187b, mayhem from a card discarded this turn: remove_permanent

    /** A's Electro's Bolt (mayhem {1}{R}) in the graveyard, discarded this turn when {@code discarded}, over {@code mountains} Mountains. */
    private static Board boltBoard(boolean discarded, int mountains) {
        Board board = new Board(PhaseType.MAIN2);
        board.subject = board.cards(board.you, "Mountain", mountains);
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

    // ── 718.3a, a prototype's cost alone: remove_permanent ──────────────

    /** A's Blitz Automaton (prototype {2}{R}, mana value 7) in hand over {@code mountains} Mountains; the first is the subject. */
    private static Board automatonBoard(int mountains) {
        Board board = new Board(PhaseType.MAIN1);
        board.subject = board.cards(board.you, "Mountain", mountains);
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

    // ── 104.4a, a draw when every player loses at once: life, inverted ──

    /** Your Plague Spitter as your upkeep begins, you at {@code yours} life and the opponent at {@code theirs}. */
    private static Board spitterBoard(int yours, int theirs) {
        Board board = new Board(PhaseType.UPKEEP);
        board.subject = board.card(board.you, "Plague Spitter");
        board.you.setLife(yours, null);
        board.opponent.setLife(theirs, null);
        return board;
    }

    @Test
    void simultaneousLossBindsAnUpkeepTriggerTakingLifeFromEachPlayerAtAnyLife() {
        Board board = spitterBoard(20, 20);
        Binding binding = SimultaneousLoss.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("permanent"));
        assertEquals(board.you, binding.player("player"));
        assertEquals(board.opponent, binding.player("opponent"));
        assertTrue(SimultaneousLoss.bind(spitterBoard(1, 20).game).isPresent(), "one player in reach already");
    }

    /** Every player within the trigger's reach is the rare board itself: the edit would add nothing to it. */
    @Test
    void simultaneousLossBindsNothingWhenEveryPlayerIsInReachAlready() {
        assertTrue(SimultaneousLoss.bind(spitterBoard(1, 1).game).isEmpty());
    }

    @Test
    void simultaneousLossBindsNothingForTheOtherPlayersTrigger() {
        Board board = new Board(PhaseType.UPKEEP);
        board.card(board.opponent, "Plague Spitter");
        assertTrue(SimultaneousLoss.bind(board.game).isEmpty());
    }

    /** Stormfist Crusader's chain draws, then takes 1 life from each player. */
    @Test
    void simultaneousLossReadsTheLifeLossBelowADraw() {
        Board board = new Board(PhaseType.UPKEEP);
        board.card(board.you, "Stormfist Crusader");
        assertTrue(SimultaneousLoss.bind(board.game).isPresent());
    }

    @Test
    void simultaneousLossBindsNothingAtAnotherBoundary() {
        Board board = new Board(PhaseType.DRAW);
        board.card(board.you, "Plague Spitter");
        assertTrue(SimultaneousLoss.bind(board.game).isEmpty());
    }

    /** Platinum Emperion keeps its controller's life from changing: the trigger would leave the opponent at 20. */
    @Test
    void simultaneousLossBindsNothingWhenAPlayerCannotLoseLife() {
        Board board = spitterBoard(20, 20);
        board.card(board.opponent, "Platinum Emperion");
        assertTrue(SimultaneousLoss.bind(board.game).isEmpty());
    }

    /** Platinum Angel's controller cannot lose the game: the 0 life loses it nothing. */
    @Test
    void simultaneousLossBindsNothingWhenAPlayerCannotLoseForZeroLife() {
        Board board = spitterBoard(20, 20);
        board.card(board.opponent, "Platinum Angel");
        assertTrue(SimultaneousLoss.bind(board.game).isEmpty());
    }

    /** A trigger asking a condition of its own, or its controller whether it happens, is no engine certainty. */
    @Test
    void simultaneousLossBindsNothingForATriggerWithAConditionOrAChoice() {
        for (Map.Entry<String, String> param : Map.of("IsPresent", "Card.Self", "CheckSVar", "X",
                "OptionalDecider", "You").entrySet()) {
            Board board = spitterBoard(20, 20);
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
            Board board = spitterBoard(20, 20);
            board.subject.setSVar("TrigDamageAll", damage + extra);
            board.subject.setSVar("DBDestroy", "DB$ Destroy | Defined$ Self");
            assertTrue(SimultaneousLoss.bind(board.game).isEmpty(), extra);
        }
        Board plain = spitterBoard(20, 20);
        plain.subject.setSVar("TrigDamageAll", damage);
        assertTrue(SimultaneousLoss.bind(plain.game).isPresent(), "the plain chain binds");
    }

    private static forge.game.trigger.Trigger upkeepTrigger(Card card) {
        return card.getTriggers().stream()
                .filter(trigger -> "Upkeep".equals(trigger.getParam("Phase"))).findFirst().orElseThrow();
    }

    @Test
    void simultaneousLossForksEachPlayerAtTheTriggersAmount() {
        Twin twin = spitterBoard(20, 20).twin(SimultaneousLoss.rule());
        assertEquals(Set.of("P0.life", "P1.life"), twin.changes());
        assertEquals(1, twin.fork().player("player").getLife());
        assertEquals(1, twin.fork().player("opponent").getLife());
    }

    /** A player already in the trigger's reach keeps its life: the edit only lowers a life total. */
    @Test
    void simultaneousLossForksOnlyThePlayerOutOfReach() {
        Board board = new Board(PhaseType.UPKEEP);
        board.card(board.you, "Plague Spitter");
        board.you.setLife(1, null);
        Twin twin = board.twin(SimultaneousLoss.rule());
        assertEquals(Set.of("P1.life"), twin.changes());
        assertEquals(1, twin.fork().player("opponent").getLife());
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
    void trampleOverPlaneswalkersBindsAnUnblockedAttackerWithoutTheKeyword() {
        Board board = planeswalkerBoard("Colossal Dreadmaw", 3);
        assertEquals(board.subject, TrampleOverPlaneswalkers.bind(board.game).orElseThrow().card("attacker"));
    }

    /** Thrasta's own attack is the side the counts hold already. */
    @Test
    void trampleOverPlaneswalkersBindsNothingWithTheKeyword() {
        assertTrue(TrampleOverPlaneswalkers.bind(planeswalkerBoard(Cards.THRASTA, 3).game).isEmpty());
    }

    /** A granted trample over planeswalkers is the keyword already, and stays on the Thrasta in its place. */
    @Test
    void trampleOverPlaneswalkersBindsNothingForAGrantedKeyword() {
        Board board = planeswalkerBoard("Colossal Dreadmaw", 3);
        board.subject.addChangedCardKeywords(List.of("Trample:Planeswalker"), null, false,
                board.game.getNextTimestamp(), null);
        assertTrue(Boards.tramplesOverPlaneswalkers(board.subject));
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isEmpty());
    }

    /** Thrasta's 7 does not top a loyalty of 7: the fork's attack would have no excess. */
    @Test
    void trampleOverPlaneswalkersBindsNothingWhenThrastaWouldHaveNoExcess() {
        assertTrue(TrampleOverPlaneswalkers.bind(planeswalkerBoard("Colossal Dreadmaw", 7).game).isEmpty());
        assertTrue(TrampleOverPlaneswalkers.bind(planeswalkerBoard("Colossal Dreadmaw", 6).game).isPresent());
    }

    @Test
    void trampleOverPlaneswalkersBindsNothingOnABlockedAttacker() {
        Board board = planeswalkerBoard("Colossal Dreadmaw", 3);
        Card bears = board.card(board.opponent, "Grizzly Bears");
        board.game.getCombat().addBlocker(board.subject, bears);
        board.game.getCombat().setBlocked(board.subject, true);
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isEmpty());
    }

    @Test
    void trampleOverPlaneswalkersBindsNothingAtAnotherBoundary() {
        assertTrue(TrampleOverPlaneswalkers.bind(planeswalkerBoard("Colossal Dreadmaw", 3,
                PhaseType.COMBAT_DECLARE_BLOCKERS).game).isEmpty());
    }

    /** The Dreadmaw's 7 marked damage would kill a 7/7 Thrasta. */
    @Test
    void trampleOverPlaneswalkersBindsNothingWhenThrastaWouldDie() {
        Board board = planeswalkerBoard("Colossal Dreadmaw", 3);
        board.subject.setDamage(5);
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isPresent());
        board.subject.setDamage(7);
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isEmpty());
    }

    /** A first striker deals its damage in the first-strike step, not in the regular one the rule forks at. */
    @Test
    void trampleOverPlaneswalkersBindsNothingForAFirstStriker() {
        Board board = planeswalkerBoard("Colossal Dreadmaw", 3);
        board.subject.addChangedCardKeywords(List.of("First Strike"), null, false,
                board.game.getNextTimestamp(), null);
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isEmpty());
    }

    @Test
    void trampleOverPlaneswalkersForksThrastaAttackingTheSamePlaneswalker() {
        Board board = planeswalkerBoard("Colossal Dreadmaw", 3);
        Twin twin = board.twin(TrampleOverPlaneswalkers.rule());
        Card thrasta = twin.forkGame().getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(Cards.THRASTA)).findFirst().orElseThrow();
        for (String change : changes(renamed(twin.real(), id(board.subject), id(thrasta)), twin.forked())) {
            assertTrue(change.startsWith(id(thrasta) + ".") || change.contains(id(thrasta)),
                    "replacing changed " + change);
        }
        assertTrue(Boards.tramplesOverPlaneswalkers(thrasta));
        Combat combat = twin.forkGame().getCombat();
        assertFalse(combat.isBlocked(thrasta));
        assertEquals(combat.getDefenderByAttacker(thrasta).getId(), board.other.getId());
    }

    // ── 702.19f, a planeswalker gone from combat: replace_card ─────────

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
    void aGoneWalkersAttackerWithoutTheKeywordBindsTheNoDamageRule() {
        Board board = goneWalkerBoard("Craw Wurm");
        assertEquals(board.subject, NoDamageToAGonePlaneswalker.bind(board.game).orElseThrow().card("attacker"));
    }

    /** Thrasta's own attack is the side 702.19f's counts hold already. */
    @Test
    void aGoneWalkersTramplerBindsNothing() {
        assertTrue(NoDamageToAGonePlaneswalker.bind(goneWalkerBoard(Cards.THRASTA).game).isEmpty());
    }

    @Test
    void theGoneWalkerRuleBindsNothingWhileThePlaneswalkerStays() {
        assertTrue(NoDamageToAGonePlaneswalker.bind(planeswalkerBoard("Craw Wurm", 3).game).isEmpty());
    }

    /** Alone in the step, the attacker would deal no damage in the live game, which then writes no combat record. */
    @Test
    void theGoneWalkerRuleBindsNothingWithoutAnotherCombatantsDamage() {
        Board board = goneWalkerBoard("Craw Wurm");
        Card bears = board.game.getCombat().getAttackers().stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        board.game.getCombat().removeFromCombat(bears);
        assertTrue(NoDamageToAGonePlaneswalker.bind(board.game).isEmpty());
    }

    @Test
    void theGoneWalkerRuleBindsNothingAtAnotherBoundary() {
        assertTrue(NoDamageToAGonePlaneswalker.bind(goneWalkerBoard("Craw Wurm",
                PhaseType.COMBAT_DECLARE_BLOCKERS).game).isEmpty());
    }

    @Test
    void theGoneWalkerRuleBindsNothingForAGrantedKeyword() {
        Board granted = goneWalkerBoard("Craw Wurm");
        granted.subject.addChangedCardKeywords(List.of("Trample:Planeswalker"), null, false,
                granted.game.getNextTimestamp(), null);
        assertTrue(NoDamageToAGonePlaneswalker.bind(granted.game).isEmpty());
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
    void aGoneWalkersAttackerForksThrasta() {
        Board board = goneWalkerBoard("Craw Wurm");
        assertReplacedAttackingTheGoneWalker(board, board.twin(NoDamageToAGonePlaneswalker.rule()),
                Cards.THRASTA);
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

    // ── 702.131a, ascend: add_permanent ─────────────────────────────────

    /** Eight Islands and Grizzly Bears of yours, Secrets of the Golden City in your hand, then {@code inHand}. */
    private static Board ascendBoard(String... inHand) {
        return ascendBoard(PhaseType.MAIN1, "Secrets of the Golden City", inHand);
    }

    /** {@link #ascendBoard(String...)} as {@code step} begins, with {@code spell} for Secrets. */
    private static Board ascendBoard(PhaseType step, String spell, String... inHand) {
        Board board = new Board(step);
        for (int i = 0; i < 8; i++) {
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
    void ascendBlessingBindsThePlayerAtNinePermanents() {
        Board board = ascendBoard();
        Binding binding = AscendBlessing.bind(board.game).orElseThrow();
        assertEquals(board.you, binding.player("player"));
        assertEquals(board.other, binding.card("spell"));
    }

    @Test
    void ascendBlessingBindsNothingAtTenPermanents() {
        Board board = ascendBoard();
        board.card(board.you, "Island");
        assertTrue(AscendBlessing.bind(board.game).isEmpty());
    }

    @Test
    void ascendBlessingBindsNothingAtAnotherBoundary() {
        assertTrue(AscendBlessing.bind(ascendBoard(PhaseType.COMBAT_DECLARE_ATTACKERS,
                "Secrets of the Golden City").game).isEmpty());
    }

    /** An ascend instant binds too: the fork forces its cast this turn (FR-010b). */
    @Test
    void ascendBlessingBindsAnInstant() {
        assertTrue(AscendBlessing.bind(ascendBoard(PhaseType.MAIN1, "Expel from Orazca").game).isPresent());
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
        board.you.getLandsInPlay().stream().limit(6).forEach(land -> land.setTapped(true));
        assertTrue(AscendBlessing.bind(board.game).isEmpty());
    }

    /** Forge counts a phased-out permanent toward ten (its battlefield zone), so the rule does too. */
    @Test
    void ascendBlessingCountsAPhasedOutPermanentAsForgeDoes() {
        Board board = ascendBoard();
        board.card(board.you, "Island").setPhasedOut(board.you);
        assertEquals(10, board.you.getZone(ZoneType.Battlefield).size());
        assertTrue(AscendBlessing.bind(board.game).isEmpty(), "ten with the phased-out one");
    }

    @Test
    void ascendBlessingForksATenthPermanentATappedIsland() {
        Board board = ascendBoard();
        Twin twin = board.twin(AscendBlessing.rule());
        List<Card> battlefield = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Battlefield));
        assertEquals(10, battlefield.size());
        Card tenth = battlefield.stream().filter(card -> !board.game.getCardsIn(ZoneType.Battlefield).contains(card))
                .findFirst().orElseThrow();
        assertEquals(Cards.ISLAND, tenth.getName());
        assertTrue(tenth.isTapped(), "the tenth makes no mana");
        for (String change : twin.changes()) {
            assertTrue(change.equals("+" + id(tenth)) || change.startsWith("+link ") && change.contains("=" + id(tenth)),
                    "adding changed " + change + "; all: " + twin.changes());
        }
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
    void plottedFromHandBindsACardTooExpensiveForTheTrigger() {
        Board board = kellanBoard("Hill Giant", "Serra Angel", "Forest");
        Binding binding = PlottedFromHand.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("spell"));
    }

    @Test
    void plottedFromHandBindsNothingWithACardTheTriggerCouldExile() {
        assertTrue(PlottedFromHand.bind(kellanBoard("Hill Giant", "Shock").game).isEmpty());
    }

    @Test
    void plottedFromHandBindsNothingWithOnlyLandsBesideKellan() {
        assertTrue(PlottedFromHand.bind(kellanBoard("Forest").game).isEmpty());
    }

    @Test
    void plottedFromHandBindsNothingWithoutTheLandsForKellan() {
        Board board = kellanBoard("Hill Giant");
        board.you.getLandsInPlay().forEach(land -> land.setTapped(true));
        assertTrue(PlottedFromHand.bind(board.game).isEmpty());
    }

    @Test
    void plottedFromHandBindsNothingAtAnotherBoundary() {
        assertTrue(PlottedFromHand.bind(kellanBoard(PhaseType.COMBAT_DECLARE_ATTACKERS, "Hill Giant").game)
                .isEmpty());
    }

    /**
     * Five lands would cast Kellan and the fork's Bears both, and Kellan's
     * trigger could find nothing to exile. Four leave one of them uncast.
     */
    @Test
    void plottedFromHandBindsNothingWhenBothCanBeCast() {
        Board board = kellanBoard("Hill Giant");
        board.card(board.you, "Forest");
        assertTrue(PlottedFromHand.bind(board.game).isPresent(), "four lands");
        board.card(board.you, "Forest");
        assertTrue(PlottedFromHand.bind(board.game).isEmpty(), "five lands");
    }

    @Test
    void plottedFromHandForksAGrizzlyBearsInTheCardsPlace() {
        Board board = kellanBoard("Hill Giant");
        Twin twin = board.twin(PlottedFromHand.rule());
        Card bears = twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand).stream()
                .filter(card -> card.getName().equals(Cards.BEARS)).findFirst().orElseThrow();
        assertOnlyChanged(twin, board.subject, bears);
    }

    // ── what makes a variant come undone (review C4, V3 #9-10) ──────────

    /**
     * C4 #2: a land in the holder's hand would be played as the turn goes on
     * and give back the mana a "one land short" variant takes, so the rule
     * does not bind; with the turn's land drop used it binds again.
     */
    private static void assertALandToPlayStopsTheBinding(Board board, DirectedRule rule) {
        assertTrue(rule.precondition(board.game).isPresent(), rule.name() + ": the base binds");
        TestBoard.inHand(board.game, board.you, "Forest");
        assertTrue(rule.precondition(board.game).isEmpty(), rule.name() + ": binds with a land to play");
        board.you.setLandsPlayedThisTurn(1);
        assertTrue(rule.precondition(board.game).isPresent(), rule.name() + ": the land drop is used");
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
    void improviseBindsNothingWithALandToPlay() {
        assertALandToPlayStopsTheBinding(engineerBoard(3), Improvise.rule());
    }

    /** A modal double-faced card whose back is a land is a land to play too. */
    @Test
    void aModalCardWithALandBackIsALandToPlay() {
        Board board = automatonBoard(3);
        TestBoard.inHand(board.game, board.you, "Turntimber Symbiosis");
        assertTrue(Prototype.bind(board.game).isEmpty());
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

    /** Serra Angel (4/4) lives with 3 damage; the Hill Giant (3/3) put in its place would not. */
    @Test
    void sharedKeywordsBindsNothingWhenTheHolderCarriesDamage() {
        Board board = concertedBoard("Serra Angel", "Grizzly Bears");
        assertTrue(SharedKeywords.bind(board.game).isPresent(), "Serra Angel has flying and vigilance");
        board.subject.setDamage(3);
        assertTrue(SharedKeywords.bind(board.game).isEmpty(), "Hill Giant is a 3/3");
    }

    /** C4 #6: taking the counter away kills a creature whose toughness rests on it. */
    @Test
    void unleashBindsNothingWhenTheCreatureLivesByItsCounter() {
        Board board = unleashBoard("Rakdos Cackler", 1);
        board.subject.setDamage(2);
        assertTrue(UnleashCantBlock.bind(board.game).isEmpty(), "a 2/2 Cackler with 2 damage dies at 1 toughness");
    }

    /**
     * C4 #8: the defender cut to the most anyone else has must not be left
     * in reach of the attack the fork's point is.
     */
    @Test
    void dethroneBindsNothingWhenTheCutWouldLeaveTheDefenderInReach() {
        Board board = dethroneBoard(20);
        board.you.setLife(3, null);
        assertTrue(Dethrone.bind(board.game).isEmpty(), "3 life against Enraged Revolutionary's 2 power and its dethrone counter");
        Board safe = dethroneBoard(20);
        safe.you.setLife(4, null);
        assertTrue(Dethrone.bind(safe.game).isPresent());
    }

    // ── what the twin itself catches ────────────────────────────────────

    /** A rule of one variant on the active player, for the twin's own tests. */
    private static DirectedRule onePlayerRule(ForkOperation operation,
                                              Function<Binding, BoardEdit> edit) {
        return DirectedRule.of("0.0", "0.0-1",
                live -> Optional.of(Binding.empty().with("player", Boards.active(live))),
                new DirectedRule.Variant("v", new Application(operation, edit)));
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
                live -> Optional.of(Binding.empty().with("creature", giant)), new DirectedRule.Variant("v",
                new Application(ForkOperation.COUNTERS,
                        fork -> new CountersOperation(fork.card("creature"), CounterEnumType.P1P1, 1))));
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

        /** {@code count} permanents named {@code name} of {@code owner}; the first, or null for none. */
        Card cards(Player owner, String name, int count) {
            Card first = null;
            for (int i = 0; i < count; i++) {
                Card card = card(owner, name);
                first = first == null ? card : first;
            }
            return first;
        }

        /** {@code attacker} attacks the opponent, no blocker declared yet. */
        void attacking(Card attacker) {
            Combat combat = new Combat(you);
            combat.addAttacker(attacker, opponent);
            game.getPhaseHandler().setCombat(combat);
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
            assertEquals(1, rule.variants().size(), rule.name() + " registers one variant");
            Binding binding = rule.precondition(game).orElseThrow(
                    () -> new AssertionError(rule.name() + ": the precondition does not hold"));
            Map<String, Object> real = BoardStates.of(game);
            GameCopier copier = new GameCopier(game);
            Game fork = copier.makeCopy();
            assertEquals(Set.of(), changes(real, BoardStates.of(fork)), "the unedited copy differs");
            assertTrue(TestForks.scoreChecks(game, copier, fork), rule.name() + ": the copy fails its score check");

            Binding forked = binding.inFork(copier);
            Application application = rule.variants().values().iterator().next();
            assertEquals(Optional.empty(), ForcedPlayRun.dryRun(application, binding, game),
                    rule.name() + ": the forced play's dry run on the live board");
            application.edit(forked).apply(fork);

            Map<String, Object> edited = BoardStates.of(fork);
            assertEquals(real, BoardStates.of(game), "the edit reached the live game");
            Map<String, String> placed = placed(fork);
            Set<String> tidied = keywordEffectCards(fork);
            fork.getAction().checkStateEffects(true);
            assertEquals(placed, placed(fork),
                    rule.name() + ": the first state-based check moved a card of the edited board");
            Map<String, Object> checked = BoardStates.of(fork);
            assertEquals(Set.of(), withoutTidied(changes(edited, checked), tidied, edited, checked),
                    rule.name() + ": the first state-based check changed the edited board");
            assertFalse(fork.isGameOver(), rule.name() + ": the first state-based check ended the fork's game");
            for (Player player : fork.getRegisteredPlayers()) {
                assertFalse(player.hasLost(), rule.name() + ": the first state-based check lost " + player);
            }
            assertEquals(real, BoardStates.of(game), "the check reached the live game");
            assertForcedPlayMayBeMade(rule, application, application.afterEdit(forked, fork), fork);
            return new Twin(changes(real, edited), real, edited, forked, fork);
        }

        /**
         * The variant's forced play (FR-010b), if it forces one, can be made
         * on the edited copy: its source, line or attack is there, and a cast
         * or an activation is playable now by the rules with its named
         * targets, the line the forcing seat would choose at this priority.
         */
        private static void assertForcedPlayMayBeMade(
                DirectedRule rule, Application application, Binding forked, Game fork) {
            if (application.forced() == null) {
                return;
            }
            ForcedPlayRun run = ForcedPlayRun.start(application, forked, fork);
            if (run.play().playsALine()) {
                assertTrue(run.choose(run.actor()) != null,
                        rule.name() + ": the forced line is playable on the edited copy");
            }
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
