package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.Binding;
import forge.StaticData;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Set;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static com.pricepredictor.connector.effects.directed.rules.RuleTwinChecks.assertOnlyAdded;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The registered directed rules of permanents added to the battlefield or
 * counted on it: a Mutagen token and a Virtuous Role (111.10v, 111.10p),
 * a Saga's last chapter (608.2p), a permanent that can't enter (614.17d),
 * bolster (701.39a) and ascend (702.131a). Each rule's scenario base
 * binds, a board missing it binds nothing, and the fork differs from its
 * twin by the variant's one fact, as {@link RegisteredRulesTest} sets out.
 */
@ExtendWith(ForgeExtension.class)
class RegisteredRulesPermanentsTest {

    // ── 111.10v, a Mutagen token: add_permanent, inverted ───────────────

    /** A Mutagen token beside Crustacean Commando, A's one Island {@code islandTapped}. */
    private static RuleBoard mutagenBoard(boolean islandTapped) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
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
        RuleBoard board = mutagenBoard(true);
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
        RuleBoard board = mutagenBoard(true);
        RuleTwin twin = board.twin(MutagenToken.rule());
        Card island = twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(Cards.ISLAND) && !card.isTapped())
                .findFirst().orElseThrow();
        for (String change : twin.changes()) {
            assertTrue(change.equals("+" + id(island)) || change.startsWith("+link ") && change.contains("=" + id(island))
                            || change.endsWith(".untapped_production"),
                    "adding changed " + change + "; all: " + twin.changes());
        }
    }

    // ── 111.10p, a Virtuous Role: add_permanent ─────────────────────────

    /** A's Grizzly Bears and four lands, Ellivere of the Wild Court in hand, as a main phase begins. */
    private static RuleBoard elliveresBoard(PhaseType phase, String... lands) {
        RuleBoard board = new RuleBoard(phase);
        for (String land : lands) {
            board.card(board.you, land);
        }
        board.other = board.card(board.you, "Grizzly Bears");
        board.subject = TestBoard.inHand(board.game, board.you, "Ellivere of the Wild Court");
        return board;
    }

    private static RuleBoard elliveresBoard() {
        return elliveresBoard(PhaseType.MAIN1, "Forest", "Forest", "Forest", "Plains");
    }

    @Test
    void virtuousRoleBindsTheCasterAndEllivere() {
        RuleBoard board = elliveresBoard();
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
        RuleBoard board = elliveresBoard();
        board.game.getAction().moveToGraveyard(board.other, null);
        assertTrue(VirtuousRole.bind(board.game).isEmpty());
    }

    @Test
    void virtuousRoleBindsNothingForACreatureWithoutTheRoleTrigger() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
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
        RuleBoard board = elliveresBoard();
        RuleTwin twin = board.twin(VirtuousRole.rule());
        assertOnlyAdded(twin, twin.fork().player("caster"), VirtuousRole.SEAL_OF_FIRE);
    }

    // ── 608.2p, a trigger on an ability resolving: counters ─────────────

    private static RuleBoard benaliaBoard(int lore) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.card(board.you, "Narci, Fable Singer");
        board.subject = board.card(board.you, "History of Benalia");
        board.subject.setCounters(CounterEnumType.LORE, lore);
        return board;
    }

    @Test
    void finalChapterResolvesBindsTheSagaTwoCountersShortOfItsLastChapter() {
        RuleBoard board = benaliaBoard(1);
        assertEquals(board.subject, FinalChapterResolves.bind(board.game).orElseThrow().card("saga"));
    }

    @Test
    void finalChapterResolvesBindsNothingWhenTheNextCounterReachesTheLastChapter() {
        assertTrue(FinalChapterResolves.bind(benaliaBoard(2).game).isEmpty());
    }

    @Test
    void finalChapterResolvesForksTheSagaWithACounterMore() {
        RuleBoard board = benaliaBoard(1);
        RuleTwin twin = board.twin(FinalChapterResolves.rule());
        String saga = id(board.subject);
        assertEquals(Set.of(saga + ".counters"), twin.changes());
        assertEquals(2, twin.fork().card("saga").getCounters(CounterEnumType.LORE));
    }

    // ── 614.17d, a permanent that can't enter: add_permanent ────────────

    private static RuleBoard zombifyBoard(boolean creatureInGraveyard) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
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
        RuleBoard board = zombifyBoard(true);
        assertEquals(board.you, CantEnterBattlefield.bind(board.game).orElseThrow().player("player"));
    }

    @Test
    void cantEnterBattlefieldBindsNothingWithNoCreatureCardToReturn() {
        assertTrue(CantEnterBattlefield.bind(zombifyBoard(false).game).isEmpty());
    }

    @Test
    void cantEnterBattlefieldBindsNothingWhenACageIsOut() {
        RuleBoard board = zombifyBoard(true);
        board.card(board.opponent, CantEnterBattlefield.GRAFDIGGERS_CAGE);
        assertTrue(CantEnterBattlefield.bind(board.game).isEmpty());
    }

    @Test
    void cantEnterBattlefieldForksTheBoardWithTheCage() {
        RuleBoard board = zombifyBoard(true);
        RuleTwin twin = board.twin(CantEnterBattlefield.rule());
        assertOnlyAdded(twin, twin.fork().player("player"), CantEnterBattlefield.GRAFDIGGERS_CAGE);
    }

    // ── 701.39a, bolster: add_permanent ─────────────────────────────────

    /** A's Cached Defenses over three Forests, beside A's {@code creatures}; the first is the subject. */
    private static RuleBoard bolsterBoard(String... creatures) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
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
        RuleBoard board = bolsterBoard();
        Card defenses = board.you.getCardsIn(ZoneType.Hand).get(0);
        assertEquals(defenses, Bolster.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void bolsterBindsNothingWhenThePlayerControlsACreature() {
        assertTrue(Bolster.bind(bolsterBoard("Grizzly Bears").game).isEmpty());
    }

    @Test
    void bolsterForksGrizzlyBearsUnderThePlayer() {
        RuleBoard board = bolsterBoard();
        RuleTwin twin = board.twin(Bolster.rule());
        assertOnlyAdded(twin, twin.fork().player("player"), Cards.BEARS);
    }

    // ── 702.131a, ascend: add_permanent ─────────────────────────────────

    /** Eight Islands and Grizzly Bears of yours, Secrets of the Golden City in your hand, then {@code inHand}. */
    private static RuleBoard ascendBoard(String... inHand) {
        return ascendBoard(PhaseType.MAIN1, "Secrets of the Golden City", inHand);
    }

    /** {@link #ascendBoard(String...)} as {@code step} begins, with {@code spell} for Secrets. */
    private static RuleBoard ascendBoard(PhaseType step, String spell, String... inHand) {
        RuleBoard board = new RuleBoard(step);
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
        RuleBoard board = ascendBoard();
        Binding binding = AscendBlessing.bind(board.game).orElseThrow();
        assertEquals(board.you, binding.player("player"));
        assertEquals(board.other, binding.card("spell"));
    }

    @Test
    void ascendBlessingBindsNothingAtTenPermanents() {
        RuleBoard board = ascendBoard();
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
        RuleBoard board = ascendBoard();
        board.you.setBlessing(true, "RIX");
        assertTrue(AscendBlessing.bind(board.game).isEmpty());
    }

    @Test
    void ascendBlessingBindsNothingWithoutTheLandsForTheSpell() {
        RuleBoard board = ascendBoard();
        board.you.getLandsInPlay().stream().limit(6).forEach(land -> land.setTapped(true));
        assertTrue(AscendBlessing.bind(board.game).isEmpty());
    }

    /** Forge counts a phased-out permanent toward ten (its battlefield zone), so the rule does too. */
    @Test
    void ascendBlessingCountsAPhasedOutPermanentAsForgeDoes() {
        RuleBoard board = ascendBoard();
        board.card(board.you, "Island").setPhasedOut(board.you);
        assertEquals(10, board.you.getZone(ZoneType.Battlefield).size());
        assertTrue(AscendBlessing.bind(board.game).isEmpty(), "ten with the phased-out one");
    }

    @Test
    void ascendBlessingForksATenthPermanentATappedIsland() {
        RuleBoard board = ascendBoard();
        RuleTwin twin = board.twin(AscendBlessing.rule());
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
}
