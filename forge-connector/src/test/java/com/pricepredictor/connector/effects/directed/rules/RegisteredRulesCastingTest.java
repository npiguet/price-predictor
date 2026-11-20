package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.Binding;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Set;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static com.pricepredictor.connector.effects.directed.rules.CardName.named;
import static com.pricepredictor.connector.effects.directed.rules.RuleTwinChecks.assertOnlyAdded;
import static com.pricepredictor.connector.effects.directed.rules.RuleTwinChecks.assertOnlyChanged;
import static com.pricepredictor.connector.effects.directed.rules.RuleTwinChecks.assertOnlyRemoved;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The registered directed rules of casting and costs: aftermath,
 * jump-start, unearth, mayhem and plot (702.127a, 702.133a, 702.84a,
 * 702.187b, 702.170c), turning a morph face up (405.6d), one target
 * chosen once (601.2c), improvise and web-slinging's costs (702.126a,
 * 702.188a) and a prototype's cost (718.3a). Each rule's scenario base
 * binds, a board missing it binds nothing, and the fork differs from its
 * twin by the variant's one fact, as {@link RegisteredRulesTest} sets out.
 */
@ExtendWith(ForgeExtension.class)
class RegisteredRulesCastingTest {

    // ── 702.127a, aftermath: move_card ──────────────────────────────────

    @Test
    void aftermathBindsTheSplitCardInTheGraveyard() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.subject = TestBoard.inGraveyard(board.game, board.you, "Cut // Ribbons");
        assertEquals(board.subject, Aftermath.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void aftermathBindsNothingInHand() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        TestBoard.inHand(board.game, board.you, "Cut // Ribbons");
        assertTrue(Aftermath.bind(board.game).isEmpty());
    }

    @Test
    void aftermathForksTheCardInHand() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.subject = TestBoard.inGraveyard(board.game, board.you, "Cut // Ribbons");
        RuleTwin twin = board.twin(Aftermath.rule());
        String cut = id(board.subject);
        assertTrue(twin.changes().containsAll(Set.of("P0.hand", "P0.graveyard")), twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(Set.of("P0.hand", "P0.graveyard").contains(change) || change.startsWith(cut + "."),
                    "moving changed " + change + "; all: " + twin.changes());
        }
        assertTrue(twin.fork().card("card").getZone().is(ZoneType.Hand));
    }

    // ── 702.133a, jump-start: move_card ─────────────────────────────────

    /** Radical Idea in A's graveyard and {@code islands} Islands, as A's first main phase begins. */
    private static RuleBoard jumpStartBoard(CardName card, int islands) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        for (int i = 0; i < islands; i++) {
            board.card(board.you, "Island");
        }
        board.subject = board.game.getAction().moveToGraveyard(TestBoard.card(board.game, board.you, card.name()), null);
        return board;
    }

    @Test
    void jumpStartFromHandBindsTheCardInTheGraveyard() {
        RuleBoard board = jumpStartBoard(named("Radical Idea"), 2);
        Binding binding = JumpStartFromHand.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.you, binding.player("caster"));
    }

    @Test
    void jumpStartFromHandBindsNothingWithoutTheLands() {
        assertTrue(JumpStartFromHand.bind(jumpStartBoard(named("Radical Idea"), 1).game).isEmpty());
    }

    @Test
    void jumpStartFromHandBindsNothingForACardWithoutJumpStart() {
        assertTrue(JumpStartFromHand.bind(jumpStartBoard(named("Divination"), 3).game).isEmpty());
    }

    /** Quasiduplicate needs a target, which the forced cast's clause would draw: a spell with one is no board the rule binds. */
    @Test
    void jumpStartFromHandBindsNothingForASpellThatTargets() {
        assertTrue(JumpStartFromHand.bind(jumpStartBoard(named("Quasiduplicate"), 3).game).isEmpty());
    }

    @Test
    void jumpStartFromHandForksTheCardInTheHand() {
        RuleBoard board = jumpStartBoard(named("Radical Idea"), 2);
        RuleTwin twin = board.twin(JumpStartFromHand.rule());
        assertTrue(twin.fork().player("caster").getCardsIn(ZoneType.Hand).stream()
                .anyMatch(card -> card.getName().equals("Radical Idea")));
        assertTrue(twin.fork().player("caster").getCardsIn(ZoneType.Graveyard).isEmpty());
    }

    // ── 405.6d, a special action: tap ───────────────────────────────────

    /** A's face-down Exalted Angel (morph {2}{W}{W}) beside {@code plains} untapped Plains; the first is {@code other}. */
    static RuleBoard angelBoard(int plains) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
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
        RuleBoard board = angelBoard(4);
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
        RuleBoard board = angelBoard(4);
        RuleTwin twin = board.twin(MorphFaceUp.rule());
        String land = id(board.other);
        assertTrue(twin.changes().contains(land + ".tapped"), twin.changes().toString());
        assertTrue(Set.of(land + ".tapped", "P0.untapped_production").containsAll(twin.changes()),
                twin.changes().toString());
        assertTrue(twin.fork().card("land").isTapped());
    }

    // ── 601.2c, one target chosen once: add_permanent ───────────────────

    private static RuleBoard undoBoard() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Island");
        }
        board.subject = board.card(board.opponent, "Grizzly Bears");
        board.other = TestBoard.inHand(board.game, board.you, "Undo");
        return board;
    }

    @Test
    void distinctTargetsBindsTheSpellAndTheOpponent() {
        RuleBoard board = undoBoard();
        Binding binding = DistinctTargets.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("spell"));
        assertEquals(board.opponent, binding.player("opponent"));
    }

    @Test
    void distinctTargetsBindsNothingWithTwoCreatures() {
        RuleBoard board = undoBoard();
        board.card(board.opponent, "Hill Giant");
        assertTrue(DistinctTargets.bind(board.game).isEmpty());
    }

    @Test
    void distinctTargetsForksASecondCreature() {
        RuleBoard board = undoBoard();
        RuleTwin twin = board.twin(DistinctTargets.rule());
        assertOnlyAdded(twin, twin.fork().player("opponent"), Cards.WURM);
    }

    // ── 702.84a, unearth from the graveyard only: move_card ─────────────

    private static RuleBoard dregscapeBoard(boolean swampUntapped) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.card(board.you, "Swamp").setTapped(!swampUntapped);
        board.subject = TestBoard.inGraveyard(board.game, board.you, "Dregscape Zombie");
        return board;
    }

    @Test
    void unearthBindsTheCardInTheGraveyard() {
        RuleBoard board = dregscapeBoard(true);
        assertEquals(board.subject, Unearth.bind(board.game).orElseThrow().card("card"));
    }

    @Test
    void unearthBindsNothingWithoutTheManaToUnearth() {
        assertTrue(Unearth.bind(dregscapeBoard(false).game).isEmpty());
    }

    @Test
    void unearthForksTheCardInHand() {
        RuleBoard board = dregscapeBoard(true);
        RuleTwin twin = board.twin(Unearth.rule());
        assertOnlyChanged(twin, board.subject, board.subject, "P0.hand", "P0.graveyard");
        assertTrue(twin.fork().card("card").getZone().is(ZoneType.Hand));
    }

    // ── 702.126a, improvise: tap ────────────────────────────────────────

    /** A's Reverse Engineer ({3}{U}{U}) in hand over two Islands and {@code ornithopters} Ornithopters; the first is the subject. */
    static RuleBoard engineerBoard(int ornithopters) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
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
        RuleBoard board = engineerBoard(3);
        assertEquals(board.subject, Improvise.bind(board.game).orElseThrow().card("artifact"));
    }

    @Test
    void improviseBindsNothingWithAnArtifactShortOrToSpare() {
        assertTrue(Improvise.bind(engineerBoard(2).game).isEmpty());
        assertTrue(Improvise.bind(engineerBoard(4).game).isEmpty());
    }

    @Test
    void improviseForksTheArtifactTapped() {
        RuleBoard board = engineerBoard(3);
        RuleTwin twin = board.twin(Improvise.rule());
        assertEquals(Set.of(id(board.subject) + ".tapped"), twin.changes());
        assertTrue(twin.fork().card("artifact").isTapped());
    }

    // ── 702.187b, mayhem from a card discarded this turn: remove_permanent

    /** A's Electro's Bolt (mayhem {1}{R}) in the graveyard, discarded this turn when {@code discarded}, over {@code mountains} Mountains. */
    static RuleBoard boltBoard(boolean discarded, int mountains) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN2);
        board.subject = board.cards(board.you, "Mountain", mountains);
        board.other = TestBoard.inGraveyard(board.game, board.you, "Electro's Bolt");
        board.other.setDiscarded(discarded);
        return board;
    }

    @Test
    void mayhemBindsALandTheMayhemCostNeeds() {
        RuleBoard board = boltBoard(true, 2);
        assertEquals(board.subject, Mayhem.bind(board.game).orElseThrow().card("land"));
    }

    @Test
    void mayhemBindsNothingForACardNotDiscardedOrWithALandToSpare() {
        assertTrue(Mayhem.bind(boltBoard(false, 2).game).isEmpty());
        assertTrue(Mayhem.bind(boltBoard(true, 3).game).isEmpty());
    }

    @Test
    void mayhemForksTheBoardWithoutTheLand() {
        RuleBoard board = boltBoard(true, 2);
        RuleTwin twin = board.twin(Mayhem.rule());
        assertOnlyRemoved(twin, board.subject, "land");
        Card bolt = twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Graveyard).get(0);
        assertTrue(bolt.wasDiscarded() && bolt.enteredThisTurn(), "the fork's Bolt was discarded this turn");
    }

    // ── 702.188a, web-slinging's tapped creature: tap ───────────────────

    /** A's Spider-Man, Web-Slinger in hand over a Plains and Grizzly Bears, tapped when {@code tapped}. */
    private static RuleBoard spiderManBoard(boolean tapped) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN2);
        board.card(board.you, "Plains");
        board.subject = board.card(board.you, "Grizzly Bears");
        board.subject.setTapped(tapped);
        TestBoard.inHand(board.game, board.you, "Spider-Man, Web-Slinger");
        return board;
    }

    @Test
    void webSlingingBindsTheOnlyTappedCreature() {
        RuleBoard board = spiderManBoard(true);
        assertEquals(board.subject, WebSlinging.bind(board.game).orElseThrow().card("creature"));
    }

    @Test
    void webSlingingBindsNothingWithoutATappedCreature() {
        assertTrue(WebSlinging.bind(spiderManBoard(false).game).isEmpty());
    }

    @Test
    void webSlingingForksTheCreatureUntapped() {
        RuleBoard board = spiderManBoard(true);
        RuleTwin twin = board.twin(WebSlinging.rule());
        assertEquals(Set.of(id(board.subject) + ".tapped"), twin.changes());
        assertFalse(twin.fork().card("creature").isTapped());
    }

    // ── 718.3a, a prototype's cost alone: remove_permanent ──────────────

    /** A's Blitz Automaton (prototype {2}{R}, mana value 7) in hand over {@code mountains} Mountains; the first is the subject. */
    static RuleBoard automatonBoard(int mountains) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.subject = board.cards(board.you, "Mountain", mountains);
        TestBoard.inHand(board.game, board.you, "Blitz Automaton");
        return board;
    }

    @Test
    void prototypeBindsALandThePrototypeCostNeeds() {
        RuleBoard board = automatonBoard(3);
        assertEquals(board.subject, Prototype.bind(board.game).orElseThrow().card("land"));
    }

    @Test
    void prototypeBindsNothingWithALandToSpare() {
        assertTrue(Prototype.bind(automatonBoard(4).game).isEmpty());
    }

    @Test
    void prototypeForksTheBoardWithoutTheLand() {
        RuleBoard board = automatonBoard(3);
        assertOnlyRemoved(board.twin(Prototype.rule()), board.subject, "land");
    }

    // ── 702.170c, a card becoming plotted: move_card ────────────────────

    /** A Forest, a Plains and an Island of yours, Kellan Joins Up in your hand, then {@code inHand}. */
    private static RuleBoard kellanBoard(Hand inHand) {
        return kellanBoard(PhaseType.MAIN1, inHand);
    }

    /** {@link #kellanBoard(Hand)} as {@code step} begins. */
    private static RuleBoard kellanBoard(PhaseType step, Hand inHand) {
        RuleBoard board = new RuleBoard(step);
        for (String land : List.of("Forest", "Plains", "Island")) {
            board.card(board.you, land);
        }
        board.other = TestBoard.inHand(board.game, board.you, PlottedFromHand.KELLAN);
        for (String name : inHand.names()) {
            Card card = TestBoard.inHand(board.game, board.you, name);
            if (board.subject == null) {
                board.subject = card;
            }
        }
        return board;
    }

    @Test
    void plottedFromHandBindsACardTooExpensiveForTheTrigger() {
        RuleBoard board = kellanBoard(Hand.of("Hill Giant", "Serra Angel", "Forest"));
        Binding binding = PlottedFromHand.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("spell"));
    }

    @Test
    void plottedFromHandBindsNothingWithACardTheTriggerCouldExile() {
        assertTrue(PlottedFromHand.bind(kellanBoard(Hand.of("Hill Giant", "Shock")).game).isEmpty());
    }

    @Test
    void plottedFromHandBindsNothingWithOnlyLandsBesideKellan() {
        assertTrue(PlottedFromHand.bind(kellanBoard(Hand.of("Forest")).game).isEmpty());
    }

    @Test
    void plottedFromHandBindsNothingWithoutTheLandsForKellan() {
        RuleBoard board = kellanBoard(Hand.of("Hill Giant"));
        board.you.getLandsInPlay().forEach(land -> land.setTapped(true));
        assertTrue(PlottedFromHand.bind(board.game).isEmpty());
    }

    @Test
    void plottedFromHandBindsNothingAtAnotherBoundary() {
        assertTrue(PlottedFromHand.bind(kellanBoard(PhaseType.COMBAT_DECLARE_ATTACKERS, Hand.of("Hill Giant")).game)
                .isEmpty());
    }

    /**
     * Five lands would cast Kellan and the fork's Bears both, and Kellan's
     * trigger could find nothing to exile. Four leave one of them uncast.
     */
    @Test
    void plottedFromHandBindsNothingWhenBothCanBeCast() {
        RuleBoard board = kellanBoard(Hand.of("Hill Giant"));
        board.card(board.you, "Forest");
        assertTrue(PlottedFromHand.bind(board.game).isPresent(), "four lands");
        board.card(board.you, "Forest");
        assertTrue(PlottedFromHand.bind(board.game).isEmpty(), "five lands");
    }

    @Test
    void plottedFromHandForksAGrizzlyBearsInTheCardsPlace() {
        RuleBoard board = kellanBoard(Hand.of("Hill Giant"));
        RuleTwin twin = board.twin(PlottedFromHand.rule());
        Card bears = twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand).stream()
                .filter(card -> card.getName().equals(Cards.BEARS)).findFirst().orElseThrow();
        assertOnlyChanged(twin, board.subject, bears);
    }
}
