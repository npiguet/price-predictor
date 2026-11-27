package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.CardStateName;
import forge.card.MagicColor;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.mana.Mana;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static com.pricepredictor.connector.effects.SnapshotReading.entitiesIn;
import static com.pricepredictor.connector.effects.SnapshotReading.entityOf;
import static com.pricepredictor.connector.effects.SnapshotReading.memberOfEach;
import static com.pricepredictor.connector.effects.SnapshotReading.valuesOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** A card slot's facts (FR-024), and its base power and toughness apart from its boosts (FR-025). */
@ExtendWith(ForgeExtension.class)
class SnapshotCardFactsTest {

    /**
     * A phased-out permanent is still on the battlefield and in the
     * snapshot, marked phased out.
     */
    @Test
    void aPhasedOutPermanentIsAnEntity() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        bears.phase(false);
        assertTrue(bears.isPhasedOut());

        Map<String, Object> entity = entitiesIn(new SnapshotBuilder(board.game).ofBoard().render())
                .stream().filter(e -> Refs.of(bears).equals(e.get("id")))
                .findFirst().orElseThrow(() -> new AssertionError("no phased-out bears"));
        assertEquals(List.of("phased_out"), entity.get("designations"));
    }

    /** A phased-out permanent takes a battlefield rank of its own, no other's. */
    @Test
    void aPhasedOutPermanentsRankIsItsOwn() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        bears.setGameTimestamp(board.game.getNextTimestamp());
        Card elves = board.battlefield("Llanowar Elves", board.you);
        elves.setGameTimestamp(board.game.getNextTimestamp());
        Card angel = board.battlefield("Serra Angel", board.you);
        angel.setGameTimestamp(board.game.getNextTimestamp());
        elves.phase(false);

        List<Map<String, Object>> entities = entitiesIn(
                new SnapshotBuilder(board.game).ofBoard().render());
        List<Object> ranks = Stream.of(bears, elves, angel)
                .map(card -> entities.stream().filter(e -> Refs.of(card).equals(e.get("id")))
                        .findFirst().orElseThrow().get("order_rank"))
                .toList();
        assertEquals(List.of(0L, 1L, 2L), ranks);
    }

    @Test
    void faceStateNamesTheFaceUp() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card delver = board.battlefield("Delver of Secrets", board.you);
        delver.setState(CardStateName.Backside, true);
        Card bushi = board.battlefield("Bushi Tenderfoot", board.you);
        bushi.setState(CardStateName.Flipped, true);
        Card guide = board.battlefield("Foothill Guide", board.you);
        guide.turnFaceDown(true);

        assertEquals(List.of("front", "back", "flipped", "face_down"),
                memberOfEach(board, "face_state", bears, delver, bushi, guide));
    }

    @Test
    void designationsAreSortedNames() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        bears.setRenowned(true);
        bears.setMonstrous(true);
        bears.addGoad(board.game.getNextTimestamp(), board.opponent);

        assertEquals(List.of("goaded", "monstrous", "renowned"),
                entityOf(board, bears).get("designations"));
        assertEquals(List.of(), entityOf(board, board.battlefield("Forest", board.you))
                .get("designations"));
    }

    @Test
    void xPaidAndCostsPaidAreTheSpells() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card mountain = board.battlefield("Mountain", board.you);
        Card fireball = board.hand("Fireball", board.you);
        Card onStack = board.game.getAction().moveToStack(fireball, null);
        onStack.setCastFrom(board.you.getZone(ZoneType.Hand));
        SpellAbility spell = board.spell(onStack, board.you);
        spell.setXManaCostPaid(2);
        for (int i = 0; i < 3; i++) {
            spell.getPayingMana().add(
                    new Mana(MagicColor.RED, mountain, null, board.you));
        }
        spell.resetTargets();
        spell.getTargets().add(board.opponent);
        board.game.getStack().add(spell);

        Map<String, Object> entity = entityOf(board, onStack);

        assertEquals(2L, entity.get("x_paid"));
        Map<String, Object> paid = RecordLines.asMap(entity.get("costs_paid"));
        assertEquals(3L, RecordLines.asMap(paid.get("mana")).get("R"));
        assertEquals("hand", paid.get("cast_from"));
        assertEquals(false, paid.get("mana_rider"));
        Map<String, Object> bears = entityOf(board, board.hand("Grizzly Bears", board.you));
        assertNull(bears.get("x_paid"));
        assertNull(bears.get("costs_paid"));
    }

    @Test
    void chosenValuesAreWhatTheCardRemembers() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card cavern = board.battlefield("Cavern of Souls", board.you);
        cavern.setChosenType("Elf");
        cavern.setChosenColors(List.of("blue"));

        Map<String, Object> chosen = RecordLines.asMap(entityOf(board, cavern).get("chosen"));

        assertEquals("Elf", chosen.get("type"));
        assertEquals(List.of("blue"), chosen.get("colors"));
        assertEquals(Map.of(), entityOf(board, board.battlefield("Forest", board.you))
                .get("chosen"));
    }

    @Test
    void historyBitsAreThisTurns() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card fresh = board.battlefield("Grizzly Bears", board.you);
        Card old = board.battlefield("Hill Giant", board.you);
        old.setTurnInZone(0);
        old.addBlockedThisTurn(fresh);
        Card elspeth = board.battlefield("Elspeth, Knight-Errant", board.you);
        elspeth.addPlaneswalkerAbilityActivated();

        Map<String, Object> freshHistory = RecordLines.asMap(entityOf(board, fresh).get("history"));
        Map<String, Object> oldHistory = RecordLines.asMap(entityOf(board, old).get("history"));

        assertEquals(List.of(true, false),
                valuesOf(freshHistory, "entered_this_turn", "blocked_this_turn"), "the fresh bears");
        assertEquals(List.of(false, true),
                valuesOf(oldHistory, "entered_this_turn", "blocked_this_turn"), "the old giant");
        assertEquals(true, RecordLines.asMap(entityOf(board, elspeth).get("history"))
                .get("loyalty_activated"));
    }

    @Test
    void orderRankIsTimestampOrderOnTheBattlefieldAndPositionElsewhere() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card older = board.battlefield("Grizzly Bears", board.you);
        Card newer = board.battlefield("Hill Giant", board.opponent);
        Card first = board.graveyard("Shock", board.you);
        Card second = board.graveyard("Giant Growth", board.you);

        assertEquals(List.of(0L, 1L, 0L, 1L),
                memberOfEach(board, "order_rank", older, newer, first, second));
    }

    @Test
    void aFaceDownCardCarriesItsRealFace() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card guide = board.battlefield("Foothill Guide", board.you);
        assertNull(entityOf(board, guide).get("real_face"));
        guide.turnFaceDown(true);

        Map<String, Object> real = RecordLines.asMap(entityOf(board, guide).get("real_face"));

        assertEquals("Foothill Guide", real.get("name"));
        assertEquals(List.of("creature"), real.get("types"));
        assertEquals(List.of(1L, 1L), real.get("base_pt"));
    }

    @Test
    void printedLoyaltyAndDefense() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Map<String, Object> elspeth = entityOf(board,
                board.battlefield("Elspeth, Knight-Errant", board.you));
        Map<String, Object> invasion = entityOf(board,
                board.hand("Invasion of Zendikar", board.you));
        Map<String, Object> bears = entityOf(board, board.hand("Grizzly Bears", board.you));

        assertEquals(Arrays.asList(4L, null, 3L, null), Arrays.asList(
                elspeth.get("printed_loyalty"), elspeth.get("printed_defense"),
                invasion.get("printed_defense"), bears.get("printed_loyalty")));
    }

    @Test
    void manaCostCountsEachSymbolKind() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);

        assertEquals(
                Arrays.asList(Map.of("generic", 1L, "G/W", 2L), Map.of("U/P", 1L),
                        Map.of("X", 1L, "R", 1L), Map.of(), null),
                Stream.of("Kitchen Finks", "Gitaxian Probe", "Fireball", "Ornithopter", "Forest")
                        .map(name -> entityOf(board, board.hand(name, board.you)).get("mana_cost"))
                        .toList());
    }

    // ── base power and toughness (FR-025) ───────────────────────────────

    /**
     * Ensoul Artifact sets the base (layer 7b): Ornithopter's base is
     * 5/5, not its printed 0/2, and nothing is a boost.
     */
    @Test
    void ensoulArtifactSetsTheBase() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card thopter = board.battlefield("Ornithopter", board.you);
        board.battlefield("Ensoul Artifact", board.you).attachToEntity(thopter, null);
        board.game.getAction().checkStaticAbilities();

        Map<String, Object> entity = entityOf(board, thopter);
        assertEquals(List.of(5L, 5L), entity.get("base_pt"));
        assertEquals(Map.of("temporary", List.of(0L, 0L), "counters", List.of(0L, 0L)),
                entity.get("pt_boosts"));
    }

    /**
     * Boosts apart from the base: an anthem's +1/+1 is temporary, and
     * every counter type adds what Forge counts for it.
     */
    @Test
    void boostsAndEveryCountersContributionAreApartFromTheBase() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        board.battlefield("Glorious Anthem", board.you);
        bears.addCounterInternal(CounterEnumType.P1P1, 1, board.you, false, null, null);
        bears.addCounterInternal(CounterEnumType.P1P2, 1, board.you, false, null, null);
        board.game.getAction().checkStaticAbilities();

        Map<String, Object> entity = entityOf(board, bears);
        assertEquals(List.of(2L, 2L), entity.get("base_pt"));
        assertEquals(Map.of("temporary", List.of(1L, 1L), "counters", List.of(2L, 3L)),
                entity.get("pt_boosts"));
        assertEquals(2, RecordLines.asMap(entity.get("counters")).size(),
                "both counter types, each by name: " + entity.get("counters"));
    }

    @Test
    void aNoncreatureHasNoPowerOrToughness() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Map<String, Object> entity = entityOf(board, board.battlefield("Forest", board.you));

        assertNull(entity.get("base_pt"));
        assertNull(entity.get("pt_boosts"));
    }
}
