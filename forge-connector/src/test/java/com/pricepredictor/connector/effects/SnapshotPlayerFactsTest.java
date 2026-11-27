package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityKey;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.SnapshotReading.entityOf;
import static com.pricepredictor.connector.effects.SnapshotReading.globalOf;
import static com.pricepredictor.connector.effects.SnapshotReading.playerOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** A player's facts and the global ones (FR-024), and an entity's owner. */
@ExtendWith(ForgeExtension.class)
class SnapshotPlayerFactsTest {

    @Test
    void speedBlessingRadMonarchAndInitiative() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.you.setSpeed(2);
        board.you.setBlessing(true, null);
        board.opponent.addCounterInternal(
                CounterEnumType.RAD, 3, board.you, false, null, AbilityKey.newMap());
        board.game.setMonarch(board.you);
        board.game.setHasInitiative(board.opponent);

        Map<String, Object> you = playerOf(board, 0);
        Map<String, Object> them = playerOf(board, 1);

        assertEquals(2L, you.get("speed"));
        assertEquals(true, you.get("city_blessing"));
        assertEquals(false, them.get("city_blessing"));
        assertEquals(3L, them.get("rad"));
        assertEquals(true, you.get("monarch"));
        assertEquals(false, them.get("monarch"));
        assertEquals(false, you.get("initiative"));
        assertEquals(true, them.get("initiative"));
    }

    @Test
    void theTurnsCountsBesideThisTurn() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.game.getAction().moveToLibrary(board.card("Forest", board.you), null, null);
        board.you.drawCard();
        board.you.gainLife(3, null, null);
        board.opponent.loseLife(2, false, false, null);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        board.game.getAction().moveTo(ZoneType.Graveyard, bears, null, null);

        Map<String, Object> you = RecordLines.asMap(playerOf(board, 0).get("turn_counts"));
        Map<String, Object> them = RecordLines.asMap(playerOf(board, 1).get("turn_counts"));

        assertEquals(1L, you.get("cards_drawn"));
        assertEquals(3L, you.get("life_gained"));
        assertEquals(2L, them.get("life_lost"));
        assertEquals(1L, you.get("permanents_to_graveyard"));
        assertEquals(1L, you.get("descended"));
        assertEquals(Set.of("cards_drawn", "life_gained", "life_lost", "damage_taken",
                "permanents_to_graveyard", "spells_cast_last_turn", "dungeons_completed",
                "descended"), you.keySet());
    }

    /**
     * Bloom Tender makes one mana of each colour among its controller's
     * permanents. The opponent's AI lists every battlefield's abilities
     * as its own candidates, which leaves the opponent as the activating
     * player of Bloom Tender's mana ability: read that way, its
     * production counted the opponent's colours. The read changes
     * nothing on the ability.
     */
    @Test
    void aManaAbilityReadingTheBoardMakesItsControllersColours() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card tender = board.battlefield("Bloom Tender", board.you);
        tender.setSickness(false);
        board.battlefield("Walking Corpse", board.you);
        board.battlefield("Serra Angel", board.opponent);
        forge.ai.ComputerUtilAbility.getSpellAbilities(
                forge.ai.ComputerUtilAbility.getAvailableCards(board.game, board.opponent),
                board.opponent);
        SpellAbility mana = tender.getManaAbilities().get(0);
        assertEquals(board.opponent, mana.getActivatingPlayer(), "the AI's scan left the opponent");
        String choice = mana.getManaPart().getExpressChoice();

        Map<String, Object> production = RecordLines.asMap(playerOf(board, 0).get("untapped_production"));

        assertEquals(Map.of("W", 0L, "U", 0L, "B", 1L, "R", 0L, "G", 1L, "C", 0L), production);
        assertEquals(board.opponent, mana.getActivatingPlayer());
        assertEquals(choice, mana.getManaPart().getExpressChoice());
    }

    /**
     * Only mana abilities that can be activated now count: Llanowar Elves
     * that came in this turn cannot pay its {T}, a Forest that did can.
     */
    @Test
    void aSummoningSickCreaturesManaAbilityMakesNothing() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card elves = board.battlefield("Llanowar Elves", board.you);
        assertTrue(elves.hasSickness());

        assertEquals(0L, production(board).get("G"), "the sick elves make nothing");

        board.battlefield("Forest", board.you);
        assertEquals(1L, production(board).get("G"), "an untapped land counts");

        elves.setSickness(false);
        assertEquals(2L, production(board).get("G"), "the elves count once they may tap");
    }

    /**
     * An ability its own restrictions forbid counts nothing: Mana Bloom's
     * "activate only once each turn" once activated this turn, or with no
     * charge counter to remove; Ashnod's Altar with no creature to
     * sacrifice.
     */
    @Test
    void aManaAbilityItsRestrictionsForbidMakesNothing() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bloom = board.battlefield("Mana Bloom", board.you);
        board.battlefield("Ashnod's Altar", board.you);
        Map<String, Object> none = Map.of("W", 0L, "U", 0L, "B", 0L, "R", 0L, "G", 0L, "C", 0L);

        assertEquals(none, production(board), "no charge counter, no creature to sacrifice");

        bloom.addCounterInternal(CounterEnumType.CHARGE, 2, board.you, false, null, new java.util.HashMap<>());
        board.battlefield("Walking Corpse", board.you);
        assertEquals(Map.of("W", 1L, "U", 1L, "B", 1L, "R", 1L, "G", 1L, "C", 1L), production(board));

        bloom.addAbilityActivated(bloom.getManaAbilities().get(0));
        assertEquals(Map.of("W", 0L, "U", 0L, "B", 0L, "R", 0L, "G", 0L, "C", 1L), production(board),
                "Mana Bloom was activated this turn; the altar still has a creature to sacrifice");
    }

    /** Reading what a player could make draws nothing from Forge's global random stream. */
    @Test
    void untappedProductionDrawsNothingFromTheRandomStream() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Forest", board.you);
        board.battlefield("Llanowar Elves", board.you).setSickness(false);
        board.battlefield("Bloom Tender", board.you);
        board.battlefield("Ashnod's Altar", board.you);
        java.util.Random was = forge.util.MyRandom.getRandom();
        forge.util.MyRandom.setRandom(new java.util.Random(7L));
        java.util.Random expected = new java.util.Random(7L);
        try {
            assertEquals(2L, production(board).get("G"));
            assertEquals(expected.nextLong(), forge.util.MyRandom.getRandom().nextLong(),
                    "the read drew from MyRandom");
        } finally {
            forge.util.MyRandom.setRandom(was);
        }
    }

    private Map<String, Object> production(CharacterizationBoard board) {
        return RecordLines.asMap(playerOf(board, 0).get("untapped_production"));
    }

    /** One land play a turn, written right after the turn's tallies it bounds. */
    @Test
    void oneLandPlayWithoutARaiseRightAfterThisTurn() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);

        assertEquals(1L, playerOf(board, 0).get("max_land_plays"));
        assertTrue(PlayerJson.render(board.game, board.you, StaticSuppression.NONE)
                .matches(".*\"this_turn\":\\{[^}]*},\"max_land_plays\":1,.*"),
                PlayerJson.render(board.game, board.you, StaticSuppression.NONE));
    }

    /** Exploration raises its controller's land plays, not the opponent's. */
    @Test
    void explorationOnTheBattlefieldAllowsTwo() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Exploration", board.you);
        board.game.getAction().checkStaticAbilities();

        assertEquals(2L, playerOf(board, 0).get("max_land_plays"));
        assertEquals(1L, playerOf(board, 1).get("max_land_plays"));
    }

    /** Explore's raise lives on its effect card in the command zone. */
    @Test
    void exploreResolvedThisTurnAllowsTwo() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.game.getAction().moveToLibrary(board.card("Forest", board.you), null, null);
        Card explore = board.hand("Explore", board.you);
        board.resolve(board.spell(explore, board.you), board.you);
        board.game.getAction().checkStaticAbilities();

        assertEquals(2L, playerOf(board, 0).get("max_land_plays"));
    }

    /** Fastbond's {@code AdjustLandPlays$ Unlimited} is null, not a number. */
    @Test
    void fastbondMakesThemUnlimited() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Fastbond", board.you);
        board.game.getAction().checkStaticAbilities();

        Map<String, Object> you = playerOf(board, 0);
        assertTrue(you.containsKey("max_land_plays"), you.toString());
        assertNull(you.get("max_land_plays"));
    }

    @Test
    void dayNightIsDayNightOrNeither() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        assertNull(globalOf(board).get("day_night"));
        board.game.setDayTime(false);
        assertEquals("day", globalOf(board).get("day_night"));
        board.game.setDayTime(true);
        assertEquals("night", globalOf(board).get("day_night"));
    }

    /** Owner is not controller: a control change leaves the owner alone. */
    @Test
    void anEntityNamesItsOwnerApartFromItsController() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        bears.addTempController(board.opponent, board.game.getNextTimestamp());

        Map<String, Object> entity = entityOf(board, bears);

        assertEquals(Refs.of(board.you), entity.get("owner"));
        assertEquals(Refs.of(board.opponent), entity.get("controller"));
    }
}
