package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.card.CardCopyService;
import forge.game.phase.PhaseType;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.SnapshotReading.entitiesIn;
import static com.pricepredictor.connector.effects.SnapshotReading.entityIn;
import static com.pricepredictor.connector.effects.SnapshotReading.idsIn;
import static com.pricepredictor.connector.effects.SnapshotReading.line;
import static com.pricepredictor.connector.effects.SnapshotReading.linesOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The entities a snapshot carries beyond the tiers (FR-024): library cards,
 * cards exiled with a permanent, mutate components and the pending subject's
 * last-known copy; and what a static line affects and the resolution's random results.
 */
@ExtendWith(ForgeExtension.class)
class SnapshotBeyondTheTiersTest {

    @Test
    void eachLibrarysTopCardIsALibraryCard() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bottom = board.game.getAction().moveToLibrary(
                board.card("Forest", board.you), null, null);
        Card top = board.game.getAction().moveToLibrary(
                board.card("Grizzly Bears", board.you), 0, null, null);

        String state = new SnapshotBuilder(board.game).ofBoard().render();
        Map<String, Object> entity = entityIn(state, top);

        assertEquals("library_card", entity.get("role"));
        assertEquals("library", entity.get("zone"));
        assertEquals(0L, entity.get("order_rank"));
        assertFalse(idsIn(state).contains(Refs.of(bottom)), state);
    }

    /**
     * Rendering a library card as it was in the library builds it from
     * its printing once per card: a snapshot taken again parses nothing,
     * so Forge's trait id counter, which every parsed ability advances,
     * moves no further.
     */
    @Test
    void aLibraryCardIsBuiltFromItsPrintingOnce() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card top = board.game.getAction().moveToLibrary(
                board.card("Prodigal Sorcerer", board.you), 0, null, null);
        SnapshotBuilder snapshots = new SnapshotBuilder(board.game);
        snapshots.ofBoard().render();

        int before = forge.game.ability.AbilityFactory.getAbility("DB$ Draw", top).getId();
        snapshots.ofBoard().render();
        int after = forge.game.ability.AbilityFactory.getAbility("DB$ Draw", top).getId();

        assertEquals(before + 1, after, "the second snapshot parsed no card");
    }

    /** A library card the record's events name joins at the write, where it sat. */
    @Test
    void aLibraryCardTheEventsNameJoinsWhereItSat() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.game.getAction().moveToLibrary(board.card("Forest", board.you), null, null);
        Card second = board.game.getAction().moveToLibrary(
                board.card("Grizzly Bears", board.you), -1, null, null);
        Snapshot before = new SnapshotBuilder(board.game).ofBoard();
        long position = board.you.getCardsIn(ZoneType.Library).indexOf(second);
        board.game.getAction().moveTo(ZoneType.Hand, second, null, null);

        String state = before.withEventCards(List.of(
                new EffectEvent(EventType.ZONE_CHANGE).subject(Refs.of(second)))).render();

        Map<String, Object> entity = entityIn(state, second);
        assertEquals("library_card", entity.get("role"));
        assertEquals("library", entity.get("zone"));
        assertEquals(position, entity.get("order_rank"));
        assertTrue(position > 0, "not the top, which the snapshot already holds");
        assertEquals(before.render(), before.withEventCards(List.of()).render());
    }

    @Test
    void theCardsExiledWithAPermanentAreSlots() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card ring = board.battlefield("Oblivion Ring", board.you);
        Card exiled = board.game.getAction().exile(
                board.battlefield("Hill Giant", board.opponent), null, null);
        exiled.setExiledWith(ring);
        ring.addExiledCard(exiled);

        Map<String, Object> entity = entityIn(
                new SnapshotBuilder(board.game).ofBoard().render(), exiled);

        assertEquals("exiled_with", entity.get("role"));
        assertEquals("exile", entity.get("zone"));
    }

    @Test
    void aMutatedPermanentsComponentsAreSlots() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card top = board.battlefield("Grizzly Bears", board.you);
        Card under = board.card("Hill Giant", board.you);
        top.addMergedCard(top);
        top.addMergedCard(under);
        top.setMutatedTimestamp(board.game.getNextTimestamp());

        String state = new SnapshotBuilder(board.game).ofBoard().render();

        assertEquals("mutate_component", entityIn(state, under).get("role"));
        assertNull(entityIn(state, top).get("role"));
    }

    /** The pending event's subject as it last existed, under an id of its own. */
    @Test
    void thePendingSubjectsLastKnownStateIsASlotOfItsOwn() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card lastKnown = CardCopyService.getLKICopy(bears);
        Card dead = board.game.getAction().moveTo(ZoneType.Graveyard, bears, null, null);
        EffectEvent died = new EffectEvent(EventType.ZONE_CHANGE)
                .subject(Refs.of(dead)).lastKnown(lastKnown);

        String state = new SnapshotBuilder(board.game).ofTrait(null, died).render();

        Map<String, Object> copy = entitiesIn(state).stream()
                .filter(e -> ("L" + bears.getId()).equals(e.get("id")))
                .findFirst().orElseThrow(() -> new AssertionError(state));
        assertEquals("last_known", copy.get("role"));
        assertEquals("battlefield", copy.get("zone"));
        assertEquals("graveyard", entityIn(state, dead).get("zone"));
    }

    /** Per static line, the permanents it affects. */
    @Test
    void aStaticLineListsThePermanentsItAffects() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        board.battlefield("Hill Giant", board.opponent);
        Card anthem = board.battlefield("Glorious Anthem", board.you);
        board.game.getAction().checkStaticAbilities();

        assertEquals(List.of(Refs.of(bears)),
                line(linesOf(board, anthem), "static").get("affected"));
    }

    /** A static that applies to a player lists the player after the permanents. */
    @Test
    void aStaticLineListsThePlayersItAffectsAfterThePermanents() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card exploration = board.battlefield("Exploration", board.you);
        board.game.getAction().checkStaticAbilities();

        assertEquals(List.of(Refs.of(board.you)),
                line(linesOf(board, exploration), "static").get("affected"));
    }

    /** The resolving ability's die results and coin flips, read off its events. */
    @Test
    void randomResultsAreTheResolutionsDiceAndCoins() {
        EffectEvent dice = new EffectEvent(EventType.DICE_ROLLED).param(EventParam.RESULTS, List.of(
                Map.of("natural", 3, "final", 4), Map.of("natural", 6, "final", 6)));
        EffectEvent coins = new EffectEvent(EventType.COIN_FLIPPED)
                .param(EventParam.RESULTS, List.of("win", "loss", "loss"));

        Map<String, Object> random = RecordLines.asMap(RecordLines.parse(
                RefsJson.renderHalf(null, new RefsJson.HalfInputs(List.of(dice, coins), null, List.of(), null, List.of()))).get("random"));

        assertEquals(List.of(4L, 6L), random.get("dice"));
        assertEquals(Map.of("won", 1L, "lost", 2L), random.get("coin_flips"));
        assertNull(RecordLines.parse(RefsJson.render(null)).get("random"),
                "unread before the resolution ends");
    }
}
