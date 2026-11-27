package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.SnapshotReading.entitiesIn;
import static com.pricepredictor.connector.effects.SnapshotReading.entityOf;
import static com.pricepredictor.connector.effects.SnapshotReading.keywordLine;
import static com.pricepredictor.connector.effects.SnapshotReading.linesOf;
import static com.pricepredictor.connector.effects.SnapshotReading.valuesOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What a card slot shows of a copy and of the face its acting ability is
 * on: a copy's lines key to what it copied, and a card in hand shows the
 * face, half or Adventure the candidate acts through.
 */
@ExtendWith(ForgeExtension.class)
class SnapshotCopiesAndFacesTest {

    /**
     * A Clone cast and resolved as a copy of a creature with an enters
     * trigger, or with a replacement: each of its lines keys to the
     * original's script, or to no line, never to {@code clone.txt}'s own
     * trigger or replacement at the copied line's index (B8), and none is
     * granted: a copy prints what it copies.
     */
    @org.junit.jupiter.params.ParameterizedTest(name = "{0}")
    @org.junit.jupiter.params.provider.CsvSource({
            "Ravenous Chupacabra, cardsfolder/r/ravenous_chupacabra.txt, trigger",
            "Dryad Militant, cardsfolder/d/dryad_militant.txt, replacement",
            "Serra Angel, cardsfolder/s/serra_angel.txt, keyword"})
    void aClonesLinesKeyToWhatItCopied(String original, String script, String kind) {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield(original, board.opponent);
        SpellAbility clone = board.spell(board.hand("Clone", board.you), board.you);
        board.game.getStack().addAndUnfreeze(clone);
        board.game.getStack().resolveStack();
        Card copy = board.game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getController() == board.you).findFirst().orElseThrow();
        assertEquals(original, copy.getName(), "the Clone entered as a copy");

        List<Map<String, Object>> lines = linesOf(board, copy);

        assertTrue(lines.stream().anyMatch(line -> kind.equals(line.get("kind"))
                && !((List<?>) line.get("keys")).isEmpty()), "the copied " + kind + ": " + lines);
        for (Map<String, Object> line : lines) {
            for (Object key : (List<?>) line.get("keys")) {
                assertEquals(script, RecordLines.asMap(key).get("script_file"), lines.toString());
            }
            assertEquals(false, line.get("granted"), "a copy prints what it copies: " + line);
        }
    }

    /**
     * A decision on an Adventure's spell, the card in hand: the host
     * shows the Adventure's face, the spell the candidate casts, not
     * the creature the hand holds it as; the hand card itself is
     * left on its front.
     */
    @Test
    void aHandCardShowsTheFaceItsActingAbilityIsOn() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card giant = board.hand("Bonecrusher Giant", board.you);
        SpellAbility stomp = giant.getAllSpellAbilities().stream()
                .filter(sa -> sa.getCardState() != null && sa.getCardState().getStateName()
                        == forge.card.CardStateName.Secondary)
                .findFirst().orElseThrow(() -> new AssertionError("no Stomp"));
        stomp.setActivatingPlayer(board.you);

        Map<String, Object> entity = entitiesIn(
                new SnapshotBuilder(board.game).ofCandidate(stomp).render()).stream()
                .filter(e -> Refs.of(giant).equals(e.get("id"))).findFirst().orElseThrow();

        assertEquals(List.of("Stomp", List.of("instant"), "hand"),
                valuesOf(entity, "name", "types", "zone"), entity.toString());
        assertEquals("Bonecrusher Giant", giant.getName(), "the game's card is untouched");
    }

    /** The acting face of a card in hand, for an ability on {@code face}. */
    private Map<String, Object> handEntityActingThrough(
            String name, forge.card.CardStateName face) {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card card = board.hand(name, board.you);
        SpellAbility acting = card.getAllSpellAbilities().stream()
                .filter(sa -> sa.getCardState() != null
                        && sa.getCardState().getStateName() == face)
                .findFirst().orElseThrow(() -> new AssertionError("no ability on " + face));
        acting.setActivatingPlayer(board.you);
        return entitiesIn(new SnapshotBuilder(board.game).ofCandidate(acting).render())
                .stream().filter(e -> Refs.of(card).equals(e.get("id")))
                .findFirst().orElseThrow();
    }

    /** Fire // Ice in hand, its Ice half acting: the host shows Ice. */
    @Test
    void aSplitCardShowsTheHalfItsActingAbilityIsOn() {
        Map<String, Object> entity = handEntityActingThrough(
                "Fire // Ice", forge.card.CardStateName.RightSplit);
        assertEquals("Ice", entity.get("name"), entity.toString());
        assertEquals("hand", entity.get("zone"), entity.toString());
    }

    /** A modal double-faced card acting through its back face shows that face. */
    @Test
    void aModalDoubleFacedCardShowsTheBackFaceItsAbilityIsOn() {
        Map<String, Object> entity = handEntityActingThrough(
                "Agadeem's Awakening",
                forge.card.CardStateName.Backside);
        assertEquals("Agadeem, the Undercrypt", entity.get("name"), entity.toString());
        assertEquals("back", entity.get("face_state"), entity.toString());
    }

    /**
     * A Clone of a transformed Delver copies its face up, Insectile
     * Aberration: the copy's Flying keys to the Aberration's face of the
     * Delver's file, not to the Clone's own line.
     */
    @Test
    void aCopyOfATransformedCardKeysToTheFaceItCopied() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card delver = board.battlefield("Delver of Secrets", board.opponent);
        delver.changeCardState("Transform", null, null);
        SpellAbility clone = board.spell(board.hand("Clone", board.you), board.you);
        board.game.getStack().addAndUnfreeze(clone);
        board.game.getStack().resolveStack();
        Card copy = board.game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getController() == board.you).findFirst().orElseThrow();
        assertEquals("Insectile Aberration", copy.getName(), "the Clone copied the face up");

        Map<String, Object> flying = keywordLine(linesOf(board, copy), "Flying");
        Map<String, Object> key = RecordLines.asMap(((List<?>) flying.get("keys")).get(0));
        assertEquals("cardsfolder/d/delver_of_secrets_insectile_aberration.txt",
                key.get("script_file"), flying.toString());
        assertEquals(1L, key.get("face"), flying.toString());
    }

    /**
     * A Clone's {@code copy_source} is the creature it copied, never the
     * Clone itself.
     */
    @Test
    void aClonesCopySourceIsWhatItCopied() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card angel = board.battlefield("Serra Angel", board.opponent);
        SpellAbility clone = board.spell(board.hand("Clone", board.you), board.you);
        board.game.getStack().addAndUnfreeze(clone);
        board.game.getStack().resolveStack();
        Card copy = board.game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getController() == board.you).findFirst().orElseThrow();

        assertEquals(Refs.of(angel), entityOf(board, copy).get("copy_source"));
        assertEquals(null, entityOf(board, angel).get("copy_source"));
    }

    /**
     * Phantasmal Image's copy keeps the copied creature's lines keyed to
     * it, and the trigger the copy effect adds ({@code AddTriggers$}) keys
     * to the Image's own line that made the copy, never to a trigger
     * index of the Image's file, which prints none.
     */
    @Test
    void aCopyEffectsAddedTriggerKeysToTheLineThatCopied() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Serra Angel", board.opponent);
        SpellAbility image = board.spell(board.hand("Phantasmal Image", board.you), board.you);
        board.game.getStack().addAndUnfreeze(image);
        board.game.getStack().resolveStack();
        Card copy = board.game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getController() == board.you).findFirst().orElseThrow();
        assertEquals("Serra Angel", copy.getName(), "the Image entered as a copy");

        List<Map<String, Object>> lines = linesOf(board, copy);

        List<Map<String, Object>> keys = new java.util.ArrayList<>();
        for (Map<String, Object> line : lines) {
            for (Object key : (List<?>) line.get("keys")) {
                keys.add(RecordLines.asMap(key));
            }
        }
        assertTrue(keys.stream().anyMatch(key -> "cardsfolder/s/serra_angel.txt".equals(
                key.get("script_file")) && "keyword".equals(key.get("trait_kind"))), lines.toString());
        assertTrue(lines.stream().anyMatch(line -> "trigger".equals(line.get("kind"))
                && ((List<?>) line.get("keys")).stream().map(RecordLines::asMap).anyMatch(key ->
                        "cardsfolder/p/phantasmal_image.txt".equals(key.get("script_file")))),
                "the added trigger keys to the Image: " + lines);
        assertTrue(keys.stream().noneMatch(key -> "cardsfolder/p/phantasmal_image.txt".equals(
                key.get("script_file")) && "trigger".equals(key.get("trait_kind"))),
                "no trigger index of the Image's file: " + lines);
    }
}
