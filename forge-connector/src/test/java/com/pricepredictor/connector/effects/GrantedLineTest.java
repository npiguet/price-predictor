package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CounterType;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A line printed on its card is not {@code granted}, and a granted line says
 * why in {@code grant}: a copy prints what it copies, a card
 * prints the lines its own statics grant it, and a keyword counter, a static
 * and a resolved effect (the card's own animation too) each grant theirs.
 */
@ExtendWith(ForgeExtension.class)
class GrantedLineTest {

    // ── printed lines ───────────────────────────────────────────────────

    /** A Clone of a Clone of Serra Angel prints Serra Angel's lines. */
    @Test
    void aCopyOfACopyPrintsWhatTheFirstCopied() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card angel = board.battlefield("Serra Angel", board.opponent);
        Card first = enterAsACopy(board, "Clone");
        board.game.getAction().moveToGraveyard(angel, null);
        Card second = enterAsACopy(board, "Clone");
        assertEquals("Serra Angel", second.getName(), "the second Clone copied the first");

        assertAllPrinted(linesOf(board, first));
        assertAllPrinted(linesOf(board, second));
    }

    /**
     * Phantasmal Image prints Serra Angel's lines, and the trigger its copy
     * effect adds is a copiable exception (707.9b), printed too.
     */
    @Test
    void aCopyPrintsTheLineItsCopyEffectAdds() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Serra Angel", board.opponent);
        Card image = enterAsACopy(board, "Phantasmal Image");

        List<Map<String, Object>> lines = linesOf(board, image);

        assertTrue(lines.stream().anyMatch(line -> "trigger".equals(line.get("kind"))), lines.toString());
        assertAllPrinted(lines);
    }

    /**
     * Ranger Class at level 3 prints every line it has: a level's granted
     * trait keys to its {@code Class:N} line and is listed beside it, printed.
     */
    @Test
    void aClassLevelsGrantedLinesArePrinted() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card ranger = board.battlefield("Ranger Class", board.you);
        ranger.setClassLevel(3);
        board.game.getAction().checkStaticAbilities();

        List<Map<String, Object>> lines = linesOf(board, ranger);

        Map<String, Object> level2 = only(lines, line -> "keyword".equals(line.get("kind"))
                && String.valueOf(line.get("fragment")).startsWith("Class:2:"));
        only(lines, line -> "trigger".equals(line.get("kind")) && line.get("keys").equals(level2.get("keys")));
        assertAllPrinted(lines);
    }

    /** Expedition Diviner's own static gives it a dies trigger: printed on the Diviner. */
    @Test
    void aTriggerACardsOwnStaticGrantsIsPrinted() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card diviner = board.battlefield("Expedition Diviner", board.you);
        board.battlefield("Prodigal Sorcerer", board.you);
        board.game.getAction().checkStaticAbilities();

        assertAllPrinted(linesOf(board, diviner));
        assertPrinted(triggerOf(board, diviner));
    }

    /** Iymrith's own static gives it ward: the ward line is printed on Iymrith. */
    @Test
    void aKeywordACardsOwnStaticGrantsIsPrinted() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card iymrith = board.battlefield("Iymrith, Desert Doom", board.you);
        board.game.getAction().checkStaticAbilities();

        Map<String, Object> ward = only(linesOf(board, iymrith), line -> "keyword".equals(line.get("kind"))
                && String.valueOf(line.get("fragment")).startsWith("Ward"));

        assertPrinted(ward);
    }

    // ── granted lines ───────────────────────────────────────────────────

    /** A trample counter grants trample: {@code grant: counter}. */
    @Test
    void aKeywordCounterGrantsItsKeyword() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        bears.addCounterInternal(CounterType.getType("Trample"), 1, board.you, false, null, Map.of());
        board.game.getAction().checkStaticAbilities();

        Map<String, Object> trample = only(linesOf(board, bears), line -> "Trample".equals(line.get("fragment")));

        assertGranted("counter", trample);
    }

    /**
     * Two Fungus Slivers: each prints the trigger its own static gives it,
     * and the trigger the other's static gives it is granted, though both
     * key to the same line of the same script.
     */
    @Test
    void theSameLineAnotherCopyOfTheCardGrantsIsGranted() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card sliver = board.battlefield("Fungus Sliver", board.you);
        board.battlefield("Fungus Sliver", board.you);
        board.game.getAction().checkStaticAbilities();

        List<Map<String, Object>> triggers = linesOf(board, sliver).stream()
                .filter(line -> "trigger".equals(line.get("kind"))).toList();

        assertEquals(2, triggers.size(), triggers.toString());
        assertEquals(triggers.get(0).get("keys"), triggers.get(1).get("keys"), triggers.toString());
        assertEquals(1, triggers.stream().filter(line -> Boolean.FALSE.equals(line.get("granted"))).count(),
                triggers.toString());
        assertGranted("static", only(triggers, line -> Boolean.TRUE.equals(line.get("granted"))));
    }

    /** Cryptolith Rite's static gives the Bears a mana ability: {@code grant: static}. */
    @Test
    void anAbilityAnotherPermanentsStaticGrantsSaysStatic() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        board.battlefield("Cryptolith Rite", board.you);
        board.game.getAction().checkStaticAbilities();

        assertGranted("static", only(linesOf(board, bears), line -> "activated".equals(line.get("kind"))));
    }

    /** Blood Moon makes Lonely Sandbar a Mountain: its red mana ability says static. */
    @Test
    void aManaAbilityATypeChangingStaticGivesSaysStatic() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card sandbar = board.battlefield("Lonely Sandbar", board.you);
        board.battlefield("Blood Moon", board.you);
        board.game.getAction().checkStaticAbilities();

        assertGranted("static", only(linesOf(board, sandbar), line -> Boolean.TRUE.equals(line.get("granted"))));
    }

    /** Jump gives the Bears flying until end of turn: {@code grant: effect}. */
    @Test
    void aKeywordAResolvedEffectGrantsSaysEffect() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        resolveAt(board, "Jump", bears);

        assertGranted("effect", only(linesOf(board, bears), line -> "Flying".equals(line.get("fragment"))));
    }

    /** Banishing Knack gives the Bears a bounce ability until end of turn: {@code grant: effect}. */
    @Test
    void anAbilityAResolvedEffectGrantsSaysEffect() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        resolveAt(board, "Banishing Knack", bears);

        assertGranted("effect", only(linesOf(board, bears), line -> "activated".equals(line.get("kind"))));
    }

    /**
     * Raging Ravine's own animation gives it an attack trigger until end of
     * turn: a resolved effect's grant, {@code grant: effect}, though the line
     * keys to the Ravine's own animation line.
     */
    @Test
    void aTriggerACardsOwnAnimationGrantsSaysEffect() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card ravine = board.battlefield("Raging Ravine", board.you);
        animate(board, ravine);

        assertGranted("effect", triggerOf(board, ravine));
    }

    /**
     * Lavaclaw Reaches' own animation gives it a pump ability until end of
     * turn: {@code grant: effect}; its printed abilities stay printed.
     */
    @Test
    void anAbilityACardsOwnAnimationGrantsSaysEffect() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card reaches = board.battlefield("Lavaclaw Reaches", board.you);
        animate(board, reaches);

        List<Map<String, Object>> abilities = linesOf(board, reaches).stream()
                .filter(line -> "activated".equals(line.get("kind"))).toList();

        assertGranted("effect", only(abilities, line -> Boolean.TRUE.equals(line.get("granted"))));
        assertEquals(2, abilities.stream().filter(line -> Boolean.FALSE.equals(line.get("granted"))).count(),
                abilities.toString());
    }

    // ── the board ───────────────────────────────────────────────────────

    /** Cast a creature that enters as a copy and resolve it; the copy that entered. */
    private static Card enterAsACopy(CharacterizationBoard board, String copier) {
        Card hand = board.hand(copier, board.you);
        SpellAbility spell = board.spell(hand, board.you);
        board.game.getStack().addAndUnfreeze(spell);
        board.game.getStack().resolveStack();
        return board.game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getController() == board.you && card.getId() == hand.getId())
                .findFirst().orElseThrow();
    }

    private static void animate(CharacterizationBoard board, Card manland) {
        board.resolve(board.ability(manland, ApiType.Animate), board.you);
        board.game.getAction().checkStaticAbilities();
    }

    private static void resolveAt(CharacterizationBoard board, String spell, Card target) {
        board.cast(board.hand(spell, board.you), board.you, target);
        board.game.getStack().resolveStack();
        board.game.getAction().checkStaticAbilities();
    }

    // ── reading the lines ───────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> linesOf(CharacterizationBoard board, Card card) {
        return (List<Map<String, Object>>) RecordLines.parse(
                EntityJson.render(board.game, card, StaticSuppression.NONE)).get("lines");
    }

    /** The card's one trigger line. */
    private static Map<String, Object> triggerOf(CharacterizationBoard board, Card card) {
        return only(linesOf(board, card), line -> "trigger".equals(line.get("kind")));
    }

    private static Map<String, Object> only(List<Map<String, Object>> lines,
                                            java.util.function.Predicate<Map<String, Object>> matches) {
        List<Map<String, Object>> found = lines.stream().filter(matches).toList();
        assertEquals(1, found.size(), lines.toString());
        return found.get(0);
    }

    private static void assertAllPrinted(List<Map<String, Object>> lines) {
        assertFalse(lines.isEmpty(), "no lines");
        lines.forEach(GrantedLineTest::assertPrinted);
    }

    private static void assertPrinted(Map<String, Object> line) {
        assertEquals(false, line.get("granted"), line.toString());
        assertFalse(line.containsKey("grant"), line.toString());
    }

    private static void assertGranted(String grant, Map<String, Object> line) {
        assertEquals(true, line.get("granted"), line.toString());
        assertEquals(grant, line.get("grant"), line.toString());
    }
}
