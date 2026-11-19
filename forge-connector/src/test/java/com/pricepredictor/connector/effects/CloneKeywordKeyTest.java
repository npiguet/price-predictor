package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import forge.ai.simulation.GameCopier;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The keyword keys of a Clone whose original changes after the Clone copied
 * it (finding 1.8). A Clone took the copiable values of what it copied when
 * it entered (707.2); a later copy effect on the original (Cytoshape) changes
 * the original, not the Clone. The Clone's keywords are printed by the card
 * whose printed values they were then, and key into that card's lines, on
 * the live board and on a fork of it; a keyword a copy effect added as an
 * exception (707.9b) keys to that effect's line. Each key is read against the
 * real sidecar of the card it names.
 */
@ExtendWith(ForgeExtension.class)
class CloneKeywordKeyTest {

    /** Moldering Karok's keywords. */
    private static final List<String> KAROKS = List.of("Lifelink", "Trample");

    private final RulesParser parser = new RulesParser();

    /**
     * Clone enters as a copy of Moldering Karok, then the opponent's
     * Cytoshape makes Karok a copy of Grizzly Bears until end of turn: the
     * Clone's Lifelink and Trample still join Karok's lines, printed, live
     * and forked.
     */
    @Test
    void aClonesKeywordsKeepTheirLinesWhenWhatItCopiedBecomesACopyOfSomethingElse() throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card karok = board.battlefield("Moldering Karok", board.you);
        ConvertedScript karoks = convertedScriptOf(karok);
        Card clone = entersAsACopy(board, "Clone", karok);
        cytoshapeIntoBears(board, karok);

        assertJoinPrinted(board, clone, karoks, KAROKS);

        Card forked = copierIn(new GameCopier(board.game).makeCopy(), "Clone");
        assertKeyedAlike(clone, forked, KAROKS);
    }

    /**
     * Dack's Duplicate enters as a copy of Moldering Karok "except it has
     * haste and dethrone", Clone enters as a copy of the Duplicate (Karok
     * tapped, so the AI rates the Duplicate best), then
     * Cytoshape makes Karok a copy of Grizzly Bears: the Clone's Lifelink and
     * Trample join Karok's lines, its Haste and Dethrone the Duplicate's copy
     * line, every line printed, live and forked.
     */
    @Test
    void aCloneOfACloneKeysThroughWhatEachCopiedWhenItEntered() throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card karok = board.battlefield("Moldering Karok", board.you);
        ConvertedScript karoks = convertedScriptOf(karok);
        Card duplicate = entersAsACopy(board, "Dack's Duplicate", karok);
        ConvertedScript duplicates = convertedScriptOf(duplicate);
        karok.setTapped(true); // the AI copies the creature it rates best: the untapped Duplicate
        Card clone = entersAsACopy(board, "Clone", duplicate);
        cytoshapeIntoBears(board, karok);

        assertJoinPrinted(board, clone, karoks, KAROKS);
        for (String keyword : List.of("Haste", "Dethrone")) {
            assertJoins(duplicates, keyOf(clone, keyword), "except it has haste and dethrone");
        }

        Card forked = copierIn(new GameCopier(board.game).makeCopy(), "Clone");
        assertKeyedAlike(clone, forked, List.of("Lifelink", "Trample", "Haste", "Dethrone"));
    }

    /**
     * Dreamtail Heron mutates onto a Clone copying Moldering Karok: the
     * merged permanent's Lifelink and Trample, which the Clone brought, join
     * Karok's lines, and Mutate the Heron's; every keyword line is printed.
     */
    @Test
    void aMutatedClonesCopiedKeywordsJoinWhatTheCloneCopied() throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card karok = board.battlefield("Moldering Karok", board.you);
        ConvertedScript karoks = convertedScriptOf(karok);
        Card clone = entersAsACopy(board, "Clone", karok);
        Card heron = board.hand("Dreamtail Heron", board.you);
        ConvertedScript herons = convertedScriptOf(heron);
        SpellAbility mutate = heron.getSpellAbilities().stream()
                .filter(SpellAbility::isMutate).findFirst().orElseThrow();
        mutate.resetTargets();
        mutate.getTargets().add(clone);
        board.resolve(mutate, board.you);
        board.game.getAction().checkStaticAbilities();
        Card merged = board.game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(Card::isMutated).findFirst().orElseThrow();

        assertJoinPrinted(board, merged, karoks, KAROKS);
        assertJoins(herons, keyOf(merged, "Mutate:3 U"), "mutate");
    }

    // ── the board ───────────────────────────────────────────────────────

    /** {@code copier} cast from hand, entered as a copy of {@code copied}, returned. */
    private static Card entersAsACopy(CharacterizationBoard board, String copier, Card copied) {
        Card inHand = board.hand(copier, board.you);
        board.resolve(board.spell(inHand, board.you), board.you);
        Card entered = board.game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getId() == inHand.getId()).findFirst().orElseThrow();
        assertSame(copied, ProvenanceResolver.copyStatesOf(entered.getCurrentState()).getOrigin(),
                copier + " entered as a copy of " + copied);
        return entered;
    }

    /**
     * The opponent's Cytoshape targeting {@code target}, which becomes a copy
     * of Grizzly Bears, the creature the opponent's AI picks to copy as the
     * worst one when the target is yours.
     */
    private static void cytoshapeIntoBears(CharacterizationBoard board, Card target) {
        board.battlefield("Grizzly Bears", board.you);
        SpellAbility cytoshape = board.spell(board.hand("Cytoshape", board.opponent), board.opponent);
        cytoshape.resetTargets();
        cytoshape.getTargets().add(target);
        board.resolve(cytoshape, board.opponent);
        board.game.getAction().checkStaticAbilities();
        assertEquals("Grizzly Bears", target.getName(), "the target is now a copy of Grizzly Bears");
    }

    /** The card on the battlefield of {@code game} whose own card is {@code copier}. */
    private static Card copierIn(Game game, String copier) {
        return game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> copier.equals(card.getRules().getName()))
                .findFirst().orElseThrow();
    }

    /** The key of the card's one keyword whose text is {@code original}. */
    private static ProvenanceKey keyOf(Card card, String original) {
        List<KeywordInterface> found = card.getCurrentState().getIntrinsicKeywords().stream()
                .filter(keyword -> original.equals(keyword.getOriginal())).toList();
        assertEquals(1, found.size(), card + "'s " + original + ": "
                + card.getCurrentState().getIntrinsicKeywords());
        return ProvenanceResolver.keyOfKeyword(card.getCurrentState(), found.get(0));
    }

    /** That a fork's card keys each of {@code keywords} as the live one does. */
    private static void assertKeyedAlike(Card card, Card forked, List<String> keywords) {
        for (String keyword : keywords) {
            assertEquals(keyOf(card, keyword), keyOf(forked, keyword), "the fork's " + keyword);
        }
    }

    /**
     * That each of {@code keywords} of the card joins its own line of
     * {@code script}, and that every keyword line the card shows is printed.
     */
    private static void assertJoinPrinted(CharacterizationBoard board, Card card, ConvertedScript script,
                                          List<String> keywords) {
        for (String keyword : keywords) {
            assertJoins(script, keyOf(card, keyword), keyword.toLowerCase(Locale.ROOT));
        }
        for (Map<String, Object> line : linesOf(board, card)) {
            if ("keyword".equals(line.get("kind"))) {
                assertEquals(false, line.get("granted"), line.toString());
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> linesOf(CharacterizationBoard board, Card card) {
        return (List<Map<String, Object>>) RecordLines.parse(
                EntityJson.render(board.game, card, StaticSuppression.NONE)).get("lines");
    }

    // ── the sidecar ─────────────────────────────────────────────────────

    /** The card's converted script and sidecar as {@code python -m price_predictor convert} writes them. */
    private ConvertedScript convertedScriptOf(Card card) throws IOException {
        String scriptFile = ScriptLocator.scriptFileOf(card);
        List<String> source = Files.readAllLines(Path.of(card.getRules().getPath()));
        return ConvertedScript.of(parser.parseScript(source, scriptFile), scriptFile);
    }

    /**
     * That the key names a line of the script, and that line is the
     * keyword's: the line declaring it, else, for a {@code built} key no
     * line declares, the line declaring its bare key, as a reader joins it.
     */
    private static void assertJoins(ConvertedScript script, ProvenanceKey key, String keyword) {
        assertNotNull(key, keyword);
        assertEquals(script.sidecar().scriptFile(), key.scriptFile(), keyword + ": " + key);
        int line = lineDeclaring(script, key);
        if (line < 0) line = lineDeclaring(script, key.withoutBuilt());
        assertTrue(line >= 0, keyword + " joins no line: " + key + " in " + script.sidecar().lines());
        String text = script.texts().get(line).toLowerCase(Locale.ROOT);
        assertTrue(text.contains(keyword), keyword + " joins \"" + text + "\"");
    }

    private static int lineDeclaring(ConvertedScript script, ProvenanceKey key) {
        for (ProvenanceSidecar.Line each : script.sidecar().lines()) {
            if (each.provenance().contains(key)) return each.lineIndex();
        }
        return -1;
    }
}
