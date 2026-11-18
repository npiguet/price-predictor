package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A keyword a keyword-built static grants its own card keys as the granting
 * keyword's line, bare (finding 1.1): Mardu Scout cast for its dash cost
 * gains haste from the static its Dash keyword builds, and the haste line
 * and the Dash line are one line of the sidecar, so they carry one key and
 * the board reads them as one slot. The haste line is printed, as any
 * keyword a card's own static gives it.
 */
@ExtendWith(ForgeExtension.class)
class KeywordStaticSelfGrantKeyTest {

    private static final String SCOUT = "Mardu Scout";

    private final RulesParser parser = new RulesParser();

    /** The dashed Scout's haste line keys as its Dash line, the key that line declares. */
    @Test
    void aKeywordItsOwnKeywordsStaticGrantsKeysAsThatKeywordsLine() throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card scout = dashed(board);
        List<Map<String, Object>> lines = linesOf(board, scout);

        Map<String, Object> dash = only(lines, fragment -> fragment.startsWith("Dash"));
        Map<String, Object> haste = only(lines, "Haste"::equals);

        assertEquals(dash.get("keys"), haste.get("keys"), lines.toString());
        assertTrue(declared(scout, keyOf(scout, "Haste")), "the sidecar declares the haste key");
    }

    /**
     * The dashed Scout's haste is printed, as a keyword a card's own static
     * gives it is (Iymrith's ward), though Forge leaves the Dash static's
     * host the Scout's object from before it entered the battlefield.
     */
    @Test
    void aKeywordItsOwnKeywordsStaticGrantsIsPrinted() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card scout = dashed(board);

        Map<String, Object> haste = only(linesOf(board, scout), "Haste"::equals);

        assertEquals(false, haste.get("granted"), haste.toString());
        assertFalse(haste.containsKey("grant"), haste.toString());
    }

    // ── the board ───────────────────────────────────────────────────────

    /** Mardu Scout cast for its dash cost and resolved; the Scout on the battlefield. */
    private static Card dashed(CharacterizationBoard board) {
        Card hand = board.hand(SCOUT, board.you);
        SpellAbility dash = hand.getSpellAbilities().stream().filter(SpellAbility::isDash)
                .findFirst().orElseThrow(() -> new AssertionError("no dash cast"));
        dash.setActivatingPlayer(board.you);
        Card onStack = board.game.getAction().moveToStack(hand, dash);
        dash.setHostCard(onStack);
        board.game.getStack().add(dash);
        board.game.getStack().resolveStack();
        board.game.getAction().checkStaticAbilities();
        return board.game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> SCOUT.equals(card.getName())).findFirst().orElseThrow();
    }

    // ── reading the lines ───────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> linesOf(CharacterizationBoard board, Card card) {
        return (List<Map<String, Object>>) RecordLines.parse(
                EntityJson.render(board.game, card, StaticSuppression.NONE)).get("lines");
    }

    /** The one keyword line whose fragment matches. */
    private static Map<String, Object> only(List<Map<String, Object>> lines, Predicate<String> fragment) {
        List<Map<String, Object>> found = lines.stream()
                .filter(line -> "keyword".equals(line.get("kind"))
                        && fragment.test(String.valueOf(line.get("fragment"))))
                .toList();
        assertEquals(1, found.size(), lines.toString());
        return found.get(0);
    }

    private static ProvenanceKey keyOf(Card card, String keyword) {
        var found = card.getKeywords().stream().filter(each -> keyword.equals(each.getOriginal())).toList();
        assertEquals(1, found.size(), card + "'s " + keyword);
        return ProvenanceResolver.keyOfKeyword(card.getCurrentState(), found.get(0));
    }

    /** Whether a line of the card's sidecar declares {@code key} itself. */
    private boolean declared(Card card, ProvenanceKey key) throws IOException {
        String scriptFile = ScriptLocator.scriptFileOf(card);
        List<String> source = Files.readAllLines(Path.of(card.getRules().getPath()));
        ProvenanceSidecar sidecar = ConvertedScript.of(parser.parseScript(source, scriptFile), scriptFile).sidecar();
        return sidecar.lines().stream().anyMatch(line -> line.provenance().contains(key));
    }
}
