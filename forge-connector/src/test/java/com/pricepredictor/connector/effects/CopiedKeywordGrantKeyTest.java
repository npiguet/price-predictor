package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.phase.PhaseType;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A keyword another permanent grants to a copy keys to the granting static
 *, as it does on a creature that copies nothing: Blue Ward on
 * a Clone of Grizzly Bears gives the Clone protection from blue, and that
 * keyword is no line of the Clone's copy effect or of Grizzly Bears, which
 * the copied state does not hold it in. A keyword the copy effect itself
 * added keeps the copy effect's line.
 */
@ExtendWith(ForgeExtension.class)
class CopiedKeywordGrantKeyTest {

    /** Blue Ward's keyword, whose host's card id Forge fills in. */
    private static final String BLUE_WARD = "Protection:Card.Blue:blue:";

    /** Blue Ward's protection keys to Blue Ward's static on a Clone of Grizzly Bears, as on the Bears. */
    @Test
    void anAurasKeywordOnACopyKeysToTheAurasStatic() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card clone = entersAsACopy(board, "Clone", bears);
        Card ward = board.battlefield("Blue Ward", board.you);
        ward.attachToEntity(clone, null);
        board.game.getAction().checkStaticAbilities();

        KeywordInterface granted = grantedKeyword(clone, BLUE_WARD);
        assertNotNull(granted.getStatic(), "the keyword carries its granting static");
        assertSame(ward, granted.getStatic().getHostCard(), "the static is Blue Ward's");
        ProvenanceKey key = ProvenanceResolver.keyOfKeyword(clone.getCurrentState(), granted);
        assertEquals(ProvenanceResolver.of(granted.getStatic()), key, "the keyword keys to Blue Ward's static");
    }

    /** A lord's keyword on a Clone of a Merfolk keys to the lord's static. */
    @Test
    void aLordsKeywordOnACopyKeysToTheLordsStatic() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card merfolk = board.battlefield("Merfolk of the Pearl Trident", board.you);
        Card clone = entersAsACopy(board, "Clone", merfolk);
        Card lord = board.battlefield("Lord of Atlantis", board.you);
        board.game.getAction().checkStaticAbilities();

        KeywordInterface granted = grantedKeyword(clone, "Landwalk:Island");
        assertNotNull(granted.getStatic());
        assertSame(lord, granted.getStatic().getHostCard());
        assertEquals(ProvenanceResolver.of(granted.getStatic()),
                ProvenanceResolver.keyOfKeyword(clone.getCurrentState(), granted));
    }

    /** What a keyword the copy effect added holds as its static: none, so its line stays the copy line. */
    @Test
    void aKeywordTheCopyEffectAddedHasNoGrantingStatic() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card duplicate = entersAsACopy(board, "Dack's Duplicate", bears);
        for (String added : List.of("Haste", "Dethrone")) {
            KeywordInterface keyword = intrinsicKeyword(duplicate, added);
            assertNull(keyword.getStatic(), added);
            assertNotNull(ProvenanceResolver.keyOfKeyword(duplicate.getCurrentState(), keyword), added);
        }
    }

    private static Card entersAsACopy(CharacterizationBoard board, String copier, Card copied) {
        Card inHand = board.hand(copier, board.you);
        board.resolve(board.spell(inHand, board.you), board.you);
        Card entered = board.game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getId() == inHand.getId()).findFirst().orElseThrow();
        assertSame(copied, ProvenanceResolver.copyStatesOf(entered.getCurrentState()).getOrigin(),
                copier + " entered as a copy of " + copied);
        return entered;
    }

    private static KeywordInterface grantedKeyword(Card card, String original) {
        List<KeywordInterface> found = card.getKeywords().stream()
                .filter(keyword -> keyword.getOriginal().startsWith(original)).toList();
        assertEquals(1, found.size(), card + "'s " + original + ": " + card.getKeywords());
        return found.get(0);
    }

    private static KeywordInterface intrinsicKeyword(Card card, String original) {
        List<KeywordInterface> found = card.getCurrentState().getIntrinsicKeywords().stream()
                .filter(keyword -> original.equals(keyword.getOriginal())).toList();
        assertTrue(found.size() == 1, card + "'s " + original + ": " + card.getCurrentState().getIntrinsicKeywords());
        return found.get(0);
    }
}
