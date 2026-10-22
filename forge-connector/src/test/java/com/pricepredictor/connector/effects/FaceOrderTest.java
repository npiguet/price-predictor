package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** The keyword ordinal the converter and the collectors both take from {@link FaceOrder}. */
@ExtendWith(ForgeExtension.class)
class FaceOrderTest {

    private static Card card(String name) {
        return TestCards.build(name);
    }

    /** The flashback keyword Blast from the Past prints. */
    private static KeywordInterface flashbackOf(CardState state) {
        for (KeywordInterface keyword : state.getIntrinsicKeywords()) {
            if (keyword.getOriginal().startsWith("Flashback")) {
                return keyword;
            }
        }
        throw new AssertionError("no flashback on " + state.getCard().getName());
    }

    /**
     * The ordinal both sides compute: the position of the keyword's printed
     * text among the card's intrinsic keywords, sorted. It cannot be a position
     * in {@code getKeywords()}: that collection is hash-ordered over an enum
     * whose {@code hashCode()} is the identity hash, so the convert JVM and the
     * collect JVM disagree about it. Pinned as literals because the sorted
     * order is a contract between those two JVMs, not something to recompute.
     */
    @Test
    void aKeywordOrdinalIsItsPrintedTextsPositionAmongTheSortedKeywords() {
        CardState state = card("Blast from the Past").getCurrentState();

        Map<String, Integer> ordinals = new LinkedHashMap<>();
        for (KeywordInterface keyword : state.getIntrinsicKeywords()) {
            ordinals.put(keyword.getOriginal().split("[: ]")[0],
                    FaceOrder.keywordIndex(state, keyword));
        }

        assertEquals(Map.of("Buyback", 0, "Cycling", 1, "Flashback", 2,
                "Kicker", 3, "Madness", 4), ordinals);
    }

    /**
     * A granted keyword names no printed line on the card that received it, so
     * it answers -1 and keys to nothing. That is strictly better than the
     * position it happens to occupy in the recipient's list, which is a wrong
     * key dressed as an answer.
     */
    @Test
    void aGrantedKeywordHasNoPrintedOrdinal() {
        Card bear = card("Grizzly Bears");
        Card blast = card("Blast from the Past");
        KeywordInterface flashback = flashbackOf(blast.getCurrentState());

        assertEquals(-1,
                FaceOrder.keywordIndex(bear.getCurrentState(), flashback),
                "a keyword the card does not print has no ordinal on it");
    }
}
