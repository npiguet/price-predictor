package com.pricepredictor.connector.effects;

import com.google.common.collect.LinkedHashMultiset;
import com.google.common.collect.Multiset;
import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.card.CounterType;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.phase.PhaseType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * One state is written as the same bytes whatever order Forge hands its
 * parts over in (finding 9.11). Forge keeps a card's counters in a hash
 * multiset over an enum hashed by identity, its keywords in the same kind of
 * order, and a creature's must-block attackers in a hash map: each differs
 * from one JVM to the next. Each writer is fed the same parts in two orders.
 */
@ExtendWith(ForgeExtension.class)
class SnapshotStableOrderTest {

    /** A card's {@code counters}, by counter name as written ({@code +1/+1} before {@code CHARGE}). */
    @Test
    void countersAreWrittenByName() {
        Multiset<CounterType> chargeFirst = LinkedHashMultiset.create();
        chargeFirst.add(CounterEnumType.CHARGE, 1);
        chargeFirst.add(CounterEnumType.P1P1, 2);
        Multiset<CounterType> plusFirst = LinkedHashMultiset.create();
        plusFirst.add(CounterEnumType.P1P1, 2);
        plusFirst.add(CounterEnumType.CHARGE, 1);

        assertEquals("{\"+1/+1\":2,\"CHARGE\":1}", EntityJson.countersJson(chargeFirst));
        assertEquals(EntityJson.countersJson(plusFirst), EntityJson.countersJson(chargeFirst));
    }

    /** {@code granted_temporary.keywords}, and every list spelled alike, by name. */
    @Test
    void keywordsAreWrittenByName() {
        KeywordInterface flying = Keyword.getInstance("Flying");
        KeywordInterface haste = Keyword.getInstance("Haste");

        assertEquals(List.of("flying", "haste"), CardSpelling.keywordNames(List.of(haste, flying)));
        assertEquals(CardSpelling.keywordNames(List.of(flying, haste)),
                CardSpelling.keywordNames(List.of(haste, flying)));
    }

    /**
     * A creature that must block two attackers: its {@code must_block}
     * links follow the attackers' ids, whichever order Forge listed them in.
     */
    @Test
    void mustBlockLinksAreWrittenByAttacker() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card giant = board.battlefield("Hill Giant", board.you);
        Card blocker = board.battlefield("Serra Angel", board.opponent);
        assertTrue(bears.getId() < giant.getId());

        blocker.addMustBlockCards(board.game.getNextTimestamp(), List.of(giant, bears));
        String giantFirst = new SnapshotBuilder(board.game).ofBoard().render();
        blocker.clearMustBlockCards();
        blocker.addMustBlockCards(board.game.getNextTimestamp(), List.of(bears, giant));
        String bearsFirst = new SnapshotBuilder(board.game).ofBoard().render();

        String byId = "{\"from\":\"" + Refs.of(blocker) + "\",\"to\":\"" + Refs.of(bears)
                + "\",\"type\":\"must_block\"},{\"from\":\"" + Refs.of(blocker) + "\",\"to\":\""
                + Refs.of(giant) + "\",\"type\":\"must_block\"}";
        assertTrue(giantFirst.contains(byId), giantFirst);
        assertEquals(bearsFirst, giantFirst);
    }
}
