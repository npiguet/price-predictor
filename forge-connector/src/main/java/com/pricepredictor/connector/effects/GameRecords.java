package com.pricepredictor.connector.effects;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/** One game's records written, and those withheld past the cap by turn and acting card. */
final class GameRecords {
    private int written;
    int withheld;
    boolean flushFailed;
    private final Map<String, Integer> byTurnAndCard = new HashMap<>();

    /**
     * Count a record against the game: written while under {@code cap},
     * or holding the game's outcome; else withheld.
     *
     * @return whether it is written
     */
    boolean admit(RecordFacts facts, int cap) {
        if (written >= cap && !facts.holdsOutcome()) {
            withhold(facts);
            return false;
        }
        written++;
        return true;
    }

    private void withhold(RecordFacts facts) {
        withheld++;
        Integer turn = facts.turn();
        String card = facts.actingScript();
        byTurnAndCard.merge("turn " + (turn == null ? "?" : turn) + ", acting card "
                + (card == null ? "none" : card), 1, Integer::sum);
    }

    String report(String gameId, int cap) {
        // The most records; of two as many, the first by turn and card.
        Map.Entry<String, Integer> most = null;
        for (Map.Entry<String, Integer> entry : new TreeMap<>(byTurnAndCard).entrySet()) {
            if (most == null || entry.getValue() > most.getValue()) {
                most = entry;
            }
        }
        return "Effect records: game " + gameId + " reached --max-records-per-game " + cap
                + "; withheld " + withheld + " more records, most often (" + most.getValue()
                + ") on " + most.getKey();
    }
}
