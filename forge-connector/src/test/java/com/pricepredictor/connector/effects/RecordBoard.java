package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.player.Player;

import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.parse;

/**
 * A record's board: the entity or player of its {@code state} that a ref names.
 */
final class RecordBoard {

    private RecordBoard() {
    }

    /** The entity {@code ref} in the record's state; fails when it holds none. */
    static Map<String, Object> entity(String record, String ref) {
        List<Map<String, Object>> found = entitiesWithId(record, ref);
        if (found.isEmpty()) {
            throw new AssertionError("no " + ref + " in " + record);
        }
        return found.get(0);
    }

    /** The entity {@code card} in the record's state. */
    static Map<String, Object> entity(String record, Card card) {
        return entity(record, Refs.of(card));
    }

    /** Every entity of the record's state written under {@code ref}. */
    static List<Map<String, Object>> entitiesWithId(String record, String ref) {
        return members(record, "entities", ref);
    }

    /** The player {@code ref} in the record's state; fails when it holds none. */
    static Map<String, Object> player(String record, String ref) {
        List<Map<String, Object>> found = members(record, "players", ref);
        if (found.isEmpty()) {
            throw new AssertionError("no " + ref + " in " + record);
        }
        return found.get(0);
    }

    /** The player {@code player} in the record's state. */
    static Map<String, Object> player(String record, Player player) {
        return player(record, Refs.of(player));
    }

    private static List<Map<String, Object>> members(String record, String block, String ref) {
        Map<String, Object> state = asMap(parse(record).get("state"));
        return ((List<?>) state.get(block)).stream().map(RecordLines::asMap)
                .filter(member -> ref.equals(member.get("id"))).toList();
    }
}
