package com.pricepredictor.connector.effects;

import forge.game.card.Card;

import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordEvents.eventsNaming;
import static com.pricepredictor.connector.effects.RecordLines.params;

/**
 * A record's events about one card, parsed: of one type, and the moves that
 * take it from one zone to another.
 */
final class RecordMoves {

    private RecordMoves() {
    }

    /** The record's events of {@code type} whose subjects name {@code card}, in order. */
    static List<Map<String, Object>> eventsOn(String record, Card card, String type) {
        return eventsOn(record, Refs.of(card), type);
    }

    /** The record's events of {@code type} whose subjects name {@code ref}, in order. */
    static List<Map<String, Object>> eventsOn(String record, String ref, String type) {
        return eventsNaming(record, ref).stream()
                .filter(event -> type.equals(event.get("type"))).toList();
    }

    /** The record's {@code zone_change} events moving {@code card} from {@code from} to {@code to}. */
    static List<Map<String, Object>> moves(String record, Card card, String from, String to) {
        return eventsOn(record, card, "zone_change").stream()
                .filter(move -> from.equals(params(move).get("from_zone")))
                .filter(move -> to.equals(params(move).get("to_zone"))).toList();
    }

    /** The record's moves of {@code card} off the stack, to whatever zone. */
    static List<Map<String, Object>> movesOffTheStack(String record, Card card) {
        return eventsOn(record, card, "zone_change").stream()
                .filter(move -> "stack".equals(params(move).get("from_zone"))).toList();
    }

    /** The record's moves of {@code card} from the battlefield to the graveyard: its deaths. */
    static List<Map<String, Object>> deathsOf(String record, Card card) {
        return moves(record, card, "battlefield", "graveyard");
    }

    /** Whether the record holds a death of {@code card}. */
    static boolean holdsTheDeathOf(String record, Card card) {
        return !deathsOf(record, card).isEmpty();
    }

    /** The {@code amount}s of the record's {@code damage_dealt} events on {@code ref}, in order. */
    static List<Object> damageTo(String record, String ref) {
        return eventsOn(record, ref, "damage_dealt").stream()
                .map(damage -> params(damage).get("amount")).toList();
    }
}
