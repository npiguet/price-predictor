package com.pricepredictor.connector.effects;

import forge.game.card.Card;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Reading a rendered snapshot: its entities, players, global facts and a card slot's lines. */
final class SnapshotReading {

    private SnapshotReading() {
    }

    static Map<String, Object> entityOf(CharacterizationBoard board, Card card) {
        return RecordLines.parse(EntityJson.render(board.game, card, StaticSuppression.NONE));
    }

    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> entitiesIn(String state) {
        return (List<Map<String, Object>>) RecordLines.parse(state).get("entities");
    }

    /** The entity of a rendered state written under this card's id. */
    static Map<String, Object> entityIn(String state, Card card) {
        return entitiesIn(state).stream()
                .filter(entity -> Refs.of(card).equals(entity.get("id")))
                .findFirst().orElseThrow(() -> new AssertionError(card + " not in " + state));
    }

    /** These members of a parsed object, in this order; an absent member reads as null. */
    static List<Object> valuesOf(Map<String, Object> object, String... members) {
        return Arrays.stream(members).map(object::get).toList();
    }

    /** This member of each card's entity, rendered one card after the other. */
    static List<Object> memberOfEach(CharacterizationBoard board, String member, Card... cards) {
        return Arrays.stream(cards).map(card -> entityOf(board, card).get(member)).toList();
    }

    static List<Object> idsIn(String state) {
        return entitiesIn(state).stream().map(entity -> entity.get("id")).toList();
    }

    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> lines(Map<String, Object> entity) {
        return (List<Map<String, Object>>) entity.get("lines");
    }

    static Map<String, Object> globalOf(CharacterizationBoard board) {
        return RecordLines.asMap(RecordLines.parse(
                new SnapshotBuilder(board.game).ofBoard().render()).get("global"));
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> playerOf(CharacterizationBoard board, int index) {
        return ((List<Map<String, Object>>) RecordLines.parse(
                new SnapshotBuilder(board.game).ofBoard().render()).get("players")).get(index);
    }

    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> linesOf(CharacterizationBoard board, Card card) {
        return (List<Map<String, Object>>) entityOf(board, card).get("lines");
    }

    static List<Object> kinds(List<Map<String, Object>> lines) {
        return lines.stream().map(line -> line.get("kind")).toList();
    }

    /** The one line of this kind. */
    static Map<String, Object> line(List<Map<String, Object>> lines, String kind) {
        List<Map<String, Object>> found = lines.stream()
                .filter(line -> kind.equals(line.get("kind"))).toList();
        assertEquals(1, found.size(), lines.toString());
        return found.get(0);
    }

    /** The one keyword line written with this fragment. */
    static Map<String, Object> keywordLine(
            List<Map<String, Object>> lines, String fragment) {
        List<Map<String, Object>> found = lines.stream()
                .filter(line -> "keyword".equals(line.get("kind"))
                        && fragment.equals(line.get("fragment"))).toList();
        assertEquals(1, found.size(), "one " + fragment + " line in " + lines);
        return found.get(0);
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> onlyKey(Map<String, Object> line) {
        List<Object> keys = (List<Object>) line.get("keys");
        assertEquals(1, keys.size(), line.toString());
        return RecordLines.asMap(keys.get(0));
    }
}
