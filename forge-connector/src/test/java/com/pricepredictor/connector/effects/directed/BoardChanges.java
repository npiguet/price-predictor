package com.pricepredictor.connector.effects.directed;

import forge.game.card.Card;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Two board-only snapshots compared member by member, for the tests that
 * check a fork edit or a directed fork changes exactly its one fact.
 */
public final class BoardChanges {

    private BoardChanges() {
    }

    public static String id(Card card) {
        return "E" + card.getId();
    }

    /**
     * What differs between two snapshots: {@code E5.tapped} for an entity
     * member, {@code P1.life} for a player member, {@code global.turn} for a
     * global one, {@code +E9} / {@code -E5} for an entity that appeared or
     * went, and {@code +link …} / {@code -link …} for a link.
     */
    public static Set<String> changes(Map<String, Object> before, Map<String, Object> after) {
        Set<String> changed = new TreeSet<>();
        members("global", asMap(before.get("global")), asMap(after.get("global")), changed);
        byId("players", before, after, changed);
        byId("entities", before, after, changed);
        Set<String> was = links(before);
        Set<String> is = links(after);
        for (String link : was) {
            if (!is.contains(link)) {
                changed.add("-link " + link);
            }
        }
        for (String link : is) {
            if (!was.contains(link)) {
                changed.add("+link " + link);
            }
        }
        return changed;
    }

    private static void byId(String block, Map<String, Object> before, Map<String, Object> after,
                             Set<String> changed) {
        Map<String, Map<String, Object>> was = indexed(before, block);
        Map<String, Map<String, Object>> is = indexed(after, block);
        for (Map.Entry<String, Map<String, Object>> entry : was.entrySet()) {
            Map<String, Object> now = is.get(entry.getKey());
            if (now == null) {
                changed.add("-" + entry.getKey());
            } else {
                members(entry.getKey(), entry.getValue(), now, changed);
            }
        }
        for (String id : is.keySet()) {
            if (!was.containsKey(id)) {
                changed.add("+" + id);
            }
        }
    }

    private static void members(String owner, Map<String, Object> was, Map<String, Object> is,
                                Set<String> changed) {
        Set<String> keys = new HashSet<>(was.keySet());
        keys.addAll(is.keySet());
        for (String key : keys) {
            if (!java.util.Objects.equals(was.get(key), is.get(key))) {
                changed.add(owner + "." + key);
            }
        }
    }

    public static Map<String, Map<String, Object>> indexed(Map<String, Object> state, String block) {
        Map<String, Map<String, Object>> byId = new LinkedHashMap<>();
        for (Object item : (List<?>) state.get(block)) {
            Map<String, Object> map = asMap(item);
            byId.put((String) map.get("id"), map);
        }
        return byId;
    }

    /** Each link as one string, its members in name order: {@code {from=P0, to=E3, type=controls}}. */
    public static Set<String> links(Map<String, Object> state) {
        Set<String> links = new TreeSet<>();
        for (Object link : (List<?>) state.get("links")) {
            links.add(new TreeMap<>(asMap(link)).toString());
        }
        return links;
    }

    public static Map<String, Object> entity(Map<String, Object> state, String id) {
        Map<String, Object> entity = indexed(state, "entities").get(id);
        assertTrue(entity != null, "no entity " + id);
        return entity;
    }

    /** {@code value} with every string equal to {@code from} read as {@code to}. */
    @SuppressWarnings("unchecked")
    public static <T> T renamed(T value, String from, String to) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> copy = new LinkedHashMap<>();
            map.forEach((key, entry) -> copy.put((String) key, renamed(entry, from, to)));
            return (T) copy;
        }
        if (value instanceof List<?> list) {
            return (T) list.stream().map(entry -> renamed(entry, from, to)).toList();
        }
        if (from.equals(value)) {
            return (T) to;
        }
        return value;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> asMap(Object value) {
        return (Map<String, Object>) value;
    }
}
