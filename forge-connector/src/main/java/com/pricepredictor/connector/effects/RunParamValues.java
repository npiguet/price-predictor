package com.pricepredictor.connector.effects;

import com.google.common.collect.Multiset;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * One run-parameter value read as what a record writes: a number, a flag,
 * a text, or the entities it names.
 *
 * <p>Forge hands a trigger's and a replacement's map over as plain objects
 * keyed by {@code AbilityKey} name; these are the readings every mode's row
 * ({@link RunParamRows}) is built from. Each answers null, or nothing, for a
 * value it cannot read: absent is the honest answer, and a zero or an empty
 * list would be a claim.
 */
final class RunParamValues {

    private RunParamValues() {
    }

    /** How many subjects one event may name before the list is capped. */
    static final int MAX_SUBJECTS = 8;

    /**
     * Add what one parameter value names to {@code into}: the value itself, or
     * each member of a collection, until {@code into} holds
     * {@value #MAX_SUBJECTS}.
     */
    static void refsIn(Object value, Set<String> into) {
        Iterable<?> members = value instanceof Iterable<?> many
                ? many : Collections.singletonList(value);
        for (Object member : members) {
            String ref = subjectOf(member);
            if (ref != null && into.size() < MAX_SUBJECTS) {
                into.add(ref);
            }
        }
    }

    /**
     * The entity a run-parameter value names, or null when it names none. A
     * spell or ability is named by the card it is on, the identity the
     * snapshot carries.
     */
    static String subjectOf(Object value) {
        if (value instanceof SpellAbility ability && ability.getHostCard() != null) {
            return Refs.of(ability.getHostCard());
        }
        return Refs.ofAny(value).orElse(null);
    }

    /** The first of these keys that names an entity, single or collection. */
    static String firstRef(Map<String, Object> byName, String... keys) {
        for (String key : keys) {
            Set<String> refs = new LinkedHashSet<>();
            refsIn(byName.get(key), refs);
            if (!refs.isEmpty()) {
                return refs.iterator().next();
            }
        }
        return null;
    }

    /** Every entity these keys name, deduplicated, or null for none. */
    static List<String> refList(Map<String, Object> byName, String... keys) {
        Set<String> refs = new LinkedHashSet<>();
        for (String key : keys) {
            refsIn(byName.get(key), refs);
        }
        return refs.isEmpty() ? null : new ArrayList<>(refs);
    }

    /** A parameter's own text, or null where it holds none. */
    static String textOf(Object value) {
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? null : text;
    }

    /** The first of these keys holding a number. */
    static Integer firstInt(Map<String, Object> byName, String... keys) {
        for (String key : keys) {
            Integer value = intOf(byName.get(key));
            if (value != null) {
                return value;
            }
        }
        return null;
    }

    static Integer intOf(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        try {
            return value == null ? null : Integer.valueOf(String.valueOf(value).trim());
        } catch (NumberFormatException e) {
            // A script variable Forge has not evaluated yet. Absent is the
            // right answer; zero would be a claim.
            return null;
        }
    }

    /** The numbers among these values, summed; an unreadable one counts 0. */
    static int sumOf(Iterable<?> values) {
        int sum = 0;
        for (Object value : values) {
            Integer each = intOf(value);
            sum += each == null ? 0 : each;
        }
        return sum;
    }

    static Boolean boolOf(Object value) {
        if (value instanceof Boolean flag) {
            return flag;
        }
        String text = value == null ? "" : String.valueOf(value);
        if ("true".equalsIgnoreCase(text)) {
            return Boolean.TRUE;
        }
        return "false".equalsIgnoreCase(text) ? Boolean.FALSE : null;
    }

    /**
     * A {@code CounterMap}'s counts per counter type, summed over its players,
     * in the order the map yields them; empty for anything that is not one.
     * {@code CounterPlayerAddedAll} hands over one player's multiset itself.
     */
    static Map<String, Integer> counterMapTotals(Object counterMap) {
        Map<String, Integer> total = new LinkedHashMap<>();
        for (Object value : multisetsIn(counterMap)) {
            if (value instanceof Multiset<?> multiset) {
                addCounts(multiset, total);
            }
        }
        return total;
    }

    private static Iterable<?> multisetsIn(Object counterMap) {
        if (counterMap instanceof Multiset<?> one) {
            return List.of(one);
        }
        return counterMap instanceof Map<?, ?> counters ? counters.values() : List.of();
    }

    private static void addCounts(Multiset<?> multiset, Map<String, Integer> total) {
        for (Multiset.Entry<?> entry : multiset.entrySet()) {
            total.merge(CardSpelling.counterName(entry.getElement()), entry.getCount(), Integer::sum);
        }
    }
}
