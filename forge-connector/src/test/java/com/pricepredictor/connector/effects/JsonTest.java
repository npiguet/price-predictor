package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The generic value writer event params and the combat payload render through.
 *
 * <p>Dedup hashes the rendered bytes, so what is pinned here is the spelling,
 * not just that the output parses.
 */
class JsonTest {

    @Test
    void scalarsRenderBareOrAsStringLiterals() {
        assertEquals("3", Json.value(3));
        assertEquals("-1.5", Json.value(-1.5));
        assertEquals("true", Json.value(true));
        assertEquals("\"E1\"", Json.value("E1"));
        assertEquals("\"MONDAY\"", Json.value(java.time.DayOfWeek.MONDAY));
    }

    @Test
    void mapsKeepTheirIterationOrderAndNest() {
        Map<String, Object> inner = new LinkedHashMap<>();
        inner.put("P1", 2);
        inner.put("E4", 1);
        Map<Object, Object> outer = new LinkedHashMap<>();
        outer.put("z", List.of("a", "b"));
        outer.put(7, inner);
        outer.put("empty", Map.of());

        assertEquals("{\"z\":[\"a\",\"b\"],\"7\":{\"P1\":2,\"E4\":1},\"empty\":{}}",
                Json.value(outer));
    }

    /**
     * A null nested in a map or a list is the string {@code "null"}: the
     * spelling events already in the corpus carry.
     */
    @Test
    void aNestedNullIsTheStringNull() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("k", null);
        List<Object> list = new ArrayList<>(Arrays.asList("a", null));

        assertEquals("{\"k\":\"null\"}", Json.value(map));
        assertEquals("[\"a\",\"null\"]", Json.value(list));
    }
}
