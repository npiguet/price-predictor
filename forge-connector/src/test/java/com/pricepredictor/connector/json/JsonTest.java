package com.pricepredictor.connector.json;

import com.pricepredictor.connector.effects.DamageStep;
import com.pricepredictor.connector.effects.EventDuration;
import com.pricepredictor.connector.effects.EventType;
import com.pricepredictor.connector.effects.RecordKind;
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
     * A null nested in a map or a list is JSON {@code null}, not a string a
     * reader would take for a value named "null".
     */
    @Test
    void aNestedNullIsJsonNull() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("k", null);
        List<Object> list = new ArrayList<>(Arrays.asList("a", null));

        assertEquals("{\"k\":null}", Json.value(map));
        assertEquals("[\"a\",null]", Json.value(list));
        assertEquals("null", Json.value(null));
    }

    /**
     * A vocabulary member is spelled by its wire name wherever it sits, never
     * by its Java constant name, which every reader refuses.
     */
    @Test
    void aWireNamedMemberIsItsWireName() {
        Map<Object, Object> map = new LinkedHashMap<>();
        map.put(DamageStep.FIRST_STRIKE, List.of(EventType.ZONE_CHANGE));

        assertEquals("\"end_of_turn\"", Json.value(EventDuration.END_OF_TURN));
        assertEquals("{\"first_strike\":[\"zone_change\"]}", Json.value(map));
        assertEquals("{\"kind\":\"rewrite\"}",
                Json.object().value("kind", RecordKind.REWRITE).render());
    }

    /** The string "null" is still a string. */
    @Test
    void theStringNullIsAString() {
        assertEquals("\"null\"", Json.value("null"));
    }

    /** Fields come out in the order they went in, whichever method added them. */
    @Test
    void anObjectWriterKeepsTheOrderItWasGiven() {
        assertEquals("{\"z\":null,\"b\":1,\"a\":[],\"y\":\"v\"}",
                Json.object()
                        .raw("z", "null")
                        .value("b", 1)
                        .value("a", List.of())
                        .value("y", "v")
                        .render());
        assertEquals("{}", Json.object().render());
    }
}
