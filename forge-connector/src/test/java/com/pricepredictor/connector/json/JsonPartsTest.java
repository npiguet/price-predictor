package com.pricepredictor.connector.json;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Splitting written JSON back into its parts, nested values kept whole as text. */
class JsonPartsTest {

    @Test
    void anArraySplitsIntoItsElementsWithNestedValuesWhole() {
        String array = "[{\"a\":[1,2],\"b\":\"x,]\"},3,\"s\\\"q,\",[]]";

        assertEquals(List.of("{\"a\":[1,2],\"b\":\"x,]\"}", "3", "\"s\\\"q,\"", "[]"),
                JsonParts.elements(array));
    }

    @Test
    void anEmptyArrayHasNoElements() {
        assertEquals(List.of(), JsonParts.elements("[]"));
        assertEquals(List.of(), JsonParts.elements(" [ ] "));
    }

    @Test
    void anObjectSplitsIntoItsFieldsInOrder() {
        String object = "{\"id\":\"C1\",\"lines\":[{\"k\":\"}\"}],\"n\":null}";

        Map<String, String> fields = JsonParts.fields(object);

        assertEquals(List.of("id", "lines", "n"), List.copyOf(fields.keySet()));
        assertEquals("\"C1\"", fields.get("id"));
        assertEquals("[{\"k\":\"}\"}]", fields.get("lines"));
        assertEquals("null", fields.get("n"));
    }

    @Test
    void whatJsonWritesSplitsBackIntoWhatWasWritten() {
        String written = Json.object().value("name", "a, \"b\"").raw("xs", "[1, 2]").render();

        Map<String, String> fields = JsonParts.fields(written);

        assertEquals(Json.string("a, \"b\""), fields.get("name"));
        assertEquals("[1, 2]", fields.get("xs"));
    }
}
