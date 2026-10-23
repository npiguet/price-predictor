package com.pricepredictor.connector;

import forge.game.keyword.Keyword;
import org.eclipse.jetty.util.ajax.JSON;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The keyword table, and the part of it that is easy to write and never notice
 * is empty.
 *
 * <p>A fresh {@code KeywordInterface} carries no traits — Forge builds them in
 * {@code createTraits}, against a host card. Read without that call every
 * keyword reports a null script, the file still looks well-formed, and the
 * script surface silently loses the thing it exists to read. So the check here
 * is that the capture produced scripts at all.
 */
@ExtendWith(ForgeExtension.class)
class KeywordDefinitionMainTest {

    @TempDir
    static Path dir;

    /** The captured table, built once: every test reads the same file. */
    private static String json;

    /** The same table, parsed: display name to entry. */
    private static Map<String, Object> table;

    @BeforeAll
    @SuppressWarnings("unchecked")
    static void captureTheTable() throws IOException {
        Path output = dir.resolve("keyword-definitions.json");
        KeywordDefinitionMain.writeTo(output);
        json = Files.readString(output, StandardCharsets.UTF_8);
        table = (Map<String, Object>) JSON.parse(json);
    }

    /** The entry of the keyword displayed as {@code name}. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> entry(String name) {
        Map<String, Object> entry = (Map<String, Object>) table.get(name);
        assertNotNull(entry, name + " is missing");
        return entry;
    }

    @Test
    void everyKeywordGetsAnEntry() {
        for (Keyword keyword : Keyword.values()) {
            if (keyword != Keyword.UNDEFINED) {
                entry(keyword.toString());
            }
        }
    }

    @Test
    void pluralMarkupIsExpandedRatherThanShipped() {
        // Forge fills the count in from the card and expands {count:noun}
        // afterwards, so an instance never sees the markup. A template has no
        // instance, so it survived — and braces delimit an atom to the
        // tokenizer, which turned each one into a single vocabulary entry:
        // "{%1$d:tapped and attacking 1/1 red Warrior creature token}".
        assertFalse(json.contains("{%"),
                "a reminder template still carries {count:noun} markup");
    }

    @Test
    void aVariableCountPluralizesItsNoun() {
        // Forge's own variable-count branch: the placeholder stays in front,
        // where the encoder's placeholder dropping removes it, and the noun is
        // pluralized because an unknown count is not one.
        assertEquals(
                "Whenever this creature attacks, defending player sacrifices "
                        + "%d permanents.",
                KeywordDefinitionMain.expandPlurals(
                        "Whenever this creature attacks, defending player "
                                + "sacrifices {%d:permanent}."));
    }

    @Test
    void aLiteralOneTakesTheArticle() {
        // The other branch, and the reason not to just delete the count:
        // Emerge's cost is one of a thing, not a plural of it.
        assertEquals(
                "sacrificing a %2$s",
                KeywordDefinitionMain.expandPlurals("sacrificing {1:%2$s}"));
    }

    @Test
    void textWithoutMarkupIsUntouched() {
        // Mana symbols are braces too, and share the noun's delimiter.
        String text = "This spell costs {1} less to cast for each %s you control.";
        assertEquals(text, KeywordDefinitionMain.expandPlurals(text));
    }

    /** The templates themselves, not just the field: one with a value slot, one without. */
    @Test
    void reminderTemplatesAreCaptured() {
        assertTrue(json.contains("\"reminder_template\":"
                        + "\"This creature can't be blocked except by creatures with flying or reach.\""),
                "Flying's reminder template is missing");
        assertTrue(json.contains("\"reminder_template\":\"%s, Discard this card: Draw a card.\""),
                "Cycling's reminder template is missing");
    }

    /**
     * The formatter is the class Forge fills the keyword's template with, read
     * straight off the enum, so the Python expansion can format a value the way
     * Forge does rather than pasting it raw (FR-018).
     */
    @Test
    void everyKeywordRecordsHowForgeFormatsItsValues() {
        assertFalse(json.contains("\"formatter\":null"),
                "a keyword reports no formatter: the reflective read of Keyword.type failed");
        assertEquals("SimpleKeyword", KeywordDefinitionMain.formatterOf(Keyword.FLYING));
        assertEquals("KeywordWithCost", KeywordDefinitionMain.formatterOf(Keyword.CYCLING));
        assertEquals("KeywordWithType", KeywordDefinitionMain.formatterOf(Keyword.ENCHANT));
        assertEquals("Ward", KeywordDefinitionMain.formatterOf(Keyword.WARD));
    }

    @Test
    void theFormatterIsWrittenIntoEachEntry() {
        table.forEach((name, entry) -> assertTrue(
                ((Map<?, ?>) entry).get("formatter") instanceof String,
                name + " carries no formatter"));
        assertEquals("SimpleKeyword", entry("Flying").get("formatter"));
    }

    @Test
    void aScriptGeneratingKeywordHasOne() {
        assertNotNull(entry("Cascade").get("generated_script"),
                "Cascade generates a trigger and should carry its script: createTraits "
                        + "was probably not called, and the script surface reads nothing");
    }
}
