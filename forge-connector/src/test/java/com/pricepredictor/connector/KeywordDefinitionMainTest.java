package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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

    private String write(@TempDir Path dir) throws IOException {
        Path output = dir.resolve("keyword-definitions.json");
        KeywordDefinitionMain.writeTo(output);
        return Files.readString(output, StandardCharsets.UTF_8);
    }

    @Test
    void everyKeywordGetsAnEntry(@TempDir Path dir) throws IOException {
        String json = write(dir);
        assertTrue(json.startsWith("{"));
        assertTrue(json.contains("\"Flying\""), "Flying is missing");
        assertTrue(json.contains("\"Cascade\""), "Cascade is missing");
    }

    @Test
    void pluralMarkupIsExpandedRatherThanShipped(@TempDir Path dir)
            throws IOException {
        // Forge fills the count in from the card and expands {count:noun}
        // afterwards, so an instance never sees the markup. A template has no
        // instance, so it survived — and braces delimit an atom to the
        // tokenizer, which turned each one into a single vocabulary entry:
        // "{%1$d:tapped and attacking 1/1 red Warrior creature token}".
        String json = write(dir);
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

    @Test
    void reminderTemplatesAreCaptured(@TempDir Path dir) throws IOException {
        assertTrue(write(dir).contains("reminder_template"));
    }

    @Test
    void someKeywordsCaptureTheirGeneratedScript(@TempDir Path dir)
            throws IOException {
        String json = write(dir);
        assertFalse(
                json.contains("\"generated_script\":null")
                        && !json.contains("\"generated_script\":\""),
                "no keyword captured a generated script: createTraits was "
                        + "probably not called, and the script surface reads "
                        + "nothing");
    }

    @Test
    void aScriptGeneratingKeywordHasOne(@TempDir Path dir) throws IOException {
        String json = write(dir);
        int at = json.indexOf("\"Cascade\"");
        assertTrue(at >= 0, "Cascade is missing");
        String entry = json.substring(at, Math.min(at + 2000, json.length()));
        int script = entry.indexOf("\"generated_script\":");
        assertTrue(script >= 0);
        assertFalse(
                entry.startsWith("null", script + "\"generated_script\":".length()),
                "Cascade generates a trigger and should carry its script");
    }
}
