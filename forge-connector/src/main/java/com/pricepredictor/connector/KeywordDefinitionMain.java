package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.Json;
import forge.game.keyword.Keyword;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Writes every Forge keyword's reminder-text template as JSON.
 *
 * <p>The encoder's keyword-expansion dropout replaces a keyword token with its
 * definition, so the model learns "flying" and "can't be blocked except by
 * creatures with flying or reach" as the same thing rather than as an opaque
 * symbol and a sentence. A keyword the vocabulary has never seen is always
 * expanded, which is what lets a brand-new set's keywords work at all — so this
 * file is the difference between an unknown keyword being meaningless and being
 * merely unfamiliar.
 *
 * <p>Templates carry Forge's own {@code %s} placeholders. A parameterized
 * keyword ("Cycling {2}") instantiates its template with the instance's values
 * at expansion time; a keyword referenced without an instance, inside another
 * keyword's definition, expands with the generic wording as written here.
 *
 * <p>From stage four the generated implementation script joins the reminder
 * text for the script-generated majority of keywords; the engine-coded minority
 * generates no script and keeps its template alone.
 */
public class KeywordDefinitionMain {

    public static void main(String[] args) {
        String outputPath = "output/effects/keyword-definitions.json";
        for (int i = 0; i < args.length; i++) {
            if ("--output".equals(args[i]) && i + 1 < args.length) {
                outputPath = args[++i];
            }
        }

        try {
            ForgeEnvironmentInitializer.initialize();
            String json = render();
            Path output = Path.of(outputPath);
            if (output.getParent() != null) {
                Files.createDirectories(output.getParent());
            }
            Files.writeString(output, json, StandardCharsets.UTF_8);
            System.out.println("Wrote " + Keyword.values().length
                    + " keyword definitions to " + output);
            System.exit(0);
        } catch (IOException | RuntimeException e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * The keyword-definition document.
     *
     * <p>{@code {"<display name>": {"reminder_template": "…",
     * "generated_script": null}}} — one entry per {@code Keyword} enum member,
     * keyed by display name because that is what a card's keyword line and the
     * converted text both spell.
     */
    static String render() {
        List<String> entries = new ArrayList<>();
        for (Keyword keyword : Keyword.values()) {
            if (keyword == Keyword.UNDEFINED) {
                // Not a keyword: the catch-all Forge routes bespoke script
                // lines through, with no name a card ever prints.
                continue;
            }
            entries.add(Json.string(keyword.toString()) + ":{"
                    + "\"reminder_template\":" + Json.string(emptyToNull(keyword.getReminderText()))
                    + ",\"generated_script\":null}");
        }
        return "{" + String.join(",", entries) + "}";
    }

    /** Null rather than "" for a keyword with no reminder text of its own. */
    private static String emptyToNull(String value) {
        return value == null || value.isEmpty() ? null : value;
    }
}
