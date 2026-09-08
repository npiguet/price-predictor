package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.Json;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.util.Lang;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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
            System.out.println("Wrote keyword definitions to "
                    + writeTo(Path.of(outputPath)));
            System.exit(0);
        } catch (IOException | RuntimeException e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /** Render the document and write it, creating the parent directory. */
    static Path writeTo(Path output) throws IOException {
        if (output.getParent() != null) {
            Files.createDirectories(output.getParent());
        }
        Files.writeString(output, render(), StandardCharsets.UTF_8);
        return output;
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
                    + "\"reminder_template\":"
                    + Json.string(emptyToNull(expandPlurals(keyword.getReminderText())))
                    + ",\"generated_script\":" + Json.string(generatedScript(keyword))
                    + "}");
        }
        return "{" + String.join(",", entries) + "}";
    }

    /**
     * Forge's plural markup, rendered the way Forge renders it.
     *
     * <p>A reminder text carries {@code {count:noun}} where the count is filled
     * in from the card and the noun is pluralized to match —
     * {@code "sacrifices {%d:permanent}"}. Forge does that substitution first
     * and expands the markup afterwards, so an instance never sees it. A
     * template has no instance and no values, so the markup survived into the
     * definition file, and the tokenizer read each one as a single symbol
     * because braces delimit an atom: {@code {%1$d:tapped and attacking 1/1 red
     * Warrior creature token}} became one vocabulary entry.
     *
     * <p>Expanded through Forge's own {@link Lang}, so the plural forms are the
     * engine's rather than a second implementation of English. A count that is
     * still a format placeholder takes Forge's variable-count branch, which
     * pluralizes the noun and leaves the placeholder in front of it — where the
     * encoder's existing placeholder dropping removes it, the same way it
     * already handles a bare {@code %s} elsewhere in a template.
     */
    static String expandPlurals(String reminderText) {
        if (reminderText == null || reminderText.indexOf('{') < 0) {
            return reminderText;
        }
        Matcher matcher = PLURAL_MARKUP.matcher(reminderText);
        StringBuilder out = new StringBuilder();
        while (matcher.find()) {
            matcher.appendReplacement(out, Matcher.quoteReplacement(
                    Lang.nounWithNumeralExceptOne(matcher.group(1), matcher.group(2))));
        }
        matcher.appendTail(out);
        return out.toString();
    }

    /**
     * {@code {count:noun}}, with a wider count than Forge's own matcher.
     *
     * <p>Forge matches the count as {@code \w+} because by the time it looks,
     * the count is a number. Here it is still {@code %d} or {@code %1$s}, which
     * {@code \w+} does not match — the reason the markup came through untouched.
     * Neither part may contain a brace, so a mana symbol beside it is not at
     * risk of being swallowed.
     */
    private static final Pattern PLURAL_MARKUP =
            Pattern.compile("\\{([^:{}]+):([^{}]+)\\}");

    /**
     * The implementation script Forge's keyword factory generates, as text.
     *
     * <p>Most keywords are implemented by generating traits from a template —
     * flying becomes a static ability, cycling becomes an activated one — and
     * that generated script is the mechanism the keyword stands for. On the
     * stage-four script surface it is what a keyword expands to, and it is far
     * closer to what the keyword *does* than its reminder text is.
     *
     * <p>Null for the engine-coded minority, which generates no script and keeps
     * its reminder template. Captured at the factory rather than read from a
     * file because no file contains it.
     */
    private static String generatedScript(Keyword keyword) {
        try {
            // The factory keys on the display name, which is what a card's
            // keyword line spells and what the definition file is keyed by.
            KeywordInterface instance = Keyword.getInstance(keyword.toString());
            if (instance == null) {
                return null;
            }
            // A fresh instance carries no traits: Forge builds them in
            // createTraits, against a host card. Without this call every
            // keyword reports an empty script and the whole capture is silently
            // null. The host is a bare Card with no game — enough for the
            // template expansion the factory does, and nothing more.
            instance.createTraits(new Card(0, null), true, true);
            List<String> parts = new ArrayList<>();
            for (Trigger trigger : instance.getTriggers()) {
                parts.add(renderParams(trigger.getMapParams()));
            }
            for (ReplacementEffect replacement : instance.getReplacements()) {
                parts.add(renderParams(replacement.getMapParams()));
            }
            for (StaticAbility staticAbility : instance.getStaticAbilities()) {
                parts.add(renderParams(staticAbility.getMapParams()));
            }
            for (SpellAbility ability : instance.getAbilities()) {
                parts.add(renderParams(ability.getMapParams()));
            }
            parts.removeIf(part -> part == null || part.isEmpty());
            return parts.isEmpty() ? null : String.join(" ;; ", parts);
        } catch (RuntimeException e) {
            // A keyword whose factory needs a host card cannot be captured
            // standalone; it keeps its reminder template.
            return null;
        }
    }

    private static String renderParams(Map<String, String> params) {
        if (params == null || params.isEmpty()) {
            return null;
        }
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> entry : new TreeMap<>(params).entrySet()) {
            if (sb.length() > 0) {
                sb.append(" | ");
            }
            sb.append(entry.getKey()).append("$ ").append(entry.getValue());
        }
        return sb.toString();
    }

    /** Null rather than "" for a keyword with no reminder text of its own. */
    private static String emptyToNull(String value) {
        return value == null || value.isEmpty() ? null : value;
    }
}
