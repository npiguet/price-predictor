package com.pricepredictor.connector;

import com.google.common.base.Strings;
import com.pricepredictor.connector.json.Json;
import forge.game.keyword.Keyword;
import forge.util.Lang;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
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
 */
public class KeywordDefinitionMain {

    private static final ToolMain.Flag OUTPUT = new ToolMain.Flag("--output");

    public static void main(String[] args) {
        ToolMain.run(() -> work(args));
    }

    /** The extraction itself; an exit it requests ends the process through {@link ToolMain#run}. */
    static void work(String[] args) throws IOException {
        String outputPath = ToolMain.args(args, Set.of(), OUTPUT).require(OUTPUT);
        ForgeEnvironmentInitializer.initialize();
        System.out.println("Wrote keyword definitions to "
                + writeTo(Path.of(outputPath)));
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
     * "formatter": "SimpleKeyword"}}} — one entry per
     * {@code Keyword} enum member, keyed by display name because that is what a
     * card's keyword line and the converted text both spell. {@code formatter}
     * is the simple name of the keyword's {@code Keyword.type} class, which is
     * how Forge formats the keyword's values into its template.
     */
    static String render() {
        Json.ObjectWriter definitions = Json.object();
        for (Keyword keyword : Keyword.values()) {
            if (keyword == Keyword.UNDEFINED) {
                // Not a keyword: the catch-all Forge routes bespoke script
                // lines through, with no name a card ever prints.
                continue;
            }
            definitions.raw(keyword.toString(), Json.object()
                    .value("reminder_template",
                            Strings.emptyToNull(expandPlurals(keyword.getReminderText())))
                    .value("formatter", formatterOf(keyword))
                    .render());
        }
        return definitions.render();
    }

    /**
     * How Forge formats a keyword's instance values into its template.
     *
     * <p>A template's {@code %s} is filled by the keyword's own class —
     * {@code KeywordWithCost} renders a mana cost as symbols,
     * {@code KeywordWithAmount} an integer, {@code KeywordWithType} a type name
     * — so an expansion done in Python has to know which class that is, and
     * {@code Keyword.type} is the one place Forge records it. The field is
     * protected and has no accessor, hence the reflective read; the enum is on
     * the plain classpath, so nothing stands in its way.
     */
    static String formatterOf(Keyword keyword) {
        try {
            Field type = Keyword.class.getDeclaredField("type");
            type.setAccessible(true);
            Object value = type.get(keyword);
            return value instanceof Class<?> c ? c.getSimpleName() : null;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
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
}
