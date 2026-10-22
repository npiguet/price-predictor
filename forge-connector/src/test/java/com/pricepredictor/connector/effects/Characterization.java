package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.SpellAbilityStackInstance;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import org.junit.jupiter.api.extension.AfterEachCallback;
import org.junit.jupiter.api.extension.BeforeEachCallback;
import org.junit.jupiter.api.extension.ExtensionContext;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.fail;

/**
 * Rendered output compared byte for byte against a checked-in expected file.
 *
 * <p>The effect-record corpus is append-only and its dedup hashes the rendered
 * bytes, so a refactor of the renderers has to reproduce them exactly; these
 * files are the bytes the renderers produced when they were captured. Set
 * {@code -Dcharacterization.update=true} to rewrite them from the current code,
 * and read the diff before committing it: a changed expected file is a changed
 * corpus contract.
 */
final class Characterization {

    private static final Path DIR = Path.of("src/test/resources/characterization");

    private static final String UPDATE_PROPERTY = "characterization.update";

    /**
     * The envelope's timestamp is the one field that reads the wall clock, so
     * it is the one field replaced, and only after it is checked to be the
     * instant it claims to be.
     */
    private static final Pattern TIMESTAMP = Pattern.compile(
            "^(\\{\"record_id\":\"[^\"]*\",\"run_id\":\"[^\"]*\",\"timestamp\":\")([^\"]*)(\")");

    private static final String TIMESTAMP_PLACEHOLDER = "<timestamp>";

    /** The literal runs of the head {@link #TIMESTAMP} matches, around the two ids. */
    private static final List<String> ENVELOPE_HEAD = List.of(
            "{\"record_id\":\"", "\",\"run_id\":\"", "\",\"timestamp\":\"");

    /** Characters either side of the first difference shown on a mismatch. */
    private static final int CONTEXT = 160;

    private Characterization() {
    }

    /** Record lines, each with its timestamp fixed, as one file's text. */
    static String recordLines(List<String> records) {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < records.size(); i++) {
            text.append(withFixedTimestamp(i + 1, records.get(i))).append('\n');
        }
        return text.toString();
    }

    private static String withFixedTimestamp(int line, String record) {
        Matcher matcher = TIMESTAMP.matcher(record);
        if (!matcher.find()) {
            int at = envelopeDeparture(record);
            fail("record " + line + " leaves the envelope head "
                    + String.join("...", ENVELOPE_HEAD) + " at offset " + at + ": "
                    + excerpt(record, at));
        }
        try {
            Instant.parse(matcher.group(2));
        } catch (DateTimeParseException e) {
            fail("the timestamp is not an ISO 8601 instant: " + matcher.group(2));
        }
        return matcher.group(1) + TIMESTAMP_PLACEHOLDER + matcher.group(3)
                + record.substring(matcher.end());
    }

    /**
     * Where a record first departs from the envelope head: the offset of the
     * first character that breaks one of its literal runs, or the record's
     * length when the head is all there but the timestamp never closes.
     */
    private static int envelopeDeparture(String record) {
        int at = 0;
        for (int run = 0; run < ENVELOPE_HEAD.size(); run++) {
            if (run > 0) {
                int close = record.indexOf('"', at);
                if (close < 0) {
                    return record.length();
                }
                at = close;
            }
            String literal = ENVELOPE_HEAD.get(run);
            for (int i = 0; i < literal.length(); i++, at++) {
                if (at >= record.length() || record.charAt(at) != literal.charAt(i)) {
                    return at;
                }
            }
        }
        return record.length();
    }

    /** Assert {@code actual} is the expected file's text, byte for byte. */
    static void assertMatches(String name, String actual) {
        Path path = DIR.resolve(name);
        try {
            if (Boolean.getBoolean(UPDATE_PROPERTY)) {
                Files.createDirectories(DIR);
                Files.writeString(path, actual, StandardCharsets.UTF_8);
            }
            if (!Files.exists(path)) {
                fail("no expected file " + path.toAbsolutePath()
                        + "; run with -D" + UPDATE_PROPERTY + "=true to capture it");
            }
            String expected = Files.readString(path, StandardCharsets.UTF_8);
            if (!expected.equals(actual)) {
                fail(describeMismatch(name, expected, actual));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Where the two texts first part, with the characters around it from each. */
    private static String describeMismatch(String name, String expected, String actual) {
        int at = 0;
        int shorter = Math.min(expected.length(), actual.length());
        while (at < shorter && expected.charAt(at) == actual.charAt(at)) {
            at++;
        }
        int line = 1;
        int lineStart = 0;
        for (int i = 0; i < at; i++) {
            if (expected.charAt(i) == '\n') {
                line++;
                lineStart = i + 1;
            }
        }
        return name + " differs at offset " + at + " (line " + line + ", column "
                + (at - lineStart + 1) + "); expected " + expected.length()
                + " chars, got " + actual.length()
                + "\n  expected: " + excerpt(expected, at)
                + "\n  actual:   " + excerpt(actual, at);
    }

    private static String excerpt(String text, int at) {
        int from = Math.max(0, at - CONTEXT);
        int to = Math.min(text.length(), at + CONTEXT);
        return (from > 0 ? "..." : "") + oneLine(text.substring(from, at))
                + " >>>" + oneLine(text.substring(at, to)) + (to < text.length() ? "..." : "");
    }

    private static String oneLine(String text) {
        return text.replace("\n", "\\n");
    }

    /**
     * Restarts Forge's JVM-wide id counters for each test at a fixed base,
     * and puts them back afterwards.
     *
     * <p>A cost half's {@code link_id} carries its ability's id, and Forge
     * numbers spell abilities, statics, triggers, replacements, stack entries
     * and games each from one counter for the whole JVM, so the ids a test's
     * objects get depend on how many every earlier test made. A fresh game per
     * test fixes the card and player ids; this fixes the ones a game does not
     * own. The base is far above any id the suite reaches, so a pinned id is
     * never one a live object already has, and putting the counters back
     * leaves every other test numbering as it would have.
     */
    static final class PinnedForgeIds implements BeforeEachCallback, AfterEachCallback {

        private static final int BASE = 1_000_000;

        private static final List<Class<?>> COUNTED = List.of(
                SpellAbility.class, StaticAbility.class, Trigger.class,
                ReplacementEffect.class, SpellAbilityStackInstance.class, Game.class);

        private static final ExtensionContext.Namespace NAMESPACE =
                ExtensionContext.Namespace.create(PinnedForgeIds.class);

        @Override
        public void beforeEach(ExtensionContext context) throws Exception {
            for (Class<?> owner : COUNTED) {
                Field counter = counter(owner);
                int found = counter.getInt(null);
                if (found >= BASE) {
                    throw new IllegalStateException(
                            owner.getSimpleName() + " ids already reach " + found
                                    + "; raise BASE");
                }
                context.getStore(NAMESPACE).put(owner, found);
                counter.setInt(null, BASE);
            }
        }

        @Override
        public void afterEach(ExtensionContext context) throws Exception {
            for (Class<?> owner : COUNTED) {
                Integer found = context.getStore(NAMESPACE).get(owner, Integer.class);
                if (found != null) {
                    counter(owner).setInt(null, found);
                }
            }
        }

        private static Field counter(Class<?> owner) throws NoSuchFieldException {
            Field field = owner.getDeclaredField("maxId");
            field.setAccessible(true);
            return field;
        }
    }
}
