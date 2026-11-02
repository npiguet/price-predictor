package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Java half of the relation-family contract pair (T058): reads
 * {@code FAMILIES} in {@code relation_types.py} and compares its spellings, in
 * order, with {@link LinkFamily}'s, so a link's {@code type} is always a
 * family the trunk has a row for, and both sides keep § 10.3's table order.
 * {@code test_relation_types.py} reads this side's source the other way.
 */
class LinkFamilyContractTest {

    private static final Path PYTHON = Path.of(
            "..", "src", "effects", "domain", "relation_types.py");
    private static final Pattern FAMILIES = Pattern.compile(
            "^FAMILIES: tuple\\[str, \\.\\.\\.] = \\((.*?)^\\)", Pattern.MULTILINE | Pattern.DOTALL);
    private static final Pattern NAME = Pattern.compile("\"([a-z_]+)\"");

    @Test
    void javaAndPythonSpellTheFamiliesAlikeInTableOrder() throws IOException {
        Matcher block = FAMILIES.matcher(Files.readString(PYTHON, StandardCharsets.UTF_8));
        assertTrue(block.find(), "no FAMILIES tuple in " + PYTHON);
        Matcher name = NAME.matcher(block.group(1));
        List<String> python = new ArrayList<>();
        while (name.find()) {
            python.add(name.group(1));
        }
        List<String> java = Arrays.stream(LinkFamily.values()).map(WireNamed::wireName).toList();
        assertEquals(python, java);
    }

    /** The four families FR-050 has the reader build from its own layout. */
    @Test
    void theReaderBuildsTheLineAndNameFamilies() {
        List<LinkFamily> notWritten = Arrays.stream(LinkFamily.values())
                .filter(family -> !family.collected()).toList();
        assertEquals(List.of(LinkFamily.LINE_OF, LinkFamily.LINE_OF_THE_SAME_CARD,
                LinkFamily.MODE_OF, LinkFamily.HAS_THE_SAME_NAME), notWritten);
    }
}
