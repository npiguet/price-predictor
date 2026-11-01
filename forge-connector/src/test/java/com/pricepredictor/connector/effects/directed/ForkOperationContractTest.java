package com.pricepredictor.connector.effects.directed;

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

/**
 * The Java half of the fork-operation contract pair (T017): reads the
 * reader's {@code ForkOperation} in {@code fork_operation.py} and compares its
 * spellings, in order, with this enum's. {@code test_fork_operation.py} reads
 * this side's source the other way.
 */
class ForkOperationContractTest {

    private static final Path PYTHON = Path.of(
            "..", "src", "effects", "domain", "fork_operation.py");
    private static final Pattern MEMBER = Pattern.compile(
            "^\\s+[A-Z_]+ = \"([a-z_]+)\"$", Pattern.MULTILINE);

    @Test
    void javaAndPythonSpellTheOperationsAlike() throws IOException {
        Matcher m = MEMBER.matcher(Files.readString(PYTHON, StandardCharsets.UTF_8));
        List<String> python = new ArrayList<>();
        while (m.find()) {
            python.add(m.group(1));
        }
        List<String> java = Arrays.stream(ForkOperation.values()).map(WireNamed::wireName).toList();
        assertEquals(python, java);
    }
}
