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
 * A decision candidate's {@code failed_checks} identifiers are the record
 * contract's eleven, spelled and ordered as its payload table lists them
 * (FR-032): the rule detectors' {@code verdict_bit} declarations name a check
 * by that spelling.
 */
class PlayabilityCheckContractTest {

    private static final Path CONTRACT = Path.of("..", "specs",
            "025-ability-effect-model-gen3", "contracts", "record-schema-delta.md");
    private static final Pattern ROW = Pattern.compile(
            "^\\| decision candidate \\| `failed_checks` \\|(.*)\\|$", Pattern.MULTILINE);
    private static final Pattern IDENTIFIER = Pattern.compile("`([a-z_]+)`");
    /** The row's "∈", after which it lists the identifiers. */
    private static final char ELEMENT_OF = '∈';

    @Test
    void theIdentifiersAreTheContractsInItsOrder() throws IOException {
        Matcher row = ROW.matcher(Files.readString(CONTRACT, StandardCharsets.UTF_8));
        assertTrue(row.find(), "no failed_checks row in " + CONTRACT);
        String shape = row.group(1);
        Matcher identifier = IDENTIFIER.matcher(shape.substring(shape.indexOf(ELEMENT_OF)));
        List<String> contract = new ArrayList<>();
        while (identifier.find()) {
            contract.add(identifier.group(1));
        }
        assertEquals(11, contract.size(), shape);
        assertEquals(contract,
                Arrays.stream(PlayabilityCheck.values()).map(WireNamed::wireName).toList());
    }
}
