package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The set code a pool line records, which every downstream join compares as a string. */
@ExtendWith(ForgeExtension.class)
class PoolMainTest {

    @TempDir
    Path tempDir;

    private List<String> poolLines(String setCode) throws IOException {
        PoolMain.writePools(setCode, 1, tempDir, null);
        List<String> lines = Files.readAllLines(tempDir.resolve("pools.txt"));
        assertEquals(1, lines.size(), lines.toString());
        return lines;
    }

    /** {@code CON} is Conflux's alias, and the pool is Conflux's under its own code. */
    @Test
    void anAliasIsWrittenUnderItsEditionsCode() throws IOException {
        String line = poolLines("CON").get(0);
        assertTrue(line.startsWith("CFX" + CardNameList.FIELD_SEPARATOR), line);
    }

    @Test
    void aPlainCodeIsWrittenAsGiven() throws IOException {
        String line = poolLines("RVR").get(0);
        assertTrue(line.startsWith("RVR" + CardNameList.FIELD_SEPARATOR), line);
    }
}
