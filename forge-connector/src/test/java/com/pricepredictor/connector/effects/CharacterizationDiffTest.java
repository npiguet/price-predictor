package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.util.List;
import java.util.zip.GZIPOutputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The characterization harness's own reading: what a regeneration prints
 * about the lines it rewrote, and how a shard's gzip members are split.
 */
class CharacterizationDiffTest {

    @Test
    void aRewrittenFileIsReportedByTheMembersThatChanged() {
        String before = "{\"a\":1,\"state\":{\"life\":20,\"tags\":[1,2]}}\n{\"b\":true}\n";
        String after = "{\"a\":1,\"state\":{\"life\":19,\"tags\":[1,3]}}\n{\"b\":true}\n{\"c\":0}\n";

        assertEquals("characterization: x.jsonl rewritten, 2 -> 3 lines"
                        + "\n  line 1: state.life, state.tags[1]"
                        + "\n  line 3: <record>",
                Characterization.changedMembers("x.jsonl", before, after));
    }

    @Test
    void aShardsMembersAreSplitInOrder() throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        for (String member : List.of("one\n", "two\nthree\n")) {
            try (GZIPOutputStream out = new GZIPOutputStream(bytes)) {
                out.write(member.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            }
        }
        byte[] shard = bytes.toByteArray();

        assertEquals(List.of("one\n", "two\nthree\n"), Characterization.gzipMembers(shard));
        byte[] cut = java.util.Arrays.copyOf(shard, shard.length - 12);
        assertThrows(AssertionError.class, () -> Characterization.gzipMembers(cut));
    }
}
