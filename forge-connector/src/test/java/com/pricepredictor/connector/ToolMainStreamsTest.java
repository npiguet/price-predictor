package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * The protocol streams must carry card names as UTF-8 whatever the platform
 * default is, because Python decodes the worker's pipes strictly as UTF-8.
 */
class ToolMainStreamsTest {

    private static final String ACCENTED = "Éowyn, Fearless Knight";

    @Test
    void utf8StreamEncodesNonAsciiNamesAsUtf8() {
        ByteArrayOutputStream sink = new ByteArrayOutputStream();
        PrintStream stream = ToolMain.utf8(sink);

        stream.println(ACCENTED);
        stream.flush();

        assertArrayEquals(
                (ACCENTED + System.lineSeparator()).getBytes(StandardCharsets.UTF_8),
                sink.toByteArray());
    }
}
