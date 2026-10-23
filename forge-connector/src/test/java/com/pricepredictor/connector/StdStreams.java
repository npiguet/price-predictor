package com.pricepredictor.connector;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * What a body printed on stderr or stdout, read back as a string.
 *
 * <p>The real stream is put back however the body ends, so a failing
 * assertion inside it cannot leave every later test writing into a buffer.
 */
public final class StdStreams {

    private StdStreams() {
    }

    /** What a test runs while the stream is captured. */
    @FunctionalInterface
    public interface Body {
        void run() throws Throwable;
    }

    public static String captureStderr(Body body) throws Throwable {
        return capture(() -> System.err, System::setErr, body);
    }

    public static String captureStdout(Body body) throws Throwable {
        return capture(() -> System.out, System::setOut, body);
    }

    private static String capture(
            Supplier<PrintStream> current, Consumer<PrintStream> install, Body body)
            throws Throwable {
        PrintStream real = current.get();
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        install.accept(new PrintStream(captured, true, StandardCharsets.UTF_8));
        try {
            body.run();
        } finally {
            install.accept(real);
        }
        return captured.toString(StandardCharsets.UTF_8);
    }
}
