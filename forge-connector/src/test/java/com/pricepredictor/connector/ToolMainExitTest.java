package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * {@link ToolMain#exit} throws rather than ending the JVM, so every exit path
 * is testable; a {@code main} that does not run under {@link ToolMain#guard}
 * or {@link ToolMain#run} would end an exit it reaches as an uncaught error
 * with status 1 instead of its code.
 */
class ToolMainExitTest {

    @Test
    void everyMainRunsUnderTheExitGuard() throws IOException {
        Path sources = Path.of("src/main/java");
        List<Path> unguarded;
        try (Stream<Path> files = Files.walk(sources)) {
            unguarded = files.filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> {
                        String text = read(path);
                        return text.contains("public static void main(")
                                && !text.contains("ToolMain.guard(")
                                && !text.contains("ToolMain.run(");
                    })
                    .toList();
        }
        assertEquals(List.of(), unguarded);
    }

    private static String read(Path path) {
        try {
            return Files.readString(path);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }
}
