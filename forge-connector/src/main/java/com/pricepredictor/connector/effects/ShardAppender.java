package com.pricepredictor.connector.effects;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Appends whole gzip members to one shard file, and puts the file back as it
 * was when an append fails.
 *
 * <p>A failed append may have written part of a member. A partial member with
 * members after it makes every reader raise, so the file is cut back to its
 * length before the append: it then holds whole members only, and the caller
 * keeps what it meant to write for the next try.
 */
final class ShardAppender {

    /** What puts bytes on the end of a file: {@link #FILES} in production. */
    @FunctionalInterface
    interface Channel {
        void append(Path path, byte[] bytes) throws IOException;
    }

    static final Channel FILES = (path, bytes) -> Files.write(
            path, bytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND);

    private final Path path;
    private Channel channel = FILES;

    ShardAppender(Path path) {
        this.path = path;
    }

    /** Append through {@code channel} from now on: a test's failing one. */
    void through(Channel channel) {
        this.channel = channel;
    }

    /**
     * Append {@code member} in one call.
     *
     * @throws IOException the append's failure, after the file is cut back to
     *                     the length it had before
     */
    void append(byte[] member) throws IOException {
        boolean existed = Files.exists(path);
        long before = existed ? Files.size(path) : 0;
        try {
            channel.append(path, member);
        } catch (IOException failure) {
            cutBack(existed, before, failure);
            throw failure;
        }
    }

    private void cutBack(boolean existed, long before, IOException failure) {
        try {
            if (existed) {
                try (FileChannel file = FileChannel.open(path, StandardOpenOption.WRITE)) {
                    file.truncate(before);
                }
            } else {
                Files.deleteIfExists(path);
            }
        } catch (IOException cutFailed) {
            failure.addSuppressed(cutFailed);
        }
    }
}
