package com.pricepredictor.connector.effects;

import com.google.common.hash.Hashing;

import java.util.HashSet;
import java.util.Set;

/**
 * The records one game has already written, by a hash of what they say.
 *
 * <p>Two records that would be byte-identical carry one observation once, so
 * collapsing them loses nothing, and the AI re-asking the same question about
 * an unmoved board makes such pairs common.
 *
 * <p>One per game, shared by every channel that deduplicates: the tag keeps
 * one kind's record from colliding with another's, and nothing else
 * separates them.
 */
final class RecordDedup {

    /** Distinct rendered records this game may hold before dedup stops. */
    private static final int COALESCE_MAX_KEYS = 200_000;

    private final Set<Long> written = new HashSet<>();

    /**
     * Whether a record saying exactly this has not been written this game yet.
     *
     * <p>A 64-bit hash rather than the 32-bit {@code String.hashCode}: over
     * the hundred thousand keys one game can produce,
     * a 32-bit space has a real chance of colliding, and a collision here
     * silently drops a record that was not a duplicate.
     *
     * <p>Bounded: past the cap it stops deduplicating rather than growing, so a
     * pathological game reverts to writing duplicates instead of exhausting a
     * 1200 MB heap. The duplicate rate of a run is therefore something to
     * measure rather than assume.
     */
    boolean allowDistinct(String tag, String payload, String state) {
        if (written.size() >= COALESCE_MAX_KEYS) {
            return true;
        }
        long digest = Hashing.murmur3_128().newHasher()
                .putUnencodedChars(tag).putChar('@')
                .putUnencodedChars(payload == null ? "" : payload).putChar('#')
                .putUnencodedChars(state == null ? "" : state)
                .hash().asLong();
        return written.add(digest);
    }
}
