package com.pricepredictor.connector.effects;

import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;

import java.util.HashSet;
import java.util.Set;

/**
 * The records one game has already written, by a hash of what they say.
 *
 * <p>What a record says is everything a reader learns from it except its
 * envelope: its kind with its moment or subkind, the player it is about, the
 * line that acted (or why none is named), the snapshot and the payload. The
 * envelope — record id, timestamp, run and game — differs on every record or
 * is fixed for the game, so it can never tell two observations apart. Two
 * records agreeing on all of the rest carry one observation, so collapsing
 * them loses nothing, and the AI re-asking the same question about an unmoved
 * board makes such pairs common.
 *
 * <p>The acting line is part of it because the snapshot does not always name
 * it. A trigger's or replacement's snapshot carries its host and the event,
 * not which of the host's traits is evaluating, so two same-mode triggers on
 * one card render the same state and payload for one event; keyed without
 * the line, only the first trigger's verdict would ever be written.
 *
 * <p>One per game, shared by every channel that deduplicates.
 */
final class RecordDedup {

    /** Distinct rendered records this game may hold before dedup stops. */
    private static final int COALESCE_MAX_KEYS = 200_000;

    private final Set<Long> written = new HashSet<>();

    /**
     * Whether a record saying what {@code record} says has not been written
     * this game yet; remembered as written when it has not.
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
    boolean allowDistinct(EffectRecord record) {
        if (written.size() >= COALESCE_MAX_KEYS) {
            return true;
        }
        Hasher hasher = Hashing.murmur3_128().newHasher();
        field(hasher, record.kindTag());
        field(hasher, record.actorPlayer());
        field(hasher, record.abilityJson());
        field(hasher, record.abilityUnresolvedWire());
        field(hasher, record.payloadJson());
        field(hasher, record.stateJson());
        return written.add(hasher.hash().asLong());
    }

    /**
     * One part, length-prefixed so that no split of the same characters
     * between two neighbouring parts hashes alike; null apart from empty.
     */
    private static void field(Hasher hasher, String value) {
        if (value == null) {
            hasher.putInt(-1);
            return;
        }
        hasher.putInt(value.length()).putUnencodedChars(value);
    }
}
