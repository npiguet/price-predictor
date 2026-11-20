package com.pricepredictor.connector.effects;

/**
 * The four members that say which record a line is and where it came from:
 * {@code record_id}, {@code run_id}, {@code timestamp} and {@code game_id},
 * as a record's envelope spells them.
 *
 * <p>They travel together from the shard writer, which hands out the id, the
 * run and the timestamp, to every record a collector builds in one game.
 * A held record takes its id and timestamp when the moment it describes
 * happens, not when it is written.
 *
 * @param recordId  {@code {run_id}.{worker}-{lifetime}.{counter}}
 * @param runId     the collecting run
 * @param timestamp ISO 8601 UTC
 * @param gameId    {@code {run_id}.{worker}-{lifetime}.{game counter}}
 */
public record RecordHeader(String recordId, String runId, String timestamp, String gameId) {
}
