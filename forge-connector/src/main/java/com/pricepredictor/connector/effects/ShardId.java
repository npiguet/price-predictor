package com.pricepredictor.connector.effects;

/**
 * Whose shard it is: the run, the worker index and the JVM lifetime, which
 * name the shard file and the middle of every id it issues.
 *
 * @param runId    the collection run's id
 * @param worker   the worker's index
 * @param lifetime the token of this JVM's lifetime ({@link RecordShardWriter#mintLifetime})
 */
public record ShardId(String runId, int worker, String lifetime) {

    /** {@code {worker}-{lifetime}}: the token that names this worker lifetime. */
    String token() {
        return worker + "-" + lifetime;
    }
}
