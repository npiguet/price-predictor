package com.pricepredictor.connector.effects;

/**
 * A finished record line and its {@link RecordFacts}: how a line travels to
 * the shard, a fork's held lines included, so the cap reads the facts its
 * maker held rather than the rendered text.
 *
 * @param json  the rendered record
 * @param facts what the cap reads of it
 */
record ShardLine(String json, RecordFacts facts) {
}
