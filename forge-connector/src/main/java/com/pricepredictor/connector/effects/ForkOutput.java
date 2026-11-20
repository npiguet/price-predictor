package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.KeptRecords;

import java.util.function.Consumer;

/**
 * Where a fork's records go and what they carry: what its recording is built
 * to write through.
 *
 * @param flags the flags every record of the fork carries
 * @param held  where the fork's finished lines go until the fork is kept
 * @param kept  the records the fork writes whatever its samplers draw
 *              (FR-010c)
 */
record ForkOutput(Consumer<EffectRecord> flags, Consumer<String> held, KeptRecords kept) {
}
