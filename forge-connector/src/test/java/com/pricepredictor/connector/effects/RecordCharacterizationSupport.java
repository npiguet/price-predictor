package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

/**
 * Every record kind's rendered line, built the way the collectors build it.
 *
 * <p>Each test drives the production collectors — the bracket on the game's
 * bus, the patched collectors' hook handlers called through a proxy as the
 * engine calls them, the fork collector's probe — over a fresh game, and
 * compares the shard's lines with an expected file. Only the envelope's
 * wall-clock timestamp is replaced.
 *
 * <p>Fork records (the damage-step probe and the interventional resolution)
 * sit in expected files of their own, so a change to what a fork writes
 * regenerates exactly those.
 *
 * <p>What every {@code RecordCharacterization*Test} shares: the sampling caps
 * and a fresh temporary directory per test, run only against the patched
 * checkout the expected records were captured on.
 */
abstract class RecordCharacterizationSupport {

    /** Every sampled channel at full rate, no forks unless a test adds them. */
    static final CollectionCaps CAPS =
            TestCaps.caps().everyDecision().interventions(0).probes(0).build();

    @TempDir
    Path tempDir;

    @BeforeEach
    void patchedCheckout() {
        Assumptions.assumeTrue(AttributionMode.detect() == AttributionMode.PATCHED,
                "the expected records were captured against the patched ../forge");
    }
}
