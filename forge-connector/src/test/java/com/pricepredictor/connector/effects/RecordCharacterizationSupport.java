package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

/**
 * What every {@code RecordCharacterization*Test} shares: the sampling caps
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
