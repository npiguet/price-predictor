package com.pricepredictor.connector;

import org.junit.jupiter.api.extension.BeforeAllCallback;
import org.junit.jupiter.api.extension.ExtensionContext;

/**
 * JUnit 5 extension that initializes the Forge environment exactly once
 * per test suite run, regardless of how many test classes use it.
 *
 * <p>{@link ForgeEnvironmentInitializer#initialize()} is itself synchronized
 * and marks itself done only once it succeeds, so a failed bootstrap is
 * retried — and its real cause reported — by every class that needs it.
 */
public class ForgeExtension implements BeforeAllCallback {

    @Override
    public void beforeAll(ExtensionContext context) {
        ForgeEnvironmentInitializer.initialize();
    }
}
