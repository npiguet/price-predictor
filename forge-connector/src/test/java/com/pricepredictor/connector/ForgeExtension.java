package com.pricepredictor.connector;

import org.junit.jupiter.api.extension.BeforeAllCallback;
import org.junit.jupiter.api.extension.ExtensionContext;

/**
 * JUnit 5 extension that initializes the Forge environment exactly once
 * per test suite run, regardless of how many test classes use it.
 */
public class ForgeExtension implements BeforeAllCallback, ExtensionContext.Store.CloseableResource {

    private static volatile boolean started = false;

    @Override
    public void beforeAll(ExtensionContext context) {
        if (!started) {
            synchronized (ForgeExtension.class) {
                if (!started) {
                    started = true;
                    ForgeEnvironmentInitializer.initialize();
                    context.getRoot().getStore(ExtensionContext.Namespace.GLOBAL)
                            .put(ForgeExtension.class.getName(), this);
                }
            }
        }
    }

    @Override
    public void close() {
        // nothing to tear down
    }
}
