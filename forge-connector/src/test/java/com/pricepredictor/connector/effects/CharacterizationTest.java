package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import org.junit.jupiter.api.extension.ExtendWith;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * A test class that pins bytes against {@code src/test/resources/characterization}:
 * Forge started, and Forge's JVM-wide ids pinned for each test
 * ({@link Characterization.PinnedForgeIds}), so the pinned bytes do not depend
 * on which tests ran first.
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith({ForgeExtension.class, Characterization.PinnedForgeIds.class})
@interface CharacterizationTest {
}
