package com.pricepredictor.connector;

import com.pricepredictor.connector.WorkerProperties.OnMalformed;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The parts of {@link WorkerProperties} that do not end the JVM. */
class WorkerPropertiesTest {

    private static final String NAME = "worker.properties.test";

    @Test
    void csvSplitsTrimsAndDropsBlankItems() {
        assertEquals(List.of("a", "b"), WorkerProperties.csv(" a , b ,, "));
        assertTrue(WorkerProperties.csv(null).isEmpty());
        assertTrue(WorkerProperties.csv("  ").isEmpty());
    }

    /**
     * Unset, empty and blank all read as the default, and under EXIT as much
     * as under WARN_AND_DEFAULT: a blank value is not a malformed one, so
     * neither policy has anything to apply to it.
     */
    @ParameterizedTest(name = "{0}")
    @EnumSource(OnMalformed.class)
    void anUnsetOrBlankValueIsTheDefaultUnderEitherPolicy(OnMalformed policy) {
        for (String value : new String[]{null, "", " "}) {
            set(value);
            try {
                assertEquals(7, WorkerProperties.intProperty(NAME, 7, policy),
                        "'" + value + "'");
                assertEquals(0.5, WorkerProperties.doubleProperty(NAME, 0.5, policy),
                        "'" + value + "'");
            } finally {
                System.clearProperty(NAME);
            }
        }
    }

    /** An optional value with no default: unset, empty and blank are all absent. */
    @ParameterizedTest(name = "''{0}''")
    @NullSource
    @ValueSource(strings = {"", " ", "\t"})
    void anUnsetOrBlankOptionalIsEmpty(String value) {
        set(value);
        try {
            assertEquals(Optional.empty(), WorkerProperties.optional(NAME));
            assertEquals(Optional.empty(), WorkerProperties.optionalPath(NAME));
        } finally {
            System.clearProperty(NAME);
        }
    }

    @Test
    void anOptionalValueIsReadTrimmed() {
        System.setProperty(NAME, " output/sealed/pools.txt ");
        try {
            assertEquals(Optional.of("output/sealed/pools.txt"), WorkerProperties.optional(NAME));
            assertEquals(Optional.of(Path.of("output/sealed/pools.txt")),
                    WorkerProperties.optionalPath(NAME));
        } finally {
            System.clearProperty(NAME);
        }
    }

    /** {@code value} as the property, or no property at all for null. */
    private static void set(String value) {
        if (value == null) {
            System.clearProperty(NAME);
        } else {
            System.setProperty(NAME, value);
        }
    }

    @Test
    void aWellFormedValueIsReadTrimmed() {
        System.setProperty(NAME, " 12 ");
        try {
            assertEquals(12, WorkerProperties.intProperty(NAME, 7, OnMalformed.WARN_AND_DEFAULT));
        } finally {
            System.clearProperty(NAME);
        }
    }

    @Test
    void aValueFailingItsConstraintKeepsTheDefaultWhenWarning() {
        System.setProperty(NAME, "-3");
        try {
            assertEquals(7, WorkerProperties.intProperty(
                    NAME, 7, OnMalformed.WARN_AND_DEFAULT, v -> v >= 0, "must be >= 0"));
        } finally {
            System.clearProperty(NAME);
        }
    }
}
