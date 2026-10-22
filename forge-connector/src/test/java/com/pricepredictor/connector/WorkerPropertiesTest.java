package com.pricepredictor.connector;

import com.pricepredictor.connector.WorkerProperties.OnMalformed;
import org.junit.jupiter.api.Test;

import java.util.List;

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

    @Test
    void anUnsetOrBlankValueIsTheDefaultUnderEitherPolicy() {
        System.clearProperty(NAME);
        assertEquals(7, WorkerProperties.intProperty(NAME, 7, OnMalformed.EXIT));
        System.setProperty(NAME, " ");
        try {
            assertEquals(0.5, WorkerProperties.doubleProperty(NAME, 0.5, OnMalformed.EXIT));
        } finally {
            System.clearProperty(NAME);
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
