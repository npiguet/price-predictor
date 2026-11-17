package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Forge hook {@link CopiedFaces} calls is probed once, before a batch
 * starts: a Forge without it ends the run naming the hook, rather than
 * failing each script on its own.
 */
class CopiedFacesHookTest {

    @Test
    void theSiblingForgeHasTheHook() {
        assertDoesNotThrow(() -> CopiedFaces.requireHook());
    }

    @Test
    void aMissingHookIsNamed() {
        assertNull(CopiedFaces.supplyMethod(Object.class));

        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> CopiedFaces.requireHook(CopiedFaces.supplyMethod(Object.class)));

        assertTrue(e.getMessage().contains("supplyPlaceholderFaces"), e.getMessage());
    }
}
