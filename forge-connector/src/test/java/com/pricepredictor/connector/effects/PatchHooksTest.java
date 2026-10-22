package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * Finding, reading and installing the engine patch's hooks, which a stock
 * checkout lacks and a patched one must have every one of.
 */
@ExtendWith(ForgeExtension.class)
class PatchHooksTest {

    @Test
    void anAbsentHookReportsAbsentRatherThanThrowing() {
        assertFalse(PatchHooks.find(
                "forge.game.trigger.TriggerHandler", "noSuchHook").present());
    }

    @Test
    void aMissingClassReportsAbsentRatherThanThrowing() {
        assertFalse(PatchHooks.find("forge.game.NotAClass", "anything").present());
    }

    @Test
    void readingAnAbsentHookYieldsNull() {
        assertEquals(null, PatchHooks.readStatic(
                "forge.game.trigger.TriggerHandler", "noSuchHook"));
    }

    @Test
    void installingAnAbsentHookIsANoOpRatherThanAFailure() {
        assertFalse(PatchHooks.install(
                "forge.game.trigger.TriggerHandler", "setNoSuchListener",
                (proxy, method, args) -> null));
    }

    @Test
    void uninstallingAnAbsentHookDoesNotThrow() {
        PatchHooks.uninstall(
                "forge.game.trigger.TriggerHandler", "setNoSuchListener");
    }

    /**
     * A patched checkout has every required hook, not just the one the mode
     * probes: an engine refactor that drops or renames a setter would
     * otherwise leave its channel empty while the count of installed hooks
     * still agrees with itself.
     */
    @Test
    void aPatchedCheckoutHasEveryRequiredHook() {
        Assumptions.assumeTrue(AttributionMode.detect() == AttributionMode.PATCHED,
                "a stock checkout is missing every hook by definition");
        assertEquals(List.of(), PatchHooks.missing(), PatchHooks.report());
    }

    /**
     * The read-only sub-ability hook answers null outside a resolution.
     *
     * <p>True on a stock checkout because the method does not exist, and true
     * on a patched one because no sub-ability is resolving on this thread —
     * the test asserts the caller-visible behaviour that holds either way.
     */
    @Test
    void theSubAbilityHookReadsNullOutsideAResolution() {
        assertEquals(null, PatchHooks.currentSubAbility());
    }
}
