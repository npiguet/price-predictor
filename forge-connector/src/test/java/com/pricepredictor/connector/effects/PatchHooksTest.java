package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Finding, reading and installing the engine patch's hooks, which a stock
 * checkout lacks and a patched one must have every one of.
 */
@ExtendWith(ForgeExtension.class)
class PatchHooksTest {

    @Test
    void anAbsentHookReportsAbsentRatherThanThrowing() {
        assertFalse(PatchHooks.find(
                "forge.game.trigger.TriggerHandler", "noSuchHook").isPresent());
    }

    @Test
    void aMissingClassReportsAbsentRatherThanThrowing() {
        assertFalse(PatchHooks.find("forge.game.NotAClass", "anything").isPresent());
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

    // ── the listener shapes ─────────────────────────────────────────────

    /**
     * Each stand-in in {@link HookShapes} is what Forge declares, method by
     * method and position by position.
     *
     * <p>The one test a Forge-side signature drift fails: a swap or an
     * insertion would not fail a build, because neither side compiles against
     * the other, and the handlers would misread every call from then on. A
     * listener may instead match an older shape a worker still accepts, which
     * for the replacement listener means three arguments rather than five.
     * Skipped on a checkout without the setter, the same degradation every
     * hook lookup makes.
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("com.pricepredictor.connector.effects.HookShapes#declared")
    void eachStandInIsForgesOwnDeclaration(HookShapes.Declared standIn) {
        Optional<Method> setter = standIn.setter().lookUp();
        Assumptions.assumeTrue(setter.isPresent(), "the checkout lacks " + standIn.setter());
        Class<?> declared = setter.get().getParameterTypes()[0];

        List<Class<?>> accepted = new ArrayList<>(List.of(standIn.shape()));
        accepted.addAll(standIn.older());
        assertTrue(accepted.stream().anyMatch(shape -> mismatch(shape, declared).isEmpty()),
                mismatch(standIn.shape(), declared));
    }

    /**
     * Each listener hook names the callbacks its stand-in declares, and reads
     * none of them past the arguments the oldest accepted shape passes.
     *
     * <p>The stand-in is checked against Forge above; this checks the names
     * the handlers dispatch on against the stand-in, so a callback renamed on
     * one side and not the other fails here rather than going quiet. Runs on
     * a stock checkout too, since neither side needs the patch.
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("com.pricepredictor.connector.effects.HookShapes#declared")
    void eachListenerHookDeclaresItsStandInsCallbacks(HookShapes.Declared standIn) {
        List<Class<?>> accepted = new ArrayList<>(List.of(standIn.shape()));
        accepted.addAll(standIn.older());
        Set<String> declared = Arrays.stream(standIn.shape().getDeclaredMethods())
                .map(Method::getName).collect(Collectors.toSet());
        Set<String> named = standIn.setter().callbacks().stream()
                .map(PatchHooks.Callback::methodName).collect(Collectors.toSet());
        assertEquals(declared, named, standIn.setter().toString());

        for (PatchHooks.Callback callback : standIn.setter().callbacks()) {
            for (Class<?> shape : accepted) {
                int passed = HookShapes.method(shape, callback.methodName()).getParameterCount();
                assertTrue(callback.arity() <= passed,
                        callback + " needs " + callback.arity() + " arguments and "
                                + shape.getSimpleName() + " passes " + passed);
            }
        }
    }

    /** How {@code shape} differs from {@code declared}, or empty when it restates it. */
    private static String mismatch(Class<?> shape, Class<?> declared) {
        for (Method ours : shape.getDeclaredMethods()) {
            Method theirs = Arrays.stream(declared.getMethods())
                    .filter(method -> method.getName().equals(ours.getName()))
                    .findFirst().orElse(null);
            if (theirs == null) {
                return declared.getName() + " declares no " + ours.getName();
            }
            if (!Arrays.equals(ours.getParameterTypes(), theirs.getParameterTypes())) {
                return ours.getName() + " takes " + Arrays.toString(theirs.getParameterTypes())
                        + " in Forge and " + Arrays.toString(ours.getParameterTypes())
                        + " in " + shape.getSimpleName();
            }
        }
        return "";
    }
}
