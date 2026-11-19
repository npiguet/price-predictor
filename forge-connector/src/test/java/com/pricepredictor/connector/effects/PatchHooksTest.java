package com.pricepredictor.connector.effects;

import com.google.common.collect.Multimap;
import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityKey;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerHandler;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
        assertNull(PatchHooks.readStatic(
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
     * The sibling {@code ../forge} checkout this suite runs against is the
     * patched one ({@code effect-record-hooks}), with every hook.
     *
     * <p>Unconditional: every test that needs a hook skips on a checkout
     * without it (the degradation a worker makes), so on a stock or stale
     * checkout about sixty tests and every golden comparison would skip and
     * the suite would still read green. This one fails instead, naming what
     * is missing.
     */
    @Test
    void theSiblingForgeCheckoutIsThePatchedOne() {
        assertEquals(AttributionMode.PATCHED, AttributionMode.detect(),
                "../forge lacks the effect-record hooks: " + PatchHooks.report());
        assertEquals(List.of(), PatchHooks.gaps(), PatchHooks.report());
    }

    // ── what the report names ───────────────────────────────────────────

    /** The trigger listener of a jar older than its run callback: it never confirms a firing. */
    interface TriggerFireWithoutRuns {
        void onConditionEvaluated(Trigger trigger, Map<AbilityKey, Object> runParams, boolean fired);
    }

    /**
     * A stale installed Forge: setters whose listeners lack what the
     * collectors read, under the names the patch gives them.
     */
    static final class StaleForge {
        public static void setEffectRecordTriggerListener(TriggerFireWithoutRuns listener) {
            // Only its parameter type is read.
        }

        public static void setEffectRecordListener(HookShapes.FiveArgumentReplacement listener) {
            // Only its parameter type is read.
        }

        private static boolean noteEffectRecordConfirm(SpellAbility sa, boolean confirmed) {
            if (confirmed) {
                throw new IllegalArgumentException("what the engine's own call throws");
            }
            return confirmed;
        }
    }

    /** {@code hook}, looked up on {@code owner} instead of its Forge class. */
    private static PatchHooks.Hook on(Class<?> owner, PatchHooks.Hook hook) {
        return new PatchHooks.Hook(owner.getName(), hook.method(), hook.unlocks(), hook.callbacks(), hook.reach());
    }

    /**
     * The private confirm hook is found among its owner's own methods and
     * called through, its answer returned and what it throws rethrown.
     */
    @Test
    void thePrivateConfirmHookIsFoundAndCalledThrough() {
        PatchHooks.Hook note = on(StaleForge.class, PatchHooks.NOTE_CONFIRM);

        assertTrue(note.present());
        assertEquals(false, note.call(null, false));
        assertThrows(IllegalArgumentException.class, () -> note.call(null, true));
    }

    /**
     * A Forge without the private confirm hook has it named by the report,
     * and a fork's call through it fails naming it rather than with a
     * reflective error from inside the engine.
     */
    @Test
    void aForgeWithoutThePrivateConfirmHookIsNamed() {
        PatchHooks.Hook note = on(TriggerFireWithoutRuns.class, PatchHooks.NOTE_CONFIRM);

        String report = PatchHooks.report(List.of(note));
        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> note.call(null, true));

        assertTrue(report.contains("noteEffectRecordConfirm"), report);
        assertTrue(failure.getMessage().contains("noteEffectRecordConfirm"), failure.getMessage());
    }

    /**
     * Each current stand-in passes every callback its hook dispatches, with
     * every argument the collectors read: the counts in {@link PatchHooks.Callback}
     * agree with the shapes {@link #eachStandInIsForgesOwnDeclaration} checks
     * against Forge.
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("com.pricepredictor.connector.effects.HookShapes#declared")
    void eachStandInLacksNoCallbackItsHookReads(HookShapes.Declared standIn) {
        assertEquals(List.of(),
                PatchHooks.callbacksLacking(standIn.shape(), standIn.setter().callbacks()));
    }

    /**
     * A listener whose interface lacks a callback is named, with the
     * callback: its setter is present, so a presence check alone read it
     * whole while the channel the callback feeds stayed empty.
     */
    @Test
    void aListenerLackingACallbackIsReportedByName() {
        String report = PatchHooks.report(List.of(on(StaleForge.class, PatchHooks.TRIGGER_LISTENER)));

        assertTrue(report.contains("setEffectRecordTriggerListener"), report);
        assertTrue(report.contains("onTriggerRun is not declared"), report);
    }

    /**
     * A listener whose callback passes fewer arguments than the collectors
     * read is named with both counts, and each callback it lacks by name.
     */
    @Test
    void aListenerPassingFewerArgumentsThanTheCollectorsReadIsReported() {
        String report = PatchHooks.report(List.of(on(StaleForge.class, PatchHooks.REPLACEMENT_LISTENER)));

        assertTrue(report.contains("onReplacement passes 5 of the 6 arguments the collectors read"), report);
        assertTrue(report.contains("onBufferedAbilityStarting is not declared"), report);
        assertTrue(report.contains("onBufferedAbilityFinished is not declared"), report);
    }

    /**
     * The sub-ability pointer is put back once a resolution is over: a
     * Lightning Bolt resolves through the hooks, its damage read the pointer
     * while it resolved (its event names the root), and the pointer reads
     * null afterwards.
     */
    @Test
    void theSubAbilityHookReadsNullOnceAResolutionIsOver(@org.junit.jupiter.api.io.TempDir
                                                        java.nio.file.Path dir) throws Throwable {
        forge.game.Game game = TestBoard.startedTwoPlayerGame(forge.game.phase.PhaseType.MAIN1);
        forge.game.player.Player you = game.getPlayers().get(0);
        forge.game.player.Player opponent = game.getPlayers().get(1);

        List<String> records = ShardRun.recordsWithClauseHook(dir, game, "", () -> {
            TestBoard.cast(game, TestBoard.inHand(game, you, "Lightning Bolt"), you, opponent);
            game.getStack().resolveStack();
            // The next step closes the held half.
            TestBoard.nextStep(game);
        });

        String half = RecordLines.only(records, "resolution", "cardsfolder/l/lightning_bolt.txt");
        assertEquals(List.of("root"), RecordLines.events(half).stream()
                .filter(event -> "damage_dealt".equals(event.get("type")))
                .map(event -> event.get("attributed_to")).toList(), half);
        assertNull(PatchHooks.currentSubAbility());
    }

    /**
     * The pending delayed triggers read back through the accessor, one list
     * per kind, each reachable by name off the record the accessor returns.
     *
     * <p>Read the way a collector reads it, reflectively: the record's own
     * accessors are the contract as much as the accessor's name. Skipped on a
     * checkout without the accessor.
     */
    @Test
    void aRegisteredDelayedTriggerReadsBackThroughTheAccessor() {
        Assumptions.assumeTrue(PatchHooks.PENDING_DELAYED_TRIGGERS.present(),
                "the checkout lacks " + PatchHooks.PENDING_DELAYED_TRIGGERS);
        Card host = TestBoard.cardInAnotherGame("Paralyze");
        Trigger trigger = host.getCurrentState().getTriggers().get(0);
        TriggerHandler handler = host.getGame().getTriggerHandler();
        handler.registerThisTurnDelayedTrigger(trigger);

        Object pending = PatchHooks.PENDING_DELAYED_TRIGGERS.read(handler);

        assertEquals(List.of(trigger), PatchHooks.read(pending, "delayed"));
        assertEquals(List.of(trigger), PatchHooks.read(pending, "thisTurn"));
        assertTrue(PatchHooks.read(pending, "playerDefined") instanceof Multimap<?, ?> byPlayer
                && byPlayer.isEmpty(), String.valueOf(pending));
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
     * for the replacement listener means three or five arguments rather than
     * six, and for the resolution listener two rather than three.
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
                if (shape != standIn.shape() && Arrays.stream(shape.getMethods())
                        .noneMatch(m -> m.getName().equals(callback.methodName()))) {
                    // An older listener that predates the callback is never
                    // called with it: onReplacing came after onReplacement.
                    continue;
                }
                int passed = HookShapes.method(shape, callback.methodName()).getParameterCount();
                assertTrue(callback.arity() <= passed,
                        callback + " needs " + callback.arity() + " arguments and "
                                + shape.getSimpleName() + " passes " + passed);
            }
        }
    }

    /**
     * How {@code shape} differs from {@code declared}, or empty when it
     * restates it: each of its methods is one Forge declares with exactly
     * those parameters. By signature rather than by name, because a listener
     * Forge widened declares the old call and the new one under one name.
     */
    private static String mismatch(Class<?> shape, Class<?> declared) {
        for (Method ours : shape.getDeclaredMethods()) {
            try {
                declared.getMethod(ours.getName(), ours.getParameterTypes());
            } catch (NoSuchMethodException absent) {
                List<String> theirs = Arrays.stream(declared.getMethods())
                        .filter(method -> method.getName().equals(ours.getName()))
                        .map(method -> Arrays.toString(method.getParameterTypes()))
                        .toList();
                return theirs.isEmpty()
                        ? declared.getName() + " declares no " + ours.getName()
                        : ours.getName() + " takes " + String.join(" or ", theirs) + " in Forge and "
                                + Arrays.toString(ours.getParameterTypes()) + " in " + shape.getSimpleName();
            }
        }
        return "";
    }

    /**
     * Every method Forge's listener declares is restated by the stand-in or
     * one of its older shapes, and so dispatched by its hook to a collector
     * ({@link #eachListenerHookDeclaresItsStandInsCallbacks}).
     *
     * <p>The other direction from {@link #eachStandInIsForgesOwnDeclaration}:
     * a notification Forge adds reaches the connector's proxy and would be
     * answered with nothing, silently. No allow-list: such a method fails
     * here until a collector reads it. Skipped on a checkout without the
     * setter, which {@link #theSiblingForgeCheckoutIsThePatchedOne} fails.
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("com.pricepredictor.connector.effects.HookShapes#declared")
    void everyMethodForgesListenerDeclaresIsRestatedOrListed(HookShapes.Declared standIn) {
        Optional<Method> setter = standIn.setter().lookUp();
        Assumptions.assumeTrue(setter.isPresent(), "the checkout lacks " + standIn.setter());
        Class<?> declared = setter.get().getParameterTypes()[0];
        List<Class<?>> accepted = new ArrayList<>(List.of(standIn.shape()));
        accepted.addAll(standIn.older());

        for (Method theirs : declared.getMethods()) {
            boolean restated = accepted.stream().anyMatch(shape -> Arrays.stream(shape.getMethods()).anyMatch(
                    ours -> ours.getName().equals(theirs.getName())
                            && Arrays.equals(ours.getParameterTypes(), theirs.getParameterTypes())));
            assertTrue(restated,
                    declared.getName() + " declares " + theirs.getName()
                            + Arrays.toString(theirs.getParameterTypes())
                            + ", which no collector reads: " + accepted + " restates none of it");
        }
    }

    /**
     * A listener whose body throws is counted, every call of it, against its
     * hook (C11): Forge contains the failure and keeps calling the listener,
     * so each failing call loses its records and only the count says how
     * many. The failure still reaches Forge, which contains it.
     */
    @Test
    void eachCallOfAListenerThatThrowsIsCountedAgainstItsHook() throws Throwable {
        PatchHooks.Hook hook = PatchHooks.PRIORITY_SETTLED_LISTENER;
        long before = HookFailures.counts().getOrDefault(hook.toString(), 0L);
        PatchHooks.Listener failing = hook.listener((callback, args) -> {
            throw new IllegalStateException("a card this listener cannot read");
        });
        Method settled = HookShapes.method(HookShapes.PrioritySettled.class, "onPrioritySettled");

        for (int call = 0; call < 3; call++) {
            assertThrows(IllegalStateException.class,
                    () -> failing.invoke(null, settled, new Object[]{null, null}));
        }

        assertEquals(before + 3, HookFailures.counts().get(hook.toString()));
    }

    /** A collection holds the proxy: Forge may keep its listener in one, and a null hash would throw. */
    @Test
    void theListenerProxyAnswersTheObjectMethods() {
        PatchHooks.Listener listener = PatchHooks.PRIORITY_SETTLED_LISTENER.listener(
                (callback, args) -> { });
        Object proxy = java.lang.reflect.Proxy.newProxyInstance(
                HookShapes.PrioritySettled.class.getClassLoader(),
                new Class<?>[]{HookShapes.PrioritySettled.class}, listener);

        Set<Object> held = new java.util.HashSet<>();
        held.add(proxy);

        assertTrue(held.contains(proxy));
        assertEquals(proxy, proxy);
        assertFalse(proxy.equals(new Object()));
        assertTrue(proxy.toString().contains("listener"), proxy.toString());
    }
}
