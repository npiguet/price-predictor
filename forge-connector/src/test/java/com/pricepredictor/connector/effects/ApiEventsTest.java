package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.StdStreams;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Set;
import java.util.TreeSet;
import java.util.function.BiConsumer;

import static com.pricepredictor.connector.effects.ApiClauseObservations.observe;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What {@link ApiEvents} is as a whole: every rule keyed by a real
 * {@code ApiType} member, an API with no rule silent, and the once-latched
 * failure reports. What each family of rules says about a clause is in the
 * {@code ApiEvents<Family>Test} classes.
 */
@ExtendWith(ForgeExtension.class)
class ApiEventsTest {

    /**
     * The failure latches are static, shared with production: without a
     * reset here, whichever test trips one first leaves every test after it --
     * including ones that never touch that latch, and any later test class in
     * the same JVM -- unable to see its own "did this print" outcome. The
     * emitter and the memo each have one, so both are reset here.
     */
    @BeforeEach
    void resetFailureLatches() {
        TestLatches.reset(ApiEvents.class, "EMITTER_FAILURE");
        TestLatches.reset(ApiEvents.class, "MEMO_FAILURE");
    }

    @Test
    void anApiWithNoEmitterIsSilent() {
        SpellAbility sa = TestCards.scriptedAbility("Lightning Bolt", "DealDamage");

        assertNull(TestCards.onlyEvent(observe(sa)),
                "damage already arrives on the bus; a second event would double-count it");
    }

    /**
     * A rule keyed by a name that is no {@code ApiType} member -- an effect
     * class's stem, say, or a member Forge renamed -- matches nothing and
     * fails silently, since {@code ruleFor} looks rules up by {@code name()}.
     */
    @Test
    void everyRuleIsKeyedByAnApiTypeMember() {
        Set<String> members = new TreeSet<>();
        for (ApiType api : ApiType.values()) {
            members.add(api.name());
        }
        Set<String> unknown = new TreeSet<>(ApiEvents.apiNames());
        unknown.removeAll(members);

        assertFalse(ApiEvents.apiNames().isEmpty());
        assertEquals(Set.of(), unknown, "RULES keys that name no ApiType member");
    }

    // ── the once-latched failure reports ─────────────────────────────────

    /** The two halves of a rule that can throw, each reported on its own latch. */
    enum Report {
        EMITTER("the emitter for", ApiEvents::reportEmitterFailure),
        MEMO("the memo for", ApiEvents::reportMemoFailure);

        /** How the report names the half that failed. */
        final String names;
        final BiConsumer<EventType, RuntimeException> report;

        Report(String names, BiConsumer<EventType, RuntimeException> report) {
            this.names = names;
            this.report = report;
        }

        Report other() {
            return this == EMITTER ? MEMO : EMITTER;
        }
    }

    /**
     * Mirrors {@code OnceOnlyWarningTest.testRaisedThreeTimesPrintsOneLine}
     * in the forge repo:
     * call the report directly with a synthetic exception rather than contrive
     * a rule that reliably throws, and assert both halves of the promise --
     * one line naming the half that failed, the event type and the exception,
     * and silence on every call after the first, which is the half inspection
     * of the {@code compareAndSet} alone cannot establish.
     */
    @ParameterizedTest
    @EnumSource(Report.class)
    void aFailureIsReportedExactlyOnceOnStderr(Report half) throws Throwable {
        String output = StdStreams.captureStderr(() -> {
            half.report.accept(EventType.TURN_ADDED, new RuntimeException("boom"));
            half.report.accept(EventType.TURN_ADDED, new RuntimeException("boom"));
            half.report.accept(EventType.X_CHANGED, new RuntimeException("boom again"));
        });

        long lineCount = output.isBlank() ? 0 : output.strip().lines().count();
        assertEquals(1, lineCount, "exactly one report despite three throws: " + output);
        assertTrue(output.contains(half.names), output);
        assertTrue(output.contains("turn_added"), output);
        assertTrue(output.contains("boom"), output);
    }

    /**
     * Each half latches its own report: a memo failure on one clause must not
     * silence a later, unrelated emitter failure on another for the rest of the
     * JVM's run, nor the reverse.
     *
     * <p>Both orders, because neither is the natural one: a report that checked
     * the other half's latch but set its own would pass one order and fail the
     * other.
     */
    @ParameterizedTest
    @EnumSource(Report.class)
    void oneHalfsFailureDoesNotSilenceALaterFailureOfTheOther(Report first) throws Throwable {
        String output = StdStreams.captureStderr(() -> {
            first.report.accept(EventType.TURN_ADDED, new RuntimeException("first boom"));
            first.other().report.accept(EventType.X_CHANGED, new RuntimeException("second boom"));
        });

        long lineCount = output.isBlank() ? 0 : output.strip().lines().count();
        assertEquals(2, lineCount, "one report from each latch, independently: " + output);
        assertTrue(output.contains("first boom"), output);
        assertTrue(output.contains("second boom"), output);
    }

    /**
     * The wiring, not just the message: {@link ApiEvents#observe} itself,
     * when a real rule's memo throws, must report through {@link
     * ApiEvents#reportMemoFailure} rather than {@link
     * ApiEvents#reportEmitterFailure}, so the line names the memo -- the
     * function to look at -- rather than the emitter. A test that only calls
     * the report directly cannot see this, since it never exercises
     * {@code observe}'s own catch block. Same throwing-memo shape {@code
     * ClauseContractTest.aThrowingMemoDoesNotDesyncTheSiblingsPop} uses.
     */
    @Test
    void observeReportsAThrowingMemoAsAMemoNotAnEmitter() throws Throwable {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ SkipTurn | ValidTgts$ Player | NumTurns$ 1",
                TestCards.build("Runeclaw Bear"));
        ApiEvents.Rule<Object> throwingMemo = ApiEvents.Rule.one(
                EventType.TURN_SKIPPED,
                (s, host) -> {
                    throw new IllegalStateException("boom (memo)");
                },
                (s, host, memo) -> new EffectEvent(EventType.TURN_SKIPPED));

        String output = StdStreams.captureStderr(() -> assertNull(
                ApiEvents.observe(sa, api -> throwingMemo).memo(),
                "a failed memo still reads back as null"));

        assertTrue(output.contains("the memo for"), output);
        assertFalse(output.contains("the emitter for"), output);
    }
}
