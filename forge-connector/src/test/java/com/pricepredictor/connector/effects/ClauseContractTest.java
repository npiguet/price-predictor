package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The clause hook's argument order.
 *
 * <p>The connector never links against {@code AbilityUtils.EffectRecordClauseListener}
 * — it is installed as a dynamic proxy over whatever the patched checkout
 * declares — so which argument is the clause and which is {@code threw} is a
 * runtime contract with no compiler behind it. A swap on either side would not
 * fail a build; it would fail a corpus, silently, forever: an installed
 * listener that reads the wrong position either throws on every call (Forge
 * contains that, logs one line per JVM, and never again) or, worse, misreads a
 * {@code Boolean} as a clause and simply produces no event, with nothing
 * anywhere to say so. These tests are what stands in for the compiler.
 *
 * <p>{@link ApiEventsTest} covers what {@link ApiEvents} says about a clause
 * once it is handed one. What is here is the wiring that hands it one at all:
 * {@link ClauseOutcomeRouter#clauseHandler()}, called the way the engine calls
 * it, with two real, distinct, non-null values so a position swap has
 * something to disagree with rather than two nulls agreeing by accident.
 */
@ExtendWith(ForgeExtension.class)
class ClauseContractTest {

    @TempDir
    Path tempDir;

    /**
     * Static, and shared with production: without a reset here, whichever
     * test trips {@code ApiEvents}' emitter-failure or memo-failure latch
     * first -- in this class or, within one JVM, {@code ApiEventsTest} --
     * leaves every test after it unable to see its own "did this print"
     * outcome. Two latches since final-fix-3.md item 4 (they no longer share
     * one), so both are reset here.
     */
    @BeforeEach
    void resetFailureLatches() {
        TestLatches.reset(ApiEvents.class, "EMITTER_FAILURE_REPORTED");
        TestLatches.reset(ApiEvents.class, "MEMO_FAILURE_REPORTED");
    }

    private static Method clauseMethod(String name) {
        return HookShapes.method(HookShapes.Clause.class, name);
    }

    // ── the argument order, pinned ──────────────────────────────────────

    /**
     * Two real, distinguishable arguments in, and each one read off its own
     * position: the clause names the event, {@code threw=false} lets it
     * through. A handler that read {@code args[1]} for the clause would find
     * a {@code Boolean} where it expected a {@code SpellAbility}, bail out on
     * the {@code instanceof} check, and this record would carry no event at
     * all.
     */
    @Test
    void theHandlerReadsEachArgumentOffItsOwnPosition() throws Throwable {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        String effectHalf = resolveClause(sa, false).get(1);

        assertTrue(effectHalf.contains("\"type\":\"turn_added\""), effectHalf);
        assertTrue(effectHalf.contains("\"count\":1"), effectHalf);
    }

    /**
     * {@code threw=true} is the other half of the same position: a clause
     * that threw writes no event, so this is what tells the two Boolean
     * values apart rather than one of them agreeing with a default.
     */
    @Test
    void aClauseThatThrewWritesNoEvent() throws Throwable {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        String effectHalf = resolveClause(sa, true).get(1);

        assertTrue(effectHalf.contains("\"moment\":\"resolution\""), effectHalf);
        assertFalse(effectHalf.contains("\"type\":\"turn_added\""), effectHalf);
    }

    /**
     * The direct demonstration: called with the two arguments where the
     * <em>other</em> method's shape would put them, {@code onClauseResolved}
     * finds a {@code Boolean} at {@code args[0]} and writes nothing, rather
     * than misreading it as a clause.
     */
    @Test
    void aSwappedCallWritesNoEvent() throws Throwable {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");
        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collector) -> {
            InvocationHandler handler = collector.clauses().clauseHandler();
            bracket.beginBracket(sa);
            handler.invoke(null, clauseMethod("onClauseResolved"),
                    new Object[]{Boolean.FALSE, sa});
            bracket.endBracket(sa.getId(), false);
            bracket.flushHeldResolution();
        });

        String effectHalf = records.get(1);
        assertFalse(effectHalf.contains("\"type\":\"turn_added\""), effectHalf);
    }

    // ── the mechanism: nested memos do not cross ─────────────────────────

    /**
     * Every rule in {@code ApiEvents} passed a null {@code Memo} until the
     * critical fix that gave {@code AddTurn}/{@code SkipTurn}/{@code
     * SkipPhase} a real one (the targeted-player list, captured before
     * resolution): before that, the {@code ThreadLocal<Deque<Object>>} this
     * handler keeps was only ever exercised on its always-empty path, and a
     * LIFO-pairing bug in it would first surface in Task 7 — someone else's
     * task, building the first production rule that genuinely needs one.
     * Pinned here instead, on the two real rules that supply one now.
     *
     * <p>Drives the documented firing order for a two-clause ability --
     * {@code resolving(root)}, {@code resolving(sub)}, {@code
     * resolved(sub, false)}, {@code resolved(root, false)} -- with the outer
     * clause ({@code AddTurn}) targeting one player and the inner
     * ({@code SkipTurn}) targeting a different one, so a crossed pop would
     * hand one clause's event the other's memo and produce an assertion
     * failure rather than an accidental pass.
     */
    @Test
    void nestedClausesDoNotCrossTheirMemos() throws Throwable {
        SpellAbility outer = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1",
                TestCards.build("Grizzly Bears"));
        SpellAbility inner = AbilityFactory.getAbility(
                "SP$ SkipTurn | ValidTgts$ Player | NumTurns$ 1",
                TestCards.build("Runeclaw Bear"));
        Player outerTarget = new Player("outer-target", TestCards.game(), 92001);
        Player innerTarget = new Player("inner-target", TestCards.game(), 92002);
        outer.resetTargets();
        outer.getTargets().add(outerTarget);
        inner.resetTargets();
        inner.getTargets().add(innerTarget);

        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collector) -> {
            InvocationHandler handler = collector.clauses().clauseHandler();
            bracket.beginBracket(outer);
            handler.invoke(null, clauseMethod("onClauseResolving"),
                    new Object[]{outer});
            handler.invoke(null, clauseMethod("onClauseResolving"),
                    new Object[]{inner});
            handler.invoke(null, clauseMethod("onClauseResolved"),
                    new Object[]{inner, false});
            handler.invoke(null, clauseMethod("onClauseResolved"),
                    new Object[]{outer, false});
            bracket.endBracket(outer.getId(), false);
            bracket.flushHeldResolution();
        });

        String effectHalf = records.get(1);
        assertTrue(effectHalf.contains("\"type\":\"turn_added\",\"subjects\":[\""
                + SnapshotBuilder.playerId(outerTarget) + "\"]"), effectHalf);
        assertTrue(effectHalf.contains("\"type\":\"turn_skipped\",\"subjects\":[\""
                + SnapshotBuilder.playerId(innerTarget) + "\"]"), effectHalf);
    }

    // ── F2: another game's clause must not reach this game's bracket ────

    /**
     * F2: a clause resolving on a DIFFERENT {@code Game} -- exactly the shape
     * {@code ForkCollector.forceResolution} produces, since {@code
     * GameSimulator.resolveStack} runs the real resolution pipeline against a
     * {@code GameCopier} clone under {@code AIOption.USE_FULL_SIMULATION}
     * regardless of how this game's own lobby seats were registered (ruling
     * R31) -- must not reach the live game's bracket. {@code GameCopier}
     * builds every copied {@code Card} against a new {@code Game} object, so
     * a fork's clause's own host card already carries a different
     * {@code Game} than this collector's.
     *
     * <p>Companion to {@code OutcomeHandlerTest
     * .anOutcomeFromAnotherGameDoesNotReachThisGamesBracket}, which covers
     * the same fix on the outcome hook's half.
     */
    @Test
    void aClauseFromAnotherGameDoesNotReachThisGamesBracket() throws Throwable {
        Card forkHost = TestBoard.cardInAnotherGame("Grizzly Bears");
        SpellAbility forkAbility = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1", forkHost);
        Player forkTarget = new Player("fork-target", forkHost.getGame(), 92201);
        forkAbility.resetTargets();
        forkAbility.getTargets().add(forkTarget);

        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collector) -> {
            InvocationHandler handler = collector.clauses().clauseHandler();
            bracket.beginBracket(forkAbility);
            handler.invoke(null, clauseMethod("onClauseResolving"),
                    new Object[]{forkAbility});
            handler.invoke(null, clauseMethod("onClauseResolved"),
                    new Object[]{forkAbility, false});
            bracket.endBracket(forkAbility.getId(), false);
            bracket.flushHeldResolution();
        });

        String shard = String.join("\n", records);
        assertFalse(shard.contains("turn_added"),
                "a fork's clause must not land in the live game's record: " + shard);
    }

    // ── F3: a throwing memo must not desync a sibling's pop ─────────────

    /**
     * F3: a clause whose own {@code Memo} throws must not desynchronize the
     * deque for the clauses around it.
     *
     * <p>Same shape as {@link #nestedClausesDoNotCrossTheirMemos}, except the
     * inner clause's memo throws -- forced by handing the clause handler a
     * rule lookup that answers a throwing rule for {@code SkipTurn} and the
     * real one for everything else: no real script is known to make
     * {@code AbilityUtils.getDefinedPlayers}/{@code getDefinedCards} throw --
     * every unrecognized input degrades to an empty list instead -- so there
     * is no real card this could be pinned to without hand-feeding the code a
     * state the engine never produces, exactly the trap this branch's own
     * test bar warns against.
     *
     * <p>The {@code onClauseResolving} calls go through {@link
     * #resolvingContained}, the way {@code
     * AbilityUtils.notifyClauseResolving} calls them in production: that
     * containment is what lets resolution continue past the throw at all,
     * and before {@link ApiEvents#before}'s own fix, it is exactly what
     * swallows the throw <em>before</em> {@code clauseMemos.push(...)} runs
     * -- the missing push this test exists to catch. Before the fix: the
     * inner clause's {@code onClauseResolving} throws out of {@link
     * ApiEvents#before} before its push runs, so the inner clause's
     * own {@code onClauseResolved} pops the OUTER clause's memo instead of
     * its own missing one, and the outer's own pop then reads an empty deque
     * and gets {@code null} -- so this asserts the OUTER clause's event, the
     * sibling the inner's failure has nothing to do with, still carries its
     * own subject.
     */
    @Test
    void aThrowingMemoDoesNotDesyncTheSiblingsPop() throws Throwable {
        SpellAbility outer = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1",
                TestCards.build("Grizzly Bears"));
        SpellAbility inner = AbilityFactory.getAbility(
                "SP$ SkipTurn | ValidTgts$ Player | NumTurns$ 1",
                TestCards.build("Runeclaw Bear"));
        Player outerTarget = new Player("outer-target", TestCards.game(), 92101);
        Player innerTarget = new Player("inner-target", TestCards.game(), 92102);
        outer.resetTargets();
        outer.getTargets().add(outerTarget);
        inner.resetTargets();
        inner.getTargets().add(innerTarget);

        ApiEvents.Rule throwingMemo = new ApiEvents.Rule(
                EffectEvent.TURN_SKIPPED,
                (sa, host) -> {
                    throw new IllegalStateException("boom (memo)");
                },
                (sa, host, memo) -> new EffectEvent(EffectEvent.TURN_SKIPPED));

        StringBuilder stderr = new StringBuilder();
        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collector) -> {
            InvocationHandler handler = collector.clauses().clauseHandler(
                    api -> api == ApiType.SkipTurn ? throwingMemo : ApiEvents.ruleFor(api));
            stderr.append(StdStreams.captureStderr(() -> {
                bracket.beginBracket(outer);
                resolvingContained(handler, outer);
                resolvingContained(handler, inner);
                handler.invoke(null, clauseMethod("onClauseResolved"),
                        new Object[]{inner, false});
                handler.invoke(null, clauseMethod("onClauseResolved"),
                        new Object[]{outer, false});
                bracket.endBracket(outer.getId(), false);
                bracket.flushHeldResolution();
            }));
        });

        String effectHalf = records.get(1);
        assertTrue(effectHalf.contains("\"type\":\"turn_added\",\"subjects\":[\""
                + SnapshotBuilder.playerId(outerTarget) + "\"]"),
                "the outer clause's own event must be unaffected by the inner "
                        + "clause's memo failure: " + effectHalf);
        assertTrue(stderr.toString().contains("boom (memo)"),
                "the memo's failure must be reported, not swallowed silently: "
                        + stderr);
    }

    /**
     * Calls {@code onClauseResolving} the way {@code
     * AbilityUtils.notifyClauseResolving} calls it in production --
     * containing a {@code RuntimeException} from the listener rather than
     * letting it escape -- so {@link #aThrowingMemoDoesNotDesyncTheSiblingsPop}
     * exercises the same shape a real game would, not a shape only a direct
     * reflective call could produce.
     */
    private static void resolvingContained(InvocationHandler handler, SpellAbility clause)
            throws Throwable {
        try {
            handler.invoke(null, clauseMethod("onClauseResolving"),
                    new Object[]{clause});
        } catch (RuntimeException e) {
            // Mirrors AbilityUtils.notifyClauseResolving's own containment,
            // which catches RuntimeException and nothing wider (ruling R8) --
            // an Error is left to propagate here too.
        }
    }

    // ── plumbing ────────────────────────────────────────────────────────

    /**
     * Drive the clause hook the way the engine drives it -- resolving then
     * resolved, on a real bracket -- and return the shard's rendered lines.
     */
    private List<String> resolveClause(SpellAbility sa, boolean threw) throws Throwable {
        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collector) -> {
            InvocationHandler handler = collector.clauses().clauseHandler();
            bracket.beginBracket(sa);
            handler.invoke(null, clauseMethod("onClauseResolving"),
                    new Object[]{sa});
            handler.invoke(null, clauseMethod("onClauseResolved"),
                    new Object[]{sa, threw});
            bracket.endBracket(sa.getId(), false);
            bracket.flushHeldResolution();
        });
        assertEquals(2, records.size(), records.toString());
        return records;
    }
}
