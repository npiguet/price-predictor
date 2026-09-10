package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.StaticData;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.GZIPInputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The clause hook's argument order, and the shape Forge declares it in.
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
 * {@link PatchedCollectors#clauseHandler()}, called the way the engine calls
 * it, with two real, distinct, non-null values so a position swap has
 * something to disagree with rather than two nulls agreeing by accident.
 */
@ExtendWith(ForgeExtension.class)
class ClauseContractTest {

    @TempDir
    Path tempDir;

    /**
     * Static, and shared with production: without a reset here, whichever
     * test trips {@code ApiEvents}' emitter-failure latch first -- in this
     * class or, within one JVM, {@code ApiEventsTest} -- leaves every test
     * after it unable to see its own "did this print" outcome.
     */
    @BeforeEach
    void resetEmitterFailureLatch() {
        ApiEvents.resetEmitterFailureLatchForTest();
    }

    /**
     * Forge's listener, copied argument for argument.
     *
     * <p>Copied rather than referenced on purpose: the connector compiles
     * against stock Forge, where this interface does not exist at all, and a
     * test that imported it would stop the whole module building on an
     * unpatched checkout. {@link #theStandInIsForgesOwnDeclaration} is what
     * keeps the copy true.
     */
    public interface ListenerShape {
        void onClauseResolving(SpellAbility clause);

        void onClauseResolved(SpellAbility clause, boolean threw);
    }

    private static Method methodNamed(Class<?> shape, String name) {
        for (Method method : shape.getMethods()) {
            if (method.getName().equals(name)) {
                return method;
            }
        }
        throw new AssertionError(shape + " declares no " + name);
    }

    // ── the argument order, pinned ──────────────────────────────────────

    /**
     * The stand-in above is what Forge declares, position by position, for
     * both halves of the listener.
     *
     * <p>Reflective, so an unpatched checkout skips the assertion rather than
     * failing it — the same degradation every hook lookup makes.
     */
    @Test
    void theStandInIsForgesOwnDeclaration() {
        PatchHooks.Lookup setter = PatchHooks.find(
                PatchHooks.ABILITY_UTILS, "setEffectRecordClauseListener");
        if (!setter.present()) {
            return;
        }
        Class<?> declared = setter.method().getParameterTypes()[0];
        for (String name : List.of("onClauseResolving", "onClauseResolved")) {
            Class<?>[] theirs = methodNamed(declared, name).getParameterTypes();
            Class<?>[] ours = methodNamed(ListenerShape.class, name).getParameterTypes();
            assertEquals(ours.length, theirs.length, name + " arity");
            for (int i = 0; i < theirs.length; i++) {
                assertEquals(ours[i], theirs[i], name + " argument " + i);
            }
        }
    }

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
        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        Path path = writer.path();
        BusBracketCollector bracket = new BusBracketCollector(
                TestCards.game(), writer, "run.0-l1.0", PatchedCollectors.CollectionCaps.defaults());
        try (PatchedCollectors collector = new PatchedCollectors(
                TestCards.game(), writer, "run.0-l1.0",
                PatchedCollectors.CollectionCaps.defaults(), 1L)) {
            collector.withBracket(bracket);
            InvocationHandler handler = collector.clauseHandler();
            bracket.beginBracket(sa);
            handler.invoke(null, methodNamed(ListenerShape.class, "onClauseResolved"),
                    new Object[]{Boolean.FALSE, sa});
            bracket.endBracket(sa.getId(), false);
        } finally {
            writer.close();
        }

        String effectHalf = readShard(path).get(1);
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

        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        Path path = writer.path();
        BusBracketCollector bracket = new BusBracketCollector(
                TestCards.game(), writer, "run.0-l1.0", PatchedCollectors.CollectionCaps.defaults());
        try (PatchedCollectors collector = new PatchedCollectors(
                TestCards.game(), writer, "run.0-l1.0",
                PatchedCollectors.CollectionCaps.defaults(), 1L)) {
            collector.withBracket(bracket);
            InvocationHandler handler = collector.clauseHandler();
            bracket.beginBracket(outer);
            handler.invoke(null, methodNamed(ListenerShape.class, "onClauseResolving"),
                    new Object[]{outer});
            handler.invoke(null, methodNamed(ListenerShape.class, "onClauseResolving"),
                    new Object[]{inner});
            handler.invoke(null, methodNamed(ListenerShape.class, "onClauseResolved"),
                    new Object[]{inner, false});
            handler.invoke(null, methodNamed(ListenerShape.class, "onClauseResolved"),
                    new Object[]{outer, false});
            bracket.endBracket(outer.getId(), false);
        } finally {
            writer.close();
        }

        String effectHalf = readShard(path).get(1);
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
     * <p>Companion to {@code PatchedCollectorTest
     * .anOutcomeFromAnotherGameDoesNotReachThisGamesBracket}, which covers
     * the same fix on the outcome hook's half.
     */
    @Test
    void aClauseFromAnotherGameDoesNotReachThisGamesBracket() throws Throwable {
        GameRules forkRules = new GameRules(GameType.Constructed);
        Game fork = new Game(List.of(), forkRules, new Match(forkRules, List.of(), "fork"));
        Card forkHost = CardFactory.getCard(
                StaticData.instance().getCommonCards().getCard("Grizzly Bears"),
                null, 900001, fork);
        SpellAbility forkAbility = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1", forkHost);
        Player forkTarget = new Player("fork-target", fork, 92201);
        forkAbility.resetTargets();
        forkAbility.getTargets().add(forkTarget);

        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        Path path = writer.path();
        try {
            BusBracketCollector bracket = new BusBracketCollector(
                    TestCards.game(), writer, "run.0-l1.0", PatchedCollectors.CollectionCaps.defaults());
            try (PatchedCollectors collector = new PatchedCollectors(
                    TestCards.game(), writer, "run.0-l1.0",
                    PatchedCollectors.CollectionCaps.defaults(), 1L)) {
                collector.withBracket(bracket);
                InvocationHandler handler = collector.clauseHandler();
                bracket.beginBracket(forkAbility);
                handler.invoke(null, methodNamed(ListenerShape.class, "onClauseResolving"),
                        new Object[]{forkAbility});
                handler.invoke(null, methodNamed(ListenerShape.class, "onClauseResolved"),
                        new Object[]{forkAbility, false});
                bracket.endBracket(forkAbility.getId(), false);
            }
        } finally {
            writer.close();
        }

        String shard = String.join("\n", readShard(path));
        assertFalse(shard.contains("turn_added"),
                "a fork's clause must not land in the live game's record: " + shard);
    }

    // ── F3: a throwing memo must not desync a sibling's pop ─────────────

    /**
     * F3: a clause whose own {@code Memo} throws must not desynchronize the
     * deque for the clauses around it.
     *
     * <p>Same shape as {@link #nestedClausesDoNotCrossTheirMemos}, except the
     * inner clause's memo throws -- forced through {@link
     * ApiEvents#setRuleOverrideForTest}, a seam built for exactly this (see
     * its own javadoc): no real script is known to make
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
     * inner clause's {@code onClauseResolving} throws out of {@code
     * ApiEvents.before(inner)} before its push runs, so the inner clause's
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

        ApiEvents.setRuleOverrideForTest("SkipTurn", new ApiEvents.Rule(
                EffectEvent.TURN_SKIPPED,
                (sa, host) -> {
                    throw new IllegalStateException("boom (memo)");
                },
                (sa, host, memo) -> new EffectEvent(EffectEvent.TURN_SKIPPED)));

        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        Path path = writer.path();
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream originalErr = System.err;
        try {
            BusBracketCollector bracket = new BusBracketCollector(
                    TestCards.game(), writer, "run.0-l1.0", PatchedCollectors.CollectionCaps.defaults());
            try (PatchedCollectors collector = new PatchedCollectors(
                    TestCards.game(), writer, "run.0-l1.0",
                    PatchedCollectors.CollectionCaps.defaults(), 1L)) {
                collector.withBracket(bracket);
                InvocationHandler handler = collector.clauseHandler();
                System.setErr(new PrintStream(captured, true));
                bracket.beginBracket(outer);
                resolvingContained(handler, outer);
                resolvingContained(handler, inner);
                handler.invoke(null, methodNamed(ListenerShape.class, "onClauseResolved"),
                        new Object[]{inner, false});
                handler.invoke(null, methodNamed(ListenerShape.class, "onClauseResolved"),
                        new Object[]{outer, false});
                bracket.endBracket(outer.getId(), false);
            } finally {
                System.setErr(originalErr);
            }
        } finally {
            writer.close();
            ApiEvents.clearRuleOverridesForTest();
        }

        String effectHalf = readShard(path).get(1);
        assertTrue(effectHalf.contains("\"type\":\"turn_added\",\"subjects\":[\""
                + SnapshotBuilder.playerId(outerTarget) + "\"]"),
                "the outer clause's own event must be unaffected by the inner "
                        + "clause's memo failure: " + effectHalf);
        assertTrue(captured.toString().contains("boom (memo)"),
                "the memo's failure must be reported, not swallowed silently: "
                        + captured);
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
            handler.invoke(null, methodNamed(ListenerShape.class, "onClauseResolving"),
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
        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        Path path = writer.path();
        BusBracketCollector bracket = new BusBracketCollector(
                TestCards.game(), writer, "run.0-l1.0", PatchedCollectors.CollectionCaps.defaults());
        try (PatchedCollectors collector = new PatchedCollectors(
                TestCards.game(), writer, "run.0-l1.0",
                PatchedCollectors.CollectionCaps.defaults(), 1L)) {
            collector.withBracket(bracket);
            InvocationHandler handler = collector.clauseHandler();
            bracket.beginBracket(sa);
            handler.invoke(null, methodNamed(ListenerShape.class, "onClauseResolving"),
                    new Object[]{sa});
            handler.invoke(null, methodNamed(ListenerShape.class, "onClauseResolved"),
                    new Object[]{sa, threw});
            bracket.endBracket(sa.getId(), false);
        } finally {
            writer.close();
        }
        List<String> records = readShard(path);
        assertEquals(2, records.size(), records.toString());
        return records;
    }

    private static List<String> readShard(Path path) throws Exception {
        try (var gzip = new GZIPInputStream(Files.newInputStream(path));
                var reader = new BufferedReader(
                        new InputStreamReader(gzip, StandardCharsets.UTF_8))) {
            return reader.lines().toList();
        }
    }
}
