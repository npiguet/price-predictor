package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.InputStreamReader;
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
