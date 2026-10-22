package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.TestCollectors.FakeTrait;
import forge.game.ability.AbilityKey;
import forge.game.card.Card;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementResult;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The rewrite payload, and the argument order it is read off.
 *
 * <p>The connector never links against the listener interface — it is installed
 * as a dynamic proxy over whatever the patched checkout declares — so the
 * argument order is a runtime contract with no compiler behind it. A swap or an
 * insertion on the Forge side would not fail a build; it would fail a corpus,
 * silently, eight hours in. These tests are what stands in for the compiler:
 * they call the handler through a stand-in interface that is a literal copy of
 * Forge's declaration, and they check that copy against the real one.
 *
 * <p>Beside that, the shape the contract fixed: one record per replacement
 * including the negatives, {@code outgoing} null exactly where nothing was
 * rewritten in place, and {@code replaced_by} keyed where an ability stood in
 * for the event; and what a record says about its acting line and its event.
 */
@ExtendWith(ForgeExtension.class)
class RewriteContractTest {

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
        void onReplacement(ReplacementEffect effect, Map<AbilityKey, Object> before,
                Map<AbilityKey, Object> after, ReplacementResult result,
                SpellAbility replacedBy);
    }

    /** The same listener before the result and the substituted ability. */
    public interface OldListenerShape {
        void onReplacement(ReplacementEffect effect, Map<AbilityKey, Object> before,
                Map<AbilityKey, Object> after);
    }

    private static Method onReplacement(Class<?> shape) {
        for (Method method : shape.getMethods()) {
            if (method.getName().equals("onReplacement")) {
                return method;
            }
        }
        throw new AssertionError(shape + " declares no onReplacement");
    }

    // ── the argument order, pinned ──────────────────────────────────────

    /**
     * The stand-in above is what Forge declares, position by position.
     *
     * <p>The one test here that a Forge-side signature drift fails. Both arities
     * are accepted because the connector must run against either — a jar built
     * before the widening still hands over three arguments — but the prefix is
     * not negotiable: {@code effect}, {@code before} and {@code after} keep
     * their positions and their types in both, which is the whole reason the
     * widening could be made without reinstalling anything.
     *
     * <p>Reflective, so an unpatched checkout skips the assertion rather than
     * failing it. That is the same degradation every hook lookup makes, and the
     * integration test's mode assertion is what says whether it happened.
     */
    @Test
    void theStandInIsForgesOwnDeclaration() {
        PatchHooks.Lookup setter = PatchHooks.find(
                PatchHooks.REPLACEMENT_HANDLER, "setEffectRecordListener");
        Assumptions.assumeTrue(setter.present(), "the checkout lacks the replacement hook");
        Class<?> declared = setter.method().getParameterTypes()[0];
        Class<?>[] theirs = onReplacement(declared).getParameterTypes();
        Class<?>[] ours = onReplacement(ListenerShape.class).getParameterTypes();

        assertTrue(theirs.length == 3 || theirs.length == 5,
                "the listener takes three arguments or five, not "
                        + theirs.length + "; an argument added anywhere but the "
                        + "end moves the ones the handler reads by position");
        for (int i = 0; i < theirs.length; i++) {
            assertEquals(ours[i], theirs[i], "argument " + i + " of onReplacement");
        }
    }

    // ── one record per replacement, whatever the result ─────────────────

    /**
     * Each of the five results reaches the payload in its wire spelling.
     *
     * <p>Including {@code NotReplaced}, which is this channel's negative and
     * used to be indistinguishable from the four others: they all left the
     * parameter map untouched, so all five arrived as an identity pair and the
     * collector dropped them. That was 87% of the candidates and left 34
     * records in 1.88M against a 7% share of the mixture.
     */
    @ParameterizedTest
    @EnumSource(ReplacementResult.class)
    void everyResultIsWrittenInItsWireSpelling(ReplacementResult result) {
        Card card = TestCards.build("Mountain");
        String json = recording().rewriteRecord(
                new FakeTrait("Moved"),
                Map.of("Card", card), Map.of("Card", card), result, null).toJson();

        assertTrue(json.contains("\"result\":\"" + WIRE.get(result) + "\""),
                "the lower snake case of " + result + ": " + json);
    }

    /** And it is snake case, which {@code NotReplaced} is the only test of. */
    @Test
    void theTwoWordResultIsSnakeCased() {
        assertEquals("not_replaced",
                EffectRecord.rewriteResult(ReplacementResult.NotReplaced));
        assertEquals("replaced",
                EffectRecord.rewriteResult(ReplacementResult.Replaced));
    }

    /**
     * A result the engine never returned is null, not a guess.
     *
     * <p>Forge notifies the listener with a null result in one case — the
     * replacement threw — because an exception is an outcome and hiding the
     * record would hide it. The reader has to see that as "no answer" rather
     * than as one of the five.
     */
    @Test
    void aReplacementWithNoAnswerSaysSoRatherThanGuessing() {
        Card card = TestCards.build("Mountain");
        String json = recording().rewriteRecord(
                new FakeTrait("Moved"),
                Map.of("Card", card), Map.of("Card", card), null, null).toJson();

        assertTrue(json.contains("\"result\":null"), json);
    }

    /** The wire spellings, written out rather than derived, so the test is a pin. */
    private static final Map<ReplacementResult, String> WIRE =
            new EnumMap<>(Map.of(
                    ReplacementResult.Replaced, "replaced",
                    ReplacementResult.NotReplaced, "not_replaced",
                    ReplacementResult.Prevented, "prevented",
                    ReplacementResult.Updated, "updated",
                    ReplacementResult.Skipped, "skipped"));

    // ── outgoing, and where it is null ──────────────────────────────────

    /**
     * A real in-place edit carries its outgoing half.
     *
     * <p>Orim's Cure taking five combat damage down to one is the shape that
     * always read correctly, and it must keep reading correctly: the null is
     * for the substitutions, not for every record.
     */
    @Test
    void aPreventedAmountIsARewriteWithBothHalves() {
        Card card = TestCards.build("Mountain");
        Map<String, Object> before = new LinkedHashMap<>(Map.of(
                "Affected", card, "DamageAmount", 5));
        Map<String, Object> after = new LinkedHashMap<>(before);
        after.put("DamageAmount", 1);

        String json = recording().rewriteRecord(
                new FakeTrait("DamageDone"), before, after,
                ReplacementResult.Updated, null).toJson();

        assertFalse(json.contains("\"outgoing\":null"),
                "five down to one is a parameter edit: " + json);
        assertTrue(json.contains("\"amount\":5"), json);
        assertTrue(json.contains("\"amount\":1"), json);
    }

    /**
     * A substitution carries none, because it edited nothing.
     *
     * <p>Rest in Peace exiles the card instead of putting it in the graveyard,
     * and it does that by running its {@code ReplaceWith$} ability — the run
     * parameters it was handed come back byte for byte. What the record says is
     * that no parameter was rewritten and that an ability stood in, which is
     * two facts an identity pair could state neither of.
     */
    @Test
    void aSubstitutionRewritesNothingAndNamesWhatRanInstead() {
        Card rest = TestCards.build("Rest in Peace");
        ReplacementEffect replacement =
                rest.getCurrentState().getReplacementEffects().iterator().next();
        SpellAbility ran = replacement.ensureAbility();
        ran.setReplacementEffect(replacement);
        Map<String, Object> params = Map.of(
                "Card", rest, "Destination", ZoneType.Graveyard);

        String json = recording().rewriteRecord(
                replacement, params, params, ReplacementResult.Replaced, ran).toJson();

        assertTrue(json.contains("\"outgoing\":null"), json);
        assertTrue(json.contains("\"result\":\"replaced\""), json);
        assertTrue(json.contains("\"replaced_by\":[{"), json);
        // Keyed through the resolver the record's own acting line uses, so the
        // two join the provenance sidecar identically.
        int at = json.indexOf("\"replaced_by\":");
        assertTrue(json.indexOf("cardsfolder/r/rest_in_peace.txt", at) > at,
                "replaced_by names the printed line that ran: " + json);
    }

    /** Where no ability ran, the field is an empty list rather than a null. */
    @Test
    void nothingRunningInsteadIsAnEmptyListNotANull() {
        Card card = TestCards.build("Mountain");
        String json = recording().rewriteRecord(
                new FakeTrait("Untap"),
                Map.of("Card", card), Map.of("Card", card),
                ReplacementResult.Prevented, null).toJson();

        assertTrue(json.contains("\"replaced_by\":[]"), json);
    }

    /**
     * An argument that is not a trait keys to nothing rather than throwing.
     *
     * <p>The same degradation every other hook read makes. A signature drift
     * that put something else in the last position must cost the field, not the
     * replacement: the exception would surface out of a dynamic proxy in the
     * middle of the engine's replacement loop.
     */
    @Test
    void aSubstitutedAbilityThatIsNotATraitCostsOnlyTheField() {
        Card card = TestCards.build("Mountain");
        String json = recording().rewriteRecord(
                new FakeTrait("Moved"),
                Map.of("Card", card), Map.of("Card", card),
                ReplacementResult.Replaced, "not an ability").toJson();

        assertTrue(json.contains("\"replaced_by\":[]"), json);
        assertTrue(json.contains("\"result\":\"replaced\""), json);
    }

    // ── the handler, called the way the engine calls it ─────────────────

    /**
     * Five arguments in, and each one read off its own position.
     *
     * <p>The values are deliberately chosen not to agree with each other: the
     * maps describe a redirected move while the result says {@code Prevented},
     * so a handler that inferred the result from the halves, or read it off the
     * wrong index, could not produce this record.
     */
    @Test
    void theHandlerReadsEachArgumentOffItsOwnPosition() throws Throwable {
        Card rest = TestCards.build("Rest in Peace");
        ReplacementEffect replacement =
                rest.getCurrentState().getReplacementEffects().iterator().next();
        SpellAbility ran = replacement.ensureAbility();
        ran.setReplacementEffect(replacement);
        Map<AbilityKey, Object> before = new LinkedHashMap<>();
        before.put(AbilityKey.Card, rest);
        before.put(AbilityKey.Destination, ZoneType.Graveyard);
        Map<AbilityKey, Object> after = new LinkedHashMap<>(before);
        after.put(AbilityKey.Destination, ZoneType.Exile);

        String line = oneRecordFrom(new Object[]{
                replacement, before, after, ReplacementResult.Prevented, ran},
                ListenerShape.class);

        assertTrue(line.contains("\"to_zone\":\"graveyard\""), line);
        assertTrue(line.contains("\"to_zone\":\"exile\""), line);
        assertTrue(line.contains("\"result\":\"prevented\""), line);
        assertTrue(line.contains("cardsfolder/r/rest_in_peace.txt"), line);
    }

    /**
     * Three arguments in, and the record says it learnt neither new one.
     *
     * <p>A worker whose Forge jar predates the widening. The listener is
     * installed by name and the setter's name did not change, so nothing warns:
     * what says so is a corpus of rewrite records whose result is null. The
     * handler must degrade to that rather than index past the end of the array
     * and throw inside the proxy.
     */
    @Test
    void anOlderJarsThreeArgumentsStillWriteARecord() throws Throwable {
        Card card = TestCards.build("Mountain");
        Map<AbilityKey, Object> params = new LinkedHashMap<>();
        params.put(AbilityKey.Card, card);

        String line = oneRecordFrom(
                new Object[]{null, params, params}, OldListenerShape.class);

        assertTrue(line.contains("\"result\":null"), line);
        assertTrue(line.contains("\"replaced_by\":[]"), line);
        assertTrue(line.contains("\"outgoing\":null"), line);
    }

    /**
     * A replacement that declined is a written record, not a dropped one.
     *
     * <p>Through the handler rather than the builder, because dropping was the
     * handler's decision to make and this is the assertion that it no longer
     * makes it.
     */
    @Test
    void aDeclinedReplacementReachesTheShard() throws Throwable {
        Card card = TestCards.build("Mountain");
        Map<AbilityKey, Object> params = new LinkedHashMap<>();
        params.put(AbilityKey.Card, card);

        String line = oneRecordFrom(new Object[]{
                null, params, params, ReplacementResult.NotReplaced, null},
                ListenerShape.class);

        assertTrue(line.contains("\"kind\":\"rewrite\""), line);
        assertTrue(line.contains("\"result\":\"not_replaced\""), line);
    }

    // ── final-fix-3.md item 2: another game's replacement ───────────────

    /**
     * final-fix-3.md item 2: a replacement reported for an effect belonging
     * to a DIFFERENT {@code Game} -- exactly {@code
     * ForkCollector.forceResolution}'s shape, since a forced ability's own
     * replacements run through {@code GameSimulator.resolveStack}'s real
     * resolution pipeline -- must not reach the live shard. This hook had no
     * game check at all before this fix (unlike the clause/outcome hooks,
     * which F2 already protected); {@code rewriteHandler} is a plain JVM
     * static exactly like the other five this brief's item 2 covers.
     *
     * <p>Companion to {@code ClauseContractTest
     * .aClauseFromAnotherGameDoesNotReachThisGamesBracket} and {@code
     * OutcomeHandlerTest.anOutcomeFromAnotherGameDoesNotReachThisGamesBracket}.
     */
    @Test
    void aReplacementFromAnotherGameDoesNotReachTheShard() throws Throwable {
        Card forkHost = TestBoard.cardInAnotherGame("Rest in Peace");
        ReplacementEffect forkReplacement =
                forkHost.getCurrentState().getReplacementEffects().iterator().next();
        Map<String, Object> params = Map.of(
                "Card", forkHost, "Destination", ZoneType.Graveyard);

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collector) ->
                collector.rewriteHandler().invoke(null, onReplacement(ListenerShape.class),
                        new Object[]{
                                forkReplacement, params, params, ReplacementResult.Replaced, null}));

        assertFalse(Files.exists(shard),
                "a fork's own replacement must not land in the live game's shard");
    }

    // ── what a rewrite record says ──────────────────────────────────────

    /**
     * The other half of the same omission: 62,546 rewrite records that named
     * no replacement effect, on the one hook every replacement passes through.
     */
    @Test
    void aRewriteRecordNamesItsReplacementEffect() {
        // Rest in Peace rather than Paralyze: its replacement redirects a move,
        // so this is the in-place rewrite the pair was always able to express,
        // and every other field on the record can be read beside it.
        Card rest = TestCards.build("Rest in Peace");
        ReplacementEffect replacement =
                rest.getCurrentState().getReplacementEffects().iterator().next();

        String json = recording().rewriteRecord(
                replacement,
                Map.of("Card", rest, "Destination", ZoneType.Graveyard),
                Map.of("Card", rest, "Destination", ZoneType.Exile),
                ReplacementResult.Replaced, null).toJson();

        assertTrue(json.contains("\"kind\":\"rewrite\""), json);
        assertTrue(json.contains("\"trait_kind\":\"replacement\""), json);
        assertTrue(json.contains("cardsfolder/r/rest_in_peace.txt"), json);
        assertTrue(json.contains(
                "\"source\":\"" + SnapshotBuilder.entityId(rest) + "\""), json);
        // And it says what the replacement actually did, which is the whole of
        // the rewrite channel's defect: a graveyard destination in, an exile
        // destination out.
        assertTrue(json.contains("\"incoming\":{\"type\":\"zone_change\""), json);
        assertTrue(json.contains("\"to_zone\":\"graveyard\""), json);
        assertTrue(json.contains("\"to_zone\":\"exile\""), json);
    }

    /**
     * A prevention that changes no parameter is a record, not a loss.
     *
     * <p>Paralyze's "doesn't untap" is the shape the whole contract is for: the
     * engine skips the event rather than rewriting a value, so both halves read
     * alike and the record used to be dropped. It was 87% of the channel. The
     * result is what makes the row readable without an outgoing — {@code skipped}
     * says outright what an identity pair could only imply — so the record is
     * written, and its outgoing is null rather than a copy of its incoming.
     */
    @Test
    void aPreventionThatRewritesNothingIsStillARecord() {
        Card paralyze = TestCards.build("Paralyze");
        ReplacementEffect prevention =
                paralyze.getCurrentState().getReplacementEffects().iterator().next();

        EffectRecord record = recording().rewriteRecord(
                prevention, Map.of("Card", paralyze), Map.of("Card", paralyze),
                ReplacementResult.Skipped, null);

        assertNotNull(record, "a skipped replacement is this channel's negative");
        String json = record.toJson();
        assertTrue(json.contains("\"result\":\"skipped\""), json);
        assertTrue(json.contains("\"outgoing\":null"),
                "null, never a copy of incoming: " + json);
        assertTrue(json.contains("\"replaced_by\":[]"),
                "a skip returns above the ability that would have run: " + json);
    }

    /**
     * A record that names its line carries no reason, and the reverse.
     *
     * <p>The two fields are one statement in two halves. A reason beside a
     * named line is a contradiction the Python reader refuses outright, so the
     * collector may never write both.
     */
    @Test
    void aRecordThatNamesItsLineCarriesNoReason() {
        Card rest = TestCards.build("Rest in Peace");
        ReplacementEffect replacement =
                rest.getCurrentState().getReplacementEffects().iterator().next();

        String json = recording().rewriteRecord(
                replacement,
                Map.of("Card", rest, "Destination", ZoneType.Graveyard),
                Map.of("Card", rest, "Destination", ZoneType.Exile),
                ReplacementResult.Replaced, null).toJson();

        assertTrue(json.contains("cardsfolder/r/rest_in_peace.txt"), json);
        assertTrue(json.contains("\"ability_unresolved\":null"), json);
    }

    /** And a rewrite that changed the destination is two different halves. */
    @Test
    void aRewrittenDestinationMakesTheTwoHalvesDiffer() {
        Card card = TestCards.build("Mountain");
        Map<String, Object> before = new LinkedHashMap<>(Map.of(
                "Origin", ZoneType.Battlefield, "Destination", ZoneType.Graveyard,
                "Card", card));
        Map<String, Object> after = new LinkedHashMap<>(before);
        after.put("Destination", ZoneType.Exile);

        EffectRecord record = recording().rewriteRecord(
                new FakeTrait("Moved"), before, after,
                ReplacementResult.Updated, null);

        assertNotNull(record, "a replacement that redirected a card is a real rewrite");
        String json = record.toJson();
        assertTrue(json.contains("\"to_zone\":\"graveyard\""), json);
        assertTrue(json.contains("\"to_zone\":\"exile\""), json);
    }

    /**
     * A rewrite whose halves read alike says so with a null, not with a copy.
     *
     * <p>Writing the copy is what made the channel unreadable: 87% of the
     * candidates were a substitution whose two halves were byte-identical
     * because Forge substitutes by running another ability rather than by
     * editing the map. Null says "not rewritten" outright, and the corpus
     * validator judges an identity pair at zero rather than letting one parse.
     */
    @Test
    void aRewriteThatChangedNoObservableParameterCarriesNoOutgoing() {
        Map<String, Object> params = Map.of("Card", TestCards.build("Mountain"));

        String json = recording().rewriteRecord(
                new FakeTrait("Moved"), params, params,
                ReplacementResult.Replaced, null).toJson();

        assertTrue(json.contains("\"outgoing\":null"), json);
        assertFalse(json.contains("\"outgoing\":{"),
                "never a copy of incoming: " + json);
    }

    /**
     * A replacement that halved a draw is a rewrite, not an identity pair.
     *
     * <p>The mode table was written from the trigger vocabulary, so 31 of
     * Forge's 42 {@code ReplacementType}s rendered as the generic outcome
     * carrying only their mode -- and an event with no value in it cannot
     * differ from itself. Every rewrite of a draw, a mill, a token count or a
     * life total was therefore dropped as an identity pair by construction,
     * whatever the replacement actually did.
     */
    @Test
    void aRewrittenQuantityOnTheReplacementSideIsNoLongerAnIdentityPair() {
        Card card = TestCards.build("Mountain");
        Map<String, Object> before = new LinkedHashMap<>(Map.of(
                "Affected", card, "Number", 3));
        Map<String, Object> after = new LinkedHashMap<>(before);
        after.put("Number", 1);

        EffectRecord record = recording().rewriteRecord(
                new FakeTrait("DrawCards"), before, after,
                ReplacementResult.Updated, null);

        assertNotNull(record, "a replacement that halved a draw is a real rewrite");
        String json = record.toJson();
        assertTrue(json.contains("\"type\":\"card_drawn\""), json);
        assertTrue(json.contains("\"count\":3"), json);
        assertTrue(json.contains("\"count\":1"), json);
    }

    /** The live path feeds the tally, which is what makes it a live number. */
    @Test
    void everyReplacementReachesTheWorkerTally() {
        Card card = TestCards.build("Mountain");
        Map<String, Object> before = new LinkedHashMap<>(Map.of(
                "Origin", ZoneType.Battlefield, "Destination", ZoneType.Graveyard,
                "Card", card));
        Map<String, Object> after = new LinkedHashMap<>(before);
        after.put("Destination", ZoneType.Exile);
        // Deltas rather than absolutes: the tally is the worker's, so every
        // other test in this class has already contributed to it.
        long recordsBefore = PatchedCollectors.JVM_REWRITES.records();
        PatchedCollectors collector = recording();

        collector.rewriteRecord(
                new FakeTrait("Moved"), before, after,
                ReplacementResult.Updated, null);
        collector.rewriteRecord(
                new FakeTrait("Moved"), before, before,
                ReplacementResult.NotReplaced, null);

        // Two records for two replacements: the one that declined is counted
        // too, because it is no longer dropped.
        assertEquals(recordsBefore + 2, PatchedCollectors.JVM_REWRITES.records());
    }

    /**
     * A replacement whose mode only the replacement side spells is still typed.
     *
     * <p>{@code ReplacementType} names {@code Untap} where {@code TriggerType}
     * names {@code Untaps}, and the mode table was written from the trigger
     * vocabulary — so an untap replacement took the generic type, carried
     * nothing but its own name, and could not differ from itself however much
     * it changed. Paralyze's is that mode. Its record still carries no outgoing,
     * because a prevention changes no parameter at all, but the incoming half is
     * now typed rather than generic.
     */
    @Test
    void anUntapReplacementIsSpelledTheReplacementSideWay() {
        Card paralyze = TestCards.build("Paralyze");
        ReplacementEffect prevention =
                paralyze.getCurrentState().getReplacementEffects().iterator().next();

        assertEquals("Untap", String.valueOf(prevention.getMode()));
        String json = recording().rewriteRecord(
                prevention, Map.of("Card", paralyze), Map.of("Card", paralyze),
                ReplacementResult.Skipped, null).toJson();

        assertTrue(json.contains("\"mode\":\"Untap\""), json);
        assertTrue(json.contains("\"outgoing\":null"), json);
    }

    // ── plumbing ────────────────────────────────────────────────────────

    private PatchedCollectors recording() {
        return new PatchedCollectors(
                TestCards.game(),
                new RecordShardWriter(tempDir, "run", 0, "l1"),
                "run.0-l1.0", CollectionCaps.defaults(), 1L);
    }

    /**
     * Everything the handler wrote for one notification, as the shard line.
     *
     * <p>Driven through {@link PatchedCollectors#rewriteHandler} with a
     * synthesized argument array, which is as close to the engine's own call as
     * this side can get without a patched jar to proxy over.
     */
    private String oneRecordFrom(Object[] args, Class<?> shape) throws Throwable {
        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collector) ->
                collector.rewriteHandler().invoke(null, onReplacement(shape), args));
        assertEquals(1, lines.size(), "one replacement writes one record");
        assertNotNull(lines.get(0));
        return lines.get(0);
    }
}
