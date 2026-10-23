package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.StdStreams;
import forge.game.ability.AbilityKey;
import forge.game.card.Card;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementResult;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static com.pricepredictor.connector.effects.TestCollectors.recording;
import static com.pricepredictor.connector.effects.TestCollectors.replacementOfMode;
import static com.pricepredictor.connector.effects.TestCollectors.unwired;
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
 * they call the handler through {@link HookShapes.Replacement}, a literal copy
 * of Forge's declaration that {@code PatchHooksTest} checks against the real
 * one.
 *
 * <p>Beside that, the shape the contract fixes: one record per replacement
 * including the negatives, {@code outgoing} null exactly where nothing was
 * rewritten in place, and {@code replaced_by} keyed where an ability stood in
 * for the event; and what a record says about its acting line and its event.
 */
@ExtendWith(ForgeExtension.class)
class RewriteContractTest {

    @TempDir
    Path tempDir;

    private static Method onReplacement(Class<?> shape) {
        return HookShapes.method(shape, "onReplacement");
    }

    /** The card's one printed replacement effect: Rest in Peace's, or Paralyze's. */
    private static ReplacementEffect replacementOf(Card card) {
        return card.getCurrentState().getReplacementEffects().iterator().next();
    }

    // ── one record per replacement, whatever the result ─────────────────

    /**
     * Each of the five results reaches the payload in its wire spelling.
     *
     * <p>Including {@code NotReplaced}, which is this channel's negative: all
     * five can leave the parameter map untouched and arrive as an identity
     * pair, so the result is the only thing that tells them apart.
     */
    @ParameterizedTest
    @EnumSource(ReplacementResult.class)
    void everyResultIsWrittenInItsWireSpelling(ReplacementResult result) {
        Card card = TestCards.build("Mountain");
        String json = recording(tempDir).rewrites().record(
                replacementOfMode("Moved"),
                Map.of("Card", card), Map.of("Card", card), result, null).toJson();

        // The spellings themselves are WireVocabularyTest's pin.
        assertTrue(json.contains("\"result\":\"" + EffectRecord.rewriteResult(result) + "\""),
                "the lower snake case of " + result + ": " + json);
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
        String json = recording(tempDir).rewrites().record(
                replacementOfMode("Moved"),
                Map.of("Card", card), Map.of("Card", card), null, null).toJson();

        assertTrue(json.contains("\"result\":null"), json);
    }

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

        String json = recording(tempDir).rewrites().record(
                replacementOfMode("DamageDone"), before, after,
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
        ReplacementEffect replacement = replacementOf(rest);
        SpellAbility ran = replacement.ensureAbility();
        ran.setReplacementEffect(replacement);
        Map<String, Object> params = Map.of(
                "Card", rest, "Destination", ZoneType.Graveyard);

        String json = recording(tempDir).rewrites().record(
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
        String json = recording(tempDir).rewrites().record(
                replacementOfMode("Untap"),
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
        String json = recording(tempDir).rewrites().record(
                replacementOfMode("Moved"),
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
        ReplacementEffect replacement = replacementOf(rest);
        SpellAbility ran = replacement.ensureAbility();
        ran.setReplacementEffect(replacement);
        Map<AbilityKey, Object> before = new LinkedHashMap<>();
        before.put(AbilityKey.Card, rest);
        before.put(AbilityKey.Destination, ZoneType.Graveyard);
        Map<AbilityKey, Object> after = new LinkedHashMap<>(before);
        after.put(AbilityKey.Destination, ZoneType.Exile);

        String line = oneRecordFrom(new Object[]{
                replacement, before, after, ReplacementResult.Prevented, ran},
                HookShapes.Replacement.class);

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
                new Object[]{null, params, params}, HookShapes.OlderReplacement.class);

        assertTrue(line.contains("\"result\":null"), line);
        assertTrue(line.contains("\"replaced_by\":[]"), line);
        assertTrue(line.contains("\"outgoing\":null"), line);
    }

    /**
     * A replacement that declined is a written record, not a dropped one.
     *
     * <p>Through the handler rather than the builder, because dropping would
     * be the handler's decision to make and this is the assertion that it does
     * not make it.
     */
    @Test
    void aDeclinedReplacementReachesTheShard() throws Throwable {
        Card card = TestCards.build("Mountain");
        Map<AbilityKey, Object> params = new LinkedHashMap<>();
        params.put(AbilityKey.Card, card);

        String line = oneRecordFrom(new Object[]{
                null, params, params, ReplacementResult.NotReplaced, null},
                HookShapes.Replacement.class);

        assertTrue(line.contains("\"kind\":\"rewrite\""), line);
        assertTrue(line.contains("\"result\":\"not_replaced\""), line);
    }

    /**
     * Two replacements on one card that both decline one event are two records.
     *
     * <p>Archelos, Lagoon Mystic prints two {@code Moved} replacements, one
     * applying while it is tapped and one while it is untapped, so on any
     * entry at least one of them declines. A rewrite snapshot names the host
     * and the event, not which of the host's replacements is answering, so the
     * two declines render the same state and payload; only the acting line
     * tells them apart, and without it the second line's negative would be
     * coalesced away as a repeat of the first's.
     */
    @Test
    void twoLinesDecliningOneEventEachReachTheShard() throws Throwable {
        Card archelos = TestCards.build("Archelos, Lagoon Mystic");
        List<ReplacementEffect> replacements =
                List.copyOf(archelos.getCurrentState().getReplacementEffects());
        assertEquals(2, replacements.size(), "Archelos prints two R: lines");
        Map<AbilityKey, Object> params = new LinkedHashMap<>();
        params.put(AbilityKey.Card, TestCards.build("Mountain"));
        params.put(AbilityKey.Destination, ZoneType.Battlefield);

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collector) -> {
            for (ReplacementEffect replacement : replacements) {
                collector.rewrites().handler().invoke(
                        null, onReplacement(HookShapes.Replacement.class),
                        new Object[]{
                                replacement, params, params, ReplacementResult.NotReplaced, null});
            }
        });

        assertEquals(2, lines.size(), "one record per declining line: " + lines);
        assertEquals(List.of(0L, 1L),
                lines.stream()
                        .map(l -> RecordLines.abilityKeys(l).get(0).get("index_within_kind"))
                        .toList(),
                lines.toString());
    }

    // ── another game's replacement ──────────────────────────────────────

    /**
     * A replacement reported for an effect belonging
     * to a DIFFERENT {@code Game} -- exactly {@code
     * ForkCollector.forceResolution}'s shape, since a forced ability's own
     * replacements run through {@code GameSimulator.resolveStack}'s real
     * resolution pipeline -- must not reach the live shard.
     * {@code RewriteChannel.handler} is a plain JVM static exactly like the
     * clause, outcome, trigger, playability and mana hooks.
     *
     * <p>Companion to {@code ClauseContractTest
     * .aClauseFromAnotherGameDoesNotReachThisGamesBracket} and {@code
     * OutcomeHandlerTest.anOutcomeFromAnotherGameDoesNotReachThisGamesBracket}.
     */
    @Test
    void aReplacementFromAnotherGameDoesNotReachTheShard() throws Throwable {
        Card forkHost = TestBoard.cardInAnotherGame("Rest in Peace");
        ReplacementEffect forkReplacement = replacementOf(forkHost);
        Map<String, Object> params = Map.of(
                "Card", forkHost, "Destination", ZoneType.Graveyard);

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collector) ->
                collector.rewrites().handler().invoke(null, onReplacement(HookShapes.Replacement.class),
                        new Object[]{
                                forkReplacement, params, params, ReplacementResult.Replaced, null}));

        assertFalse(Files.exists(shard),
                "a fork's own replacement must not land in the live game's shard");
    }

    // ── what a rewrite record says ──────────────────────────────────────

    /**
     * A rewrite record names its replacement effect, on the one hook every
     * replacement passes through.
     *
     * <p>And a record that names its line carries no reason for an empty one.
     * The two fields are one statement in two halves: a reason beside a named
     * line is a contradiction the Python reader refuses outright, so the
     * collector may never write both.
     */
    @Test
    void aRewriteRecordNamesItsReplacementEffect() {
        // Rest in Peace rather than Paralyze: its replacement redirects a move,
        // so this is the in-place rewrite the pair was always able to express,
        // and every other field on the record can be read beside it.
        Card rest = TestCards.build("Rest in Peace");
        ReplacementEffect replacement = replacementOf(rest);

        String json = recording(tempDir).rewrites().record(
                replacement,
                Map.of("Card", rest, "Destination", ZoneType.Graveyard),
                Map.of("Card", rest, "Destination", ZoneType.Exile),
                ReplacementResult.Replaced, null).toJson();

        assertTrue(json.contains("\"kind\":\"rewrite\""), json);
        assertTrue(json.contains("\"trait_kind\":\"replacement\""), json);
        assertTrue(json.contains("cardsfolder/r/rest_in_peace.txt"), json);
        assertTrue(json.contains("\"ability_unresolved\":null"), json);
        assertTrue(json.contains(
                "\"source\":\"" + SnapshotBuilder.entityId(rest) + "\""), json);
        assertTrue(json.contains("\"incoming\":{\"type\":\"zone_change\""), json);
    }

    /**
     * A prevention that changes no parameter is a record, not a loss, and its
     * mode is spelled the replacement side's way.
     *
     * <p>Paralyze's "doesn't untap" is the shape the whole contract is for: the
     * engine skips the event rather than rewriting a value, so both halves read
     * alike. The result is what makes the row readable without an outgoing —
     * {@code skipped}
     * says outright what an identity pair could only imply — so the record is
     * written, and its outgoing is null rather than a copy of its incoming.
     *
     * <p>{@code ReplacementType} names {@code Untap} where {@code TriggerType}
     * names {@code Untaps}, so without its own entry in the mode table an untap
     * replacement would take the generic type, carry nothing but its own name,
     * and could not differ from itself however much it changed. Paralyze's is
     * that mode, and its incoming half is typed rather than generic.
     */
    @Test
    void aPreventionThatRewritesNothingIsStillARecord() {
        Card paralyze = TestCards.build("Paralyze");
        ReplacementEffect prevention = replacementOf(paralyze);
        assertEquals("Untap", String.valueOf(prevention.getMode()));

        EffectRecord record = recording(tempDir).rewrites().record(
                prevention, Map.of("Card", paralyze), Map.of("Card", paralyze),
                ReplacementResult.Skipped, null);

        assertNotNull(record, "a skipped replacement is this channel's negative");
        String json = record.toJson();
        assertTrue(json.contains("\"result\":\"skipped\""), json);
        assertTrue(json.contains("\"outgoing\":null"),
                "null, never a copy of incoming: " + json);
        assertTrue(json.contains("\"replaced_by\":[]"),
                "a skip returns above the ability that would have run: " + json);
        assertTrue(json.contains("\"mode\":\"Untap\""), json);
    }

    /**
     * A replacement that halved a draw is a rewrite, not an identity pair.
     *
     * <p>The replacement side has quantity keys of its own, beyond the trigger
     * vocabulary. Read as the generic outcome, carrying only its mode, such an
     * event has no value in it and cannot differ from itself: every rewrite of
     * a draw, a mill, a token count or a life total would be dropped as an
     * identity pair, whatever the replacement actually did.
     */
    @Test
    void aRewrittenQuantityOnTheReplacementSideIsNotAnIdentityPair() {
        Card card = TestCards.build("Mountain");
        Map<String, Object> before = new LinkedHashMap<>(Map.of(
                "Affected", card, "Number", 3));
        Map<String, Object> after = new LinkedHashMap<>(before);
        after.put("Number", 1);

        EffectRecord record = recording(tempDir).rewrites().record(
                replacementOfMode("DrawCards"), before, after,
                ReplacementResult.Updated, null);

        assertNotNull(record, "a replacement that halved a draw is a real rewrite");
        String json = record.toJson();
        assertTrue(json.contains("\"type\":\"card_drawn\""), json);
        assertTrue(json.contains("\"count\":3"), json);
        assertTrue(json.contains("\"count\":1"), json);
    }

    /**
     * The live path feeds the worker's tally: every replacement is a record,
     * the one that declined included, and only one that edited a parameter in
     * place counts as a rewrite.
     *
     * <p>Read off the summary the collector prints when it closes, which is the
     * number an operator reads. The tally is the worker's, so every test in
     * this JVM has already added to it, and it prints only every tenth game:
     * each step closes games until a summary prints and compares it with the
     * one before.
     */
    @Test
    void anUpdatedRewriteCountsTowardTheTallyAndADeclinedOneDoesNot() throws Throwable {
        Card card = TestCards.build("Mountain");
        Map<String, Object> before = new LinkedHashMap<>(Map.of(
                "Origin", ZoneType.Battlefield, "Destination", ZoneType.Graveyard,
                "Card", card));
        Map<String, Object> after = new LinkedHashMap<>(before);
        after.put("Destination", ZoneType.Exile);

        // A first record, so the tally has something to print at all.
        Tally start = tallyAfter(before, before, ReplacementResult.NotReplaced);
        Tally updated = tallyAfter(before, after, ReplacementResult.Updated);
        Tally declined = tallyAfter(before, before, ReplacementResult.NotReplaced);

        assertEquals(start.records() + 1, updated.records());
        assertEquals(start.rewritten() + 1, updated.rewritten(),
                "a destination moved in place is a rewrite");
        assertEquals(updated.records() + 1, declined.records(),
                "a replacement that declined is still a record");
        assertEquals(updated.rewritten(), declined.rewritten(),
                "a replacement that declined rewrote nothing");
    }

    /** The headline counts of one printed tally summary. */
    private record Tally(long records, long rewritten) {
    }

    private static final Pattern HEADLINE =
            Pattern.compile("(\\d+) records, (\\d+) rewrote a parameter in place");

    /** A summary prints on the worker's first game and every tenth after it. */
    private static final int GAMES_PER_SUMMARY = 10;

    /**
     * One game whose only replacement is the one described, then as many
     * games without one as it takes the tally to print, and what it printed.
     */
    private Tally tallyAfter(Map<String, Object> before, Map<String, Object> after,
            ReplacementResult result) throws Throwable {
        Channels game = recording(tempDir);
        game.rewrites().record(replacementOfMode("Moved"), before, after, result, null);
        String printed = StdStreams.captureStdout(game::close);
        for (int games = 1; printed.isEmpty() && games < GAMES_PER_SUMMARY; games++) {
            printed = StdStreams.captureStdout(() -> unwired(CollectionCaps.defaults()).close());
        }
        Matcher headline = HEADLINE.matcher(printed);
        assertTrue(headline.find(), "no tally summary in " + GAMES_PER_SUMMARY + " games: " + printed);
        return new Tally(Long.parseLong(headline.group(1)), Long.parseLong(headline.group(2)));
    }

    // ── plumbing ────────────────────────────────────────────────────────

    /**
     * Everything the handler wrote for one notification, as the shard line.
     *
     * <p>Driven through {@link RewriteChannel#handler} with a
     * synthesized argument array, which is as close to the engine's own call as
     * this side can get without a patched jar to proxy over.
     */
    private String oneRecordFrom(Object[] args, Class<?> shape) throws Throwable {
        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collector) ->
                collector.rewrites().handler().invoke(null, onReplacement(shape), args));
        assertEquals(1, lines.size(), "one replacement writes one record");
        assertEquals("rewrite", RecordLines.parse(lines.get(0)).get("kind"), lines.get(0));
        return lines.get(0);
    }
}
