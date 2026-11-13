package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementResult;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.TestCollectors.replacementOfMode;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * A damage replacement's buffered ability (WP2's hooks, for WP7): its rewrite
 * record names it in {@code buffered}, never in {@code replaced_by}, waits for
 * its run, and takes the events published between
 * {@code onBufferedAbilityStarting} and {@code onBufferedAbilityFinished};
 * one never run says so. And a null {@code before} is unknown, never diffed.
 */
@ExtendWith(ForgeExtension.class)
class BufferedRewriteTest {

    private static final Method ON_REPLACEMENT =
            HookShapes.method(HookShapes.Replacement.class, "onReplacement");
    private static final Method STARTING =
            HookShapes.method(HookShapes.Replacement.class, "onBufferedAbilityStarting");
    private static final Method FINISHED =
            HookShapes.method(HookShapes.Replacement.class, "onBufferedAbilityFinished");

    @TempDir
    Path tempDir;

    private static Map<String, Object> damage(Card target) {
        return new LinkedHashMap<>(Map.of("DamageAmount", 2, "Affected", target));
    }

    private static SpellAbility draw() {
        return AbilityFactory.getAbility("DB$ Draw | Defined$ You", TestCards.build("Swans of Bryn Argoll"));
    }

    private static EffectEvent drawn() {
        return new EffectEvent(EventType.CARD_DRAWN).subject("P1").param("count", 1);
    }

    private static Map<String, Object> rewrite(String line) {
        assertEquals("rewrite", RecordLines.parse(line).get("kind"), line);
        return RecordLines.payload(line);
    }

    @Test
    void aBufferedAbilitysRunIsItsRecordsEvents() throws Throwable {
        ReplacementEffect prevention = replacementOfMode("DamageDone");
        SpellAbility buffered = draw();
        Card target = TestCards.build("Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            var handler = collectors.rewrites().handler();
            handler.invoke(null, ON_REPLACEMENT, new Object[]{
                    prevention, damage(target), damage(target), ReplacementResult.Prevented,
                    null, buffered});
            handler.invoke(null, STARTING, new Object[]{prevention, buffered, Map.of()});
            bracket.recordClauseEvent(drawn());
            handler.invoke(null, FINISHED, new Object[]{prevention, buffered, Map.of()});
        });

        assertEquals(1, lines.size(), "the draw went to the rewrite record and nowhere else: " + lines);
        Map<String, Object> payload = rewrite(lines.get(0));
        assertEquals(List.of(), payload.get("replaced_by"), "a buffered ability never stood in");
        assertEquals(List.of(), payload.get("buffered"), "named, keyed to no printed line here");
        assertEquals(true, payload.get("buffered_ran"));
        List<?> events = (List<?>) payload.get("buffered_events");
        assertEquals(1, events.size(), lines.get(0));
        assertEquals("card_drawn", RecordLines.asMap(events.get(0)).get("type"));
    }

    @Test
    void severalReplacementsSummedIntoOneRunGiveItsEventsToTheFirst() throws Throwable {
        ReplacementEffect prevention = replacementOfMode("DamageDone");
        SpellAbility buffered = draw();

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            var handler = collectors.rewrites().handler();
            for (String name : List.of("Grizzly Bears", "Hill Giant")) {
                Card target = TestCards.build(name);
                handler.invoke(null, ON_REPLACEMENT, new Object[]{
                        prevention, damage(target), damage(target), ReplacementResult.Prevented,
                        null, buffered});
            }
            handler.invoke(null, STARTING, new Object[]{prevention, buffered, Map.of()});
            bracket.recordClauseEvent(drawn());
            handler.invoke(null, FINISHED, new Object[]{prevention, buffered, Map.of()});
        });

        assertEquals(2, lines.size(), lines.toString());
        List<Integer> taken = lines.stream()
                .map(line -> ((List<?>) rewrite(line).get("buffered_events")).size()).toList();
        assertEquals(List.of(1, 0), taken, "the oldest takes the run's events");
        lines.forEach(line -> assertEquals(true, rewrite(line).get("buffered_ran"), line));
    }

    @Test
    void aBufferedAbilityThatNeverRanSaysSo() throws Throwable {
        ReplacementEffect prevention = replacementOfMode("DamageDone");
        Card target = TestCards.build("Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.rewrites().handler().invoke(null, ON_REPLACEMENT, new Object[]{
                        prevention, damage(target), damage(target), ReplacementResult.Prevented,
                        null, draw()}));

        assertEquals(1, lines.size(), "written at the game's end: " + lines);
        assertEquals(false, rewrite(lines.get(0)).get("buffered_ran"));
        assertEquals(List.of(), rewrite(lines.get(0)).get("buffered_events"));
    }

    @Test
    void anEventOutsideARunStaysWithTheBracket() throws Throwable {
        ReplacementEffect prevention = replacementOfMode("DamageDone");
        Card target = TestCards.build("Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            collectors.rewrites().handler().invoke(null, ON_REPLACEMENT, new Object[]{
                    prevention, damage(target), damage(target), ReplacementResult.Prevented,
                    null, draw()});
            bracket.recordClauseEvent(drawn());
        });

        lines.stream().filter(line -> line.contains("\"kind\":\"rewrite\"")).forEach(line ->
                assertEquals(List.of(), rewrite(line).get("buffered_events"), line));
    }

    @Test
    void aReplacementWhoseEventCouldNotBeCopiedIsUnknownNotRewritten() throws Throwable {
        ReplacementEffect moved = replacementOfMode("Moved");
        Card card = TestCards.build("Mountain");
        Map<String, Object> after = new LinkedHashMap<>(Map.of(
                "Origin", ZoneType.Battlefield, "Destination", ZoneType.Exile, "Card", card));

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.rewrites().handler().invoke(null, ON_REPLACEMENT, new Object[]{
                        moved, null, after, ReplacementResult.Updated, null, null}));

        Map<String, Object> payload = rewrite(lines.get(0));
        assertEquals(true, payload.get("incoming_unknown"), lines.get(0));
        assertNull(payload.get("outgoing"), "nothing is diffed against an unknown event");
        assertEquals("zone_change", RecordLines.asMap(payload.get("incoming")).get("type"), lines.get(0));
    }
}
