package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
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

import static com.pricepredictor.connector.effects.TestParsed.replacementOfMode;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A damage replacement's buffered ability: its rewrite
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
        return new EffectEvent(EventType.CARD_DRAWN).subject("P1").param(EventParam.COUNT, 1);
    }

    private static Map<String, Object> rewrite(String line) {
        assertEquals("rewrite", RecordLines.parse(line).get("kind"), line);
        return RecordEvents.payload(line);
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

    /** A damage replacement naming a buffered draw that never runs, then the given bracket events. */
    private List<String> unrunBufferedReplacementThen(List<EffectEvent> events) throws Throwable {
        ReplacementEffect prevention = replacementOfMode("DamageDone");
        Card target = TestCards.build("Grizzly Bears");

        return ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            collectors.rewrites().handler().invoke(null, ON_REPLACEMENT, new Object[]{
                    prevention, damage(target), damage(target), ReplacementResult.Prevented,
                    null, draw()});
            events.forEach(bracket::recordClauseEvent);
        });
    }

    @Test
    void aBufferedAbilityThatNeverRanSaysSo() throws Throwable {
        List<String> lines = unrunBufferedReplacementThen(List.of());

        assertEquals(1, lines.size(), "written at the game's end: " + lines);
        assertEquals(false, rewrite(lines.get(0)).get("buffered_ran"));
        assertEquals(List.of(), rewrite(lines.get(0)).get("buffered_events"));
    }

    @Test
    void anEventOutsideARunStaysWithTheBracket() throws Throwable {
        List<String> lines = unrunBufferedReplacementThen(List.of(drawn()));

        lines.stream().filter(line -> line.contains("\"kind\":\"rewrite\"")).forEach(line ->
                assertEquals(List.of(), rewrite(line).get("buffered_events"), line));
    }

    // ── played through the engine ──────────────────────────────────────

    private static final String WHY = "for the replacement hook's buffered ability";

    private static List<String> rewrites(List<String> records) {
        return records.stream().filter(line -> line.contains("\"kind\":\"rewrite\"")).toList();
    }

    /**
     * Shock at Swans of Bryn Argoll, played: the damage is prevented, the
     * buffered draw runs, and the rewrite record takes the draws.
     */
    @Test
    void swansOfBrynArgollsDrawIsItsRewritesBufferedEvents() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        for (int i = 0; i < 3; i++) {
            game.getAction().moveToLibrary(TestBoard.inHand(game, you, "Island"), null, null);
        }
        Card swans = TestBoard.inPlay(game, opponent, "Swans of Bryn Argoll");
        Card shock = TestBoard.inHand(game, you, "Shock");
        int hand = you.getCardsIn(ZoneType.Hand).size();

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, shock, you, swans);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        assertEquals(hand - 1 + 2, you.getCardsIn(ZoneType.Hand).size(), "Shock cast, two drawn");
        List<String> rewrites = rewrites(records);
        assertEquals(1, rewrites.size(), records.toString());
        Map<String, Object> payload = rewrite(rewrites.get(0));
        assertEquals(List.of(), payload.get("replaced_by"));
        assertEquals(1, ((List<?>) payload.get("buffered")).size(), rewrites.get(0));
        assertEquals(true, payload.get("buffered_ran"));
        List<String> types = ((List<?>) payload.get("buffered_events")).stream()
                .map(event -> String.valueOf(RecordLines.asMap(event).get("type"))).toList();
        assertEquals(List.of("zone_change", "card_drawn", "zone_change", "card_drawn"), types,
                rewrites.get(0));
        for (String other : records) {
            if (!other.contains("\"kind\":\"rewrite\"")) {
                assertEquals(false, RecordEvents.eventTypes(other).contains("card_drawn"),
                        "the draws are the rewrite's alone: " + other);
            }
        }
    }

    /**
     * Two blockers' damage to Rock Hydra, played: two replacements of one
     * damage event, one buffered removal run. The first record takes the
     * removal's counter change, the second says it ran.
     */
    @Test
    void twoDamagesToRockHydraShareOneBufferedRun() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card hydra = TestBoard.inPlay(game, you, "Rock Hydra");
        hydra.addCounterInternal(forge.game.card.CounterEnumType.P1P1, 5, you, false, null,
                Map.of());
        hydra.setSickness(false);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card ogre = TestBoard.inPlay(game, opponent, "Gray Ogre");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getPhaseHandler().devModeSet(PhaseType.COMBAT_DAMAGE, you);
            forge.game.combat.Combat combat = new forge.game.combat.Combat(you);
            combat.addAttacker(hydra, opponent);
            combat.addBlocker(hydra, bears);
            combat.addBlocker(hydra, ogre);
            combat.setBlocked(hydra, true);
            combat.orderBlockersForDamageAssignment();
            combat.orderAttackersForDamageAssignment();
            game.getPhaseHandler().setCombat(combat);
            combat.assignCombatDamage(false);
            combat.dealAssignedDamage();
            game.getAction().checkStateEffects(false);
            game.getPhaseHandler().devModeSet(PhaseType.COMBAT_END, you);
        });

        assertEquals(1, hydra.getCounters(forge.game.card.CounterEnumType.P1P1),
                "four damage prevented, four counters removed");
        List<String> rewrites = rewrites(records);
        assertEquals(2, rewrites.size(), records.toString());
        List<Integer> taken = rewrites.stream()
                .map(line -> ((List<?>) rewrite(line).get("buffered_events")).size()).toList();
        assertEquals(List.of(1, 0), taken, "the oldest takes the run: " + rewrites);
        assertEquals(-4L, ((Number) RecordLines.params(RecordLines.asMap(
                ((List<?>) rewrite(rewrites.get(0)).get("buffered_events")).get(0)))
                .get("delta")).longValue(), "the summed removal");
        rewrites.forEach(line -> assertEquals(true, rewrite(line).get("buffered_ran"), line));
    }

    @Test
    void aReplacementWhoseEventCouldNotBeCopiedIsUnknownNotRewritten() throws Throwable {
        ReplacementEffect moved = replacementOfMode("Moved");
        Card card = TestCards.build("Mountain");
        Map<String, Object> after = new LinkedHashMap<>(Map.of(
                "Origin", ZoneType.Battlefield, "Destination", ZoneType.Exile, "Affected", card));

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.rewrites().handler().invoke(null, ON_REPLACEMENT, new Object[]{
                        moved, null, after, ReplacementResult.Updated, null, null}));

        Map<String, Object> payload = rewrite(lines.get(0));
        assertEquals(true, payload.get("incoming_unknown"), lines.get(0));
        assertNull(payload.get("outgoing"), "nothing is diffed against an unknown event");
        assertEquals("zone_change", RecordLines.asMap(payload.get("incoming")).get("type"), lines.get(0));
    }
}
