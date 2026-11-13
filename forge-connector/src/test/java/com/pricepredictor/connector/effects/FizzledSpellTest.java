package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.GameActionUtil;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.outcomeOf;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * A fizzled item's effect half (608.2b): a spell's holds exactly its own
 * card's move off the stack, wherever the card goes; an ability's holds
 * nothing. The move is that half's and no other record's, and is not
 * counted as dropped.
 */
@ExtendWith(ForgeExtension.class)
class FizzledSpellTest {

    private static final String WHY = "for the resolution hook's fizzle";

    @TempDir
    Path tempDir;

    /** What a fizzle run wrote: its records and its dropped-event lines. */
    private record Run(List<String> records, List<String> dropped) {
    }

    private Run played(Game game, ShardRun.Played body) throws Throwable {
        CollectionCaps counting = TestCaps.caps().interventions(0).probes(0)
                .countDroppedEvents().build();
        RecordShardWriter writer = ShardRun.recordedWithClauseHook(
                tempDir, game, counting, WHY, body);
        Path counts = GameRecording.droppedEventsFile(writer);
        return new Run(ShardRun.readShard(writer.path()),
                Files.exists(counts) ? Files.readAllLines(counts) : List.of());
    }

    /** The events of every record that move {@code card} off the stack. */
    private static long movesOffTheStack(List<String> records, Card card) {
        return records.stream().flatMap(record -> events(record).stream())
                .filter(event -> "zone_change".equals(event.get("type"))
                        && List.of(Refs.of(card)).equals(event.get("subjects"))
                        && "stack".equals(params(event).get("from_zone")))
                .count();
    }

    private static void resolveAndMoveOn(Game game) {
        game.getStack().resolveStack();
        TestBoard.priority(game);
        TestBoard.nextStep(game);
    }

    @Test
    void aFizzledShocksHalfHoldsItsMoveToTheGraveyardAlone() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        Card shock = TestBoard.inHand(game, caster, "Shock");

        Card second = TestBoard.inHand(game, caster, "Shock");

        Run run = played(game, () -> {
            TestBoard.cast(game, shock, caster, bears);
            // A move by hand at priority, which no record takes: dropped.
            game.getAction().exile(bears, null, null);
            resolveAndMoveOn(game);
            // The next cast, which discards a move still unfiled.
            TestBoard.cast(game, second, caster, game.getPlayers().get(1));
            resolveAndMoveOn(game);
        });

        String half = RecordLines.records(run.records(), "resolution").stream()
                .filter(record -> "fizzled".equals(RecordLines.payload(record).get("outcome")))
                .findFirst().orElseThrow(() -> new AssertionError(run.records().toString()));
        assertEquals("fizzled", outcomeOf(half), half);
        List<Map<String, Object>> events = events(half);
        assertEquals(1, events.size(), half);
        assertEquals("zone_change", events.get(0).get("type"), half);
        assertEquals(List.of(Refs.of(shock)), events.get(0).get("subjects"), half);
        assertEquals("graveyard", params(events.get(0)).get("to_zone"), half);
        assertEquals(1, movesOffTheStack(run.records(), shock),
                "the move is in one record: " + run.records());
        assertEquals(1, run.dropped().stream().filter(line -> line.startsWith("zone_change\t"))
                        .mapToInt(line -> Integer.parseInt(line.substring(line.lastIndexOf('\t') + 1)))
                        .sum(),
                "only the exile made by hand was dropped: " + run.dropped());
    }

    /** A fizzled flashback spell is exiled: its half holds that move. */
    @Test
    void aFizzledFlashbackSpellsHalfHoldsItsExile() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        Card firebolt = TestBoard.inGraveyard(game, caster, "Firebolt");
        SpellAbility flashback = GameActionUtil.getAlternativeCosts(
                        TestBoard.spellOf(firebolt, caster), caster, false).stream()
                .filter(SpellAbility::isFlashback).findFirst()
                .orElseThrow(() -> new AssertionError("no flashback cast"));
        flashback.setActivatingPlayer(caster);

        Run run = played(game, () -> {
            Card onStack = game.getAction().moveToStack(firebolt, flashback);
            flashback.setHostCard(onStack);
            flashback.resetTargets();
            flashback.getTargets().add(bears);
            game.getStack().add(flashback);
            game.getAction().exile(bears, null, null);
            resolveAndMoveOn(game);
        });

        String half = RecordLines.only(run.records(), "resolution", "cardsfolder/f/firebolt.txt");
        assertEquals("fizzled", outcomeOf(half), half);
        List<Map<String, Object>> events = events(half);
        assertEquals(1, events.size(), half);
        assertEquals("exile", params(events.get(0)).get("to_zone"), half);
    }

    /** A fizzled ability has no card to move: its half holds nothing. */
    @Test
    void aFizzledAbilitysHalfHoldsNothing() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card sorcerer = TestBoard.inPlay(game, caster, "Prodigal Sorcerer");
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        SpellAbility ping = TestBoard.ability(sorcerer, ApiType.DealDamage);
        ping.setActivatingPlayer(caster);

        Run run = played(game, () -> {
            ping.resetTargets();
            ping.getTargets().add(bears);
            game.getStack().add(ping);
            game.getAction().exile(bears, null, null);
            resolveAndMoveOn(game);
        });

        List<String> halves = RecordLines.records(run.records(), "resolution").stream()
                .filter(record -> "cardsfolder/p/prodigal_sorcerer.txt".equals(RecordLines.scriptOf(record)))
                .toList();
        assertEquals(1, halves.size(), run.records().toString());
        assertEquals("fizzled", outcomeOf(halves.get(0)), halves.get(0));
        assertFalse(events(halves.get(0)).iterator().hasNext(), halves.get(0));
    }
}
