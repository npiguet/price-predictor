package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.DirectedRules;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEventGameFinished;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Predicate;

import static com.pricepredictor.connector.effects.RecordEvents.events;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * What the {@code DirectedForkPlayedForward*Test} classes share: one
 * registered rule played forward: a live game set
 * up as the rule's base, recorded by a {@code GameRecording} that takes the
 * rule's directed fork at the boundary ({@code DirectedFork.take}), the live
 * turn played out and the fork played to its end; the fork's records then
 * show the variant's point happening, not only the edited board. Each case
 * picks a rule whose point is the engine's own (a turn-based action, a
 * trigger, combat damage, the collector's own reading of a priority stop's
 * candidates), so neither seat's choices decide it. Run with
 * {@code -Ddirected.records.dump=DIR}, every case's records are copied to
 * {@code DIR} for tagging with the Python detectors.
 *
 * <p>Each class extends this one, which holds the temporary directory its
 * games' shards are written to.
 *
 * <p>{@code pt} is played forward by
 * {@code ForkRecordsTest.theRegisteredToughnessZeroRuleForksATougherCreatureToTheEndOfTheTurn}.
 * The families left out have no registered rule whose point happens
 * without a seat's choice ({@code tap}: a spell cast, an attack, a block),
 * or no registered rule at all ({@code keyword}, {@code colour_type}).
 */
abstract class DirectedForkPlayedForwardSupport {

    @TempDir
    Path tempDir;

    /** A started two-seat game in {@code step} of the first seat's third turn. */
    static Game board(PhaseType step) {
        Game game = ForkPlayForwardTest.board(TestBoard.twoPlayerGame(PhaseType.MAIN1));
        game.getPhaseHandler().devModeSet(step, game.getPlayers().get(0), 3);
        return game;
    }

    /** A permanent since the turn began, with a timestamp of its own. */
    static Card permanent(Game game, Player owner, String name) {
        Card card = TestBoard.inPlay(game, owner, name);
        card.setGameTimestamp(game.getNextTimestamp());
        card.setSickness(false);
        return card;
    }

    /**
     * Record the game from the boundary beginning {@code step}, taking
     * {@code rule}'s directed fork there, and play the live turn out; the
     * fork's records, each checked to name the rule's registration.
     */
    List<String> forkOf(Game game, PhaseType step, String rule, String variant) throws Throwable {
        return forkOf(played(game, step, rule, Decisions.SAMPLED), rule, variant);
    }

    /**
     * Every record of the game recorded from the boundary beginning
     * {@code step}, {@code rule}'s directed fork taken there, and the live
     * turn played out; with {@code step} null, the live game reaches the
     * boundary itself, its turn-based actions included. A live game the turn
     * ended writes its end, as {@code Match.startGame} has it do.
     */
    List<String> played(Game game, PhaseType step, String rule, Decisions decisions) throws Throwable {
        TestCaps built = TestCaps.caps().manaCap(2000).interventions(0).probes(0);
        built = switch (decisions) {
            case SAMPLED -> built;
            case NONE -> built.playabilityRate(0.0).candidateRates(new TestCaps.CandidateRates(0.0, 0.0));
            case EVERY -> built.everyDecision();
        };
        CollectionCaps caps = built.directed(1, 1.0, rule).build();
        // A directory of its own per game: one test may play several.
        RecordShardWriter shard = ShardRun.writer(
                java.nio.file.Files.createDirectories(tempDir.resolve("game-" + DUMPED.incrementAndGet())));
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(game, shard, ShardRun.GAME_ID, caps)
                        .seeded(7L))) {
            recording.open();
            game.getTriggerHandler().resetActiveTriggers();
            if (step != null) {
                TestBoard.enterStep(game, step);
            }
            ForkPlayForwardTest.playTheTurnOut(game);
            if (game.isGameOver()) {
                game.fireEvent(new GameEventGameFinished());
            }
        } finally {
            shard.close();
        }
        List<String> records = ShardRun.readShard(shard.path());
        dump(rule, records);
        return records;
    }

    /** How a case's decision candidates are sampled, live and in the fork alike. */
    enum Decisions {
        /** At the defaults' rates. */
        SAMPLED,
        /** At rate 0: only the candidates a fork keeps (FR-010c) are written. */
        NONE,
        /** At rate 1: every candidate is written. */
        EVERY
    }

    /**
     * Where each case's records are copied when the property
     * {@code directed.records.dump} names a directory: one
     * {@code <rule>-<n>.jsonl} per game played, for tagging with the Python
     * detectors (the brief's check that both sides of a rule are tagged).
     */
    static final String DUMP = System.getProperty("directed.records.dump");
    private static final java.util.concurrent.atomic.AtomicInteger DUMPED = new java.util.concurrent.atomic.AtomicInteger();

    private static void dump(String rule, List<String> records) throws java.io.IOException {
        if (DUMP == null) {
            return;
        }
        Path dir = Path.of(DUMP);
        java.nio.file.Files.createDirectories(dir);
        java.nio.file.Files.write(dir.resolve(rule + "-" + DUMPED.incrementAndGet() + ".jsonl"), records,
                java.nio.charset.StandardCharsets.UTF_8);
    }

    /**
     * The fork's records among {@code all}, each checked to name the variant
     * and, in {@code aimed_at}, every rule of the rule's registration.
     */
    static List<String> forkOf(List<String> all, String rule, String variant) {
        List<String> forked = new ArrayList<>();
        for (String record : all) {
            Map<String, Object> parsed = parse(record);
            if (Boolean.TRUE.equals(parsed.get("fork"))) {
                assertEquals(DirectedRules.registered().find(rule).orElseThrow().rules(), parsed.get("aimed_at"),
                        record);
                assertEquals(variant, RecordLines.asMap(parsed.get("directed")).get("variant"), record);
                forked.add(record);
            }
        }
        assertFalse(forked.isEmpty(), rule + ": the fork was kept and recorded");
        return forked;
    }

    /** Every event of {@code records} of {@code type} naming {@code ref}. */
    static List<Map<String, Object>> eventsOf(List<String> records, String type, String ref) {
        List<Map<String, Object>> found = new ArrayList<>();
        for (String record : records) {
            for (Map<String, Object> event : events(record)) {
                if (type.equals(event.get("type")) && subjectsOf(event).contains(ref)) {
                    found.add(event);
                }
            }
        }
        return found;
    }

    /** The event's {@code subjects}, or none when it carries no list of them. */
    private static List<?> subjectsOf(Map<String, Object> event) {
        return event.get("subjects") instanceof List<?> subjects ? subjects : Collections.emptyList();
    }

    static String ref(Card card) {
        return "E" + card.getId();
    }

    /** A started two-seat game in {@code step} of the first seat's third turn, both libraries ten Forests. */
    static Game bare(PhaseType step) {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        game.getPhaseHandler().devModeSet(step, active, 3);
        for (Player player : game.getPlayers()) {
            for (int i = 0; i < 10; i++) {
                game.getAction().moveToLibrary(TestBoard.card(game, player, "Forest"), null, null);
            }
        }
        return game;
    }

    static Card lands(Game game, Player owner, String name, int count) {
        Card first = null;
        for (int i = 0; i < count; i++) {
            Card land = permanent(game, owner, name);
            first = first == null ? land : first;
        }
        return first;
    }

    /** The fired trigger records among {@code records} whose trigger's host is {@code host}. */
    static List<String> fired(List<String> records, Card host) {
        return sourcedBy(records, SourcedKind.TRIGGER, host, firedIs(Boolean.TRUE));
    }

    /** The not-fired trigger records among {@code records} whose trigger's host is {@code host}. */
    static List<String> notFired(List<String> records, Card host) {
        return sourcedBy(records, SourcedKind.TRIGGER, host, firedIs(Boolean.FALSE));
    }

    /** The fork's resolution records whose source is {@code card}. */
    static List<String> resolutionsOf(List<String> fork, Card card) {
        return sourcedBy(fork, SourcedKind.RESOLUTION, card, record -> true);
    }

    /** Keeps a trigger record whose payload's {@code fired} is {@code fired}. */
    private static Predicate<String> firedIs(Boolean fired) {
        return record -> fired.equals(RecordEvents.payload(record).get("fired"));
    }

    /** The kinds of record {@link #sourcedBy} selects by their source: each its wire name, lower-cased. */
    enum SourcedKind {
        TRIGGER,
        RESOLUTION;

        boolean isKindOf(Map<String, Object> parsed) {
            return name().toLowerCase(Locale.ROOT).equals(parsed.get("kind"));
        }
    }

    /**
     * The records among {@code records} of {@code kind} whose source is
     * {@code card} and that {@code kept} keeps, checked in that order.
     */
    private static List<String> sourcedBy(List<String> records, SourcedKind kind, Card card,
                                          Predicate<String> kept) {
        return records.stream().filter(record -> isSourcedBy(parse(record), kind, card)).filter(kept).toList();
    }

    /** Whether a parsed record is of {@code kind} and its state's source is {@code card}. */
    private static boolean isSourcedBy(Map<String, Object> parsed, SourcedKind kind, Card card) {
        Object source = RecordLines.asMap(RecordLines.asMap(parsed.get("state")).get("refs")).get("source");
        return kind.isKindOf(parsed) && ref(card).equals(source);
    }

    /** The fork's records among {@code records} holding an event of {@code type}, as kind/moment. */
    static List<String> holding(List<String> records, String type) {
        List<String> found = new ArrayList<>();
        for (String record : records) {
            if (RecordEvents.eventTypes(record).contains(type)) {
                found.add(parse(record).get("kind") + "/" + parse(record).get("moment"));
            }
        }
        return found;
    }

    /** A started two-seat game in {@code step} of the first seat's third turn, both libraries empty. */
    static Game empty(PhaseType step) {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        game.getPhaseHandler().devModeSet(step, game.getPlayers().get(0), 3);
        return game;
    }

    /** {@code names} at the bottom of {@code owner}'s library in order. */
    static List<Card> library(Game game, Player owner, String... names) {
        List<Card> cards = new ArrayList<>();
        for (String name : names) {
            cards.add(game.getAction().moveToLibrary(TestBoard.card(game, owner, name), -1, null, null));
        }
        return cards;
    }
}
